# -*- coding: utf-8 -*-
import sqlite3
import os
import logging

logger = logging.getLogger("Database")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "radar_hub.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def preprocess_for_fts(text: str) -> str:
    """
    为 FTS5 搜索引擎对中文和英文混合文本进行预处理：
    在每个中文字符前后插入空格，保证英文单词/数字和中文字符被完全分离开来，
    且每个中文字符都成为独立的 Token，从而绕过 FTS5 无分词器无法检索中文的问题。
    """
    if not text:
        return ""
    processed = []
    for char in text:
        # 判断是否在 CJK 统一汉字区间
        if '\u4e00' <= char <= '\u9fff':
            processed.append(f" {char} ")
        else:
            processed.append(char)
    temp = "".join(processed)
    # 合并多余的空格，去除首尾空格
    return " ".join(temp.split())

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn

def init_db():
    """初始化本地关系数据库架构，同时保留向后兼容性，并创建 v2 核心表与 FTS5 虚拟索引与触发器"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # === 1. 保留老的 V1 结构以供平滑迁移或兼容读取 ===
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT,
            venue TEXT,
            year INTEGER,
            citations INTEGER DEFAULT 0,
            abstract TEXT,
            pdf_path TEXT,
            source_engine TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_summaries (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT UNIQUE,
            model_name TEXT,
            dialectical_analysis TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id) REFERENCES papers (paper_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_archives (
            archive_id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            results_json TEXT NOT NULL,
            archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # === 2. 部署全新的 V2 一体化多模态知识湖结构 ===
    # 2.1 文档元数据总表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL CHECK(source_type IN ('arxiv_paper', 'local_pdf', 'ext_blog', 'daily_brief', 'weekly_insight')),
            title TEXT NOT NULL,
            authors TEXT,
            canonical_url TEXT NOT NULL UNIQUE,
            local_path TEXT,
            content_hash TEXT NOT NULL,
            origin_provider TEXT NOT NULL CHECK(origin_provider IN ('arxiv', 'manual', 'local_fs', 'publisher_site')),
            discovery_provider TEXT NOT NULL CHECK(discovery_provider IN ('arxiv_api', 'exa', 'gemini_grounding', 'manual')),
            crawl_provider TEXT NOT NULL CHECK(crawl_provider IN ('requests', 'firecrawl', 'browser', 'native')),
            analysis_model TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'ingested', 'failed')),
            llm_score REAL CHECK(llm_score >= 0.0 AND llm_score <= 10.0),
            score_reason_json TEXT,
            scored_by_model TEXT,
            scored_at TIMESTAMP,
            published_at TIMESTAMP NOT NULL,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            license_hint TEXT
        )
    ''')
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_published_at ON documents(published_at);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_source_type_score ON documents(source_type, llm_score);")


    # 2.2 细粒度物理切片全文搜索物化表 (search_chunks)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_chunks (
            search_chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            section_path TEXT,
            body TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_chunks_chunk_id ON search_chunks(chunk_id);")

    # 2.3 FTS5 虚拟索引表 (unified_knowledge_fts)
    # 注意: 如果已创建了虚拟表，这里会跳过。
    try:
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS unified_knowledge_fts USING fts5(
                title,
                section_path,
                body,
                chunk_id UNINDEXED,
                doc_id UNINDEXED,
                content='search_chunks',
                content_rowid='search_chunk_id'
            );
        ''')
    except sqlite3.OperationalError as e:
        logger.warning(f"虚拟表创建或已存在: {e}")

    # 2.4 FTS5 触发器同步逻辑 (同步 search_chunks ➔ unified_knowledge_fts)
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS t_fts_chunk_ai AFTER INSERT ON search_chunks BEGIN
          INSERT INTO unified_knowledge_fts(rowid, title, section_path, body, chunk_id, doc_id) 
          VALUES (new.search_chunk_id, new.title, new.section_path, new.body, new.chunk_id, new.doc_id);
        END;
    ''')

    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS t_fts_chunk_ad AFTER DELETE ON search_chunks BEGIN
          INSERT INTO unified_knowledge_fts(unified_knowledge_fts, rowid, title, section_path, body, chunk_id, doc_id) 
          VALUES('delete', old.search_chunk_id, old.title, old.section_path, old.body, old.chunk_id, old.doc_id);
        END;
    ''')

    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS t_fts_chunk_au AFTER UPDATE ON search_chunks BEGIN
          INSERT INTO unified_knowledge_fts(unified_knowledge_fts, rowid, title, section_path, body, chunk_id, doc_id) 
          VALUES('delete', old.search_chunk_id, old.title, old.section_path, old.body, old.chunk_id, old.doc_id);
          INSERT INTO unified_knowledge_fts(rowid, title, section_path, body, chunk_id, doc_id) 
          VALUES (new.search_chunk_id, new.title, new.section_path, new.body, new.chunk_id, new.doc_id);
        END;
    ''')

    # 2.5 全文与AI高密度资产表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_contents (
            doc_id TEXT PRIMARY KEY,
            full_text_markdown TEXT NOT NULL,
            ai_summary TEXT NOT NULL,
            structured_takeaways_json TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        )
    ''')

    # 2.6 长文本物理切片表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            section_path TEXT,
            page_number INTEGER,
            token_count INTEGER NOT NULL,
            text TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);")
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_doc_hash_index ON chunks(doc_id, chunk_index);")

    # 2.7 轻量级实体词典表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entity_lexicon (
            entity_id TEXT PRIMARY KEY,
            entity_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL UNIQUE,
            entity_type TEXT NOT NULL CHECK(entity_type IN ('model', 'infra', 'algorithm', 'org')),
            alias_json TEXT
        )
    ''')

    # 2.8 显式拓扑图关系边表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entity_relations (
            source_entity_id TEXT NOT NULL,
            target_entity_id TEXT NOT NULL,
            relation_type TEXT NOT NULL CHECK(relation_type IN ('component_of', 'optimized_by', 'competes_with', 'uses')),
            weight REAL DEFAULT 1.0,
            evidence_doc_id TEXT NOT NULL,
            evidence_chunk_id TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source_entity_id, target_entity_id, relation_type),
            FOREIGN KEY(source_entity_id) REFERENCES entity_lexicon(entity_id) ON DELETE CASCADE,
            FOREIGN KEY(target_entity_id) REFERENCES entity_lexicon(entity_id) ON DELETE CASCADE,
            FOREIGN KEY(evidence_doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        )
    ''')

    # 2.9 计费与配额审计明细账本
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS quota_ledger (
            ledger_id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_provider TEXT NOT NULL CHECK(api_provider IN ('exa', 'firecrawl', 'dashscope', 'google', 'deepseek')),
            model_name TEXT NOT NULL,
            api_metric TEXT NOT NULL,
            pricing_rule_id INTEGER NOT NULL,
            tokens_in INTEGER DEFAULT 0,
            tokens_out INTEGER DEFAULT 0,
            cache_hit_tokens INTEGER DEFAULT 0,
            credits_spent INTEGER DEFAULT 0,
            cost_usd REAL NOT NULL,
            request_payload_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2.10 真实商业模型动态计费规则表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS provider_pricing_rules (
            rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider_name TEXT NOT NULL,
            model_name TEXT NOT NULL,
            metric_key TEXT NOT NULL,
            unit_price_usd REAL NOT NULL,
            effective_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(provider_name, model_name, metric_key)
        )
    ''')

    # 2.11 分布式摄取事务异常追踪与灾难回滚补偿记录表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ingestion_errors (
            error_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            tx_id TEXT NOT NULL,
            step_failed TEXT NOT NULL,
            error_log TEXT NOT NULL,
            resolved_status INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()

    # === 3. 播种初始商业计费规则 (Seed Pricing Data) ===
    # 按照 Qwen, DeepSeek, Gemini, Firecrawl, Exa 的标准计费标准设置默认值
    seed_rules = [
        # DeepSeek V4 (Qwen/compatible)
        ('deepseek', 'deepseek-v4-flash', 'input_cache_hit', 0.0028), # 每百万
        ('deepseek', 'deepseek-v4-flash', 'input_cache_miss', 0.14),
        ('deepseek', 'deepseek-v4-flash', 'output', 0.28),
        # Firecrawl Scraper
        ('firecrawl', 'firecrawl-scraper', 'credit_cost', 0.0010),   # 1 credit = $0.001
        # Exa Search
        ('exa', 'exa-engine', 'search_request', 0.0070),              # 单次 search = $0.007
        ('exa', 'exa-engine', 'contents_request', 0.0010),            # 单页 content = $0.001
        ('exa', 'exa-engine', 'summary_request', 0.0020),
        # Google Gemini 2.5 Flash
        ('google', 'gemini-2.5-flash', 'input', 0.075),               # 每百万
        ('google', 'gemini-2.5-flash', 'output', 0.30),
        ('google', 'gemini-2.5-flash', 'grounding_prompt', 0.010),    # 单次联网 = $0.01
        # Alibaba DashScope (Qwen 3.7 Max)
        ('dashscope', 'qwen3.7-max', 'input', 0.012),                 # 每千
        ('dashscope', 'qwen3.7-max', 'output', 0.048),
    ]
    
    for provider, model, metric, price in seed_rules:
        cursor.execute('''
            INSERT OR IGNORE INTO provider_pricing_rules (provider_name, model_name, metric_key, unit_price_usd)
            VALUES (?, ?, ?, ?)
        ''', (provider, model, metric, price))
        
    conn.commit()
    conn.close()
    logger.info("数据库核心表及规则种子完成初始化。")

# === 4. 保留并重构 V1 的数据操作函数以保持系统稳定性 ===

def insert_paper(paper_data):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR IGNORE INTO papers (paper_id, title, authors, venue, year, citations, abstract, pdf_path, source_engine)
            VALUES (:paper_id, :title, :authors, :venue, :year, :citations, :abstract, :pdf_path, :source_engine)
        ''', paper_data)
        conn.commit()
    finally:
        conn.close()

def save_ai_summary(paper_id, model_name, analysis_text):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO ai_summaries (paper_id, model_name, dialectical_analysis)
            VALUES (?, ?, ?)
        ''', (paper_id, model_name, analysis_text))
        conn.commit()
    finally:
        conn.close()

def resolve_pdf_path(db_path):
    """将数据库中存储的 PDF 路径动态解析为当前运行环境下的有效路径，保证完美移植性"""
    if not db_path:
        return ""
    if os.path.exists(db_path):
        return db_path
        
    normalized = db_path.replace("\\", "/")
    if "storage/library/" in normalized:
        relative_part = normalized.split("storage/library/")[-1]
        local_path = os.path.join("storage", "library", relative_part)
        if os.path.exists(local_path):
            return local_path
            
    filename = os.path.basename(db_path)
    local_path = os.path.join("storage", "library", filename)
    if os.path.exists(local_path):
        return local_path
        
    return db_path

def insert_search_archive(archive_id, query, results_json):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO search_archives (archive_id, query, results_json)
            VALUES (?, ?, ?)
        ''', (archive_id, query, results_json))
        conn.commit()
    finally:
        conn.close()

def get_search_archives():
    conn = get_db_connection()
    try:
        rows = conn.execute('SELECT * FROM search_archives ORDER BY archived_at DESC').fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def delete_search_archive(archive_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM search_archives WHERE archive_id = ?', (archive_id,))
        conn.commit()
    finally:
        conn.close()

def delete_paper_metadata(paper_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Delete from dependencies first (ai_summaries has foreign key to papers)
        cursor.execute('DELETE FROM ai_summaries WHERE paper_id = ?', (paper_id,))
        # Then delete from papers table
        cursor.execute('DELETE FROM papers WHERE paper_id = ?', (paper_id,))
        conn.commit()
    finally:
        conn.close()