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
        CREATE TABLE IF NOT EXISTS paper_analysis_versions (
            version_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT NOT NULL,
            version_num INTEGER NOT NULL,
            model_name TEXT,
            dialectical_analysis TEXT NOT NULL,
            is_default INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id) REFERENCES papers (paper_id) ON DELETE CASCADE,
            UNIQUE(paper_id, version_num)
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_analysis_versions_paper_id ON paper_analysis_versions(paper_id);")

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

    # 2.12 联网搜索中断挂起与断点恢复任务表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS suspended_search_tasks (
            task_id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            filter_type TEXT NOT NULL DEFAULT 'arxiv_paper',
            allow_web_search INTEGER DEFAULT 1,
            force_penetrate INTEGER DEFAULT 0,
            model_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('suspended', 'running', 'completed', 'failed')),
            current_step TEXT NOT NULL,
            error_step TEXT,
            error_message TEXT,
            completed_steps_json TEXT NOT NULL DEFAULT '[]',
            remaining_steps_json TEXT NOT NULL DEFAULT '[]',
            intermediate_data_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_suspended_tasks_status ON suspended_search_tasks(status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_suspended_tasks_created ON suspended_search_tasks(created_at);")

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
    
    # 同步 V2 沉淀库与 V1 大仓，确保两端资产双向对齐
    sync_ingested_documents_to_papers()

def sync_ingested_documents_to_papers():
    """将 V2 documents 中已通过 2PC 沉淀的文献同步至 papers 与 ai_summaries 表（INSERT OR IGNORE），确保大仓视图完整统一"""
    conn = get_db_connection()
    try:
        conn.execute("""
            INSERT OR IGNORE INTO papers (paper_id, title, authors, venue, year, abstract, pdf_path, source_engine)
            SELECT 
                d.doc_id, 
                d.title, 
                COALESCE(d.authors, '未知作者'), 
                CASE WHEN d.doc_id LIKE 'arxiv%' OR d.canonical_url LIKE '%arxiv.org%' THEN 'arXiv' ELSE '网络文献大仓' END, 
                2025, 
                COALESCE(c.ai_summary, d.title), 
                d.local_path, 
                'v2_sync'
            FROM documents d
            LEFT JOIN document_contents c ON d.doc_id = c.doc_id
            WHERE d.status = 'ingested'
        """)
        conn.execute("""
            INSERT OR IGNORE INTO ai_summaries (paper_id, model_name, dialectical_analysis)
            SELECT 
                d.doc_id, 
                COALESCE(d.analysis_model, 'AI 知识大仓沉淀'), 
                COALESCE(c.full_text_markdown, c.ai_summary, '文档已通过 2PC 完整沉淀入库。')
            FROM documents d
            LEFT JOIN document_contents c ON d.doc_id = c.doc_id
            WHERE d.status = 'ingested'
        """)
        conn.commit()
    except Exception as e:
        logger.warning(f"同步 V2 documents 到 papers 失败: {e}")
    finally:
        conn.close()

    # 存量 ai_summaries 自动平滑升版为 paper_analysis_versions 的第 1 版
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO paper_analysis_versions (paper_id, version_num, model_name, dialectical_analysis, is_default, created_at)
            SELECT s.paper_id, 1, s.model_name, s.dialectical_analysis, 1, COALESCE(s.updated_at, CURRENT_TIMESTAMP)
            FROM ai_summaries s
            WHERE s.dialectical_analysis IS NOT NULL 
              AND s.dialectical_analysis != '' 
              AND s.dialectical_analysis NOT LIKE '❌%'
              AND NOT EXISTS (
                  SELECT 1 FROM paper_analysis_versions v WHERE v.paper_id = s.paper_id
              )
        ''')
        conn.commit()
    except Exception as e:
        logger.warning(f"存量 ai_summaries 升版至 paper_analysis_versions 失败: {e}")
    finally:
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

def get_paper_analysis_versions(paper_id):
    """获取指定论文的所有历史与当前 AI 解构版本列表（按版本号升序排列）"""
    conn = get_db_connection()
    try:
        rows = conn.execute('''
            SELECT version_id, paper_id, version_num, model_name, dialectical_analysis, is_default, created_at
            FROM paper_analysis_versions
            WHERE paper_id = ?
            ORDER BY version_num ASC
        ''', (paper_id,)).fetchall()
        
        versions = [dict(r) for r in rows]
        # 若版本表暂无记录但 ai_summaries 中有有效记录，自动补充为第 1 版
        if not versions:
            s_row = conn.execute('SELECT model_name, dialectical_analysis, updated_at FROM ai_summaries WHERE paper_id = ?', (paper_id,)).fetchone()
            if s_row and s_row["dialectical_analysis"] and not s_row["dialectical_analysis"].startswith("❌"):
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR IGNORE INTO paper_analysis_versions (paper_id, version_num, model_name, dialectical_analysis, is_default, created_at)
                    VALUES (?, 1, ?, ?, 1, ?)
                ''', (paper_id, s_row["model_name"], s_row["dialectical_analysis"], s_row["updated_at"]))
                conn.commit()
                rows = conn.execute('''
                    SELECT version_id, paper_id, version_num, model_name, dialectical_analysis, is_default, created_at
                    FROM paper_analysis_versions
                    WHERE paper_id = ?
                    ORDER BY version_num ASC
                ''', (paper_id,)).fetchall()
                versions = [dict(r) for r in rows]
        return versions
    finally:
        conn.close()

def save_paper_analysis_version(paper_id, model_name, analysis_text, set_as_default=True):
    """保存论文 AI 解构内容。成功解构时新增独立版本并完整保留所有历史版本；失败时不覆盖既有有效版本"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        is_error = not analysis_text or analysis_text.strip().startswith("❌")
        if not is_error:
            # 1. 计算递增版本号
            cursor.execute("SELECT IFNULL(MAX(version_num), 0) FROM paper_analysis_versions WHERE paper_id = ?", (paper_id,))
            max_v = cursor.fetchone()[0]
            new_v = max_v + 1
            
            # 2. 若设为默认展示，更新旧版本状态
            if set_as_default:
                cursor.execute("UPDATE paper_analysis_versions SET is_default = 0 WHERE paper_id = ?", (paper_id,))
                
            cursor.execute('''
                INSERT INTO paper_analysis_versions (paper_id, version_num, model_name, dialectical_analysis, is_default, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (paper_id, new_v, model_name, analysis_text, 1 if set_as_default else 0))
            
            # 3. 同步更新 ai_summaries 保证全局向后兼容
            if set_as_default:
                cursor.execute('''
                    INSERT OR REPLACE INTO ai_summaries (paper_id, model_name, dialectical_analysis, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (paper_id, model_name, analysis_text))
            conn.commit()
            return True, new_v
        else:
            # 解构失败：若此前无任何有效版本，则将错误信息暂存到 ai_summaries 便于 UI 提示
            cursor.execute("SELECT COUNT(*) FROM paper_analysis_versions WHERE paper_id = ? AND dialectical_analysis NOT LIKE '❌%'", (paper_id,))
            has_valid = cursor.fetchone()[0] > 0
            if not has_valid:
                cursor.execute('''
                    INSERT OR REPLACE INTO ai_summaries (paper_id, model_name, dialectical_analysis, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (paper_id, model_name, analysis_text))
            conn.commit()
            return False, 0
    finally:
        conn.close()

def set_paper_default_version(paper_id, version_num):
    """将指定的历史解构版本设为该文献的默认展示版本，并同步更新 ai_summaries"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # 1. 更新 paper_analysis_versions 中的 is_default 标识
        cursor.execute("UPDATE paper_analysis_versions SET is_default = 0 WHERE paper_id = ?", (paper_id,))
        cursor.execute("UPDATE paper_analysis_versions SET is_default = 1 WHERE paper_id = ? AND version_num = ?", (paper_id, version_num))
        
        # 2. 查询该版本具体内容并同步回 ai_summaries
        row = cursor.execute('''
            SELECT model_name, dialectical_analysis, created_at 
            FROM paper_analysis_versions 
            WHERE paper_id = ? AND version_num = ?
        ''', (paper_id, version_num)).fetchone()
        
        if row:
            cursor.execute('''
                INSERT OR REPLACE INTO ai_summaries (paper_id, model_name, dialectical_analysis, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (paper_id, row["model_name"], row["dialectical_analysis"], row["created_at"]))
        conn.commit()
    finally:
        conn.close()

def save_ai_summary(paper_id, model_name, analysis_text):
    """兼容旧接口：将解构报告保存为一个新版本并默认展示"""
    return save_paper_analysis_version(paper_id, model_name, analysis_text, set_as_default=True)

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
        # Delete from dependencies first (ai_summaries and paper_analysis_versions have foreign key to papers)
        cursor.execute('DELETE FROM paper_analysis_versions WHERE paper_id = ?', (paper_id,))
        cursor.execute('DELETE FROM ai_summaries WHERE paper_id = ?', (paper_id,))
        # Then delete from papers table
        cursor.execute('DELETE FROM papers WHERE paper_id = ?', (paper_id,))
        conn.commit()
    finally:
        conn.close()

# === 5. 挂起任务与断点恢复数据操作函数 ===

def insert_suspended_task(task_dict):
    """插入或更新挂起任务"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO suspended_search_tasks (
                task_id, query, filter_type, allow_web_search, force_penetrate,
                model_id, status, current_step, error_step, error_message,
                completed_steps_json, remaining_steps_json, intermediate_data_json,
                updated_at
            ) VALUES (
                :task_id, :query, :filter_type, :allow_web_search, :force_penetrate,
                :model_id, :status, :current_step, :error_step, :error_message,
                :completed_steps_json, :remaining_steps_json, :intermediate_data_json,
                CURRENT_TIMESTAMP
            )
        ''', task_dict)
        conn.commit()
    finally:
        conn.close()

def update_suspended_task_status(task_id, status, current_step=None, error_step=None, error_message=None, completed_steps_json=None, remaining_steps_json=None, intermediate_data_json=None):
    """更新挂起任务的状态和进度"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        updates = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        params = [status]
        if current_step is not None:
            updates.append("current_step = ?")
            params.append(current_step)
        if error_step is not None:
            updates.append("error_step = ?")
            params.append(error_step)
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        if completed_steps_json is not None:
            updates.append("completed_steps_json = ?")
            params.append(completed_steps_json)
        if remaining_steps_json is not None:
            updates.append("remaining_steps_json = ?")
            params.append(remaining_steps_json)
        if intermediate_data_json is not None:
            updates.append("intermediate_data_json = ?")
            params.append(intermediate_data_json)
        params.append(task_id)
        
        sql = f"UPDATE suspended_search_tasks SET {', '.join(updates)} WHERE task_id = ?"
        cursor.execute(sql, tuple(params))
        conn.commit()
    finally:
        conn.close()

def get_suspended_tasks(status=None):
    """获取所有挂起/未完全完成的联网搜索任务（按时间倒序）"""
    conn = get_db_connection()
    try:
        if status:
            rows = conn.execute(
                'SELECT * FROM suspended_search_tasks WHERE status = ? ORDER BY created_at DESC', 
                (status,)
            ).fetchall()
        else:
            rows = conn.execute(
                'SELECT * FROM suspended_search_tasks ORDER BY created_at DESC'
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def get_suspended_task_by_id(task_id):
    """根据 task_id 获取单个任务详情"""
    conn = get_db_connection()
    try:
        row = conn.execute('SELECT * FROM suspended_search_tasks WHERE task_id = ?', (task_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def delete_suspended_task(task_id):
    """删除指定的挂起任务记录"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM suspended_search_tasks WHERE task_id = ?', (task_id,))
        conn.commit()
    finally:
        conn.close()

def extract_arxiv_id(text):
    """从文本、URL 或 ID 中提取标准的 4位年份.4-5位序列号 arXiv ID"""
    if not text:
        return None
    import re
    m = re.search(r'(\d{4}\.\d{4,5})', str(text))
    return m.group(1) if m else None

def get_paper_deconstruction_status(title=None, doc_id=None, url=None):
    """
    检查某篇文献或报告在本地大仓中是否已经完成 AI 深度解构或 2PC 沉淀。
    多维度对账检索：
    1. 优先按 doc_id / paper_id 对账
    2. 按 URL / Canonical URL / 自动提取的 arXiv ID (如 2502.14051) 对账
    3. 按标题精确 / 核心前缀 / 规范化词频对账
    4. 支持跨 V1 (papers + ai_summaries) 与 V2 (documents + document_contents) 统一自动桥接
    返回 (is_deconstructed: bool, paper_id: str or None)
    """
    conn = get_db_connection()
    try:
        # 1. 提取可能包含的 arXiv ID (如 2502.14051)
        arx_id = extract_arxiv_id(url) or extract_arxiv_id(doc_id) or extract_arxiv_id(title)
        
        # 2. 第一路：检索 V1 papers + ai_summaries (已有全景解构报告)
        if doc_id:
            row = conn.execute("""
                SELECT p.paper_id, s.dialectical_analysis 
                FROM papers p 
                LEFT JOIN ai_summaries s ON p.paper_id = s.paper_id 
                WHERE p.paper_id = ?
            """, (doc_id,)).fetchone()
            if row and row["dialectical_analysis"] and not row["dialectical_analysis"].startswith("❌"):
                return True, row["paper_id"]

        if arx_id:
            row = conn.execute("""
                SELECT p.paper_id, s.dialectical_analysis 
                FROM papers p 
                LEFT JOIN ai_summaries s ON p.paper_id = s.paper_id 
                WHERE (p.paper_id LIKE ? OR p.pdf_path LIKE ? OR p.title LIKE ?)
                  AND s.dialectical_analysis IS NOT NULL 
                  AND s.dialectical_analysis NOT LIKE '❌%'
            """, (f"%{arx_id}%", f"%{arx_id}%", f"%{arx_id}%")).fetchone()
            if row and row["dialectical_analysis"] and not row["dialectical_analysis"].startswith("❌"):
                return True, row["paper_id"]

        if title:
            safe_t = title.strip()
            # 2.1 精确匹配
            row = conn.execute("""
                SELECT p.paper_id, s.dialectical_analysis 
                FROM papers p 
                LEFT JOIN ai_summaries s ON p.paper_id = s.paper_id 
                WHERE p.title = ?
            """, (safe_t,)).fetchone()
            if row and row["dialectical_analysis"] and not row["dialectical_analysis"].startswith("❌"):
                return True, row["paper_id"]
                
            # 2.2 核心前缀匹配 (如 "RocketKV: ...")
            core_prefix = safe_t.split(":")[0].strip() if ":" in safe_t else (safe_t[:30] if len(safe_t) >= 6 else safe_t)
            if len(core_prefix) >= 5:
                row = conn.execute("""
                    SELECT p.paper_id, s.dialectical_analysis 
                    FROM papers p 
                    LEFT JOIN ai_summaries s ON p.paper_id = s.paper_id 
                    WHERE (p.title LIKE ? OR p.title LIKE ?)
                      AND s.dialectical_analysis IS NOT NULL 
                      AND s.dialectical_analysis NOT LIKE '❌%'
                """, (f"{core_prefix}%", f"%{core_prefix}%")).fetchone()
                if row and row["dialectical_analysis"] and not row["dialectical_analysis"].startswith("❌"):
                    return True, row["paper_id"]

        # 3. 第二路：检索 V2 documents (已通过 2PC 沉淀入库)
        doc_row = None
        if doc_id:
            doc_row = conn.execute("""
                SELECT d.*, c.full_text_markdown, c.ai_summary 
                FROM documents d 
                LEFT JOIN document_contents c ON d.doc_id = c.doc_id 
                WHERE d.doc_id = ? AND d.status = 'ingested'
            """, (doc_id,)).fetchone()

        if not doc_row and arx_id:
            doc_row = conn.execute("""
                SELECT d.*, c.full_text_markdown, c.ai_summary 
                FROM documents d 
                LEFT JOIN document_contents c ON d.doc_id = c.doc_id 
                WHERE (d.canonical_url LIKE ? OR d.doc_id LIKE ? OR d.title LIKE ?) AND d.status = 'ingested'
            """, (f"%{arx_id}%", f"%{arx_id}%", f"%{arx_id}%")).fetchone()

        if not doc_row and url:
            doc_row = conn.execute("""
                SELECT d.*, c.full_text_markdown, c.ai_summary 
                FROM documents d 
                LEFT JOIN document_contents c ON d.doc_id = c.doc_id 
                WHERE d.canonical_url = ? AND d.status = 'ingested'
            """, (url,)).fetchone()

        if not doc_row and title:
            safe_t = title.strip()
            core_prefix = safe_t.split(":")[0].strip() if ":" in safe_t else (safe_t[:30] if len(safe_t) >= 6 else safe_t)
            doc_row = conn.execute("""
                SELECT d.*, c.full_text_markdown, c.ai_summary 
                FROM documents d 
                LEFT JOIN document_contents c ON d.doc_id = c.doc_id 
                WHERE (d.title = ? OR d.title LIKE ? OR d.title LIKE ?) AND d.status = 'ingested'
            """, (safe_t, f"{core_prefix}%", f"%{core_prefix}%")).fetchone()

        if doc_row:
            d_id = doc_row["doc_id"]
            d_title = doc_row["title"]
            d_authors = doc_row["authors"] or "未知作者"
            d_venue = "arXiv" if ("arxiv" in d_id.lower() or "arxiv.org" in str(doc_row["canonical_url"])) else "网络大仓沉淀"
            d_analysis = doc_row["full_text_markdown"] or doc_row["ai_summary"] or "文档已通过 2PC 完整沉淀入库。"
            d_model = doc_row["analysis_model"] or "AI 知识大仓沉淀"
            
            # 自动将 V2 沉淀文献桥接至 papers & ai_summaries，确保 Tab 1 本地大仓无缝可读
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO papers (paper_id, title, authors, venue, year, abstract, pdf_path, source_engine)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (d_id, d_title, d_authors, d_venue, 2025, d_analysis[:300], None, "v2_ingested_sync"))
                
                conn.execute("""
                    INSERT OR IGNORE INTO ai_summaries (paper_id, model_name, dialectical_analysis)
                    VALUES (?, ?, ?)
                """, (d_id, d_model, d_analysis))
                conn.commit()
            except Exception as bridge_e:
                logger.warning(f"桥接同步 V2 沉淀文献至 papers 失败: {bridge_e}")
                
            return True, d_id

        return False, None
    except Exception as e:
        logger.warning(f"检查解构状态异常: {e}")
        return False, None
    finally:
        conn.close()