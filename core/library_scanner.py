# -*- coding: utf-8 -*-
import os
import glob
import hashlib
import logging
from datetime import datetime
from .database import get_db_connection, DB_PATH

logger = logging.getLogger("LibraryScanner")

LIBRARY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "library")
os.makedirs(LIBRARY_DIR, exist_ok=True)

def _preregister_doc_in_v2(conn, doc_id: str, title: str, pdf_path: str, source_type: str = "local_pdf"):
    """
    在 V2 documents 表中以 status='pending' 轻量预注册一篇文档。
    仅建立 ID 映射，不做切片/向量化（这些在 AI 解构完成后触发）。
    使用 INSERT OR IGNORE，保证幂等性。
    """
    abs_path = os.path.abspath(pdf_path)
    canonical_url = "file://" + abs_path.replace("\\", "/")
    content_hash = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO documents
                (doc_id, source_type, title, authors, canonical_url, local_path,
                 content_hash, origin_provider, discovery_provider, crawl_provider,
                 analysis_model, status, published_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'local_fs', 'manual', 'native',
                    'pending', 'pending', ?)
            """,
            (
                doc_id, source_type, title,
                "手动导入 (Local Import)",
                canonical_url, abs_path, content_hash,
                datetime.now().isoformat()
            )
        )
        logger.debug(f"V2 预注册完成 (pending): doc_id={doc_id}")
    except Exception as e:
        logger.warning(f"V2 预注册失败 (不影响主流程): doc_id={doc_id}, 错误: {e}")



def sync_local_library():
    """扫描 storage/library 目录中的所有 PDF，并将尚未在数据库注册的文献登记入库。
    同时在 V2 documents 表中以 status='pending' 预注册，等待后续 AI 解构触发完整 V2 摄取。
    """
    pdf_files = glob.glob(os.path.join(LIBRARY_DIR, "*.pdf"))
    conn = get_db_connection()
    cursor = conn.cursor()
    
    added_count = 0
    
    try:
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            rel_pdf_path = os.path.join("storage", "library", filename)
            # 校验路径是否已在数据库中 (为了极佳的移植性，兼容绝对路径与相对路径)
            cursor.execute("SELECT 1 FROM papers WHERE pdf_path = ? OR pdf_path LIKE ?", (rel_pdf_path, "%" + filename))
            if cursor.fetchone():
                continue
                
            # 如果不存在，解析文件名进行注册
            basename_no_ext = os.path.splitext(filename)[0]
            
            # 创建唯一的 paper_id (以文件名做 MD5)
            hasher = hashlib.md5()
            hasher.update(basename_no_ext.encode('utf-8'))
            paper_id = "manual_" + hasher.hexdigest()[:16]
            
            # 再校验一次以 paper_id 为主键是否存在
            cursor.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,))
            if cursor.fetchone():
                paper_id += "_alt"
            
            # 提取可读标题并去除前缀
            title = basename_no_ext
            if title.startswith("arxiv_"):
                title = title[6:]
                parts = title.split("_", 1)
                if len(parts) > 1:
                    title = parts[1]
            elif title.startswith("scholar_"):
                title = title[8:]
                parts = title.split("_", 1)
                if len(parts) > 1:
                    title = parts[1]
                    
            # 还原空格
            title = title.replace("_", " ").replace("-", " ").strip()
            if not title:
                title = basename_no_ext
                
            paper_data = {
                "paper_id": paper_id,
                "title": title,
                "authors": "手动导入 (Local Import)",
                "venue": "Manual",
                "year": datetime.now().year,
                "citations": 0,
                "abstract": "（此文献为手动放置于 storage/library 目录的本地 PDF，暂无学术图谱元数据。请激活右侧 AI 大脑对其进行辩证技术解构！）",
                "pdf_path": rel_pdf_path,
                "source_engine": "manual"
            }
            
            # 插入 V1 papers 表
            cursor.execute('''
                INSERT OR IGNORE INTO papers (paper_id, title, authors, venue, year, citations, abstract, pdf_path, source_engine)
                VALUES (:paper_id, :title, :authors, :venue, :year, :citations, :abstract, :pdf_path, :source_engine)
            ''', paper_data)

            # 在 V2 documents 表中以 pending 状态预注册（建立 ID 映射桥梁）
            _preregister_doc_in_v2(conn, paper_id, title, pdf_path, source_type="local_pdf")

            added_count += 1
            logger.info(f"新 PDF 已注册（V1+V2 pending）: {filename} -> {paper_id}")
            
        conn.commit()
    finally:
        conn.close()
        
    return added_count

def get_unanalyzed_papers():
    """获取所有在数据库中登记但尚未进行 AI 深度剖析的论文列表"""
    conn = get_db_connection()
    try:
        query = """
            SELECT p.paper_id, p.title, p.pdf_path
            FROM papers p
            LEFT JOIN ai_summaries s ON p.paper_id = s.paper_id
            WHERE s.dialectical_analysis IS NULL
        """
        rows = conn.execute(query).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_papers_pending_v2_ingestion():
    """
    获取所有已在 V1 papers 表中存在（且有 AI 解构报告），
    但尚未完成 V2 摄取（documents.status != 'ingested'）的论文列表。
    用于存量迁移脚本识别待处理目标。
    """
    conn = get_db_connection()
    try:
        query = """
            SELECT p.paper_id, p.title, p.pdf_path, p.authors,
                   s.dialectical_analysis,
                   COALESCE(d.status, 'not_registered') AS v2_status
            FROM papers p
            INNER JOIN ai_summaries s ON p.paper_id = s.paper_id
            LEFT JOIN documents d ON p.paper_id = d.doc_id
            WHERE s.dialectical_analysis IS NOT NULL
              AND s.dialectical_analysis != ''
              AND COALESCE(d.status, '') != 'ingested'
            ORDER BY p.created_at DESC
        """
        rows = conn.execute(query).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

