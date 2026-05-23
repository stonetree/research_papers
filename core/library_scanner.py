# -*- coding: utf-8 -*-
import os
import glob
import hashlib
from datetime import datetime
from .database import get_db_connection

LIBRARY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "library")
os.makedirs(LIBRARY_DIR, exist_ok=True)

def sync_local_library():
    """扫描 storage/library 目录中的所有 PDF，并将尚未在数据库注册的文献登记入库"""
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
            
            # 插入数据库
            cursor.execute('''
                INSERT OR IGNORE INTO papers (paper_id, title, authors, venue, year, citations, abstract, pdf_path, source_engine)
                VALUES (:paper_id, :title, :authors, :venue, :year, :citations, :abstract, :pdf_path, :source_engine)
            ''', paper_data)
            added_count += 1
            
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
