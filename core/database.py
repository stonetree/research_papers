# -*- coding: utf-8 -*-
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "radar_hub.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化本地关系数据库架构"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
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
    conn.commit()
    conn.close()

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
    """将数据库中存储的 PDF 路径（可能是其他机器上的绝对路径）动态解析为当前运行环境下的有效路径，保证完美移植性"""
    if not db_path:
        return ""
    # 1. 如果路径在当前环境下直接存在（无论相对还是绝对），直接返回
    if os.path.exists(db_path):
        return db_path
        
    # 2. 尝试从路径中分离出 storage/library 部分，并拼接为当前目录下的相对路径
    normalized = db_path.replace("\\", "/")
    if "storage/library/" in normalized:
        relative_part = normalized.split("storage/library/")[-1]
        local_path = os.path.join("storage", "library", relative_part)
        if os.path.exists(local_path):
            return local_path
            
    # 3. 兜底策略：直接在本地的 storage/library 下寻找同名文件
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