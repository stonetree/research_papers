# -*- coding: utf-8 -*-
import os
import re
import arxiv
from datetime import datetime
from .database import get_db_connection, insert_paper

LIBRARY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "library")
os.makedirs(LIBRARY_DIR, exist_ok=True)

def execute_arxiv_search(query_string, limit=3):
    """从 arXiv 抓取最新的研究快讯并沉淀至本地"""
    search = arxiv.Search(
        query=query_string,
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    conn = get_db_connection()
    cursor = conn.cursor()
    new_papers = []
    
    try:
        client = arxiv.Client()
        for result in client.results(search):
            # 提取唯一 ID 作为主键
            paper_id = result.entry_id.split("/abs/")[-1].split("v")[0]
            
            # 数据库防重校验
            cursor.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,))
            if cursor.fetchone():
                print(f"⏭️  [arXiv 拦截] 文献已存在于知识库，跳过: {result.title}")
                continue
                
            safe_title = re.sub(r'[\\/*?:"<>|]', "", result.title)[:60].strip()
            pdf_filename = f"arxiv_{paper_id}_{safe_title}.pdf"
            local_pdf_path = os.path.join(LIBRARY_DIR, pdf_filename)
            pdf_path_rel = os.path.join("storage", "library", pdf_filename)
            
            print(f"📥 [arXiv 管道] 发现最新成果，正在物理落盘: {result.title}")
            
            pdf_url = result.pdf_url
            if not pdf_url:
                print(f"⚠️  [arXiv 跳过] 未找到 PDF 链接: {result.title}")
                continue
                
            import requests
            res = requests.get(pdf_url, stream=True, timeout=20)
            if res.status_code == 200:
                with open(local_pdf_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                print(f"❌  [arXiv 错误] 物理 PDF 下载失败 (HTTP {res.status_code}): {result.title}")
                continue
            
            authors_str = ", ".join([a.name for a in result.authors])
            paper_data = {
                "paper_id": paper_id,
                "title": result.title,
                "authors": authors_str,
                "venue": "arXiv",
                "year": result.published.year,
                "citations": 0,  # 新论文初设为 0
                "abstract": result.summary,
                "pdf_path": pdf_path_rel,
                "source_engine": "arxiv"
            }
            
            insert_paper(paper_data)
            new_papers.append(paper_data)
            
    finally:
        conn.close()
        
    return new_papers