# -*- coding: utf-8 -*-
import os
import re
import requests
import hashlib
from .database import insert_paper, get_db_connection
from .ai_analyst import analyze_and_store_paper

LIBRARY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "library")
os.makedirs(LIBRARY_DIR, exist_ok=True)

def translate_arxiv_url(url):
    """如果 URL 是 arXiv 的抽象页面，则自动转换成直接 PDF 链接"""
    url = url.strip()
    if "arxiv.org/abs/" in url:
        url = url.replace("arxiv.org/abs/", "arxiv.org/pdf/")
        if not url.endswith(".pdf"):
            url += ".pdf"
    elif "arxiv.org/pdf/" in url and not url.endswith(".pdf"):
        url += ".pdf"
    return url

def download_and_import_paper(paper_dict, model_id):
    """
    下载单篇论文，写入本地大仓并进行 AI 深度解构
    返回 (success, message)
    """
    title = paper_dict.get("title", "").strip()
    url = paper_dict.get("url", "").strip()
    
    if not title:
        return False, "论文标题为空，无法导入。"
    if not url:
        return False, f"《{title}》未提供可下载链接。"
        
    # 自动转换 arXiv URL
    pdf_url = translate_arxiv_url(url)
    
    # 提取年份和会议/期刊
    year_venue = paper_dict.get("year_venue", "").strip()
    year = None
    venue = "联网检索"
    
    # 试图从 year_venue 中提取 4 位数字作为年份
    year_match = re.search(r'\b(20\d{2}|19\d{2})\b', year_venue)
    if year_match:
        year = int(year_match.group(1))
        # 去掉年份后作为 venue
        venue = year_venue.replace(year_match.group(1), "").strip().strip("(),.-")
    else:
        venue = year_venue if year_venue else "联网检索"
        
    # 生成唯一的 paper_id (使用 title 的 MD5)
    paper_id = hashlib.md5(title.encode('utf-8')).hexdigest()[:8]
    
    # 去除文件名非法字符
    safe_title = re.sub(r'[\\/*?:"<>|]', "", title)[:60].strip()
    pdf_filename = f"model_search_{paper_id}_{safe_title}.pdf"
    local_pdf_path = os.path.join(LIBRARY_DIR, pdf_filename)
    pdf_path_rel = os.path.join("storage", "library", pdf_filename)
    
    # 去重检查：如果数据库中已存在同名的 paper_id，直接使用已有数据
    conn = get_db_connection()
    try:
        exists = conn.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        if exists:
            # 已经存在，尝试对其运行 AI 解析
            res = analyze_and_store_paper(paper_id, pdf_path_rel, title, model_id=model_id)
            if res.startswith("❌"):
                return False, f"论文已存在，但 AI 深度解构失败: {res}"
            return True, f"《{title}》已存在于库中，已重新生成 AI 解构报告。"
    finally:
        conn.close()

    # 开始下载物理 PDF
    print(f"📥 开始下载 PDF: {title} from {pdf_url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        res = requests.get(pdf_url, stream=True, timeout=45, headers=headers)
        if res.status_code != 200:
            return False, f"下载 PDF 失败 (HTTP {res.status_code}): {pdf_url}"
            
        with open(local_pdf_path, 'wb') as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        return False, f"网络下载过程中出现异常: {e}"
        
    # 写入 papers 数据库
    paper_data = {
        "paper_id": paper_id,
        "title": title,
        "authors": paper_dict.get("authors", "未知"),
        "venue": venue,
        "year": year,
        "citations": 0,
        "abstract": paper_dict.get("summary", "暂无摘要描述。"),
        "pdf_path": pdf_path_rel,
        "source_engine": "model_web_search"
    }
    
    try:
        insert_paper(paper_data)
    except Exception as e:
        return False, f"数据库写入失败: {e}"
        
    # 并发/串行触发 AI 首席科学家分析
    try:
        res = analyze_and_store_paper(paper_id, pdf_path_rel, title, model_id=model_id)
        if res.startswith("❌"):
            return True, f"《{title}》已成功物理下载入库，但大模型分析遇到异常: {res}"
        return True, f"《{title}》成功下载并完成 AI 首席科学家深度解构！"
    except Exception as e:
        return True, f"《{title}》成功下载入库，但触发大模型解构时抛出异常: {e}"
