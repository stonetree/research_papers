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
    """如果 URL 是 arXiv 的抽象页面、HTML 页面或未加 .pdf 后缀的链接，自动转换成直接 PDF 链接"""
    url = url.strip()
    # 匹配各类 arXiv 格式：arxiv.org/abs/..., arxiv.org/html/..., arxiv.org/pdf/...
    # 例如：https://arxiv.org/html/2407.12820v1#S3 -> https://arxiv.org/pdf/2407.12820v1.pdf
    arxiv_pattern = re.compile(
        r'https?://(?:export\.)?arxiv\.org/(?:abs|html|pdf)/([0-9]+\.[0-9]+(?:v[0-9]+)?|[a-zA-Z\-]+/[0-9]+)',
        re.IGNORECASE
    )
    match = arxiv_pattern.search(url)
    if match:
        paper_id = match.group(1)
        return f"https://arxiv.org/pdf/{paper_id}.pdf"
    
    if "arxiv.org/abs/" in url:
        url = url.replace("arxiv.org/abs/", "arxiv.org/pdf/")
        if not url.endswith(".pdf"):
            url += ".pdf"
    elif "arxiv.org/html/" in url:
        url = url.replace("arxiv.org/html/", "arxiv.org/pdf/")
        if not url.endswith(".pdf"):
            url += ".pdf"
    elif "arxiv.org/pdf/" in url and not url.endswith(".pdf"):
        url += ".pdf"
    return url

def download_and_import_paper(paper_dict, model_id):
    """
    下载单篇论文，写入本地大仓并进行 AI 深度解构
    返回 (success, message, paper_id)
    """
    title = paper_dict.get("title", "").strip()
    url = paper_dict.get("url", "").strip()
    
    if not title:
        return False, "论文标题为空，无法导入。", None
    if not url:
        return False, f"《{title}》未提供可下载链接。", None
        
    # 自动转换 arXiv URL
    pdf_url = translate_arxiv_url(url)
    
    # 提取年份 and 会议/期刊
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
    
    # 去重检查：如果数据库中已存在同名的 paper_id，检查是否需要重新生成报告
    conn = get_db_connection()
    try:
        existing_paper = conn.execute("SELECT pdf_path FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        existing_summary = conn.execute("SELECT dialectical_analysis FROM ai_summaries WHERE paper_id = ?", (paper_id,)).fetchone()
        if existing_paper:
            # 如果本地已有文件并且已有有效解构报告（非报错信息），直接复用
            current_summary_text = existing_summary["dialectical_analysis"] if existing_summary else ""
            if current_summary_text and not current_summary_text.startswith("❌"):
                return True, f"《{title}》已存在于库中，已重新载入已有 AI 解构报告。", paper_id
            
            # 如果之前解构失败，尝试重新解析
            res = analyze_and_store_paper(paper_id, pdf_path_rel, title, model_id=model_id)
            if res.startswith("❌"):
                return False, f"论文已存在，但 AI 深度解构失败: {res}", paper_id
            return True, f"《{title}》已存在于库中，已重新生成 AI 解构报告。", paper_id
    finally:
        conn.close()
 
    # 开始下载物理 PDF
    print(f"📥 开始下载 PDF: {title} from {pdf_url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        res = requests.get(pdf_url, stream=True, timeout=45, headers=headers, allow_redirects=True)
        if res.status_code != 200:
            return False, f"下载 PDF 失败 (HTTP {res.status_code}): {pdf_url}", None
            
        content_bytes = bytearray()
        for chunk in res.iter_content(chunk_size=8192):
            if chunk:
                content_bytes.extend(chunk)

        # 检查是否为有效 PDF (以 %PDF- 开头)
        if not content_bytes.startswith(b'%PDF-'):
            text_prefix = content_bytes[:1000].decode("utf-8", errors="ignore").lower()
            if "<html" in text_prefix or "<!doctype html" in text_prefix or "arxiv" in text_prefix:
                print(f"⚠️ 下载内容为 HTML 网页并非原生 PDF (URL: {pdf_url})，正在自动补救提取 arXiv 官方原生 PDF...")
                arxiv_match = re.search(r'arxiv\.org/(?:abs|html|pdf)/([0-9]+\.[0-9]+(?:v[0-9]+)?)', pdf_url + " " + text_prefix)
                if arxiv_match:
                    repaired_pdf_url = f"https://arxiv.org/pdf/{arxiv_match.group(1)}.pdf"
                    print(f"🔄 补救重定向至 arXiv 原生 PDF: {repaired_pdf_url}")
                    res2 = requests.get(repaired_pdf_url, stream=True, timeout=45, headers=headers, allow_redirects=True)
                    if res2.status_code == 200:
                        content_bytes2 = bytearray()
                        for chunk in res2.iter_content(chunk_size=8192):
                            if chunk:
                                content_bytes2.extend(chunk)
                        if content_bytes2.startswith(b'%PDF-'):
                            content_bytes = content_bytes2
                            print(f"✅ 成功补救下载 arXiv 原生 PDF: {repaired_pdf_url}")

        with open(local_pdf_path, 'wb') as f:
            f.write(content_bytes)
    except Exception as e:
        return False, f"网络下载过程中出现异常: {e}", None
        
    # 写入 papers 数据库
    paper_data = {
        "paper_id": paper_id,
        "title": title,
        "authors": paper_dict.get("authors", "未知作者"),
        "venue": venue,
        "year": year,
        "citations": 0,
        "abstract": paper_dict.get("abstract") or paper_dict.get("summary", "暂无摘要描述。"),
        "pdf_path": pdf_path_rel,
        "source_engine": paper_dict.get("source_engine", "online_search_funnel")
    }
    
    try:
        insert_paper(paper_data)
    except Exception as e:
        pass
        
    # 并发/串行触发 AI 深度解构
    try:
        res = analyze_and_store_paper(paper_id, pdf_path_rel, title, model_id=model_id)
        if res.startswith("❌"):
            return True, f"《{title}》已成功下载入库，但大模型解构报告生成遇到异常: {res}", paper_id
        return True, f"《{title}》成功完成 AI 深度解构并沉淀入库！", paper_id
    except Exception as e:
        return True, f"《{title}》成功下载入库，但触发大模型解构时抛出异常: {e}", paper_id
