# -*- coding: utf-8 -*-
import os
import re
import requests
import arxiv
from .database import get_db_connection, insert_paper
from .ai_analyst import arbitrate_papers

LIBRARY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "library")
os.makedirs(LIBRARY_DIR, exist_ok=True)

TARGET_VENUES = ["ASPLOS", "OSDI", "SOSP", "ISCA", "MICRO", "VLDB", "arXiv"]

def fetch_semantic_scholar_candidates(query_string, limit=15):
    """阶段 1 初审宽进：只抓取 Semantic Scholar 元数据，绝对不下载物理文件"""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query_string,
        "limit": limit * 2,
        "fields": "paperId,title,authors,venue,year,citationCount,openAccessPdf,abstract"
    }
    
    conn = get_db_connection()
    cursor = conn.cursor()
    candidates = []
    
    try:
        response = requests.get(url, params=params, timeout=10, headers={"Connection": "close"})
        if response.status_code != 200:
            print(f"⚠️ [Scholar Funnel Fetch 失败] HTTP {response.status_code}: {response.text}")
            return []
            
        papers = response.json().get("data", [])
        for p in papers:
            paper_id = p.get("paperId")
            if not paper_id or not p.get("openAccessPdf"):
                continue
                
            venue = p.get("venue", "")
            is_target_venue = any(tv.lower() in venue.lower() for tv in TARGET_VENUES) if venue else True
            if not is_target_venue:
                continue
                
            # 去重校验：如果数据库里已有该论文，则跳过
            cursor.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,))
            if cursor.fetchone():
                continue
                
            authors_str = ", ".join([a['name'] for a in p.get('authors', [])])
            candidates.append({
                "paper_id": paper_id,
                "title": p.get("title", ""),
                "authors": authors_str,
                "venue": p.get("venue", "arXiv"),
                "year": p.get("year"),
                "citations": p.get("citationCount", 0),
                "abstract": p.get("abstract", "暂无摘要描述。"),
                "pdf_url": p["openAccessPdf"]["url"],
                "source_engine": "semantic_scholar"
            })
            if len(candidates) >= limit:
                break
    except Exception as e:
        print(f"⚠️ [Scholar Funnel Fetch 异常]: {e}")
    finally:
        conn.close()
        
    return candidates

def fetch_arxiv_candidates(query_string, limit=15):
    """阶段 1 初审宽进：只抓取 ArXiv 元数据，绝对不下载物理文件"""
    search = arxiv.Search(
        query=query_string,
        max_results=limit * 2,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    conn = get_db_connection()
    cursor = conn.cursor()
    candidates = []
    
    try:
        client = arxiv.Client()
        for result in client.results(search):
            paper_id = result.entry_id.split("/abs/")[-1].split("v")[0]
            
            cursor.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,))
            if cursor.fetchone():
                continue
                
            pdf_url = result.pdf_url
            if not pdf_url:
                continue
                
            authors_str = ", ".join([a.name for a in result.authors])
            candidates.append({
                "paper_id": paper_id,
                "title": result.title,
                "authors": authors_str,
                "venue": "arXiv",
                "year": result.published.year,
                "citations": 0,
                "abstract": result.summary,
                "pdf_url": pdf_url,
                "source_engine": "arxiv"
            })
            if len(candidates) >= limit:
                break
    except Exception as e:
        print(f"⚠️ [ArXiv Funnel Fetch 异常]: {e}")
    finally:
        conn.close()
        
    return candidates

def execute_two_stage_funnel_search(topic_name, query_string, target_limit, model_id):
    """
    双阶段漏斗（Two-stage Filtering）检索管道：
    1. 【宽进阶段】：大量抓取学术元数据候选（Title & Abstract），但不做物理 PDF 下载；
    2. 【语义仲裁阶段】：让 AI 科学家大脑从第一性原理，快速过滤剔除水文；
    3. 【精确下载】：只精准下载大模型仲裁返回的黄金 Paper 列表；
    4. 【深度沉淀】：交给 AI 大脑进行多模态全景技术解构。
    """
    print(f"📡 [漏斗阶段 1：宽进] 开始大规模检索候选学术论文元数据...")
    # 大幅拉取 15 篇作为第一层漏斗候选池
    candidates = fetch_semantic_scholar_candidates(query_string, limit=15)
    source_engine = "Semantic Scholar"
    
    if not candidates:
        print(f"⚠️ [漏斗自动降级] Semantic Scholar 降级或未发现新文，正在切换至 ArXiv 宽进抓取...")
        candidates = fetch_arxiv_candidates(query_string, limit=15)
        source_engine = "arXiv"
        
    if not candidates:
        return [], "多源宽进管道未捕获任何全新论文（已被本地历史完美拦截）。"
        
    print(f"🧠 [漏斗阶段 2：闪电初审] 成功捕获 {len(candidates)} 篇候选文献，启动 AI 首席科学家语义仲裁...")
    
    # 调用 AI 仲裁，返回黄金 ID 列表
    golden_ids = []
    try:
        golden_ids = arbitrate_papers(candidates, topic_name, model_id)
        print(f"⚖️ [大模型仲裁结果] 从 {len(candidates)} 篇文献中，仲裁筛选出 {len(golden_ids)} 篇黄金强相关文献！")
    except Exception as e:
        print(f"❌ [大模型仲裁异常]: {e}")
        
    # 如果仲裁失败或未筛出结果，执行安全回退（取前 target_limit 篇）
    if not golden_ids:
        print("⚠️ [仲裁降级] 大模型未返回有效黄金 ID 列表，系统启动启发式回退，直接选取前几篇候选人...")
        golden_ids = [c["paper_id"] for c in candidates[:target_limit]]
        
    # 限制最终获取数量符合用户的 limit 参数
    selected_golden_ids = golden_ids[:target_limit]
    
    # 筛选黄金文献数据
    golden_papers = [c for c in candidates if c["paper_id"] in selected_golden_ids]
    
    new_papers = []
    
    # 阶段 3 & 4：靶向精准物理落盘
    print(f"📥 [漏斗阶段 3：精确收割] 开始对选定的 {len(golden_papers)} 篇黄金文献进行物理下载落盘...")
    for p in golden_papers:
        paper_id = p["paper_id"]
        safe_title = re.sub(r'[\\/*?:"<>|]', "", p["title"])[:60].strip()
        
        pdf_prefix = "scholar" if p["source_engine"] == "semantic_scholar" else "arxiv"
        pdf_filename = f"{pdf_prefix}_{paper_id}_{safe_title}.pdf"
        local_pdf_path = os.path.join(LIBRARY_DIR, pdf_filename)
        pdf_path_rel = os.path.join("storage", "library", pdf_filename)
        
        pdf_url = p["pdf_url"]
        print(f"🎯 [精准下载] 物理落盘黄金文献: {p['title']} ({pdf_url})")
        
        try:
            res = requests.get(pdf_url, stream=True, timeout=30, headers={"Connection": "close"})
            if res.status_code == 200:
                with open(local_pdf_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                paper_data = {
                    "paper_id": paper_id,
                    "title": p["title"],
                    "authors": p["authors"],
                    "venue": p["venue"],
                    "year": p["year"],
                    "citations": p["citations"],
                    "abstract": p["abstract"],
                    "pdf_path": pdf_path_rel,
                    "source_engine": p["source_engine"]
                }
                
                # 入库 papers
                insert_paper(paper_data)
                new_papers.append(paper_data)
            else:
                print(f"❌ [精确下载失败] HTTP {res.status_code}: {p['title']}")
        except Exception as e:
            print(f"❌ [精准下载异常] 《{p['title']}》下载物理文件时出错: {e}")
            
    return new_papers, source_engine
