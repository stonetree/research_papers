# -*- coding: utf-8 -*-
import requests
import os
import re
from .database import get_db_connection, insert_paper

LIBRARY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "library")
os.makedirs(LIBRARY_DIR, exist_ok=True)

TARGET_VENUES = ["ASPLOS", "OSDI", "SOSP", "ISCA", "MICRO", "VLDB", "arXiv"]

def execute_semantic_search(query_string, limit=3):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query_string,
        "limit": limit * 2,
        "fields": "paperId,title,authors,venue,year,citationCount,openAccessPdf,abstract"
    }
    
    conn = get_db_connection()
    cursor = conn.cursor()
    new_papers = []
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            print(f"⚠️ [Semantic Scholar API Error] HTTP {response.status_code}: {response.text}")
            return []
        papers = response.json().get("data", [])
        
        for p in papers:
            paper_id = p["paperId"]
            if not p.get("openAccessPdf"): continue
            
            venue = p.get("venue", "")
            is_target_venue = any(tv.lower() in venue.lower() for tv in TARGET_VENUES) if venue else True
            if not is_target_venue: continue
            
            cursor.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,))
            if cursor.fetchone(): continue
                
            safe_title = re.sub(r'[\\/*?:"<>|]', "", p["title"])[:60].strip()
            pdf_filename = f"scholar_{paper_id}_{safe_title}.pdf"
            local_pdf_path = os.path.join(LIBRARY_DIR, pdf_filename)
            pdf_path_rel = os.path.join("storage", "library", pdf_filename)
            
            print(f"📥 [Scholar 管道] 捕获硬核成果: {p['title']}")
            pdf_url = p["openAccessPdf"]["url"]
            
            res = requests.get(pdf_url, stream=True, timeout=20)
            if res.status_code == 200:
                with open(local_pdf_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=8192): f.write(chunk)
                
                authors_str = ", ".join([a['name'] for a in p.get('authors', [])])
                paper_data = {
                    "paper_id": paper_id,
                    "title": p["title"],
                    "authors": authors_str,
                    "venue": p.get("venue", "arXiv"),
                    "year": p.get("year"),
                    "citations": p.get("citationCount", 0),
                    "abstract": p.get("abstract", ""),
                    "pdf_path": pdf_path_rel,
                    "source_engine": "semantic_scholar"
                }
                insert_paper(paper_data)
                new_papers.append(paper_data)
                if len(new_papers) >= limit: break
                
    finally:
        conn.close()
        
    return new_papers