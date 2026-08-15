import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from core.database import get_search_archives

def test_archive_parsing():
    archives = get_search_archives()
    print(f"Total archives found in DB: {len(archives)}")
    for arc in archives:
        arc_id = arc["archive_id"]
        parsed_res = json.loads(arc["results_json"])
        if isinstance(parsed_res, list) and len(parsed_res) > 0 and isinstance(parsed_res[0], dict):
            titles = [p.get("title") for p in parsed_res]
            print(f"✅ [V1 List] ArcID={arc_id}, Query='{arc['query']}', Titles count={len(titles)}:")
            for idx, t in enumerate(titles[:3]):
                print(f"   [{idx+1}] {t}")
        elif isinstance(parsed_res, dict) and "evidences" in parsed_res:
            ev_list = parsed_res.get("evidences", [])
            articles_map = {}
            for ev in ev_list:
                t = ev.get("title") or "未知标题"
                doc_id = ev.get("doc_id") or t
                if doc_id not in articles_map:
                    articles_map[doc_id] = t
            print(f"✅ [V2 Studio] ArcID={arc_id}, Query='{arc['query']}', Unique Titles count={len(articles_map)}:")
            for idx, (doc_id, t) in enumerate(list(articles_map.items())[:3]):
                print(f"   [{idx+1}] {t} ({doc_id})")
        elif isinstance(parsed_res, dict) and "candidates" in parsed_res:
            cand_list = parsed_res.get("candidates", [])
            print(f"✅ [Candidate List] ArcID={arc_id}, Query='{arc['query']}', Candidates count={len(cand_list)}:")
            for idx, c in enumerate(cand_list[:3]):
                print(f"   [{idx+1}] {c.get('title')}")
        else:
            print(f"⚠️ [Fallback] ArcID={arc_id}")

if __name__ == "__main__":
    test_archive_parsing()
