# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import uuid
from core.database import get_search_archives, insert_search_archive, delete_search_archive

def test_archive_and_load():
    print("Testing Archive Insertion and Loading logic across all formats...")
    
    # 1. Test inserting V2 studio dict archive
    arc_id_v2 = f"test_arc_v2_{uuid.uuid4().hex[:6]}"
    v2_data = {
        "answer": "Test answer for deterministic workflow",
        "evidences": [{"title": "Paper A", "canonical_url": "https://arxiv.org/abs/2401.001", "doc_id": "doc_001", "hybrid_score": 0.95, "text": "Evidence snippet"}],
        "routing_path": "local_cache_hit",
        "query": "deterministic agent workflow",
        "model_id": "qwen3.7-max"
    }
    insert_search_archive(arc_id_v2, v2_data["query"], json.dumps(v2_data, ensure_ascii=False))
    
    # 2. Test inserting V1 paper list archive
    arc_id_v1 = f"test_arc_v1_{uuid.uuid4().hex[:6]}"
    v1_data = [
        {"title": "Paper 1", "authors": "Author A", "year_venue": "OSDI 2024", "summary": "Summary 1", "url": "https://arxiv.org/abs/2401.002"},
        {"title": "Paper 2", "authors": "Author B", "year_venue": "SOSP 2023", "summary": "Summary 2", "url": "https://arxiv.org/abs/2401.003"}
    ]
    insert_search_archive(arc_id_v1, "V1 list query", json.dumps(v1_data, ensure_ascii=False))
    
    # 3. Read back all archives and test load rendering logic
    archives = get_search_archives()
    print(f"Total archives read from DB: {len(archives)}")
    
    found_v1 = False
    found_v2 = False
    
    for arc in archives:
        arc_id = arc["archive_id"]
        parsed = json.loads(arc["results_json"])
        
        # Test rendering branch selection
        if isinstance(parsed, dict) and "answer" in parsed:
            print(f"✅ Archive {arc_id} correctly routed to V2 Studio renderer (query: {arc['query']})")
            if arc_id == arc_id_v2:
                found_v2 = True
        elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            print(f"✅ Archive {arc_id} correctly routed to V1 Paper List renderer ({len(parsed)} papers, query: {arc['query']})")
            if arc_id == arc_id_v1:
                found_v1 = True
        elif isinstance(parsed, dict) and "candidates" in parsed:
            print(f"✅ Archive {arc_id} correctly routed to Candidate List renderer (query: {arc['query']})")
        else:
            print(f"⚠️ Archive {arc_id} fallback format")

    assert found_v1, "V1 test archive not found!"
    assert found_v2, "V2 test archive not found!"
    
    # Clean up test records
    delete_search_archive(arc_id_v1)
    delete_search_archive(arc_id_v2)
    print("🧹 Test archive records cleaned up.")
    print("🟢 All archive creation, retrieval, and multi-version loading tests passed!")

if __name__ == "__main__":
    test_archive_and_load()
