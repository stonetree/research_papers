# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import asyncio
from core.database import get_db_connection, get_paper_deconstruction_status, get_search_archives
from core.ai_analyst import evaluate_candidates_relevance_batch
from core.search_engine import execute_two_stage_online_academic_search

def test_deconstruction_status_check():
    print("⏳ 1. Testing get_paper_deconstruction_status...")
    
    # Test RocketKV by title, url, arxiv_id, and partial title
    rocket_title = "RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression"
    rocket_url = "https://arxiv.org/html/2502.14051v3"
    
    # 1.1 Match by URL
    is_dec_u, pid_u = get_paper_deconstruction_status(url=rocket_url)
    assert is_dec_u and pid_u, f"Failed URL lookup for RocketKV ({rocket_url})"
    print(f"✅ RocketKV found by URL: status={is_dec_u}, paper_id={pid_u}")
    
    # 1.2 Match by Title
    is_dec_t, pid_t = get_paper_deconstruction_status(title=rocket_title)
    assert is_dec_t and pid_t, f"Failed Title lookup for RocketKV ({rocket_title})"
    print(f"✅ RocketKV found by Title: status={is_dec_t}, paper_id={pid_t}")
    
    # 1.3 Match by Core Prefix
    is_dec_p, pid_p = get_paper_deconstruction_status(title="RocketKV: Accelerating...")
    assert is_dec_p and pid_p, "Failed Prefix lookup for RocketKV"
    print(f"✅ RocketKV found by Core Prefix: status={is_dec_p}, paper_id={pid_p}")
    
    # 1.4 Match by combined title + url
    is_dec_c, pid_c = get_paper_deconstruction_status(title=rocket_title, url=rocket_url)
    assert is_dec_c and pid_c, "Failed Combined lookup for RocketKV"
    print(f"✅ RocketKV found by Combined: status={is_dec_c}, paper_id={pid_c}")
    
    # Test non-existent paper
    is_dec_fake, fake_pid = get_paper_deconstruction_status(title="NonExistentPaperTitle_XYZ_123456789")
    assert not is_dec_fake and fake_pid is None, "Fake paper should not be deconstructed!"
    print("✅ Non-existent paper correctly marked as pending deconstruction.")

def test_evaluate_candidates_relevance():
    print("\n⏳ 2. Testing evaluate_candidates_relevance_batch fallback and enrichment...")
    sample_candidates = [
        {
            "title": "RocketKV: Accelerating LLM Inference via Speculative KV Cache Streaming",
            "authors": "Zhang et al.",
            "venue": "OSDI 2025",
            "abstract": "RocketKV introduces high-throughput DMA and hierarchical memory compression to reduce memory footprint by 80% during long-context decoding.",
            "url": "https://arxiv.org/abs/2501.0001"
        },
        {
            "title": "Generic Unrelated Topic on Gardening in Spring",
            "authors": "Green et al.",
            "venue": "Nature 2023",
            "abstract": "A review of spring flowers and soil conditions for household plants.",
            "url": "https://example.com/gardening"
        }
    ]
    
    # Call with a model ID (will fallback gracefully if API key isn't present, or call LLM)
    evaluated = evaluate_candidates_relevance_batch(sample_candidates, "RocketKV memory optimization for LLM inference", "qwen3.7-max")
    assert len(evaluated) == 2, "Candidate count should match!"
    for c in evaluated:
        assert "score" in c, "Candidate must have 'score'"
        assert "recommendation" in c, "Candidate must have 'recommendation'"
        assert "critique" in c, "Candidate must have 'critique'"
        print(f"✅ Candidate 《{c['title'][:35]}...》: ⭐ {c['recommendation']} ({c['score']}分) | 点评: {c['critique'][:45]}...")

def test_two_stage_search_flow():
    print("\n⏳ 3. Testing execute_two_stage_online_academic_search pipeline...")
    async def _run():
        res = await execute_two_stage_online_academic_search(
            query="CXL 3.0 cache coherence and pooled memory",
            filter_type="arxiv_paper",
            relevance_model_id="qwen3.7-max",
            target_limit=4
        )
        assert res.get("status") == "success", f"Search flow failed: {res}"
        results = res.get("results", [])
        assert len(results) > 0, "Should return at least 1 evaluated candidate"
        print(f"✅ Two-stage search returned {len(results)} evaluated candidates.")
        for idx, r in enumerate(results[:2]):
            print(f"   [{idx+1}] 《{r['title'][:40]}...》 | 得分: {r.get('score')} | 推荐: {r.get('recommendation')}")
    
    asyncio.run(_run())

def test_archives_deconstruction_inspection():
    print("\n⏳ 4. Testing historical archives deconstruction status inspection...")
    archives = get_search_archives()
    print(f"Read {len(archives)} archives from database.")
    for arc in archives:
        arc_id = arc["archive_id"]
        parsed = json.loads(arc["results_json"])
        if isinstance(parsed, list):
            for p in parsed:
                is_d, pid = get_paper_deconstruction_status(title=p.get("title"), doc_id=p.get("paper_id"))
        elif isinstance(parsed, dict) and "evidences" in parsed:
            for ev in parsed["evidences"]:
                is_d, pid = get_paper_deconstruction_status(title=ev.get("title"), doc_id=ev.get("doc_id"))
    print("✅ All historical archives successfully scanned for deconstruction status without errors.")

if __name__ == "__main__":
    test_deconstruction_status_check()
    test_evaluate_candidates_relevance()
    test_two_stage_search_flow()
    test_archives_deconstruction_inspection()
    print("\n🟢 All Two-Stage Search and Deconstruction tests PASSED!")
