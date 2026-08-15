# -*- coding: utf-8 -*-
import os
import sys
import json
import asyncio
import logging
from unittest.mock import patch, AsyncMock

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.database import (
    init_db,
    get_db_connection,
    insert_suspended_task,
    get_suspended_tasks,
    get_suspended_task_by_id,
    delete_suspended_task
)
from core.suspended_task_manager import (
    record_suspended_search_task,
    resume_suspended_search_task_async,
    resume_suspended_search_task_sync
)
from core.search_engine import execute_unified_studio_search_flow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TestTaskSuspension")

async def run_tests():
    logger.info("=========================================")
    logger.info("⏳ 1. 初始化数据库并测试基础 CRUD 接口...")
    init_db()
    
    # 1. 模拟记录一个挂起任务
    mock_task_id = "test_susp_503_001"
    # 先清理可能的旧数据
    delete_suspended_task(mock_task_id)
    
    completed_steps = [
        {"step_id": 1, "step_name": "本地双路混合检索与置信度初审", "summary": "召回 12 条候选切片", "timestamp": "2026-08-15 10:00:00", "status": "success"},
        {"step_id": 2, "step_name": "全网智能打捞与文献抓取沉淀", "summary": "抓取并 2PC 入库 3 篇文献", "timestamp": "2026-08-15 10:00:05", "status": "success"},
        {"step_id": 3, "step_name": "2PC 混合检索索引刷新与证据精排", "summary": "筛选出 5 条高密证据切片", "timestamp": "2026-08-15 10:00:08", "status": "success"}
    ]
    remaining_steps = [
        {"step_id": 4, "step_name": "AI 大脑事实对账与解答合成", "summary": "基于 5 条事实切片调用大模型生成带引用的解答", "status": "pending"}
    ]
    intermediate_data = {
        "routing_path": "exa_penetrate_funnel",
        "evidences": [
            {"doc_id": "test_doc_01", "text": "CXL 3.0 provides low latency cache coherent shared memory across heterogenous hosts.", "hybrid_score": 0.052, "title": "CXL 3.0 Interconnect Spec", "canonical_url": "https://arxiv.org/abs/2601.0001"},
            {"doc_id": "test_doc_02", "text": "Hardware-enforced coherency reduces Host OS page migration overhead.", "hybrid_score": 0.048, "title": "Heterogeneous Coherence at Scale", "canonical_url": "https://arxiv.org/abs/2601.0002"}
        ],
        "scraped_urls": ["https://arxiv.org/abs/2601.0001", "https://arxiv.org/abs/2601.0002"],
        "ingested_doc_ids": ["test_doc_01", "test_doc_02"]
    }
    
    task_id_recorded = record_suspended_search_task(
        query="CXL 3.0 cache coherence and memory sharing",
        filter_type="arxiv_paper",
        allow_web_search=True,
        force_penetrate=False,
        model_id="deepseek-v4",
        error_step="Step 4: AI 大脑事实对账与解答合成 (LLM 503/瞬态故障)",
        error_message="HTTP 503 Service Unavailable: Model provider overloaded, please retry later.",
        completed_steps=completed_steps,
        remaining_steps=remaining_steps,
        intermediate_data=intermediate_data,
        task_id=mock_task_id
    )
    
    assert task_id_recorded == mock_task_id, "返回的 task_id 应一致"
    
    # 2. 检查从数据库查询任务
    task = get_suspended_task_by_id(mock_task_id)
    assert task is not None, "应该成功查询到保存的挂起任务"
    assert task["status"] == "suspended", f"任务状态应为 suspended，实际为: {task['status']}"
    assert "503" in task["error_message"], "错误日志应记录 503 报错信息"
    
    all_suspended = get_suspended_tasks(status="suspended")
    assert any(t["task_id"] == mock_task_id for t in all_suspended), "挂起任务列表中应包含当前任务"
    logger.info("✅ 挂起任务注册与持久化读取测试通过！")
    
    # 3. 测试断点恢复与重试逻辑 (模拟 API answer 恢复正常)
    logger.info("=========================================")
    logger.info("⏳ 2. 测试断点恢复重试与状态流转...")
    
    mock_success_answer = {
        "status": "success",
        "answer": "根据文献 [test_doc_01]，CXL 3.0 提供了低时延硬件一致性共享内存。进一步依据文献 [test_doc_02]，硬件一致性有效削减了 Host OS 换页迁移开销。",
        "model": "deepseek-v4",
        "usage": {"total_tokens": 150}
    }
    
    with patch("core.search_engine.api_answer", new=AsyncMock(return_value=mock_success_answer)):
        res_resume = await resume_suspended_search_task_async(mock_task_id)
        assert res_resume["status"] == "success", f"断点恢复应当成功，实际返回: {res_resume}"
        assert len(res_resume["evidences"]) == 2, "应准确复用挂起前已沉淀的证据切片"
        assert "[test_doc_01]" in res_resume["answer"], "回答中应绑定证据引用"
        
        # 验证数据库状态更新为 completed
        task_after = get_suspended_task_by_id(mock_task_id)
        assert task_after["status"] == "completed", f"任务状态应更新为 completed，实际为: {task_after['status']}"
        completed_steps_after = json.loads(task_after["completed_steps_json"])
        assert len(completed_steps_after) == 4, f"完成步骤数量应为 4，实际为: {len(completed_steps_after)}"
        logger.info("✅ 断点恢复重试执行与状态完备推进测试通过！")
        
    # 4. 测试搜索流中大模型遭遇 503 时的全自动挂起集成
    logger.info("=========================================")
    logger.info("⏳ 3. 测试 execute_unified_studio_search_flow 遭遇 503 时的自动挂起流水线...")
    
    mock_503_answer = {
        "status": "failed",
        "error": "HTTP 503 Service Unavailable: Backpressure reached."
    }
    
    # 模拟本地检索召回与大模型 503
    mock_local_evidences = [
        {"doc_id": "doc_cache_01", "text": "vLLM PagedAttention v2 reduces memory fragmentation.", "hybrid_score": 0.08, "title": "PagedAttention v2", "canonical_url": "https://arxiv.org/abs/2602.0001"}
    ]
    
    with patch("core.search_engine.execute_production_hybrid_retrieval", new=AsyncMock(return_value=mock_local_evidences)), \
         patch("core.search_engine.api_answer", new=AsyncMock(return_value=mock_503_answer)):
        
        search_res = await execute_unified_studio_search_flow(
            query="vLLM PagedAttention v2 memory efficiency",
            filter_type="arxiv_paper",
            allow_web_search=True,
            force_penetrate=False,
            model_id="deepseek-v4"
        )
        
        assert search_res["status"] == "suspended", f"搜索流在 503 时应返回 suspended 状态，实际为: {search_res}"
        assert "task_id" in search_res, "返回结果中应包含注册的 task_id"
        auto_task_id = search_res["task_id"]
        
        # 验证数据库中真实写入了该挂起任务
        auto_task = get_suspended_task_by_id(auto_task_id)
        assert auto_task is not None, "数据库中应存在自动创建的挂起任务"
        assert auto_task["status"] == "suspended"
        assert "503" in auto_task["error_message"]
        logger.info(f"✅ 全自动 503 捕获并挂起任务成功！Doc/TaskID={auto_task_id}")
        
        # 清理自动创建的任务
        delete_suspended_task(auto_task_id)
        
    # 清理测试任务
    delete_suspended_task(mock_task_id)
    
    logger.info("=========================================")
    logger.info("🟢 所有任务挂起、时序列表保存、503 异常现场捕获与断点恢复测试全部通过！")

if __name__ == "__main__":
    asyncio.run(run_tests())
