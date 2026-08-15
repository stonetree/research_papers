# -*- coding: utf-8 -*-
import json
import uuid
import logging
import asyncio
import concurrent.futures
from datetime import datetime
from typing import Dict, Any, List, Optional
from .database import (
    insert_suspended_task,
    update_suspended_task_status,
    get_suspended_tasks,
    get_suspended_task_by_id,
    delete_suspended_task
)

logger = logging.getLogger("SuspendedTaskManager")

def record_suspended_search_task(
    query: str,
    filter_type: str,
    allow_web_search: bool,
    force_penetrate: bool,
    model_id: str,
    error_step: str,
    error_message: str,
    completed_steps: List[Dict[str, Any]],
    remaining_steps: List[Dict[str, Any]],
    intermediate_data: Dict[str, Any],
    task_id: Optional[str] = None
) -> str:
    """
    记录或更新因瞬态错误（如 HTTP 503 / 429 / 超时）导致中断的联网搜索任务。
    返回 task_id。
    """
    if not task_id:
        task_id = f"susp_task_{uuid.uuid4().hex[:10]}"
        
    current_step_desc = error_step or "已暂停"
    task_dict = {
        "task_id": task_id,
        "query": query,
        "filter_type": filter_type,
        "allow_web_search": 1 if allow_web_search else 0,
        "force_penetrate": 1 if force_penetrate else 0,
        "model_id": model_id,
        "status": "suspended",
        "current_step": current_step_desc,
        "error_step": error_step,
        "error_message": error_message,
        "completed_steps_json": json.dumps(completed_steps, ensure_ascii=False),
        "remaining_steps_json": json.dumps(remaining_steps, ensure_ascii=False),
        "intermediate_data_json": json.dumps(intermediate_data, ensure_ascii=False)
    }
    insert_suspended_task(task_dict)
    logger.info(f"⏸️ 联网搜索任务已安全挂起并记录现场: Doc/TaskID={task_id}, Query='{query}', 出错步骤='{error_step}'")
    return task_id

async def resume_suspended_search_task_async(
    task_id: str,
    override_model_id: Optional[str] = None,
    on_progress = None
) -> Dict[str, Any]:
    """
    异步断点恢复并继续执行挂起的联网搜索任务。
    直接利用已抓取和沉淀的证据切片，跳过前期重复消耗，直接重试大模型事实对账与解答合成。
    """
    task = get_suspended_task_by_id(task_id)
    if not task:
        return {"status": "failed", "error": f"未找到任务编号为 '{task_id}' 的挂起记录。"}
        
    query = task.get("query", "")
    filter_type = task.get("filter_type", "arxiv_paper")
    model_id = override_model_id or task.get("model_id", "qwen3.7-max")
    
    # 标记状态为正在恢复中
    update_suspended_task_status(task_id, status="running", current_step="正在断点恢复执行中...")
    
    if on_progress:
        on_progress(f"⚡ 正在恢复任务 [{task_id}]... 读取已沉淀的证据切片与现场快照")
        
    try:
        intermediate_data = json.loads(task.get("intermediate_data_json", "{}"))
    except Exception:
        intermediate_data = {}
        
    try:
        completed_steps = json.loads(task.get("completed_steps_json", "[]"))
    except Exception:
        completed_steps = []
        
    try:
        remaining_steps = json.loads(task.get("remaining_steps_json", "[]"))
    except Exception:
        remaining_steps = []

    # 1. 提取或重构证据切片
    evidences = intermediate_data.get("evidences", [])
    routing_path = intermediate_data.get("routing_path", "resume_from_suspended")
    
    from .search_engine import execute_production_hybrid_retrieval, api_answer
    
    if not evidences:
        if on_progress:
            on_progress("🔎 正在从本地混合知识湖（FTS5 + LanceDB）重新提取高密证据切片...")
        evidences = await execute_production_hybrid_retrieval(query, filter_type, top_k_raw=10)
        intermediate_data["evidences"] = evidences
        
    top_evidences = evidences[:5]
    
    # 2. 调用目标 AI 大脑进行解答合成与事实对账
    if on_progress:
        on_progress(f"🧠 正在调用 AI 大脑 ({model_id}) 重新合成严谨解答与证据绑定...")
        
    ans_res = await api_answer(
        model_provider="dashscope",
        model_name="qwen3.7-max",
        user_query=query,
        context_evidences=top_evidences,
        model_id=model_id
    )
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if ans_res.get("status") == "success":
        # 成功合成解答，将剩余步骤推进为已完成
        completed_step_item = {
            "step_id": 4,
            "step_name": "AI 大脑事实对账与解答合成",
            "summary": f"由大模型 ({model_id}) 完成严谨学术解答生成与 doc_id 事实对账",
            "timestamp": now_str,
            "status": "success"
        }
        completed_steps.append(completed_step_item)
        
        update_suspended_task_status(
            task_id=task_id,
            status="completed",
            current_step="全流程执行完成",
            error_step="",
            error_message="",
            completed_steps_json=json.dumps(completed_steps, ensure_ascii=False),
            remaining_steps_json=json.dumps([], ensure_ascii=False),
            intermediate_data_json=json.dumps(intermediate_data, ensure_ascii=False)
        )
        logger.info(f"🎉 挂起任务 {task_id} 成功恢复并执行完成！")
        return {
            "status": "success",
            "task_id": task_id,
            "routing_path": routing_path,
            "answer": ans_res.get("answer", ""),
            "evidences": top_evidences,
            "query": query,
            "model_id": model_id
        }
    else:
        err_msg = ans_res.get("error", "AI 大脑合成解答失败。")
        logger.warning(f"⚠️ 挂起任务 {task_id} 恢复重试依然失败: {err_msg}")
        update_suspended_task_status(
            task_id=task_id,
            status="suspended",
            current_step="AI 大脑解答合成重试失败",
            error_step="AI 大脑事实对账与解答合成",
            error_message=err_msg
        )
        return {
            "status": "suspended",
            "task_id": task_id,
            "error": err_msg,
            "message": f"重试未能成功 ({err_msg})，现场已更新并继续保持挂起状态。"
        }

def resume_suspended_search_task_sync(
    task_id: str,
    override_model_id: Optional[str] = None,
    on_progress = None
) -> Dict[str, Any]:
    """
    同步包装函数：便于在 Streamlit 主线程或其他同步上下文中直接调用断点恢复。
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run, 
                resume_suspended_search_task_async(task_id, override_model_id, on_progress)
            ).result()
    else:
        return asyncio.run(resume_suspended_search_task_async(task_id, override_model_id, on_progress))
