# -*- coding: utf-8 -*-
import json
import logging
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
import numpy as np
import aiosqlite

from .database import DB_PATH, preprocess_for_fts
from .write_worker import WriteWorker
from .lancedb_client import LanceDBClient
from .api_clients import LocalComputeKernelClient
from .ingestion import HeterogeneousIngestionEngine
from .cost_manager import api_billing_audit
from .config_loader import get_model_config

logger = logging.getLogger("SearchEngine")

async def execute_production_hybrid_retrieval(
    query_string: str,
    filter_source_type: Any,
    top_k_raw: int = 50,
    db_path: str = DB_PATH
) -> List[Dict[str, Any]]:
    """
    双路并行检索与 RRF 排名融合算法引擎。
    从 FTS5 虚拟表关键词 Match 通路和 LanceDB 向量拓扑通路拉取候选，利用 RRF(k=60) 对等合并。
    """
    lancedb_client = LanceDBClient()
    compute_client = LocalComputeKernelClient()
    
    # 增加内部候选检出限制，保持较大的候选池防止目标文档在融合阶段被推后或截断
    internal_k = max(top_k_raw, 50)
    
    # 归一化为列表
    if isinstance(filter_source_type, str):
        source_types = [filter_source_type]
    elif isinstance(filter_source_type, (list, tuple, set)):
        source_types = list(filter_source_type)
    else:
        source_types = []
        
    if not source_types:
        return []
        
    # 通路 1: SQLite FTS5 MATCH 全文检索
    fts_candidates = []
    import re
    # 预处理中文/英文混合文本，使中文字符被空格分离
    preprocessed_query = preprocess_for_fts(query_string)
    # 清洗特殊字符，防止 FTS5 MATCH 解析列报错（例如 CPU-GPU 中的减号被当作列名/操作符）
    clean_query = preprocessed_query.replace('"', ' ')
    clean_query = re.sub(r'[^\w\s]', ' ', clean_query)
    words = [w for w in clean_query.split() if w]
    if not words:
        return []
    fts_query_str = " AND ".join(words)
        
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        placeholders = ",".join(["?"] * len(source_types))
        fts_sql = f"""
            SELECT sc.chunk_id, sc.doc_id, sc.title, sc.section_path 
            FROM unified_knowledge_fts f
            JOIN search_chunks sc ON f.rowid = sc.search_chunk_id
            JOIN documents d ON sc.doc_id = d.doc_id
            WHERE unified_knowledge_fts MATCH ? AND d.source_type IN ({placeholders}) 
            LIMIT ?
        """
        try:
            # 搜索时使用 BM25 排序偏好
            params = [fts_query_str] + source_types + [internal_k]
            async with db.execute(fts_sql, params) as cursor:
                rows = await cursor.fetchall()
                for idx, row in enumerate(rows):
                    fts_candidates.append({
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "fts_rank": idx + 1
                    })
        except Exception as fts_err:
            logger.error(f"FTS5 MATCH 检索异常: {str(fts_err)}")

    # 通路 2: LanceDB 向量数据库语义模糊搜索
    vector_candidates = []
    try:
        # 获取 Query 的 1024 维特征向量
        query_vector = await compute_client.get_embedding(query_string)
        if query_vector:
            all_vec_results = []
            for st in source_types:
                results = lancedb_client.search_vector(st, query_vector, limit=internal_k)
                all_vec_results.extend(results)
            # 按余弦距离从小到大排序
            all_vec_results.sort(key=lambda x: x.get("_distance", 1.0))
            for idx, res in enumerate(all_vec_results[:internal_k]):
                meta = res.get("metadata", {})
                parent_id = meta.get("parent_id") or res.get("id", "").replace("chunk_", "")
                vector_candidates.append({
                    "chunk_id": res.get("id"),
                    "doc_id": parent_id,
                    "vector_rank": idx + 1
                })
    except Exception as vec_err:
        logger.error(f"LanceDB 向量空间检索异常: {str(vec_err)}")

    # 相互倒排排名融合 (RRF, k=60)
    K_CONSTANT = 60
    rrf_scoreboard = {}

    for cand in fts_candidates:
        c_id = cand["chunk_id"]
        rrf_scoreboard[c_id] = rrf_scoreboard.get(c_id, 0.0) + (1.0 / (K_CONSTANT + cand["fts_rank"]))

    for cand in vector_candidates:
        c_id = cand["chunk_id"]
        rrf_scoreboard[c_id] = rrf_scoreboard.get(c_id, 0.0) + (1.0 / (K_CONSTANT + cand["vector_rank"]))

    # 取得分前 top_k_raw 名进行精排
    sorted_rrf = sorted(rrf_scoreboard.items(), key=lambda x: x[1], reverse=True)[:top_k_raw]
    if not sorted_rrf:
        return []

    # 核心物理真空反查：反向映射 chunks 获取干净的长文本正文
    final_evidence_payload = []
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        placeholders = ",".join(["?"] * len(sorted_rrf))
        target_chunk_ids = [item[0] for item in sorted_rrf]
        
        extraction_sql = f"""
            SELECT c.chunk_id, c.doc_id, c.section_path, c.page_number, c.text, d.title, d.canonical_url, d.source_type 
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.chunk_id IN ({placeholders})
        """
        try:
            async with db.execute(extraction_sql, target_chunk_ids) as cursor:
                db_rows = await cursor.fetchall()
                row_map = {row["chunk_id"]: row for row in db_rows}
                
                for chunk_id, rrf_score in sorted_rrf:
                    if chunk_id in row_map:
                        r = row_map[chunk_id]
                        final_evidence_payload.append({
                            "doc_id": r["doc_id"],
                            "chunk_id": r["chunk_id"],
                            "title": r["title"],
                            "canonical_url": r["canonical_url"],
                            "section_path": r["section_path"],
                            "page_number": r["page_number"],
                            "text": r["text"],
                            "source_type": r["source_type"],
                            "hybrid_score": float(rrf_score)
                        })
        except Exception as ext_err:
            logger.error(f"物理反查 Chunks 失败: {str(ext_err)}")
            
    return final_evidence_payload


# === 2. 四大解耦微服务接口 ===

async def api_retrieve(query: str, filter_type: Any, top_k_raw: int = 50) -> Dict[str, Any]:
    """
    原子检索层接口：仅 RRF 粗筛融合。
    """
    candidates = await execute_production_hybrid_retrieval(query, filter_type, top_k_raw)
    return {
        "status": "success",
        "candidates": candidates
    }

async def api_rerank(query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    原子重排层接口：输入 candidates，调用本地 8082 Reranker 微服务。
    当 8082 服务离线或不在线时，给出警告提示并平滑跳过重排步骤，直接透传 RRF 排序顺序。
    """
    if not candidates:
        return {"results": [], "skipped_rerank": False}
        
    compute_client = LocalComputeKernelClient()
    is_rerank_ready = await compute_client.check_rerank_service_health()
    
    if not is_rerank_ready:
        logger.warning("⚠️ [Rerank 8082 离线] 本地 Reranker 服务未在 8082 端口响应，自动跳过 Logits 重排步骤，直接保持 RRF 混合融合顺序输出。")
        formatted_results = []
        for rank_idx, cand in enumerate(candidates):
            cand_copy = cand.copy()
            cand_copy["rerank_score"] = cand.get("hybrid_score", 0.0)
            formatted_results.append({
                "id": cand["chunk_id"],
                "relevance_score": cand.get("hybrid_score", 0.0),
                "rank_index": rank_idx,
                "evidence": cand_copy
            })
        return {"results": formatted_results, "skipped_rerank": True}

    # 限制 Query 长度不超过 200 字符，整体截断限制在 600 字符以内防止超出 512 batch size
    safe_query = query[:200]
    texts = [f"标题: {c['title']} | 章节: {c['section_path']} | 正文: {c['text']}"[:600] for c in candidates]
    
    scores = await compute_client.get_rerank_scores(safe_query, texts)
    if not scores:
        logger.warning("⚠️ [Rerank 计算空回退] Reranker 未返回得分序列，自动跳过 Logits 重排步骤，透传 RRF 结果。")
        formatted_results = []
        for rank_idx, cand in enumerate(candidates):
            cand_copy = cand.copy()
            cand_copy["rerank_score"] = cand.get("hybrid_score", 0.0)
            formatted_results.append({
                "id": cand["chunk_id"],
                "relevance_score": cand.get("hybrid_score", 0.0),
                "rank_index": rank_idx,
                "evidence": cand_copy
            })
        return {"results": formatted_results, "skipped_rerank": True}

    results = []
    for idx, cand in enumerate(candidates):
        score = scores[idx] if idx < len(scores) else 0.0
        cand_copy = cand.copy()
        cand_copy["rerank_score"] = score
        results.append(cand_copy)
        
    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    formatted_results = []
    for rank_idx, r in enumerate(results):
        formatted_results.append({
            "id": r["chunk_id"],
            "relevance_score": r["rerank_score"],
            "rank_index": rank_idx,
            "evidence": r
        })
    return {"results": formatted_results, "skipped_rerank": False}

@api_billing_audit(api_provider="dashscope", api_metric="tokens")
async def api_answer(
    model_provider: str,
    model_name: str,
    user_query: str,
    context_evidences: List[Dict[str, Any]],
    model_id: str = "qwen3.7-max"
) -> Dict[str, Any]:
    """
    原子大模型合成接口：读取高密证据与提问，绑定 doc_id 生成事实回答。
    """
    cfg = get_model_config(model_id)
    if not cfg:
        return {"status": "failed", "error": "Model configuration not found"}
        
    api_key = cfg.get("resolved_api_key", "").strip()
    api_url = cfg.get("url", "").strip()
    actual_model_name = cfg.get("model", model_name)
    
    # 组装上下文证据
    evidence_text = ""
    for idx, ev in enumerate(context_evidences):
        evidence_text += f"证据[{idx+1}] (doc_id: {ev['doc_id']}): {ev['text']}\n---\n"
        
    system_instruction = (
        "你是一个极其严谨的 AI 硬件专家与系统架构师。\n"
        "请严格基于下面提供的证据上下文回答用户的提问。\n"
        "回答中每个推论和论断的句尾，必须强制逆向显式绑定证据中提供的 [doc_id]（例如: 依据文献 [doc_id] 所示）。\n"
        "严禁胡思乱想，凡是无法在证据中找到支持的论断，必须明确表示信息缺失。"
    )
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"证据上下文：\n{evidence_text}\n\n问题：{user_query}"}
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json={
                "model": actual_model_name,
                "messages": messages,
                "temperature": 0.1
            }, headers=headers, timeout=35) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # 抓取 token 消耗
                    usage = data.get("usage", {})
                    
                    # 兼容百炼 Responses API 格式
                    if "output" in data and isinstance(data["output"], list):
                        answer_text = ""
                        for out_item in data["output"]:
                            if out_item.get("type") == "message" and out_item.get("role") == "assistant":
                                contents = out_item.get("content", [])
                                if contents and isinstance(contents, list):
                                    answer_text = contents[0].get("text", "").strip()
                                    break
                        if not answer_text:
                            # 降级尝试提取任意文本内容
                            for out_item in data["output"]:
                                contents = out_item.get("content", [])
                                if isinstance(contents, list) and contents:
                                    for c in contents:
                                        if "text" in c:
                                            answer_text = c["text"].strip()
                                            break
                                if answer_text:
                                    break
                    else:
                        # 标准 OpenAI chat/completions 格式
                        answer_text = data["choices"][0]["message"]["content"].strip()
                    
                    return {
                        "model": actual_model_name,
                        "usage": usage,
                        "status": "success",
                        "answer": answer_text
                    }
                else:
                    err_text = await resp.text()
                    return {"status": "failed", "error": f"HTTP {resp.status}: {err_text}"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


async def api_search(
    query: str,
    filter_type: Any = "ext_blog",
    routing_mode: str = "retrieve_rerank_answer",
    model_id: str = "qwen3.7-max"
) -> Dict[str, Any]:
    """
    聚合编排路由接口。
    """
    # 1. 检索 (RRF)
    ret_res = await api_retrieve(query, filter_type)
    candidates = ret_res.get("candidates", [])
    
    if routing_mode == "retrieve_only" or not candidates:
        return {
            "status": "success",
            "routing_path": "retrieve_only",
            "candidates": candidates
        }
        
    # 2. 重排
    rerank_res = await api_rerank(query, candidates)
    results = rerank_res.get("results", [])
    reranked_evidences = [r["evidence"] for r in results]
    
    if routing_mode == "retrieve_rerank":
        return {
            "status": "success",
            "routing_path": "retrieve_rerank",
            "candidates": reranked_evidences
        }
        
    # 3. 问答生成 (取前 5 篇精排文献喂大模型)
    top_evidences = reranked_evidences[:5]
    ans_res = await api_answer(
        model_provider="dashscope",
        model_name="qwen3.7-max",
        user_query=query,
        context_evidences=top_evidences,
        model_id=model_id
    )
    
    if ans_res.get("status") == "failed":
        return {
            "status": "failed",
            "error": ans_res.get("error", "解答合成失败。")
        }
        
    return {
        "status": "success",
        "routing_path": "retrieve_rerank_answer",
        "answer": ans_res.get("answer", "解答合成失败。"),
        "evidences": top_evidences,
        "cost": ans_res.get("usage")
    }


# === 3. 一体化 Pipeline B 智能交互式学术探测通路 ===

async def execute_unified_studio_search_flow(
    query: str,
    filter_type: str = "ext_blog",
    force_penetrate: bool = False,
    allow_web_search: bool = True,
    model_id: str = "qwen3.7-max",
    on_progress = None
) -> Dict[str, Any]:
    """
    智能交互问答层核心控制：Pipeline B 事实绑定打捞与解答合成流程。
    """
    # Step 1: 触发本地双路混合检索与 RRF 融合反查
    if on_progress:
        on_progress("🔎 正在启动本地 FTS5 + LanceDB 双路混合检索...")
    local_evidences = await execute_production_hybrid_retrieval(query, filter_type, top_k_raw=30)
    
    # 判定本地最高评分是否击穿置信度门槛 (RRF 复合得分 > 0.04)
    has_high_confidence_local_hit = len(local_evidences) > 0 and local_evidences[0]["hybrid_score"] > 0.04
    
    # 路由判定：
    # 若 (本地高置信度命中 且 未开启强制穿透) 或 选项已关闭联网检索 (allow_web_search=False)
    # 则完全使用本地知识大仓结果；否则触发联网打捞。
    if (has_high_confidence_local_hit and not force_penetrate) or not allow_web_search:
        if not allow_web_search:
            logger.info("🔒 联网检索已被选项关闭，直接使用本地知识大仓进行解答合成。")
            if on_progress:
                on_progress("🔒 联网检索已被选项关闭，直接使用本地大仓进行解答合成...")
            routing_path = "local_only"
        else:
            logger.info("🎯 本地知识大仓高置信度精确命中！执行 0 成本本地解答合成。")
            if on_progress:
                on_progress("🎯 本地知识大仓高置信度命中！正在执行零成本本地解答合成...")
            routing_path = "local_cache_hit"
        evidence_payload = local_evidences
    else:
        logger.warning("🔍 本地大仓未检索到相关高密细节，或者用户开启强力穿透打捞！启动 Exa 神经网络精准打捞...")
        if on_progress:
            on_progress("🌐 本地大仓未检索到相关高密细节，或者用户开启强力穿透打捞！正在启动 Exa 神经网络打捞...")
        # Step 2: 驱动 Exa 探测
        ingestion_engine = HeterogeneousIngestionEngine()
        # 实际开发中，这里会调用 ExaApiClient 检索网页并自适应下载
        # 此处我们调用 execute_intelligent_active_pull，由于我们想要将结果沉淀落库
        # 如果 query 包含 url，直接抓取 URL；如果是普通关键词，在 Exa 打捞
        # 我们使用 ExaApiClient 提取 highlights 并落库
        try:
            include_domains = [
                "arxiv.org",
                "semanticscholar.org",
                "biorxiv.org",
                "medrxiv.org",
                "pubmed.ncbi.nlm.nih.gov",
                "researchgate.net",
                "ieeexplore.ieee.org",
                "dl.acm.org",
                "link.springer.com",
                "sciencedirect.com",
                "nature.com",
                "science.org",
                "mdpi.com",
                "frontiersin.org"
            ] if filter_type == "arxiv_paper" else None
            category = "research paper" if filter_type == "arxiv_paper" else None
            exa_response = await ingestion_engine.exa_client.search_and_extract_highlights(
                query, 
                num_results=3,
                include_domains=include_domains,
                category=category
            )
            results = exa_response.get("results", [])
            
            # 对抓取的每一页自动通过 IngestionEngine 进行 2PC 沉淀落库
            for idx, r in enumerate(results):
                raw_url = r.get("url")
                raw_title = r.get("title", "未知标题")
                if on_progress:
                    on_progress(f"📥 正在抓取并沉淀第 {idx+1}/{len(results)} 页: {raw_title[:20]}...")
                # 触发 2PC 自动落库
                await ingestion_engine.execute_intelligent_active_pull(
                    discovery_query=f"url: {raw_url}",
                    user_confirmed_firecrawl=True,
                    source_type=filter_type,
                    on_progress=on_progress
                )
        except Exception as err:
            logger.error(f"Exa 网络探针打捞失败: {str(err)}")
            
        # Step 3: 数据沉淀落库完毕，重新跑一次本地混合检索获取最新真实物理切片
        if on_progress:
            on_progress("🔎 数据大仓已沉淀完毕，重新运行本地双路检索以更新真实切片缓存...")
        evidence_payload = await execute_production_hybrid_retrieval(query, filter_type, top_k_raw=10)
        routing_path = "exa_penetrate_funnel"

    # Step 4: 交付精排与 LLM 问答生成
    top_evidences = evidence_payload[:5]
    
    # 如果没有证据，使用空列表以防止 LLM 抛出未绑定异常
    if on_progress:
        on_progress(f"🧠 正在交付 AI 大脑 ({model_id}) 进行事实对账并合成严谨解答...")
    ans_res = await api_answer(
        model_provider="dashscope",
        model_name="qwen3.7-max",
        user_query=query,
        context_evidences=top_evidences,
        model_id=model_id
    )
    
    if ans_res.get("status") == "failed":
        return {
            "status": "failed",
            "error": ans_res.get("error", "未能成功合成解答。")
        }
        
    return {
        "status": "success",
        "routing_path": routing_path,
        "answer": ans_res.get("answer", "未能成功合成解答。"),
        "evidences": top_evidences
    }
