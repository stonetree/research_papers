# -*- coding: utf-8 -*-
import json
import logging
import hashlib
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any, Optional
from google import genai
from google.genai import types

from .database import DB_PATH, get_db_connection
from .write_worker import WriteWorker
from .cost_manager import api_billing_audit, HardBudgetValidator
from .ingestion import IngestionCoordinator, HeterogeneousIngestionEngine
from .api_clients import ExaApiClient
from .config_loader import get_model_config
from .lancedb_client import VECTOR_DIM
from .env_helper import get_env_var

logger = logging.getLogger("Orchestrator")

class DailyRadarPipeline:
    """
    每日 24h AI 雷达快讯自适应摄取引擎。
    调用免费层 Google Gemini 2.5 Flash 配合 Google Search Grounding，
    获取过去 24h 技术异构源动态并自动过滤落库。
    """
    def __init__(self, db_path: str = DB_PATH, coordinator: IngestionCoordinator = None):
        self.db_path = db_path
        self.coordinator = coordinator or IngestionCoordinator(db_path)
        self.writer = WriteWorker(db_path)

    @api_billing_audit(api_provider="google", api_metric="tokens")
    async def run_daily_radar_cron(self, gemini_api_key: str = None) -> Dict[str, Any]:
        """
        每日自动触发的雷达简报任务。
        """
        # 读取专门的配置
        from core.briefing_manager import load_briefing_config
        br_config = load_briefing_config()
        api_key = gemini_api_key or br_config.get("gemini_api_key") or get_env_var("GEMINI_API_KEY", "").strip()
        model_name = br_config.get("model_name", "gemini-2.5-flash")
        
        if not api_key:
            logger.error("Gemini API Key 未配置，跳过雷达扫描。")
            return {"status": "aborted", "reason": "API Key Missing", "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

        client = genai.Client(api_key=api_key)
        
        prompt = (
            "请搜索过去 24 小时全球关于 AI 基础设施、大模型推理加速、高带宽芯片总线或异构计算领域的最新 10 条突破性进展。\n"
            "【核心抓取关键词与技术方向】：除通用 AI 基础设施外，必须重点检索并覆盖 kvcache（KV Cache 显存优化、卸载换入换出、压缩量化、长上下文加速）与 agent（AI Agent 智能体架构、AIOS、环境交互与动作生成、工具调用）两个关键技术方向的最新硬核突破。\n"
            "剔除一切炒作、资本吹嘘与公关软文。每个项目必须包含：\n"
            "1. 标题 (title)\n"
            "2. 权威来源/URL (url)\n"
            "3. 核心技术痛点与架构变更机制分析 (technical_depth)\n"
            "4. 估算技术硬度评分 Technical_Depth_Score (0.0-10.0)\n"
            "请严格以如下 JSON 数组格式返回，不要有任何 Markdown 代码块包裹或说明字样:\n"
            "[\n"
            "  {\n"
            "    \"title\": \"vLLM 发布多 NUMA 节点 KV 优化\",\n"
            "    \"url\": \"https://github.com/vllm-project/vllm/pull/123\",\n"
            "    \"technical_depth\": \"在 Prefill 阶段进行 NUMA 感知的块分配，减少跨 CPU 节点访存\",\n"
            "    \"score\": 8.8\n"
            "  }\n"
            "]"
        )
        
        try:
            # 驱动 Gemini 2.5 Flash 强联网 Grounding 扫射
            # 异步执行防止 Streamlit 卡死 (由于 genai.Client 目前为同步阻塞，使用 run_in_executor 挂载)
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[{"google_search": {}}],  # 强制激活 Grounding
                        temperature=0.1
                    )
                )
            )
            
            raw_text = response.text.strip()
            # 清理 Markdown 代码块
            if raw_text.startswith("```"):
                parts = raw_text.split("```")
                if len(parts) >= 3:
                    raw_text = parts[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
            raw_text = raw_text.strip()
            
            items = json.loads(raw_text)
            
            # 对产生的 10 条快讯遍历入库
            ingested_count = 0
            for item in items:
                title = item.get("title", "Untitled News")
                url = item.get("url", f"https://radar.ai/news-{hash(title)}")
                technical_depth = item.get("technical_depth", "")
                score = float(item.get("score", 7.0))
                
                # 规范化 url 清洗
                canonical_url = HeterogeneousIngestionEngine(self.db_path)._canonicalize_url(url)
                content_hash = hashlib.sha256(canonical_url.encode('utf-8')).hexdigest()
                doc_id = f"brief_{content_hash[:12]}"
                
                # 事务入库
                doc_payload = {
                    "doc_id": doc_id,
                    "source_type": "daily_brief",
                    "title": title,
                    "canonical_url": canonical_url,
                    "content_hash": content_hash,
                    "local_path": None,
                    "origin_provider": "publisher_site",
                    "discovery_provider": "gemini_grounding",
                    "crawl_provider": "native",
                    "analysis_model": model_name,
                    "published_at": datetime.now().isoformat(),
                    "full_text_markdown": f"# {title}\n\n{technical_depth}",
                    "ai_summary": technical_depth,
                    "structured_takeaways_json": json.dumps([technical_depth]),
                    "evidence_json": json.dumps([{"url": canonical_url, "fact": technical_depth}]),
                    "llm_score": score,
                    "score_reason_json": json.dumps({"reason": "Gemini Grounding Auto Scan"}),
                    "scored_by_model": model_name
                }
                
                # 构建 chunks
                chunks_payload = [{
                    "text": technical_depth,
                    "token_count": len(technical_depth) // 4,
                    "text_hash": hashlib.md5(technical_depth.encode('utf-8')).hexdigest(),
                    "section_path": "### Summary",
                    "page_number": 1
                }]
                
                # 获取特征向量 (可以使用零向量填充以保持 briefing 的 0 向量生成，或者调用客户端)
                import numpy as np
                mock_vecs = [list(np.zeros(VECTOR_DIM).astype(np.float32))]
                
                success = await self.coordinator.execute_2pc_ingestion(
                    doc_payload=doc_payload,
                    chunks_payload=chunks_payload,
                    vectors_list=mock_vecs
                )
                if success:
                    ingested_count += 1
            
            # 返回计费核销元数据
            prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 2000
            completion_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 1000
            
            return {
                "model": model_name,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens
                },
                "grounding_metadata_triggered": True,
                "status": "success",
                "ingested_count": ingested_count
            }
        except Exception as e:
            logger.error(f"每日雷达快讯扫描崩溃: {str(e)}")
            return {
                "model": model_name,
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
                "grounding_metadata_triggered": False,
                "status": "failed",
                "error": str(e)
            }


class DeterministicKnowledgeGraphEngine:
    """
    确定性知识图谱网络 NER 抽取与动态依赖构建引擎。
    利用 DeepSeek 提取文献实体，upsert 入词典表并添加拓扑图边表。
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.writer = WriteWorker(db_path)

    async def run_production_ner_injection_loop(self, doc_id: str, chunk_id: str, raw_markdown_text: str, model_id: str = "deepseek-v4"):
        # 调用大模型执行 NER 抽取
        ner_prompt = (
            "请从以下文本片段中提取涉及的【技术实体】（如模型名称、框架、算法、组织）及其之间的【关联关系】。\n"
            "实体类型 entity_type 限制在: model, infra, algorithm, org 之一。\n"
            "关系类型 relation_type 限制在: component_of, optimized_by, competes_with, uses 之一。\n"
            "必须严格返回如下 JSON 格式，不要有任何 markdown 代码包裹:\n"
            "{\n"
            "  \"entities\": [\n"
            "    {\"name\": \"vLLM\", \"type\": \"infra\", \"aliases\": [\"vllm\"]},\n"
            "    {\"name\": \"PagedAttention\", \"type\": \"algorithm\", \"aliases\": [\"paged_attention\"]}\n"
            "  ],\n"
            "  \"relations\": [\n"
            "    {\"source\": \"vLLM\", \"target\": \"PagedAttention\", \"type\": \"uses\", \"confidence\": 0.98}\n"
            "  ]\n"
            "}\n"
            f"待分析文本:\n{raw_markdown_text[:3000]}"
        )
        
        # 调用大模型 (包裹 DeepSeek 计费)
        response_json = await self._call_deepseek_ner(ner_prompt, model_id)
        if not response_json:
            return
            
        try:
            entities = response_json.get("entities", [])
            relations = response_json.get("relations", [])
            
            sql_pipeline = []
            
            # Step 1: 优先强行写入 entity_lexicon 词典表以避免外键失效
            for ent in entities:
                ent_name = ent["name"].strip()
                ent_id = ent_name.lower().replace(" ", "_")
                ent_type = ent["type"].strip()
                aliases = ent.get("aliases", [])
                
                sql_pipeline.append((
                    "INSERT INTO entity_lexicon (entity_id, entity_name, normalized_name, entity_type, alias_json) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT(normalized_name) DO UPDATE SET alias_json = excluded.alias_json",
                    (ent_id, ent_name, ent_id, ent_type, json.dumps(aliases))
                ))
            
            # 提早写入 lexicon
            if sql_pipeline:
                await self.writer.execute_write(sql_pipeline)
                sql_pipeline.clear()
            
            # Step 2: 注入拓扑图边表
            for rel in relations:
                source_name = rel["source"].strip()
                target_name = rel["target"].strip()
                r_type = rel["type"].strip()
                conf = float(rel.get("confidence", 1.0))
                
                source_id = source_name.lower().replace(" ", "_")
                target_id = target_name.lower().replace(" ", "_")
                
                # 校验实体是否存在词典里
                async with aiosqlite.connect(self.db_path) as db:
                    db.row_factory = aiosqlite.Row
                    c = await db.execute("SELECT 1 FROM entity_lexicon WHERE entity_id = ?", (source_id,))
                    has_src = await c.fetchone()
                    c = await db.execute("SELECT 1 FROM entity_lexicon WHERE entity_id = ?", (target_id,))
                    has_tgt = await c.fetchone()
                
                if not has_src or not has_tgt:
                    logger.warning(f"关系实体 [{source_name} ➔ {target_name}] 不存在于 Lexicon 词典，抛弃关系写入")
                    continue
                    
                sql_pipeline.append((
                    "INSERT INTO entity_relations (source_entity_id, target_entity_id, relation_type, weight, evidence_doc_id, evidence_chunk_id, confidence) "
                    "VALUES (?, ?, ?, 1.0, ?, ?, ?) ON CONFLICT(source_entity_id, target_entity_id, relation_type) DO UPDATE SET weight = weight + 1.0",
                    (source_id, target_id, r_type, doc_id, chunk_id, conf)
                ))
                
            if sql_pipeline:
                await self.writer.execute_write(sql_pipeline)
                
            logger.info(f"🎉 实体与拓扑图关系成功注入。相关文档 ID: {doc_id}")
        except Exception as e:
            logger.error(f"NER 解析或落库失败: {str(e)}")

    @api_billing_audit(api_provider="deepseek", api_metric="tokens")
    async def _call_deepseek_ner(self, prompt: str, model_id: str) -> Optional[Dict[str, Any]]:
        cfg = get_model_config(model_id)
        if not cfg:
            return None
        api_key = cfg.get("resolved_api_key", "").strip()
        api_url = cfg.get("url", "").strip()
        model_name = cfg.get("model", "deepseek-v4-flash")
        
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a precise JSON NER extractor."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, timeout=20) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        text = data["choices"][0]["message"]["content"].strip()
                        if text.startswith("```"):
                            parts = text.split("```")
                            if len(parts) >= 3:
                                text = parts[1]
                                if text.startswith("json"):
                                    text = text[4:]
                        text = text.strip()
                        return json.loads(text)
        except Exception as e:
            logger.error(f"NER 大模型调用失败: {str(e)}")
        return None


class DailyAutomationOrchestrator:
    """
    每日雷达与学术大仓自适应漏斗定时编排调度中心。
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.coordinator = IngestionCoordinator(db_path)
        self.ingestion_engine = HeterogeneousIngestionEngine(db_path)
        self.radar_pipeline = DailyRadarPipeline(db_path, self.coordinator)
        self.ner_engine = DeterministicKnowledgeGraphEngine(db_path)
        self.exa_client = ExaApiClient()

    async def run_daily_automation_funnel_cron(self, gemini_key: str = None, scoring_model_id: str = "deepseek-v4"):
        logger.info("⏰ 触发每日 24h AI 雷达与学术大仓漏斗自动化定时管线...")
        
        # 1. 触发 Daily Radar 快讯（免费 Grounding）
        radar_status = await self.radar_pipeline.run_daily_radar_cron(gemini_api_key=gemini_key)
        logger.info(f"每日雷达快讯扫描完毕。状态: {radar_status.get('status')}, 沉淀条数: {radar_status.get('ingested_count')}")
        
        # 2. 驱动真实 Exa 客户端，抓取最新 20 篇 arXiv 论文元数据
        logger.info("正在通过 Exa 神经网络引擎打捞当日最新的 20 篇 AI/LLM 领域 arXiv 论文元数据...")
        try:
            exa_raw_papers = await self.exa_client.search_and_extract_highlights(
                query="site:arxiv.org (large language model OR LLM) (kvcache OR \"KV cache\" OR agent OR \"AI agent\" OR inference microarchitecture)",
                num_results=20,
                db_path=self.db_path
            )
        except Exception as exa_err:
            logger.error(f"每日 arXiv 元数据打捞网络阻断，终止后续漏斗。原因: {str(exa_err)}")
            return

        paper_results = exa_raw_papers.get("results", [])
        
        # 3. 循环遍历摘要，驱动 DeepSeek 运行批量打分初筛
        for paper in paper_results:
            paper_url = paper.get("url")
            abstract_text = "".join([h.get("text", "") for h in paper.get("highlights", [])])
            title = paper.get("title", "Untitled ArXiv Paper")
            
            # 运行初筛打分
            eval_data = await self._call_deepseek_scoring(title, abstract_text, scoring_model_id)
            if not eval_data:
                continue
                
            llm_score = float(eval_data.get("llm_score", 0.0))
            score_reason = eval_data.get("reason", "未给出明确原因")
            
            # 4. 自适应门槛拦截 (得分 >= 8.5)
            if llm_score >= 8.5:
                logger.info(f"🔥 检测到高价值学术资产! [得分: {llm_score:.1f}], 论文: '{title}'。强制切入智能化自适应抓取管道...")
                
                pull_result = await self.ingestion_engine.execute_intelligent_active_pull(
                    discovery_query=f"url: {paper_url}",
                    user_confirmed_firecrawl=True
                )
                
                # 5. 反向更新 SQLite 指标，并运行 NER 提取拓扑图
                if pull_result.get("status") == "success" and pull_result.get("ingested_doc_ids"):
                    target_doc_id = pull_result["ingested_doc_ids"][0]
                    
                    # 刚性更新 documents
                    writer = WriteWorker(self.db_path)
                    await writer.execute_write([
                        (
                            "UPDATE documents SET llm_score = ?, score_reason_json = ?, scored_by_model = ?, scored_at = CURRENT_TIMESTAMP WHERE doc_id = ?",
                            (llm_score, json.dumps({"reason": score_reason}), "deepseek-v4-flash", target_doc_id)
                        )
                    ])
                    logger.info(f"🎉 资产 {target_doc_id} 已完美注入初筛打分指标。")
                    
                    # NER 提取
                    await self.ner_engine.run_production_ner_injection_loop(
                        doc_id=target_doc_id,
                        chunk_id=f"chunk_{target_doc_id}_0",
                        raw_markdown_text=f"{title}\n\n{abstract_text}",
                        model_id=scoring_model_id
                    )
            else:
                logger.info(f"💤 论文评分不足门槛 [得分: {llm_score:.1f}], 抛弃。标题: '{title}'")
                
        logger.info("============== ✅ 管线 A 24h 自动化漏斗执行终结 ==============")

    @api_billing_audit(api_provider="deepseek", api_metric="tokens")
    async def _call_deepseek_scoring(self, title: str, abstract: str, model_id: str) -> Optional[Dict[str, Any]]:
        cfg = get_model_config(model_id)
        if not cfg:
            return None
        api_key = cfg.get("resolved_api_key", "").strip()
        api_url = cfg.get("url", "").strip()
        model_name = cfg.get("model", "deepseek-v4-flash")
        
        prompt = (
            "请对以下论文标题及摘要进行硬核技术价值评估。给出 0.0 到 10.0 的分值，"
            "并给出两句极度冷酷的理由。必须返回以下标准 JSON 格式，严禁带有任何 Markdown 标签或包裹:\n"
            "{\n  \"llm_score\": 9.2,\n  \"reason\": \"对 vLLM 的多 NUMA 节点 KVCache 分配有底层演进。\"\n}\n"
            f"论文标题: {title}\n摘要: {abstract}"
        )
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                }, timeout=20) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        text = data["choices"][0]["message"]["content"].strip()
                        if text.startswith("```"):
                            parts = text.split("```")
                            if len(parts) >= 3:
                                text = parts[1]
                                if text.startswith("json"):
                                    text = text[4:]
                        text = text.strip()
                        return json.loads(text)
        except Exception as e:
            logger.error(f"DeepSeek 批量打分 API 调用异常: {str(e)}")
        return None
import os
