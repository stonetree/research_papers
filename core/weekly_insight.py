# -*- coding: utf-8 -*-
import os
import json
import logging
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any
import numpy as np
import aiosqlite
from google import genai
from google.genai import types

from .database import DB_PATH
from .lancedb_client import LanceDBClient, VECTOR_DIM
from .api_clients import LocalComputeKernelClient
from .cost_manager import api_billing_audit
from .write_worker import WriteWorker
from .config_loader import get_model_config
from .ingestion import IngestionCoordinator
from .env_helper import get_env_var

logger = logging.getLogger("WeeklyInsight")

class WeeklyInsightPipeline:
    """
    算法化周报高密度二次熔炼管线。
    1. Candidate Pool 过滤筛选本周高硬度文献；
    2. 本地向量空间余弦聚类去重，计算时空新颖度 Novelty_Score；
    3. 构建标准工程事实证据卡片（Evidence Cards）；
    4. 驱动大语言模型严格基于证据卡片矩阵合成强引用周报，并写入大仓。
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.lancedb_client = LanceDBClient()
        self.compute_client = LocalComputeKernelClient()
        self.writer = WriteWorker(db_path)
        self.coordinator = IngestionCoordinator(db_path)

    async def run_weekly_synthesis_pipeline(self, gemini_api_key: str = None) -> Dict[str, Any]:
        logger.info("📅 启动每周技术深度洞察算法化熔炼管线...")
        
        # Step 1: Candidate Pool 自动筛选池
        # 筛选过去 7 天内落库，且评分 >= 8.0 的高价值原始资产
        seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
        
        candidates = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            sql = """
                SELECT doc_id, title, canonical_url, llm_score, source_type 
                FROM documents 
                WHERE source_type IN ('arxiv_paper', 'local_pdf', 'ext_blog') 
                  AND llm_score >= 8.0 
                  AND ingested_at >= ?
            """
            async with db.execute(sql, (seven_days_ago,)) as cursor:
                rows = await cursor.fetchall()
                candidates = [dict(row) for row in rows]
                
        if not candidates:
            logger.warning("📭 过去 7 天内大仓无评级分数 >= 8.0 的高硬度资产，周报熔炼取消。")
            return {"status": "empty", "reason": "No high-score candidates in the past 7 days."}

        # Step 2: 向量空间聚类去重与新颖度审查
        # 2.1 获取本周候选文档的全局 Embedding 向量
        valid_candidates = []
        candidate_vectors = []
        
        for cand in candidates:
            doc_id = cand["doc_id"]
            source_type = cand["source_type"]
            # 从 LanceDB 中读取该文档的第一个切片向量作为代表向量
            table_name = self.lancedb_client.get_table_name(source_type)
            try:
                tbl = self.lancedb_client.get_or_create_table(source_type)
                # 读取该 parent_id 的所有向量
                res = tbl.search(list(np.zeros(VECTOR_DIM))).where(f"metadata.parent_id = '{doc_id}'").limit(1).to_list()
                if res:
                    vec = res[0]["vector"]
                    candidate_vectors.append(vec)
                    valid_candidates.append(cand)
            except Exception as e:
                logger.error(f"提取 LanceDB 代表向量失败. DocID: {doc_id}. 原因: {str(e)}")
                
        if not valid_candidates:
            return {"status": "empty", "reason": "No vector chunks found in LanceDB for candidates."}

        # 2.2 本地余弦去重 (Similarity > 0.85 视为同源/同质化技术事件，只保留分数高者)
        n_cands = len(valid_candidates)
        keep_indices = set(range(n_cands))
        
        for i in range(n_cands):
            if i not in keep_indices:
                continue
            for j in range(i + 1, n_cands):
                if j not in keep_indices:
                    continue
                v_i = np.array(candidate_vectors[i])
                v_j = np.array(candidate_vectors[j])
                # 计算余弦相似度
                cos_sim = np.dot(v_i, v_j) / (np.linalg.norm(v_i) * np.linalg.norm(v_j) + 1e-9)
                if cos_sim > 0.85:
                    # 去除分数低的一个
                    if valid_candidates[i]["llm_score"] >= valid_candidates[j]["llm_score"]:
                        keep_indices.discard(j)
                    else:
                        keep_indices.discard(i)
                        break
                        
        filtered_candidates = [valid_candidates[idx] for idx in keep_indices]
        filtered_vectors = [candidate_vectors[idx] for idx in keep_indices]
        
        if not filtered_candidates:
            return {"status": "empty", "reason": "All candidates were clustered and removed."}

        # 2.3 计算 Novelty Score 时空新颖度
        # 拉取过去 4 周 (排除本周) 沉淀的背景技术代表向量
        four_weeks_ago = (datetime.now() - timedelta(days=28)).isoformat()
        baseline_vectors = []
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            sql = """
                SELECT doc_id, source_type FROM documents 
                WHERE source_type IN ('arxiv_paper', 'local_pdf', 'ext_blog')
                  AND ingested_at >= ? AND ingested_at < ?
            """
            async with db.execute(sql, (four_weeks_ago, seven_days_ago)) as cursor:
                base_rows = await cursor.fetchall()
                for brow in base_rows:
                    b_id = brow["doc_id"]
                    b_type = brow["source_type"]
                    try:
                        tbl = self.lancedb_client.get_or_create_table(b_type)
                        res = tbl.search(list(np.zeros(VECTOR_DIM))).where(f"metadata.parent_id = '{b_id}'").limit(1).to_list()
                        if res:
                            baseline_vectors.append(res[0]["vector"])
                    except:
                        pass
                        
        # 如果背景基准库为空，则默认全部视为全新
        if baseline_vectors:
            baseline_mean = np.mean(baseline_vectors, axis=0)
            final_selected_candidates = []
            for cand, vec in zip(filtered_candidates, filtered_vectors):
                cos_sim = np.dot(vec, baseline_mean) / (np.linalg.norm(vec) * np.linalg.norm(baseline_mean) + 1e-9)
                novelty_score = 1.0 - cos_sim
                cand["novelty_score"] = float(novelty_score)
                # 过滤掉陈旧/炒作概念 (Novelty_Score 必须大于 0.15 作为硬门槛)
                if novelty_score > 0.15:
                    final_selected_candidates.append(cand)
        else:
            final_selected_candidates = filtered_candidates
            for cand in final_selected_candidates:
                cand["novelty_score"] = 1.0

        if not final_selected_candidates:
            return {"status": "empty", "reason": "All candidates filtered out by novelty threshold."}

        # Step 3: 标准事实证据卡片矩阵（Evidence Card Matrix）组装
        evidence_cards = []
        for cand in final_selected_candidates:
            doc_id = cand["doc_id"]
            # 读取该 doc_id 的结构化内容
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute("SELECT full_text_markdown, ai_summary FROM document_contents WHERE doc_id = ?", (doc_id,)) as cur:
                    content_row = await cur.fetchone()
                    if content_row:
                        evidence_cards.append({
                            "doc_id": doc_id,
                            "title": cand["title"],
                            "url": cand["canonical_url"],
                            "summary": content_row["ai_summary"],
                            "novelty": cand.get("novelty_score", 1.0)
                        })

        # Step 4: 驱动大模型强事实绑定周报合成
        from core.briefing_manager import load_briefing_config
        br_config = load_briefing_config()
        api_key = gemini_api_key or br_config.get("gemini_api_key") or get_env_var("GEMINI_API_KEY", "").strip()
        model_name = br_config.get("model_name", "gemini-2.5-flash")
        
        if not api_key:
            return {"status": "failed", "reason": "Gemini API key missing for weekly synthesis."}

        client = genai.Client(api_key=api_key)
        
        # 将卡片矩阵格式化为 prompt 输入
        cards_text = ""
        for idx, card in enumerate(evidence_cards):
            cards_text += (
                f"=== 证据卡片 [{idx+1}] ===\n"
                f"名称: {card['title']}\n"
                f"doc_id: {card['doc_id']}\n"
                f"物理源地址: {card['url']}\n"
                f"技术概要: {card['summary']}\n"
                f"新颖度分数: {card['novelty']:.2f}\n---\n"
            )
            
        system_instruction = (
            "你是一个权威的 AI 软硬件协同首席科学家与系统架构总监。\n"
            "用户向你交付了本周打捞沉淀的【高硬度事实证据卡片矩阵】。\n"
            "请撰写一份本周 AI 基础设施与异构计算前沿的技术洞察白皮书（周报）。\n\n"
            "你的白皮书必须严格满足以下事实约束：\n"
            "1. 严禁模型自由发挥与想象。只能基于提供的卡片内容提炼机制、边界差异与行业落地价值。\n"
            "2. 在作出的任何技术性断论和陈述句尾，必须强制显式绑定数据湖中的 `[doc_id]` 引用标签（例如：... NUMA 节点访存开销能消减 40% [ext_blog_cf123]）。\n"
            "3. 周报必须排版清晰，结构完整，包含背景说明、本周重大突破分析、以及未来工程复刻建议。"
        )
        
        try:
            # 异步调度 Gemini
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=f"证据卡片矩阵：\n{cards_text}\n\n请严格基于此矩阵撰写每周深度洞察白皮书。",
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.1,
                        max_output_tokens=65536
                    )
                )
            )
            
            report_md = response.text.strip()
            
            # 刚性过滤检查：如果生成内容中未携带任何 [doc_id] 引用，直接打回或强制添加警告
            if not any(card["doc_id"] in report_md for card in evidence_cards):
                logger.warning("大模型生成的周报未携带 doc_id 引用，强制注入引用尾标。")
                # 强制追加事实绑定附录
                report_md += "\n\n### ⚖️ 本周事实溯源审计列表\n"
                for card in evidence_cards:
                    report_md += f"- 依据大仓文献 [{card['doc_id']}]: {card['title']} (地址: {card['url']})\n"
            
            # 写入 SQLite documents 归档落盘
            content_hash = hashlib.sha256(report_md.encode('utf-8')).hexdigest()
            doc_id = f"weekly_{content_hash[:12]}"
            title = f"每周 AI 技术深入洞察白皮书 ({datetime.now().strftime('%Y-%m-%d')})"
            
            doc_payload = {
                "doc_id": doc_id,
                "source_type": "weekly_insight",
                "title": title,
                "canonical_url": f"https://radar.ai/weekly-{doc_id}",
                "content_hash": content_hash,
                "local_path": None,
                "origin_provider": "publisher_site",
                "discovery_provider": "manual",
                "crawl_provider": "native",
                "analysis_model": model_name,
                "published_at": datetime.now().isoformat(),
                "full_text_markdown": report_md,
                "ai_summary": "基于本周高分数文献自动熔炼产生的每周技术深入洞察报告。",
                "structured_takeaways_json": json.dumps(filtered_candidates),
                "evidence_json": json.dumps(evidence_cards),
                "llm_score": 9.5,
                "score_reason_json": json.dumps({"reason": "Weekly synthesis logic"}),
                "scored_by_model": model_name
            }
            
            chunks_payload = [{
                "text": report_md[:1000],
                "token_count": 250,
                "text_hash": hashlib.md5(report_md[:1000].encode('utf-8')).hexdigest(),
                "section_path": "### Introduction",
                "page_number": 1
            }]
            
            # 使用 mock 向量注入协调器提交
            mock_vecs = [list(np.zeros(VECTOR_DIM).astype(np.float32))]
            
            is_ok = await self.coordinator.execute_2pc_ingestion(
                doc_payload=doc_payload,
                chunks_payload=chunks_payload,
                vectors_list=mock_vecs
            )
            
            if is_ok:
                # 顺便写一份实体 markdown 文件落入本地 storage 目录，供前端 markdown 归档阅读器完美读取
                brief_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "briefings")
                os.makedirs(brief_dir, exist_ok=True)
                
                # 命名规则符合 app.py 归档阅读器逻辑
                report_file = os.path.join(brief_dir, f"weekly_{datetime.now().strftime('%Y%m%d')}_{doc_id[:8]}.md")
                with open(report_file, "w", encoding="utf-8") as f:
                    f.write(report_md)
                    
                logger.info(f"✅ 周报成功生成并落盘登记！DocID: {doc_id}, 物理文件: {report_file}")
                return {"status": "success", "doc_id": doc_id, "file_path": report_file}
            else:
                return {"status": "failed", "reason": "2PC database commit failed for weekly synthesis."}
                
        except Exception as e:
            logger.error(f"大模型周报生成异常: {str(e)}")
            return {"status": "failed", "reason": str(e)}
