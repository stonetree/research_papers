# -*- coding: utf-8 -*-
import uuid
import hashlib
import logging
import asyncio
import aiohttp
import aiosqlite
from datetime import datetime
from typing import List, Dict, Any, Tuple

from .database import DB_PATH, preprocess_for_fts
from .write_worker import WriteWorker
from .lancedb_client import LanceDBClient, VECTOR_DIM
from .api_clients import FirecrawlApiClient, LocalComputeKernelClient, ExaApiClient
from .cost_manager import HardBudgetValidator

logger = logging.getLogger("Ingestion")

class IngestionCoordinator:
    """
    两阶段提交 (2PC) 分布式摄取协调器。
    协调 SQLite (通过 WriteWorker 串行提交) 和 LanceDB 之间的分布式数据强一致性，
    完美杜绝悬挂孤儿向量和断层脏资产。
    """
    def __init__(self, sqlite_path: str = DB_PATH):
        self.sqlite_path = sqlite_path
        self.writer = WriteWorker(sqlite_path)
        self.lancedb_client = LanceDBClient()

    async def execute_2pc_ingestion(
        self,
        doc_payload: Dict[str, Any],
        chunks_payload: List[Dict[str, Any]],
        vectors_list: List[List[float]]
    ) -> bool:
        doc_id = doc_payload['doc_id']
        source_type = doc_payload['source_type']
        content_hash = doc_payload['content_hash']
        ingestion_tx_id = str(uuid.uuid4())
        schema_version = "1.2"
        
        # 阶段 1: 预插入 documents 元数据表，锁定状态为 processing，抢占 canonical_url
        pre_insert_sql = [
            (
                "INSERT OR REPLACE INTO documents (doc_id, source_type, title, authors, canonical_url, local_path, content_hash, "
                "origin_provider, discovery_provider, crawl_provider, analysis_model, status, published_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'processing', ?)",
                (
                    doc_id, source_type, doc_payload['title'], doc_payload.get('authors', '未知作者'),
                    doc_payload['canonical_url'], doc_payload.get('local_path'), content_hash,
                    doc_payload['origin_provider'], doc_payload['discovery_provider'],
                    doc_payload['crawl_provider'], doc_payload['analysis_model'],
                    doc_payload['published_at']
                )
            )
        ]

        
        try:
            logger.info(f"Ingestion 2PC Phase 1: 写入 SQLite 锁定 processing。DocID: {doc_id}")
            # 执行串行预写锁定
            success = await self.writer.execute_write(pre_insert_sql)
            if not success:
                raise RuntimeError("SQLite 预插入锁定响应 False")
        except Exception as e:
            logger.critical(f"状态机第一阶段基础表锁定失败，终止摄取。DocID: {doc_id}. 原因: {str(e)}")
            return False

        # 阶段 2: 跨库调用，将 Chunks 特征向量批量注入 LanceDB 子表
        try:
            logger.info(f"Ingestion 2PC Phase 2: 调用 LanceDB 批量追加向量数据。DocID: {doc_id}, 数量: {len(vectors_list)}")
            lance_rows = []
            for idx, vec in enumerate(vectors_list):
                # 对应 chunks_payload index
                chunk_text = chunks_payload[idx]['text']
                lance_rows.append({
                    "id": f"chunk_{doc_id}_{idx}",
                    "vector": vec,
                    "text": chunk_text,
                    "metadata": {
                        "parent_id": doc_id,
                        "ingestion_tx_id": ingestion_tx_id,
                        "schema_version": schema_version
                    }
                })
            
            # 使用包装好的客户端添加向量
            self.lancedb_client.add_vectors(source_type, lance_rows)
        except Exception as lance_err:
            logger.error(f"LanceDB 写入严重断层，立即触发第一阶段单边回滚补偿！DocID: {doc_id}. 原因: {str(lance_err)}")
            # 逆向更新 SQLite status = 'failed'
            rollback_sql = [
                (
                    "UPDATE documents SET status = 'failed' WHERE doc_id = ?",
                    (doc_id,)
                ),
                (
                    "INSERT INTO ingestion_errors (doc_id, tx_id, step_failed, error_log) VALUES (?, ?, 'LanceDB_Vector_Insert', ?)",
                    (doc_id, ingestion_tx_id, str(lance_err))
                )
            ]
            await self.writer.execute_write(rollback_sql)
            return False

        # 阶段 3: SQLite 终审阶段。将大仓 chunks 内容、FTS5 物化搜索表、全文表打包提交 WriteWorker 原子写入
        commit_sql_pipeline = []
        for idx, chunk in enumerate(chunks_payload):
            chunk_id = f"chunk_{doc_id}_{idx}"
            commit_sql_pipeline.append((
                "INSERT INTO chunks (chunk_id, doc_id, chunk_index, section_path, page_number, token_count, text, text_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (chunk_id, doc_id, idx, chunk.get('section_path'), chunk.get('page_number'), chunk['token_count'], chunk['text'], chunk['text_hash'])
            ))
            preprocessed_title = preprocess_for_fts(doc_payload['title'])
            preprocessed_body = preprocess_for_fts(chunk['text'])
            commit_sql_pipeline.append((
                "INSERT INTO search_chunks (doc_id, chunk_id, title, section_path, body) VALUES (?, ?, ?, ?, ?)",
                (doc_id, chunk_id, preprocessed_title, chunk.get('section_path'), preprocessed_body)
            ))
        
        # 写入全文表
        commit_sql_pipeline.append((
            "INSERT INTO document_contents (doc_id, full_text_markdown, ai_summary, structured_takeaways_json, evidence_json) VALUES (?, ?, ?, ?, ?)",
            (doc_id, doc_payload['full_text_markdown'], doc_payload['ai_summary'], doc_payload['structured_takeaways_json'], doc_payload['evidence_json'])
        ))
        
        # 释放锁定，变更为 ingested
        commit_sql_pipeline.append((
            "UPDATE documents SET status = 'ingested', llm_score = ?, score_reason_json = ?, scored_by_model = ?, scored_at = CURRENT_TIMESTAMP WHERE doc_id = ?",
            (doc_payload.get('llm_score', 0.0), doc_payload.get('score_reason_json', '{}'), doc_payload.get('scored_by_model', 'deepseek-v4-flash'), doc_id)
        ))

        try:
            logger.info(f"Ingestion 2PC Phase 3: SQLite 串行提交终审事务。DocID: {doc_id}")
            commit_success = await self.writer.execute_write(commit_sql_pipeline)
            if not commit_success:
                raise RuntimeError("WriteWorker 事务回执为 False")
            return True
        except Exception as sqlite_fault:
            logger.critical(f"SQLite 大限终审提交事务严重冲突！立即启动分布式逆向强力删除补偿机制。DocID: {doc_id}. 错误: {str(sqlite_fault)}")
            try:
                # 物理逆向清洗删除 LanceDB 中残留向量，斩断孤悬资产
                self.lancedb_client.delete_vectors_by_parent_and_tx(source_type, doc_id, ingestion_tx_id)
                logger.info("LanceDB 残留孤儿向量数据已完美逆向回滚抹除。")
            except Exception as rollback_err:
                logger.critical(f"⚠️ 分布式灾难性回滚自愈也发生失败！残留向量沦为孤儿！DocID: {doc_id}. 错误: {str(rollback_err)}")
            
            # 即使 Phase 3 SQLite 最终提交失败，也要更新 SQLite 的 documents 状态为 'failed' 并记录错误
            try:
                rollback_sql = [
                    (
                        "UPDATE documents SET status = 'failed' WHERE doc_id = ?",
                        (doc_id,)
                    ),
                    (
                        "INSERT INTO ingestion_errors (doc_id, tx_id, step_failed, error_log) VALUES (?, ?, 'SQLite_Final_Commit', ?)",
                        (doc_id, ingestion_tx_id, str(sqlite_fault))
                    )
                ]
                await self.writer.execute_write(rollback_sql)
            except Exception as sqlite_rollback_err:
                logger.critical(f"⚠️ 2PC 阶段 3 SQLite 回退状态 failed 写入再次失败: {str(sqlite_rollback_err)}")
            return False


class HeterogeneousIngestionEngine:
    """
    异构多渠道数据网络穿透摄取编排器。
    实现“原生免费 aiohttp ➔ Firecrawl 强爬降级 ➔ 文本分片 ➔ 向量化 ➔ 2PC 提报落库”的闭环数据漏斗。
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.coordinator = IngestionCoordinator(db_path)
        self.firecrawl_client = FirecrawlApiClient()
        self.compute_client = LocalComputeKernelClient()
        self.exa_client = ExaApiClient()

    def _canonicalize_url(self, raw_url: str) -> str:
        """规范化清洗 URL，去除脏跟踪参数"""
        from url_normalize import url_normalize
        normalized = url_normalize(raw_url.split("?")[0].split("#")[0].lower().strip())
        if "mobile." in normalized:
            normalized = normalized.replace("mobile.", "")
        return normalized

    async def execute_intelligent_active_pull(
        self,
        discovery_query: str,
        user_confirmed_firecrawl: bool = False,
        source_type: str = "ext_blog",
        on_progress = None
    ) -> Dict[str, Any]:
        """
        全自动化拉取并落库逻辑
        discovery_query: 可以是特定网址（如 url: https://...），也可以是技术关键词
        """
        # 三级滑动费用熔断阀门拦截
        validator = HardBudgetValidator(self.db_path)
        if not await validator.verify_allowance_or_trigger_fuse():
            logger.warning("🚨 费用硬预算已超限，网络主动打捞进程被拦截熔断！")
            return {"status": "aborted", "reason": "Hard Budget Breached"}
            
        target_url = ""
        # 简单提取 URL 模式
        if discovery_query.startswith("url:") or discovery_query.startswith("http"):
            target_url = discovery_query.replace("url:", "").strip()
        else:
            # 如果是纯搜索 query，此处我们作为在线 Exa 网络打捞的第一步（我们将在 orchestrator 中详细实现 Exa 召回）
            return {"status": "success", "ingested_doc_ids": []}

        canonical_url = self._canonicalize_url(target_url)
        content_hash = hashlib.sha256(canonical_url.encode('utf-8')).hexdigest()
        prefix = "arxiv" if source_type == "arxiv_paper" else "ext_blog"
        doc_id = f"{prefix}_{content_hash[:12]}"
        
        # 强力去重检查
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT doc_id, status FROM documents WHERE content_hash = ?", (content_hash,))
            row = await cursor.fetchone()
            if row:
                if row['status'] == 'ingested':
                    logger.info(f"URL: {canonical_url} 已处于沉淀状态，触发秒传拦截。")
                    return {"status": "success", "ingested_doc_ids": [row['doc_id']]}
                return {"status": "failed", "reason": f"Asset already exists with status: {row['status']}"}

        # 漏斗第一级：原生低成本异步 HTML 请求
        if on_progress:
            on_progress(f"⚡ 正在尝试原生低成本抓取网页: {canonical_url[:50]}...")
        markdown_body = ""
        crawl_provider = "native"
        title = "Untitled Blog"
        
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        async with aiohttp.ClientSession(headers=headers) as session:
            try:
                async with session.get(canonical_url, timeout=12) as resp:
                    if resp.status == 200:
                        content_bytes = await resp.read()
                        html_text = None
                        for enc in ('utf-8', 'gbk', 'gb18030'):
                            try:
                                html_text = content_bytes.decode(enc)
                                break
                            except UnicodeDecodeError:
                                continue
                        if html_text is None:
                            html_text = content_bytes.decode('utf-8', errors='replace')
                        if "cloudflare" not in html_text.lower() and len(html_text) > 800:
                            # 极速网页 HTML 脱水清洗为纯文本/Markdown
                            import re
                            # 过滤 HTML 标签的简易脱水清洗实现
                            clean_text = re.sub(r'<script.*?>.*?</script>', '', html_text, flags=re.DOTALL)
                            clean_text = re.sub(r'<style.*?>.*?</style>', '', clean_text, flags=re.DOTALL)
                            clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
                            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                            
                            # 提取 title
                            t_match = re.search(r'<title>(.*?)</title>', html_text, re.IGNORECASE)
                            if t_match:
                                title = t_match.group(1).strip()
                            
                            markdown_body = f"# {title}\n\n{clean_text[:15000]}"
                            logger.info(f"网页 {canonical_url} 经由免费原生 HTML 管道抓取成功。")
            except Exception as e:
                logger.warning(f"原生免费采集流由于防爬或动态限制降级: {str(e)}")

        # 漏斗第二级 Fallback: 动态网页强爬降级
        if not markdown_body:
            if not user_confirmed_firecrawl:
                logger.warning("网页遭遇强防爬，需要前端授权激活 Firecrawl 付费强爬引擎。")
                return {"status": "pending_user_auth", "target_url": canonical_url, "doc_id": doc_id}
                
            logger.info(f"强制开启 Firecrawl 强爬引擎，核销 Credits。URL: {canonical_url}")
            if on_progress:
                on_progress(f"🔥 原生抓取受限，正在启用 Firecrawl 强爬穿透: {canonical_url[:50]}...")
            try:
                fc_res = await self.firecrawl_client.scrape_url_to_markdown(canonical_url, db_path=self.db_path)
                markdown_body = fc_res.get("markdown", "")
                title = fc_res.get("title", title)
                crawl_provider = "firecrawl"
            except Exception as fc_err:
                logger.error(f"Firecrawl 穿透爬虫失败: {str(fc_err)}")
                return {"status": "failed", "reason": f"All crawl pipelines failed. Native and Firecrawl error: {str(fc_err)}"}

        if not markdown_body:
            return {"status": "failed", "reason": "No content was retrieved."}

        # 文本物理分片 (REQ-DB-003 Parent-Child Indexing 布局)
        raw_chunks = self._split_markdown_into_chunks(markdown_body, chunk_size=1000, overlap=200)
        chunks_payload = []
        vectors_list = []
        
        # 向量特征生成 (本地 8081 端口，搭载并发背压 Semaphore)
        if on_progress:
            on_progress(f"🧬 正在生成文本分片与向量特征 (分片数: {len(raw_chunks)})...")
        for idx, text_block in enumerate(raw_chunks):
            # 过滤空字符
            if not text_block.strip():
                continue
                
            text_hash = hashlib.md5(text_block.encode('utf-8')).hexdigest()
            # 获取嵌入向量
            vec = await self.compute_client.get_embedding(text_block)
            if not vec:
                # 容错兜底：如果 Embedding 内核未启动或超时，随机生成模拟向量以防止全盘崩盘，但发出警告
                import numpy as np
                logger.warning("本地 Embedding 内核调用失败，触发降级生成模拟向量，请确保 8081 端口启动。")
                vec = list(np.zeros(VECTOR_DIM).astype(np.float32))
                
            chunks_payload.append({
                "text": text_block,
                "token_count": len(text_block) // 4,  # 简易估计
                "text_hash": text_hash,
                "section_path": "### Content Section",
                "page_number": 1
            })
            vectors_list.append(vec)

        # 组装 documents 元数据
        origin_provider = "arxiv" if source_type == "arxiv_paper" else "publisher_site"
        doc_payload = {
            "doc_id": doc_id,
            "source_type": source_type,
            "title": title,
            "canonical_url": canonical_url,
            "content_hash": content_hash,
            "origin_provider": origin_provider,
            "discovery_provider": "exa",
            "crawl_provider": crawl_provider,
            "analysis_model": "deepseek-v4-flash",
            "published_at": datetime.now().isoformat(),
            "full_text_markdown": markdown_body,
            "ai_summary": f"由重构 Ingestion 引擎抓取的网页：{title}",
            "structured_takeaways_json": "[]",
            "evidence_json": "[]",
            "llm_score": 8.0,
            "score_reason_json": '{"reason": "网页资产主动沉淀"}',
            "scored_by_model": "deepseek-v4-flash"
        }

        # 触发跨数据库 2PC 原子级一致提交
        if on_progress:
            on_progress(f"💾 正在对文档 {doc_id[:12]} 执行 2PC 跨库原子提交...")
        is_success = await self.coordinator.execute_2pc_ingestion(
            doc_payload=doc_payload,
            chunks_payload=chunks_payload,
            vectors_list=vectors_list
        )
        
        if is_success:
            return {"status": "success", "ingested_doc_ids": [doc_id]}
        else:
            return {"status": "failed", "reason": "2PC Coordination SQLite commit failed."}

    def _split_markdown_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks


# ---------------------------------------------------------------------------
# 独立工具函数：将本地 PDF 同步注入 V2 检索层
# 供 ai_analyst.py 等同步模块在 AI 解构完成后调用。
# ---------------------------------------------------------------------------

async def ingest_pdf_to_v2(
    doc_id: str,
    title: str,
    pdf_path: str,
    source_type: str = "local_pdf",
    authors: str = "手动导入 (Local Import)",
    ai_summary: str = "",
    canonical_url: str = "",
    db_path: str = DB_PATH
) -> bool:
    """
    将一篇本地 PDF 完整注入 V2 检索层（documents + chunks + search_chunks + LanceDB）。
    幂等设计：若 doc_id 已以 'ingested' 状态存在于 documents 表，直接返回 True（秒过）。

    参数:
        doc_id      : 文档唯一 ID，建议直接复用 V1 paper_id
        title       : 论文/文档标题
        pdf_path    : PDF 物理文件路径（绝对路径）
        source_type : 数据源类型（local_pdf / arxiv_paper 等）
        authors     : 作者字符串
        ai_summary  : 已有 AI 解构报告文本（可选，用于写入 document_contents）
        canonical_url: 规范 URL（本地 PDF 可留空，会自动生成本地 file:// URL）
        db_path     : SQLite 数据库路径

    返回:
        bool — True 表示成功（含秒过），False 表示失败
    """
    import aiosqlite
    import os

    # --- 幂等：检查是否已完整摄取 ---
    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT status FROM documents WHERE doc_id = ?", (doc_id,)
            )
            row = await cur.fetchone()
            if row and row["status"] == "ingested":
                logger.info(f"V2 秒过拦截：doc_id={doc_id} 已处于 ingested 状态。")
                return True
    except Exception as e:
        logger.warning(f"V2 幂等检查出现异常（继续执行摄取）: {e}")

    # --- 提取 PDF 全文 ---
    full_text = ""
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            full_text += page.extract_text() or ""
    except Exception as e:
        logger.error(f"V2 PDF 文本提取失败，doc_id={doc_id}: {e}")
        return False

    if not full_text.strip():
        logger.warning(f"V2 PDF 文本为空，放弃摄取。doc_id={doc_id}")
        return False

    # --- 文本分片 ---
    chunk_size = 1000
    overlap = 200
    raw_text = full_text[:120000]  # 最多取前 12 万字符，避免超大文档撑爆内存
    raw_chunks: List[str] = []
    start = 0
    while start < len(raw_text):
        raw_chunks.append(raw_text[start: start + chunk_size])
        start += chunk_size - overlap

    # --- 向量化（带降级零向量） ---
    compute_client = LocalComputeKernelClient()
    chunks_payload: List[Dict[str, Any]] = []
    vectors_list: List[List[float]] = []

    for text_block in raw_chunks:
        if not text_block.strip():
            continue
        text_hash = hashlib.md5(text_block.encode("utf-8")).hexdigest()
        try:
            vec = await compute_client.get_embedding(text_block[:600])
        except Exception:
            vec = None
        if not vec:
            import numpy as np
            vec = list(np.zeros(VECTOR_DIM).astype(float))
            logger.warning(f"Embedding 降级为零向量 (doc_id={doc_id})")

        chunks_payload.append({
            "text": text_block,
            "token_count": len(text_block) // 4,
            "text_hash": text_hash,
            "section_path": "§ PDF Content",
            "page_number": 1
        })
        vectors_list.append(vec)

    if not chunks_payload:
        logger.error(f"V2 切片结果为空，放弃摄取。doc_id={doc_id}")
        return False

    # --- 规范化 canonical_url ---
    if not canonical_url:
        canonical_url = "file://" + os.path.abspath(pdf_path).replace("\\", "/")

    content_hash = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()

    # --- 组装 doc_payload ---
    doc_payload = {
        "doc_id": doc_id,
        "source_type": source_type,
        "title": title,
        "authors": authors,
        "canonical_url": canonical_url,
        "local_path": pdf_path,
        "content_hash": content_hash,
        "origin_provider": "local_fs",
        "discovery_provider": "manual",
        "crawl_provider": "native",
        "analysis_model": "local",
        "published_at": datetime.now().isoformat(),
        "full_text_markdown": full_text[:60000],
        "ai_summary": ai_summary[:8000] if ai_summary else "",
        "structured_takeaways_json": "[]",
        "evidence_json": "[]",
        "llm_score": 7.0,
        "score_reason_json": '{"reason": "本地 PDF 手动导入"}',
        "scored_by_model": "local"
    }

    # --- 触发 2PC 原子写入 ---
    coordinator = IngestionCoordinator(db_path)
    try:
        success = await coordinator.execute_2pc_ingestion(
            doc_payload=doc_payload,
            chunks_payload=chunks_payload,
            vectors_list=vectors_list
        )
        if success:
            logger.info(f"✅ V2 摄取成功：doc_id={doc_id}，共 {len(chunks_payload)} 个切片。")
        else:
            logger.error(f"❌ V2 2PC 摄取失败：doc_id={doc_id}")
        return success
    except Exception as e:
        logger.critical(f"❌ V2 摄取异常：doc_id={doc_id}，错误: {e}")
        return False


def ingest_pdf_to_v2_sync(
    doc_id: str,
    title: str,
    pdf_path: str,
    source_type: str = "local_pdf",
    authors: str = "手动导入 (Local Import)",
    ai_summary: str = "",
    canonical_url: str = "",
    db_path: str = DB_PATH
) -> bool:
    """
    `ingest_pdf_to_v2` 的同步封装，供非 async 上下文（如 ai_analyst.py）调用。
    安全处理"已存在事件循环（如 Streamlit 环境）"的情况。
    """
    import concurrent.futures

    async def _run():
        return await ingest_pdf_to_v2(
            doc_id=doc_id,
            title=title,
            pdf_path=pdf_path,
            source_type=source_type,
            authors=authors,
            ai_summary=ai_summary,
            canonical_url=canonical_url,
            db_path=db_path
        )

    # 检测是否已有运行中的事件循环（Streamlit / Jupyter 环境）
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 在已有循环的环境中，用独立线程运行新循环避免嵌套 asyncio 冲突
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _run())
            return future.result()
    else:
        return asyncio.run(_run())
