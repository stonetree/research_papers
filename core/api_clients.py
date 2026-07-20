# -*- coding: utf-8 -*-
import os
import asyncio
import logging
import hashlib
import aiohttp
from typing import List, Dict, Any, Optional

from .cost_manager import api_billing_audit
from .database import DB_PATH
from .env_helper import get_env_var

logger = logging.getLogger("ApiClients")

class LocalComputeKernelClient:
    """
    本地高性能计算内核客户端 (llama-server CLI 包装器)
    使用 asyncio.Semaphore(8) 提供反向背压，保障本地微服务高负荷下不崩溃。
    """
    def __init__(self, embedding_port: int = 8081, rerank_port: int = 8082):
        self.embedding_url = f"http://127.0.0.1:{embedding_port}/v1/embeddings"
        self.rerank_url = f"http://127.0.0.1:{rerank_port}/v1/rerank"
        # 进程级全局信号量限制，确保请求进入本地算力时维持在合理并发内
        self.embedding_sem = asyncio.Semaphore(8)
        self.rerank_sem = asyncio.Semaphore(8)

    async def check_embedding_service_health(self) -> bool:
        """
        心跳检测：检查本地 8081 端口 Embedding 服务是否处于 Ready 就绪状态
        """
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"input": "healthcheck", "model": "qwen3-embedding"}
                async with session.post(self.embedding_url, json=payload, timeout=2) as resp:
                    return resp.status == 200
        except Exception:
            return False

    @staticmethod
    def check_service_health_sync() -> bool:
        """类静态同步版本的 8081 Embedding 服务心跳健康探测"""
        return check_embedding_service_health_sync()

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        向本地 8081 端口获取高维稠密特征向量
        """
        async with self.embedding_sem:
            payload = {"input": text, "model": "qwen3-embedding"}
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(self.embedding_url, json=payload, timeout=30) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # 提取 embedding list
                            if "data" in data and len(data["data"]) > 0:
                                return data["data"][0]["embedding"]
                        else:
                            # 尝试兼容旧版本或非 OpenAI 协议格式
                            fallback_url = self.embedding_url.replace("/v1/embeddings", "/embedding")
                            async with session.post(fallback_url, json={"content": text}, timeout=10) as f_resp:
                                if f_resp.status == 200:
                                    f_data = await f_resp.json()
                                    return f_data.get("embedding")
                            logger.error(f"本地 Embedding 内核响应错误。HTTP: {resp.status}")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ 本地 Embedding 内核响应超时，将自动降级使用 FTS5 全文切片落库。")
                except Exception as e:
                    logger.warning(f"⚠️ 本地 Embedding 内核 (127.0.0.1:8081) 离线或未启动 ({e})，将平滑降级使用 FTS5 全文切片落库。")
                return None

    async def get_rerank_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        向本地 8082 端口获取候选切片与 Query 的相关性分数 (Logits)
        """
        if not documents:
            return []
            
        async with self.rerank_sem:
            # 兼容标准 LLM reranker API
            payload = {"query": query, "documents": documents, "model": "qwen3-reranker"}
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(self.rerank_url, json=payload, timeout=30) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # 标准返回 results: [{"index": 0, "relevance_score": 4.82}, ...]
                            results = data.get("results", [])
                            # 根据 index 还原列表顺序的分数
                            scores = [0.0] * len(documents)
                            for res in results:
                                idx = res.get("index")
                                score = res.get("relevance_score", 0.0)
                                if 0 <= idx < len(documents):
                                    scores[idx] = float(score)
                            return scores
                        else:
                            # 兼容 llama-server 另一种形式 /rerank
                            fallback_url = self.rerank_url.replace("/v1/rerank", "/rerank")
                            async with session.post(fallback_url, json=payload, timeout=10) as f_resp:
                                if f_resp.status == 200:
                                    f_data = await f_resp.json()
                                    results = f_data.get("results", [])
                                    scores = [0.0] * len(documents)
                                    for res in results:
                                        idx = res.get("index")
                                        score = res.get("relevance_score", 0.0)
                                        if 0 <= idx < len(documents):
                                            scores[idx] = float(score)
                                    return scores
                            logger.error(f"本地 Reranker 内核响应错误。HTTP: {resp.status}")
                except asyncio.TimeoutError:
                    logger.error("🚨 本地 Reranker 内核排队响应超时，自动实施平滑降级，各候选切片重排分数兜底置零！")
                except Exception as e:
                    logger.error(f"🚨 本地 Reranker 内核连接异常: {str(e)}")
                return [0.0] * len(documents)


class ExaApiClient:
    """
    Exa 神经网络语义检索异步客户端，仅使用 aiohttp 确保完全非阻塞。
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or get_env_var("EXA_API_KEY")
        self.base_url = "https://api.exa.ai"
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": self.api_key or ""
        }

    @api_billing_audit(api_provider="exa", api_metric="search_request")
    async def search_and_extract_highlights(
        self, 
        query: str, 
        num_results: int = 10, 
        db_path: str = DB_PATH,
        include_domains: List[str] = None,
        category: str = None
    ) -> Dict[str, Any]:
        """
        向 Exa 发起语义打捞请求，强制提取 highlights 作为高密度脱水内容。
        """
        if not self.api_key:
            logger.warning("EXA_API_KEY 未设置，网络请求将不可避免失败。")
            
        endpoint = f"{self.base_url}/search"
        payload = {
            "query": query,
            "numResults": num_results,
            "useAutoprompt": True,
            "highlights": {
                "numSentences": 3,
                "highlightsPerUrl": 2
            }
        }
        if include_domains:
            payload["includeDomains"] = include_domains
        if category:
            payload["category"] = category
        
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(endpoint, json=payload, timeout=25) as response:
                if response.status != 200:
                    err_text = await response.text()
                    raise RuntimeError(f"Exa HTTP Error {response.status}: {err_text}")
                
                raw_data = await response.json()
                # 填充审计装饰器所依赖的核心参数
                raw_data["call_type"] = "search_request"
                raw_data["query_hash"] = query_hash
                return raw_data


class FirecrawlApiClient:
    """
    Firecrawl 复杂网页强力穿透抓取 Markdown 客户端，仅在原生 requests 失败时熔断降级进入。
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or get_env_var("FIRECRAWL_API_KEY")
        self.base_url = "https://api.firecrawl.dev/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json"
        }

    @api_billing_audit(api_provider="firecrawl", api_metric="credit_cost")
    async def scrape_url_to_markdown(self, target_url: str, db_path: str = DB_PATH) -> Dict[str, Any]:
        """
        调用 Firecrawl 对高难度 Cloudflare/JavaScript 渲染单页执行强力解析脱水。
        """
        if not self.api_key:
            logger.warning("FIRECRAWL_API_KEY 未设置，Firecrawl 无法工作。")
            
        endpoint = f"{self.base_url}/scrape"
        payload = {
            "url": target_url,
            "formats": ["markdown"],
            "onlyMainContent": True,
            "waitFor": 3000
        }
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(endpoint, json=payload, timeout=45) as response:
                if response.status != 200:
                    err_text = await response.text()
                    raise RuntimeError(f"Firecrawl HTTP Error {response.status}: {err_text}")
                
                raw_data = await response.json()
                if not raw_data.get("success"):
                    raise RuntimeError(f"Firecrawl scrape failed: {raw_data.get('error')}")
                
                # 返回核心解析数据与统一 1 Credit 开销标记
                data = raw_data.get("data", {})
                return {
                    "markdown": data.get("markdown", ""),
                    "title": data.get("metadata", {}).get("title", target_url),
                    "credit_cost": 1.0
                }


def check_embedding_service_health_sync() -> bool:
    """同步版本的 8081 Embedding 服务心跳健康探测"""
    import asyncio
    import concurrent.futures

    client = LocalComputeKernelClient()

    async def _run():
        return await client.check_embedding_service_health()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _run())
            return future.result()
    else:
        return asyncio.run(_run())
