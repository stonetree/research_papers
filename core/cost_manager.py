# -*- coding: utf-8 -*-
import time
import logging
from functools import wraps
import aiosqlite
from typing import Dict, Any

from .database import DB_PATH
from .write_worker import WriteWorker

logger = logging.getLogger("CostManager")

def api_billing_audit(api_provider: str, api_metric: str):
    """
    大模型及外部渠道三段式精细化收费对账装饰器。
    拦截返回值并在异步 WriteWorker 队列中登记账单明细。
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 执行底层网络/模型调用
            response = await func(*args, **kwargs)
            if not response:
                return response
                
            db_path = kwargs.get('db_path', DB_PATH)
            writer = WriteWorker(db_path)
            
            try:
                # 级联读取最新 pricing 规则表
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    cursor = await db.execute(
                        "SELECT rule_id, metric_key, unit_price_usd FROM provider_pricing_rules WHERE provider_name = ?",
                        (api_provider,)
                    )
                    rows = await cursor.fetchall()
                    # 规则哈希映射
                    rules = {row['metric_key']: (row['rule_id'], row['unit_price_usd']) for row in rows}
                
                # 如果数据库中规则未初始化，使用备用定价硬编码
                fallback_rules = {
                    "deepseek": {
                        "input_cache_hit": (1, 0.0028), "input_cache_miss": (2, 0.14), "output": (3, 0.28)
                    },
                    "firecrawl": {
                        "credit_cost": (4, 0.0010)
                    },
                    "exa": {
                        "search_request": (5, 0.0070), "contents_request": (6, 0.0010), "summary_request": (7, 0.0020)
                    },
                    "google": {
                        "input": (8, 0.075), "output": (9, 0.30), "grounding_prompt": (10, 0.010)
                    },
                    "dashscope": {
                        "input": (11, 0.012), "output": (12, 0.048)
                    }
                }
                
                def get_rule(key, default_rule):
                    return rules.get(key, fallback_rules.get(api_provider, {}).get(key, default_rule))

                audit_queries = []
                
                # 1. DeepSeek 官方原生三段式缓存敏感型计费
                if api_provider == "deepseek":
                    usage = response.get("usage", {})
                    cache_hit_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    cache_miss_tokens = max(0, prompt_tokens - cache_hit_tokens)
                    output_tokens = usage.get("completion_tokens", 0)
                    model_name = response.get("model", "deepseek-v4-flash")
                    
                    hit_rule = get_rule("input_cache_hit", (1, 0.0028))
                    miss_rule = get_rule("input_cache_miss", (2, 0.14))
                    out_rule = get_rule("output", (3, 0.28))
                    
                    # 价格单位: 美元/百万 tokens
                    cost = (
                        (cache_hit_tokens * hit_rule[1]) +
                        (cache_miss_tokens * miss_rule[1]) +
                        (output_tokens * out_rule[1])
                    ) / 1000000.0
                    
                    audit_queries.append((
                        "INSERT INTO quota_ledger (api_provider, model_name, api_metric, pricing_rule_id, tokens_in, tokens_out, cache_hit_tokens, cost_usd, request_payload_summary) "
                        "VALUES ('deepseek', ?, 'tokens', ?, ?, ?, ?, ?, ?)",
                        (model_name, miss_rule[0], prompt_tokens, output_tokens, cache_hit_tokens, cost, f"DeepSeek execution: in={prompt_tokens}, out={output_tokens}, hit={cache_hit_tokens}")
                    ))
                    
                # 2. Firecrawl 积分制核销
                elif api_provider == "firecrawl":
                    credits_spent = response.get("credits_spent", 1)
                    credit_rule = get_rule("credit_cost", (4, 0.0010))
                    cost = credits_spent * credit_rule[1]
                    url_str = response.get("success_url", "unknown url")
                    
                    audit_queries.append((
                        "INSERT INTO quota_ledger (api_provider, model_name, api_metric, pricing_rule_id, credits_spent, cost_usd, request_payload_summary) "
                        "VALUES ('firecrawl', 'firecrawl-scraper', 'credits', ?, ?, ?, ?)",
                        (credit_rule[0], credits_spent, cost, f"Scrape URL: {url_str}")
                    ))

                # 3. Exa 神经网络独立分项计费对账
                elif api_provider == "exa":
                    call_type = response.get("call_type", "search_request")
                    exa_rule = get_rule(call_type, (5, 0.0070))
                    cost = exa_rule[1]
                    q_hash = response.get("query_hash", "none")
                    
                    audit_queries.append((
                        "INSERT INTO quota_ledger (api_provider, model_name, api_metric, pricing_rule_id, cost_usd, request_payload_summary) "
                        "VALUES ('exa', 'exa-engine', ?, ?, ?, ?)",
                        (call_type, exa_rule[0], cost, f"Exa operation: {call_type}, q_hash={q_hash}")
                    ))

                # 4. Google Gemini 强联网 API 审计
                elif api_provider == "google":
                    usage = response.get("usage", {})
                    t_in = usage.get("prompt_tokens", 0)
                    t_out = usage.get("completion_tokens", 0)
                    model_name = response.get("model", "gemini-2.5-flash")
                    
                    in_rule = get_rule("input", (8, 0.075))
                    out_rule = get_rule("output", (9, 0.30))
                    ground_rule = get_rule("grounding_prompt", (10, 0.010))
                    
                    # 计费单位: 美元/百万 tokens
                    cost = ((t_in * in_rule[1]) + (t_out * out_rule[1])) / 1000000.0
                    
                    # 超额追加 Search Grounding 费用
                    if response.get("grounding_metadata_triggered", False):
                        cost += ground_rule[1]
                        
                    audit_queries.append((
                        "INSERT INTO quota_ledger (api_provider, model_name, api_metric, pricing_rule_id, tokens_in, tokens_out, cost_usd, request_payload_summary) "
                        "VALUES ('google', ?, 'tokens', ?, ?, ?, ?, ?)",
                        (model_name, in_rule[0], t_in, t_out, cost, f"Gemini call: in={t_in}, out={t_out}, grounding={response.get('grounding_metadata_triggered')}")
                    ))

                # 5. Alibaba DashScope Qwen 高推理计费 (单位: 美元/千 tokens)
                elif api_provider == "dashscope":
                    usage = response.get("usage", {})
                    t_in = usage.get("input_tokens", 0)
                    t_out = usage.get("output_tokens", 0)
                    model_name = response.get("model", "qwen3.7-max")
                    
                    in_rule = get_rule("input", (11, 0.012))
                    out_rule = get_rule("output", (12, 0.048))
                    
                    cost = ((t_in * in_rule[1]) + (t_out * out_rule[1])) / 1000.0
                    
                    audit_queries.append((
                        "INSERT INTO quota_ledger (api_provider, model_name, api_metric, pricing_rule_id, tokens_in, tokens_out, cost_usd, request_payload_summary) "
                        "VALUES ('dashscope', ?, 'tokens', ?, ?, ?, ?, ?)",
                        (model_name, in_rule[0], t_in, t_out, cost, f"Qwen execution: in={t_in}, out={t_out}")
                    ))

                if audit_queries:
                    await writer.execute_write(audit_queries)
                    
            except Exception as audit_err:
                logger.error(f"计费审计拦截日志登记失败！原因: {str(audit_err)}")
                
            return response
        return wrapper
    return decorator

class HardBudgetValidator:
    def __init__(self, database_path: str = None):
        self.db_path = database_path or DB_PATH

    async def verify_allowance_or_trigger_fuse(self, daily_limit: float = None, weekly_limit: float = None) -> bool:
        """
        验证每日/每周配额是否透支，如果透支则触发熔断机制。
        返回 True 表示正常，False 表示触发熔断。
        """
        from .config_loader import get_global_settings
        settings = get_global_settings()
        if daily_limit is None:
            daily_limit = float(settings.get("daily_budget", 2.0))
        if weekly_limit is None:
            weekly_limit = float(settings.get("weekly_budget", 10.0))

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            daily_query = "SELECT SUM(cost_usd) as daily_cost FROM quota_ledger WHERE created_at >= datetime('now', 'start of day')"
            weekly_query = "SELECT SUM(cost_usd) as weekly_cost FROM quota_ledger WHERE created_at >= datetime('now', '-7 days')"
            
            try:
                async with db.execute(daily_query) as cursor:
                    day_row = await cursor.fetchone()
                    d_cost = day_row['daily_cost'] if day_row and day_row['daily_cost'] else 0.0
                    
                async with db.execute(weekly_query) as cursor:
                    week_row = await cursor.fetchone()
                    w_cost = week_row['weekly_cost'] if week_row and week_row['weekly_cost'] else 0.0
                    
                if d_cost >= daily_limit:
                    logger.critical(f"🚨【每日硬限额熔断】今日累计开销已达 ${d_cost:.4f} (每日配额上限为 ${daily_limit:.2f})，触发商业API强行关闭！")
                    return False
                if w_cost >= weekly_limit:
                    logger.critical(f"🚨【每周硬限额熔断】本周累计开销已达 ${w_cost:.4f} (每周配额上限为 ${weekly_limit:.2f})，触发商业API强行关闭！")
                    return False
                return True
            except Exception as e:
                logger.error(f"滑动硬预算验证异常，默认放行。原因: {str(e)}")
                return True
