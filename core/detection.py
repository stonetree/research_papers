# -*- coding: utf-8 -*-

def model_supports_web_search(cfg):
    """
    判断一个模型配置是否支持原生 responses / grounding 联网搜索功能（路径 A 增强节点）。
    """
    if not cfg:
        return False
    provider = cfg.get("provider", "")
    url = cfg.get("url", "")
    return (provider in ["openai_compatible", "deepseek"] and "/responses" in url) or (provider == "gemini")


def can_be_used_for_web_search(cfg):
    """
    判断一个模型是否可以配置为 AI 联网学术探测大脑。
    双路径支持规则：
    - 路径 A: 原生 responses 端点 (使用模型原生的工具联网)
    - 路径 B: 标准 Chat 端口模型 (框架自动驱动 Exa/Firecrawl 全网搜集学术文献切片并注入 Prompt 上下文)
    只要配置了合法的 API Key 与 URL，均允许配置使用。
    """
    if not cfg:
        return False
    api_key = cfg.get("resolved_api_key") or cfg.get("api_key")
    url = cfg.get("url")
    provider = cfg.get("provider")
    if provider == "gemini" and api_key:
        return True
    return bool(api_key and url)


def get_search_capable_models(api_models):
    """
    过滤并返回所有可配置为 AI 联网学术探测大脑的模型配置字典
    """
    if not api_models:
        return {}
    return {k: v for k, v in api_models.items() if can_be_used_for_web_search(v)}
