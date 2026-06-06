# -*- coding: utf-8 -*-

def model_supports_web_search(cfg):
    """
    判断一个模型配置是否支持内置联网搜索功能。
    目前在本项目中，仅支持 provider == "openai_compatible" 或 "deepseek" 且 url 包含 "/responses" 的模型。
    """
    if not cfg:
        return False
    provider = cfg.get("provider", "")
    url = cfg.get("url", "")
    return (provider in ["openai_compatible", "deepseek"] and "/responses" in url)

def get_search_capable_models(api_models):
    """
    过滤并返回所有具有联网搜索能力的模型配置列表
    """
    if not api_models:
        return {}
    return {k: v for k, v in api_models.items() if model_supports_web_search(v)}
