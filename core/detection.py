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
    自动兼容配置文件中的 api_key、resolved_api_key 以及 api_key_env 环境变量/注册表回退。
    """
    if not cfg:
        return False
    from .env_helper import get_env_var
    api_key = (cfg.get("resolved_api_key") or cfg.get("api_key") or "").strip()
    if not api_key:
        env_var = cfg.get("api_key_env", "")
        if env_var:
            api_key = get_env_var(env_var, "").strip()

    url = cfg.get("url", "").strip()
    provider = cfg.get("provider", "")
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
