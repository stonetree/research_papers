# -*- coding: utf-8 -*-
import os
import json

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "api_config.json")

def load_raw_config():
    """读取原始 JSON 配置字典（包含元数据和全局设置）"""
    if not os.path.exists(CONFIG_PATH):
        default_config = {
            "_default_model": "deepseek-v4",
            "_global_settings": {
                "max_concurrent_analysis": 2,
                "max_papers_per_batch": 3,
                "analysis_granularity": "summary"  # 'summary' (概要) 或 'detailed' (完整)
            },
            "deepseek-v4": {
                "name": "DeepSeek-V4 (高性能推理)",
                "provider": "openai_compatible",
                "model": "deepseek-v4-flash",
                "api_key": "",
                "api_key_env": "DEEPSEEK_API_KEY",
                "url": "https://api.deepseek.com/chat/completions"
            }
        }
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"创建默认配置文件失败: {e}")
        return default_config
        
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 确保含有 _global_settings 键
            if "_global_settings" not in data:
                data["_global_settings"] = {
                    "max_concurrent_analysis": 2,
                    "max_papers_per_batch": 3,
                    "analysis_granularity": "summary"
                }
            return data
    except Exception as e:
        print(f"读取 API 配置文件失败: {e}")
        return {}

def save_raw_config(raw):
    """保存并持久化原始 JSON 配置"""
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(raw, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"写入 API 配置失败: {e}")
        return False

def load_api_config():
    """加载可用的 API 模型配置列表（过滤掉元数据键如 _default_model, _global_settings）"""
    raw = load_raw_config()
    return {k: v for k, v in raw.items() if not k.startswith("_")}

def get_default_model():
    """获取设置的默认模型 ID"""
    raw = load_raw_config()
    return raw.get("_default_model", "deepseek-v4")

def set_default_model(model_id):
    """保存并持久化默认模型 ID"""
    raw = load_raw_config()
    raw["_default_model"] = model_id
    return save_raw_config(raw)

def get_global_settings():
    """获取全局配置字典"""
    raw = load_raw_config()
    return raw.get("_global_settings", {
        "max_concurrent_analysis": 2,
        "max_papers_per_batch": 3,
        "analysis_granularity": "summary"
    })

def update_global_settings(settings):
    """保存并持久化全局配置"""
    raw = load_raw_config()
    raw["_global_settings"] = settings
    return save_raw_config(raw)

def update_model_config(model_id, name, provider, model_name, api_key, url, api_key_env=""):
    """更新或添加特定模型的配置"""
    raw = load_raw_config()
    raw[model_id] = {
        "name": name,
        "provider": provider,
        "model": model_name,
        "api_key": api_key,
        "api_key_env": api_key_env or f"{model_id.replace('-', '_').upper()}_API_KEY",
        "url": url
    }
    return save_raw_config(raw)

def delete_model_config(model_id):
    """删除特定模型配置"""
    raw = load_raw_config()
    if model_id in raw:
        del raw[model_id]
        # 如果默认模型被删除了，重置默认模型
        if raw.get("_default_model") == model_id:
            models = [k for k in raw.keys() if not k.startswith("_")]
            raw["_default_model"] = models[0] if models else ""
        return save_raw_config(raw)
    return False

def get_model_config(model_id):
    """获取特定模型的解析配置，自动处理 API Key 环境变量回退"""
    configs = load_api_config()
    cfg = configs.get(model_id)
    if not cfg:
        return None
        
    # 处理 API Key 优先级：配置文件配置 > 环境变量
    api_key = cfg.get("api_key", "").strip()
    if not api_key:
        env_var = cfg.get("api_key_env", "")
        if env_var:
            api_key = os.environ.get(env_var, "").strip()
            
    resolved_cfg = cfg.copy()
    resolved_cfg["resolved_api_key"] = api_key
    return resolved_cfg
