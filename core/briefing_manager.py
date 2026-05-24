# -*- coding: utf-8 -*-
import os
import re
import math
import json
import datetime
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "briefing_config.json")
BASE_FOLDER_NAME = os.path.join(PROJECT_ROOT, "storage", "briefings")

def load_briefing_config():
    """读取简报独立配置文件"""
    if not os.path.exists(CONFIG_PATH):
        default_config = {
            "gemini_api_key": "",
            "model_name": "gemini-1.5-pro",
            "daily_briefing_time": "09:00",
            "weekly_insight_time": "10:00",
            "weekly_insight_day": "Monday",
            "auto_scheduled": True
        }
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"创建简报默认配置失败: {e}")
        return default_config
        
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"读取简报配置失败: {e}")
        return {}

def save_briefing_config(config):
    """保存简报独立配置文件"""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"保存简报配置失败: {e}")
        return False

def get_gemini_api_key(config):
    """提取有效的 Gemini API Key (配置优先，环境变量后备)"""
    key = config.get("gemini_api_key", "").strip()
    if not key:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key

def call_gemini_api_with_search(prompt, config=None):
    """强联网版 Gemini API 调用，执行网络搜索并抓取 Grounding 追踪日志"""
    if config is None:
        config = load_briefing_config()
        
    api_key = get_gemini_api_key(config)
    if not api_key:
        return "❌ 错误: 未配置 Gemini API Key，且未检测到全局 GEMINI_API_KEY 环境变量。"
        
    model_name = config.get("model_name", "gemini-1.5-pro")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    # 显式注入 Google Search 工具以启用强联网搜索
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "tools": [
            {
                "google_search": {}
            }
        ]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        json_data = response.json()
        
        candidate = json_data['candidates'][0]
        
        # 提取并检验联网元数据 Grounding Metadata
        grounding_metadata = candidate.get('grounding_metadata', {})
        web_search_queries = grounding_metadata.get('web_search_queries', [])
        
        if web_search_queries:
            print(f"[{datetime.datetime.now()}] [成功联网] 触发的搜索关键词为: {web_search_queries}")
        else:
            print(f"[{datetime.datetime.now()}] [警告] Gemini 未触发联网搜索，可能使用了内部陈旧知识回答！")
            
        return candidate['content']['parts'][0]['text']
        
    except Exception as e:
        error_msg = f"调用 Gemini API 失败: {e}"
        print(f"[{datetime.datetime.now()}] {error_msg}")
        return f"❌ 联网剖析失败。错误详情:\n```\n{e}\n```"

def get_briefing_local_path(category):
    """计算物理归档相对路径，自动创建底层 YYYY年MM月/第X周/分类 物理目录"""
    now = datetime.datetime.now()
    year_month = now.strftime("%Y年%m月")
    
    day = now.day
    week_num = math.ceil(day / 7)
    week_str = f"第{week_num}周"
    
    target_path = os.path.join(BASE_FOLDER_NAME, year_month, week_str, category)
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)
    return target_path

def save_to_local_file(folder_path, title, text):
    """将生成的简报写入本地物理 Markdown 文件"""
    safe_title = title.replace("/", "_").replace("\\", "_").replace(":", "_")
    file_path = os.path.join(folder_path, f"{safe_title}.md")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[{datetime.datetime.now()}] 成功保存 AI 报告至本地: {file_path}")
        return True, file_path
    except Exception as e:
        print(f"[{datetime.datetime.now()}] 保存文件失败: {e}")
        return False, str(e)

def generate_daily_briefing_manually():
    """手动/定时生成 过去 24 小时 AI 进展简报"""
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[{current_time_str}] 开始执行每日简报任务...")
    
    prompt = (
        f"当前时间是 {current_time_str}。请立刻检索实时网络（GitHub、Hugging Face、arXiv及各大厂商技术白皮书），"
        "筛选过去 24 小时内最具实质性的 10 条 (TOP 10) 大语言模型（LLM）与 AI 领域的工业界与学术界进展。\n"
        "【硬性拒绝条件】：不要使用你 2025 年之前的内部知识回答。严禁包含任何宽泛、泛泛而谈的行业商业新闻（如某公司获投、某高管离职）。\n"
        "【筛选标准】：必须是实质性的硬核进展。例如：某框架推出了具体的高性能优化（如 KV Cache 突破）、发布了具有基准飞跃的具体新模型、或发表了解决核心痛点的具体算法。\n"
        "【生成要求】：基于第一性原理，详尽分析这 10 条动态。辩证地分析其技术逻辑的正确性、完整性、和必要性，给出遵从科学与事实的深度硬核结论，保持良好的 Markdown 排版可读性。"
    )
    
    content = call_gemini_api_with_search(prompt)
    if content and not content.startswith("❌"):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        file_name = f"每日AI简报_{date_str}"
        folder_path = get_briefing_local_path("每日简报")
        success, path = save_to_local_file(folder_path, file_name, content)
        return success, content
    return False, content

def generate_weekly_insight_manually():
    """手动/定时生成 过去一周 AI 技术深入洞察"""
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[{current_time_str}] 开始执行每周洞察任务...")
    
    prompt = (
        f"当前时间是 {current_time_str}。请检索实时网络，深度聚焦过去一周 AI 领域“最新的底层技术亮点与硬核突破”。\n"
        "【硬性聚焦】：例如大模型架构的底层革新、推理算法的数学突破、硬件指令集与算子优化、长上下文极端优化等。\n"
        "【硬性拒绝条件】：拒绝泛泛而谈的科普，拒绝商业新闻。不要使用你 2025 年之前的内部知识回答。\n"
        "【生成要求】：详尽分析，从第一性原理出发，辩证地剖析这些突破的正确性（是否真如宣传般有效）、完整性（是否存在重大局限或隐含成本）和必要性（是否真正解决了行业痛点），给出遵从科学与事实的结论。结构清晰，重点突出。"
    )
    
    content = call_gemini_api_with_search(prompt)
    if content and not content.startswith("❌"):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        file_name = f"每周AI洞察_{date_str}"
        folder_path = get_briefing_local_path("每周洞察报告")
        success, path = save_to_local_file(folder_path, file_name, content)
        return success, content
    return False, content

def list_archived_reports():
    """层级化扫描本地 storage/briefings 下所有保存的 Markdown 报告"""
    if not os.path.exists(BASE_FOLDER_NAME):
        return []
        
    reports = []
    for root, dirs, files in os.walk(BASE_FOLDER_NAME):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, PROJECT_ROOT).replace("\\", "/")
                
                # 分割路径提取层级信息
                # 相对路径格式：storage/briefings/YYYY年MM月/第X周/Category/filename.md
                sub_parts = os.path.relpath(full_path, BASE_FOLDER_NAME).replace("\\", "/").split("/")
                if len(sub_parts) >= 4:
                    year_month, week_str, category, filename = sub_parts[0], sub_parts[1], sub_parts[2], sub_parts[3]
                else:
                    year_month, week_str, category, filename = "未知日期", "未知周数", "其他", file
                    
                reports.append({
                    "path": rel_path,
                    "year_month": year_month,
                    "week": week_str,
                    "category": category,
                    "title": os.path.splitext(filename)[0],
                    "filename": filename,
                    "mtime": os.path.getmtime(full_path)
                })
                
    # 按文件修改时间降序排列
    reports.sort(key=lambda x: x["mtime"], reverse=True)
    return reports

def test_briefing_api_connection(api_key, model_name):
    """专属诊断工具：诊断简报模型 API 连通性与响应延时"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [{
            "parts": [{
                "text": "Hello, confirm you are Google Gemini. Answer in exactly 5 words."
            }]
        }]
    }
    
    try:
        import time
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        latency = round(time.time() - start_time, 2)
        
        if response.status_code == 200:
            res_json = response.json()
            reply = res_json['candidates'][0]['content']['parts'][0]['text'].strip()
            return True, f"成功连接至 {model_name}！响应回复: '{reply}'", latency
        else:
            return False, f"HTTP {response.status_code}: {response.text}", 0.0
    except Exception as e:
        return False, str(e), 0.0
