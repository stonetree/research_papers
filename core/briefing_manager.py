# -*- coding: utf-8 -*-
import os
import re
import math
import json
import datetime
import requests
from .env_helper import get_env_var

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "briefing_config.json")
BASE_FOLDER_NAME = os.path.join(PROJECT_ROOT, "storage", "briefings")

def load_briefing_config():
    """读取简报独立配置文件"""
    if not os.path.exists(CONFIG_PATH):
        default_config = {
            "gemini_api_key": "",
            "model_name": "gemini-2.5-flash",
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
        key = get_env_var("GEMINI_API_KEY", "").strip()
    return key

def call_gemini_api_with_search(prompt, system_instruction=None, config=None):
    """强联网版 Gemini API 调用，执行网络搜索并抓取 Grounding 追踪日志"""
    if config is None:
        config = load_briefing_config()
        
    api_key = get_gemini_api_key(config)
    if not api_key:
        return "❌ 错误: 未配置 Gemini API Key，且未检测到全局 GEMINI_API_KEY 环境变量。"
        
    model_name = config.get("model_name", "gemini-2.5-flash")
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
    
    if system_instruction:
        payload["system_instruction"] = {
            "parts": [{
                "text": system_instruction
            }]
        }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=3600)
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
    """手动/定时生成 过去 24 小时 AI 进展简报 (对接新版 DailyRadarPipeline)"""
    import asyncio
    from core.orchestrator import DailyRadarPipeline
    from core.database import get_db_connection
    
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[{current_time_str}] 开始执行新版每日雷达简报任务...")
    
    pipeline = DailyRadarPipeline()
    try:
        res = asyncio.run(pipeline.run_daily_radar_cron())
        if res.get("status") == "success":
            conn = get_db_connection()
            cursor = conn.cursor()
            # 获取最近入库的 10 条快讯
            cursor.execute("""
                SELECT d.title, c.full_text_markdown 
                FROM documents d 
                JOIN document_contents c ON d.doc_id = c.doc_id 
                WHERE d.source_type = 'daily_brief' 
                ORDER BY d.ingested_at DESC LIMIT 10
            """)
            rows = cursor.fetchall()
            conn.close()
            
            if rows:
                content = f"# 🪐 每日 AI 进展简报 (TOP 10) - {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n"
                for idx, r in enumerate(rows):
                    content += f"### {idx+1}. {r[0]}\n{r[1]}\n\n"
                    
                date_str = datetime.datetime.now().strftime("%Y-%m-%d")
                file_name = f"每日AI简报_{date_str}"
                folder_path = get_briefing_local_path("每日简报")
                success, path = save_to_local_file(folder_path, file_name, content)
                return success, content
        return False, res.get("error", "未抓取到有效快讯")
    except Exception as e:
        print(f"每日简报执行失败: {e}")
        return False, str(e)

def generate_weekly_insight_manually():
    """手动/定时生成 过去一周 AI 技术深入洞察 (对接新版 WeeklyInsightPipeline)"""
    import asyncio
    from core.weekly_insight import WeeklyInsightPipeline
    from core.database import get_db_connection
    
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[{current_time_str}] 开始执行新版每周技术洞察任务...")
    
    pipeline = WeeklyInsightPipeline()
    try:
        res = asyncio.run(pipeline.run_weekly_synthesis_pipeline())
        if res.get("status") == "success":
            doc_id = res["doc_id"]
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT full_text_markdown FROM document_contents WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                content = row[0]
                date_str = datetime.datetime.now().strftime("%Y-%m-%d")
                file_name = f"每周AI洞察_{date_str}"
                folder_path = get_briefing_local_path("每周洞察报告")
                success, path = save_to_local_file(folder_path, file_name, content)
                return success, content
        return False, res.get("reason", "生成周报失败")
    except Exception as e:
        print(f"每周技术洞察执行失败: {e}")
        return False, str(e)


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
