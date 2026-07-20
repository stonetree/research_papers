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
            "auto_scheduled": True,
            "proxy": ""
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
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # 代理支持
    proxies = None
    proxy_url = config.get("proxy", "").strip()
    if proxy_url:
        proxies = {
            "http": proxy_url,
            "https": proxy_url
        }
        
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
    
    max_retries = 3
    last_err = None
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"[{datetime.datetime.now()}] [重试提示] 正在重试调用 Gemini API (第 {attempt+1}/{max_retries} 次尝试)...")
            response = requests.post(url, json=payload, headers=headers, proxies=proxies, timeout=180)
            response.raise_for_status()
            
            from .ai_analyst import _audit_api_call
            _audit_api_call(provider="gemini", model_name=model_name, api_url=url, request_label="generate_briefing_report")
            
            json_data = response.json()
            
            candidate = json_data['candidates'][0]
            
            # 提取并检验联网元数据 Grounding Metadata (兼容 Gemini REST API 的驼峰体与下划线命名)
            grounding_metadata = candidate.get('groundingMetadata') or candidate.get('grounding_metadata') or {}
            web_search_queries = grounding_metadata.get('webSearchQueries') or grounding_metadata.get('web_search_queries') or []
            search_chunks = grounding_metadata.get('groundingChunks') or grounding_metadata.get('searchChunks') or grounding_metadata.get('search_chunks') or []
            
            if web_search_queries or search_chunks:
                print(f"[{datetime.datetime.now()}] [成功联网] 触发的搜索关键词: {web_search_queries or '已抓取全网最新动态'}")
            else:
                print(f"[{datetime.datetime.now()}] [警告] Gemini 未触发联网搜索，可能使用了内部陈旧知识回答！")
                
            return candidate['content']['parts'][0]['text']
            
        except requests.exceptions.RequestException as e:
            last_err = e
            print(f"[{datetime.datetime.now()}] 第 {attempt+1} 次调用 Gemini API 失败: {e}")
            # 输出环境诊断信息方便调试后台任务代理配置
            import urllib.request
            sys_proxies = urllib.request.getproxies()
            print(f"[{datetime.datetime.now()}] [网络诊断] 当前配置代理: {proxy_url or '无'} | 系统级检测代理: {sys_proxies} | 环境变量: HTTP_PROXY={os.environ.get('HTTP_PROXY', '无')}, HTTPS_PROXY={os.environ.get('HTTPS_PROXY', '无')}")
            
            if attempt < max_retries - 1:
                import time
                sleep_time = 2 * (attempt + 1)
                time.sleep(sleep_time)
        except Exception as e:
            # 针对 JSON 解析或其它非网络异常不重试直接失败
            error_msg = f"调用 Gemini API 发生非网络异常: {e}"
            print(f"[{datetime.datetime.now()}] {error_msg}")
            return f"❌ 联网剖析失败。错误详情:\n```\n{e}\n```"
            
    # 如果所有重试都失败了
    error_msg = f"调用 Gemini API 失败 (经过 {max_retries} 次尝试): {last_err}"
    print(f"[{datetime.datetime.now()}] {error_msg}")
    return f"❌ 联网剖析失败 (网络与SSL握手在多次尝试后均断开)。错误详情:\n```\n{last_err}\n```"

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

def wait_for_network_connectivity(timeout_seconds=90, check_interval=5):
    """等待网络连接就绪（通过测试连接公共服务如 baidu.com 或 google.com）"""
    import urllib.request
    print(f"🌐 [网络就绪检测] 开始检测网络连接，超时限制: {timeout_seconds}秒...")
    start_time = datetime.datetime.now()
    while (datetime.datetime.now() - start_time).total_seconds() < timeout_seconds:
        try:
            # 尝试连接百度
            urllib.request.urlopen("https://www.baidu.com", timeout=3)
            print("🟢 [网络就绪检测] 检测到网络连接已畅通！")
            return True
        except Exception:
            try:
                # 尝试连接 Google (如果代理已经就位)
                urllib.request.urlopen("https://www.google.com", timeout=3)
                print("🟢 [网络就绪检测] 检测到网络/代理连接已畅通！")
                return True
            except Exception:
                pass
        print(f"⏳ [网络就绪检测] 网络尚未就绪，等待 {check_interval} 秒后重试...")
        import time
        time.sleep(check_interval)
    print("🔴 [网络就绪检测] 达到超时时间，未检测到可用网络连接。")
    return False

def generate_daily_briefing_manually():
    """手动/定时生成 过去 24 小时 AI 进展简报（基于 Gemini 强联网 Grounding 与第一性原理分析）"""
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[{current_time_str}] 开始执行每日 AI 简报生成任务...")
    
    # 执行网络就绪前置检测
    if not wait_for_network_connectivity(timeout_seconds=90):
        err_msg = "❌ 网络连接不可用，每日简报任务被终止。"
        print(f"[{current_time_str}] {err_msg}")
        return False, err_msg
    
    prompt = (
        f"【强制联网指令】：请你务必且必须首先使用内置的 google_search 检索工具，搜索并获取截至目前（{current_time_str}）过去 24 小时内全网 AI 领域的 10 条核心技术动态与突破。\n"
        "必须严格聚焦于“工业界与学术界的最新实质性动态”，例如厂商推出了具体的新模型，或学术界发表了解决突出问题的具体新算法。拒绝宽泛的行业新闻。不要使用你 2025 年之前的内部知识回答。\n"
        "要求：详尽分析，从第一性原理出发，辩证分析问题的正确性、完整性、和必要性，给出遵从科学与事实的结论，生成的报告需具有良好的可读性，突出重点。"
    )
    
    system_instruction = "你是一个顶级的 AI 系统架构师。你在生成报告前必须首先调用 Google Search 工具在全网检索过去 24 小时内最新的技术新闻与论文。基于实际检索到的事实，运用第一性原理进行辩证分析。必须确保输出的是最新信息，严禁胡编乱造。"
    
    content = call_gemini_api_with_search(prompt, system_instruction=system_instruction)
    if content and not content.startswith("❌"):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        file_name = f"每日AI简报_{date_str}"
        folder_path = get_briefing_local_path("每日简报")
        success, path = save_to_local_file(folder_path, file_name, content)
        
        # 调用 V2 通用纯文本分块沉淀，全量构建 documents + document_contents + chunks + search_chunks(FTS5) + LanceDB
        try:
            from core.ingestion import ingest_markdown_text_to_v2_sync
            doc_id = f"daily_brief_{date_str}"
            status_res = ingest_markdown_text_to_v2_sync(
                doc_id=doc_id,
                title=f"每日 AI 进展简报 ({date_str})",
                full_text_markdown=content,
                source_type="daily_brief",
                authors="Google Gemini Grounding",
                canonical_url=f"file://storage/briefings/daily_brief_{date_str}.md"
            )
            if status_res == "ingested":
                print(f"[{datetime.datetime.now()}] 成功将每日简报全量分块向量化沉淀至 V2 数据大仓与 LanceDB 向量库。")
            elif status_res == "pending":
                print(f"[{datetime.datetime.now()}] ⚠️ 每日简报生成完毕，但由于 Embedding 服务 (127.0.0.1:8081) 未启动，已自动存储全文本并进入【待向量化队列】(status='pending')。开启服务后可一键同步！")
            else:
                print(f"[{datetime.datetime.now()}] 警告: 每日简报沉淀至 V2 发生失败。")
        except Exception as sync_ex:
            print(f"同步简报至 V2 数据大仓发生异常: {sync_ex}")
            
        return success, content
    return False, content

def generate_weekly_insight_manually():
    """手动/定时生成 过去一周 AI 技术深入洞察（基于 Gemini 强联网 Grounding 与系统架构师级辩证剖析）"""
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[{current_time_str}] 开始执行每周 AI 技术洞察任务...")
    
    # 执行网络就绪前置检测
    if not wait_for_network_connectivity(timeout_seconds=90):
        err_msg = "❌ 网络连接不可用，每周技术洞察任务被终止。"
        print(f"[{current_time_str}] {err_msg}")
        return False, err_msg
    
    prompt = (
        f"【强制联网指令】：根据第一性原理，请你务必首先调用内置的 google_search 检索工具，搜索并获取截至目前（{current_time_str}）过去 7 天内全网 AI 领域“最新的技术亮点与硬核突破”（如架构的底层革新、推理算法的数学突破、硬件指令集的更新）。不要使用你 2025 年之前的内部知识回答。\n"
        "严禁对一般技术进行泛泛而谈，必须进行深度辩证分析。\n"
        "要求：详尽分析，从第一性原理出发，辩证分析问题的正确性、完整性、和必要性，给出遵从科学与事实的结论，生成的报告需具有良好的可读性，结构清晰，重点突出。"
    )
    
    system_instruction = "你是一个顶级的 AI 系统架构师。你在生成洞察前必须首先调用 Google Search 工具在全网检索上周（7天）最新的极客技术新闻与顶级论文。基于检索到的事实，运用第一性原理进行辩证分析。必须确保输出的是最新信息，严禁胡编乱造。"
    
    content = call_gemini_api_with_search(prompt, system_instruction=system_instruction)
    if content and not content.startswith("❌"):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        file_name = f"每周AI洞察_{date_str}"
        folder_path = get_briefing_local_path("每周洞察报告")
        success, path = save_to_local_file(folder_path, file_name, content)
        
        # 调用 V2 通用纯文本分块沉淀，全量构建 documents + document_contents + chunks + search_chunks(FTS5) + LanceDB
        try:
            from core.ingestion import ingest_markdown_text_to_v2_sync
            doc_id = f"weekly_insight_{date_str}"
            status_res = ingest_markdown_text_to_v2_sync(
                doc_id=doc_id,
                title=f"每周 AI 技术深入洞察 ({date_str})",
                full_text_markdown=content,
                source_type="weekly_insight",
                authors="Google Gemini Grounding",
                canonical_url=f"file://storage/briefings/weekly_insight_{date_str}.md"
            )
            if status_res == "ingested":
                print(f"[{datetime.datetime.now()}] 成功将每周洞察全量分块向量化沉淀至 V2 数据大仓与 LanceDB 向量库。")
            elif status_res == "pending":
                print(f"[{datetime.datetime.now()}] ⚠️ 每周洞察生成完毕，但由于 Embedding 服务 (127.0.0.1:8081) 未启动，已自动存储全文本并进入【待向量化队列】(status='pending')。开启服务后可一键同步！")
            else:
                print(f"[{datetime.datetime.now()}] 警告: 每周洞察沉淀至 V2 发生失败。")
        except Exception as sync_ex:
            print(f"同步周报至 V2 数据大仓发生异常: {sync_ex}")
            
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

def test_briefing_api_connection(api_key, model_name, proxy_url=None):
    """专属诊断工具：诊断简报模型 API 连通性与响应延时"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    payload = {
        "contents": [{
            "parts": [{
                "text": "Hello, confirm you are Google Gemini. Answer in exactly 5 words."
            }]
        }]
    }
    
    proxies = None
    if proxy_url:
        proxies = {
            "http": proxy_url,
            "https": proxy_url
        }
        
    try:
        import time
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, proxies=proxies, timeout=15)
        latency = round(time.time() - start_time, 2)
        
        if response.status_code == 200:
            res_json = response.json()
            reply = res_json['candidates'][0]['content']['parts'][0]['text'].strip()
            return True, f"成功连接至 {model_name}！响应回复: '{reply}'", latency
        else:
            return False, f"HTTP {response.status_code}: {response.text}", 0.0
    except Exception as e:
        import urllib.request
        sys_proxies = urllib.request.getproxies()
        return False, f"连接失败: {e} | 诊断环境代理: {sys_proxies}", 0.0
