# -*- coding: utf-8 -*-
import os
import sqlite3
import threading
import time
from datetime import datetime
from .database import get_db_connection, resolve_pdf_path
from .library_scanner import sync_local_library, get_unanalyzed_papers
from .ai_analyst import analyze_and_store_paper

# 全局控制锁与状态标志
_scheduler_lock = threading.Lock()
_scheduler_started = False

def init_scheduler_db():
    """初始化数据库定时任务表"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scheduler_tasks (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type TEXT NOT NULL, -- 'one_shot' 或 'daily'
            scheduled_time TEXT NOT NULL, -- 'YYYY-MM-DD HH:MM' 或 'HH:MM'
            model_id TEXT NOT NULL,
            is_completed INTEGER DEFAULT 0,
            last_run_date TEXT, -- 防止每日任务在同一分钟内重复运行
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 动态向下兼容追加新字段以区分本地大仓全盘扫描或线上雷达自动探测
    try:
        cursor.execute("ALTER TABLE scheduler_tasks ADD COLUMN task_goal TEXT DEFAULT 'local_scan'")
    except sqlite3.OperationalError:
        pass # 字段已存在
    try:
        cursor.execute("ALTER TABLE scheduler_tasks ADD COLUMN topic_key TEXT")
    except sqlite3.OperationalError:
        pass # 字段已存在
    try:
        cursor.execute("ALTER TABLE scheduler_tasks ADD COLUMN search_limit INTEGER DEFAULT 15")
    except sqlite3.OperationalError:
        pass # 字段已存在
        
    conn.commit()
    conn.close()

def add_scheduler_task(task_type, scheduled_time, model_id, task_goal="local_scan", topic_key=None, search_limit=15):
    """添加新的定时扫描或线上雷达自动检索分析任务"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO scheduler_tasks (task_type, scheduled_time, model_id, task_goal, topic_key, search_limit)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (task_type, scheduled_time, model_id, task_goal, topic_key, search_limit))
        conn.commit()
    finally:
        conn.close()

def delete_scheduler_task(task_id):
    """取消/删除指定的定时任务"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM scheduler_tasks WHERE task_id = ?', (task_id,))
        conn.commit()
    finally:
        conn.close()

def get_active_tasks():
    """获取所有处于激活（未完成或每日重复）状态的定时任务列表"""
    conn = get_db_connection()
    try:
        tasks = conn.execute('''
            SELECT * FROM scheduler_tasks 
            WHERE is_completed = 0 
            ORDER BY created_at DESC
        ''').fetchall()
        return [dict(task) for task in tasks]
    finally:
        conn.close()

def execute_scan_and_analysis(model_id):
    """执行底层全盘扫描和缺失 AI 分析报告的批量并发补全任务"""
    print(f"⏰ [定时器激活] 开始执行定时全盘扫描任务... (AI 大脑: {model_id})")
    try:
        from .config_loader import get_global_settings
        from concurrent.futures import ThreadPoolExecutor
        
        # 1. 物理目录扫描入库
        added = sync_local_library()
        print(f"⏰ [定时器同步] 本地目录同步完毕。新增文献数: {added}")
        
        # 2. 诊断并批量补全缺失分析报告
        unanalyzed = get_unanalyzed_papers()
        if not unanalyzed:
            print("⏰ [定时器报告] 大仓检查完毕：所有文献已具有完整的 AI 分析报告。")
            return
            
        # 读取全局并发与批次上限配置
        settings = get_global_settings()
        max_workers = settings.get("max_concurrent_analysis", 2)
        max_batch = settings.get("max_papers_per_batch", 3)
        
        # Enforce batch limit
        papers_to_process = unanalyzed[:max_batch]
        total_papers = len(papers_to_process)
        
        print(f"⏰ [定时器解构] 诊断发现有 {len(unanalyzed)} 篇缺失报告。批次上限为 {max_batch}，本批次开始并发补全 {total_papers} 篇... (并发线程数: {max_workers})")
        
        def run_single_analysis(paper):
            print(f"⏰ [定时器并发任务开始] 正在解析: {paper['title']}")
            resolved_pdf = resolve_pdf_path(paper["pdf_path"])
            res = analyze_and_store_paper(paper["paper_id"], resolved_pdf, paper["title"], model_id=model_id)
            if res.startswith("❌"):
                print(f"⏰ [定时器并发任务报错] 《{paper['title']}》解构失败: {res}")
            else:
                print(f"⏰ [定时器并发任务成功] 《{paper['title']}》解构报告生成并成功入库。")
                
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(run_single_analysis, papers_to_process)
                
        print("⏰ [定时任务完成] 全盘自动定时扫描与大模型联合并发分析工作圆满结束！")
    except Exception as e:
        print(f"⏰ [定时任务出错] 执行大仓定时扫描时发生未捕获异常: {e}")

def execute_online_search_and_analysis(topic_key, search_limit, model_id):
    """后台执行线上雷达定时拉取探测、自动初审下载并并发技术解构"""
    print(f"⏰ [定时器激活] 开始执行线上雷达定时探测任务... (技术方向: {topic_key}, 数量: {search_limit}, AI 大脑: {model_id})")
    try:
        from config.research_topics import TOPIC_REGISTRY
        from .funnel_search import execute_two_stage_funnel_search
        from .ai_analyst import analyze_and_store_paper
        from .config_loader import get_global_settings
        
        if topic_key not in TOPIC_REGISTRY:
            print(f"⏰ [定时器错误] 未找到指定的技术方向: {topic_key}")
            return
            
        topic = TOPIC_REGISTRY[topic_key]
        
        # 1. 启动双阶段漏斗检索
        print("⏰ [定时器漏斗] 启动双阶段漏斗线上拉取检索...")
        new_items, used_engine = execute_two_stage_funnel_search(
            topic_name=topic["name"],
            query_string=topic["mapping_query"],
            target_limit=search_limit,
            model_id=model_id
        )
        
        if new_items:
            print(f"⏰ [定时器漏斗] 成功探测抓取并初审沉淀 {len(new_items)} 篇黄金文献。开始并发生成剖析报告...")
            # 2. 补全 AI 解析
            from concurrent.futures import ThreadPoolExecutor
            settings = get_global_settings()
            max_workers = settings.get("max_concurrent_analysis", 2)
            
            def run_single_analysis(item):
                print(f"⏰ [定时器并发任务开始] 正在剖析线上新文献: {item['title']}")
                resolved_pdf = resolve_pdf_path(item["pdf_path"])
                res = analyze_and_store_paper(item["paper_id"], resolved_pdf, item["title"], model_id=model_id)
                if res.startswith("❌"):
                    print(f"⏰ [定时器并发任务报错] 《{item['title']}》解构失败: {res}")
                else:
                    print(f"⏰ [定时器并发任务成功] 《{item['title']}》解构报告生成并成功入库。")
                    
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(run_single_analysis, new_items)
                
            print("⏰ [定时任务完成] 线上雷达定时探测与大模型全自动解构分析完成！")
        else:
            print("⏰ [定时器漏斗] 探测完毕，目前无新文献符合筛选标准。")
            
    except Exception as e:
        print(f"⏰ [定时任务出错] 执行线上雷达定时探测时发生异常: {e}")

def execute_scheduled_task(task):
    """根据定时任务的目标执行相应的工作流分流"""
    task_goal = task.get("task_goal", "local_scan")
    model_id = task.get("model_id")
    
    if task_goal == "local_scan":
        execute_scan_and_analysis(model_id)
    elif task_goal == "online_search":
        topic_key = task.get("topic_key")
        search_limit = task.get("search_limit", 15)
        execute_online_search_and_analysis(topic_key, search_limit, model_id)

def _scheduler_loop():
    """后台轮询主线程"""
    print("⏰ [轮询守护线程启动] 智能定时任务守护进程 RadarSchedulerDaemon 正在运行中...")
    while True:
        try:
            # 每隔 15 秒扫描一次数据库任务
            time.sleep(15)
            
            now = datetime.now()
            current_date_str = now.strftime("%Y-%m-%d")
            current_time_str = now.strftime("%H:%M")
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # 1. 处理单次定时任务 (One-shot)
            cursor.execute('''
                SELECT * FROM scheduler_tasks 
                WHERE task_type = 'one_shot' AND is_completed = 0
            ''')
            one_shot_tasks = cursor.fetchall()
            
            for task in one_shot_tasks:
                task_id = task['task_id']
                target_time = task['scheduled_time'] # 'YYYY-MM-DD HH:MM'
                
                try:
                    target_dt = datetime.strptime(target_time, "%Y-%m-%d %H:%M")
                    if now >= target_dt:
                        # 立即置为已完成，防止并发冲突
                        cursor.execute('UPDATE scheduler_tasks SET is_completed = 1 WHERE task_id = ?', (task_id,))
                        conn.commit()
                        
                        # 转换并异步启动分流处理
                        task_dict = dict(task)
                        threading.Thread(target=execute_scheduled_task, args=(task_dict,), daemon=True).start()
                except Exception as ex:
                    print(f"⏰ [定时任务时间格式错误] 任务 ID {task_id}: {target_time}, Error: {ex}")
            
            # 2. 处理每日定时任务 (Daily)
            cursor.execute('''
                SELECT * FROM scheduler_tasks 
                WHERE task_type = 'daily' AND scheduled_time = ?
            ''', (current_time_str,))
            daily_tasks = cursor.fetchall()
            
            for task in daily_tasks:
                task_id = task['task_id']
                last_run = task['last_run_date']
                
                # 若今天尚未执行过该时刻的任务，则触发
                if last_run != current_date_str:
                    cursor.execute('UPDATE scheduler_tasks SET last_run_date = ? WHERE task_id = ?', (current_date_str, task_id))
                    conn.commit()
                    
                    # 转换并异步启动分流处理
                    task_dict = dict(task)
                    threading.Thread(target=execute_scheduled_task, args=(task_dict,), daemon=True).start()
                    
            conn.close()
            
            # 3. 处理独立 AI 简报与技术洞察自动定时任务 (decoupled briefing/insight scheduling)
            try:
                from .briefing_manager import load_briefing_config, generate_daily_briefing_manually, generate_weekly_insight_manually, BASE_FOLDER_NAME
                import json
                
                br_config = load_briefing_config()
                if br_config.get("auto_scheduled", True):
                    current_day_name = now.strftime("%A") # e.g. "Monday"
                    state_path = os.path.join(BASE_FOLDER_NAME, ".scheduler_state.json")
                    os.makedirs(os.path.dirname(state_path), exist_ok=True)
                    
                    br_state = {}
                    if os.path.exists(state_path):
                        try:
                            with open(state_path, "r", encoding="utf-8") as fs:
                                br_state = json.load(fs)
                        except Exception:
                            pass
                            
                    # 每日简报判定
                    daily_time = br_config.get("daily_briefing_time", "09:00")
                    if current_time_str == daily_time and br_state.get("last_daily_date") != current_date_str:
                        br_state["last_daily_date"] = current_date_str
                        try:
                            with open(state_path, "w", encoding="utf-8") as fs:
                                json.dump(br_state, fs)
                        except Exception:
                            pass
                        print(f"⏰ [简报定时激活] 到了每日简报设定时刻 {daily_time}，启动后台强联网抓取...")
                        threading.Thread(target=generate_daily_briefing_manually, daemon=True).start()
                        
                    # 每周技术洞察判定
                    weekly_day = br_config.get("weekly_insight_day", "Monday")
                    weekly_time = br_config.get("weekly_insight_time", "10:00")
                    if current_day_name == weekly_day and current_time_str == weekly_time and br_state.get("last_weekly_date") != current_date_str:
                        br_state["last_weekly_date"] = current_date_str
                        try:
                            with open(state_path, "w", encoding="utf-8") as fs:
                                json.dump(br_state, fs)
                        except Exception:
                            pass
                        print(f"⏰ [洞察定时激活] 到了每周洞察设定时刻 {weekly_day} {weekly_time}，启动后台强联网抓取...")
                        threading.Thread(target=generate_weekly_insight_manually, daemon=True).start()
            except Exception as br_ex:
                print(f"⏰ [简报守护轮询异常] {br_ex}")
                
        except Exception as e:
            print(f"⏰ [守护进程轮询异常] 扫描触发任务时出错: {e}")

def start_scheduler():
    """启动后台定时任务轮询服务（线程安全）"""
    global _scheduler_started
    with _scheduler_lock:
        if not _scheduler_started:
            init_scheduler_db()
            t = threading.Thread(target=_scheduler_loop, name="RadarSchedulerDaemon", daemon=True)
            t.start()
            _scheduler_started = True
            print("⏰ [定时服务启动成功] 后台智能扫描分析守护服务成功激活运行！")
