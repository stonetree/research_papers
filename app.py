# -*- coding: utf-8 -*-
import builtins
import datetime
import re

_original_print = builtins.print

def timestamped_print(*args, **kwargs):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if args:
        first_arg = args[0]
        if isinstance(first_arg, str):
            # Avoid double-timestamping if the first argument already starts with a timestamp like "[2026-06-10 ..."
            if re.match(r"^\[\d{4}-\d{2}-\d{2}", first_arg):
                _original_print(*args, **kwargs)
            else:
                _original_print(f"[{now_str}] {first_arg}", *args[1:], **kwargs)
        else:
            _original_print(f"[{now_str}]", *args, **kwargs)
    else:
        _original_print(f"[{now_str}]", **kwargs)

builtins.print = timestamped_print

import streamlit as st
import os
import json
import threading
from core.database import init_db, get_db_connection, resolve_pdf_path, insert_search_archive, get_search_archives, delete_search_archive, delete_paper_metadata
from core.engine_semantic import execute_semantic_search
from core.engine_arxiv import execute_arxiv_search
from core.ai_analyst import analyze_and_store_paper, test_api_connection, model_web_search
from core.downloader import download_and_import_paper
from core.detection import get_search_capable_models, model_supports_web_search
from core.config_loader import load_api_config, get_default_model, set_default_model, get_global_settings, update_global_settings, update_model_config, delete_model_config
from core.library_scanner import sync_local_library, get_unanalyzed_papers
from core.scheduler import start_scheduler, add_scheduler_task, delete_scheduler_task, get_active_tasks
from core.funnel_search import execute_two_stage_funnel_search
from config.research_topics import TOPIC_REGISTRY
from core.env_helper import get_env_var

init_db()
api_models = load_api_config()
start_scheduler()

def format_utc_to_local(utc_str):
    if not utc_str:
        return "未知"
    try:
        from datetime import datetime, timezone
        # Handle SQLite CURRENT_TIMESTAMP with or without millisecond parts
        dt = datetime.strptime(utc_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=timezone.utc).astimezone(None)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return utc_str

def extract_snippet_with_highlight(text, keyword, length=200):
    if not text or not keyword:
        return ""
    lower_text = text.lower()
    lower_keyword = keyword.lower()
    idx = lower_text.find(lower_keyword)
    if idx == -1:
        return text[:length] + "..." if len(text) > length else text
    start = max(0, idx - 80)
    end = min(len(text), idx + len(keyword) + 100)
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    escaped_keyword = re.escape(keyword)
    highlighted = re.sub(
        f"({escaped_keyword})", 
        r'<span style="background-color: #ffd43b; color: #1e1e1e; padding: 2px 6px; border-radius: 4px; font-weight: bold;">\1</span>', 
        snippet, 
        flags=re.IGNORECASE
    )
    return highlighted

def format_evidence_url(url: str) -> str:
    if not url:
        return "#"
    if url.startswith("https://radar.ai/migrated/"):
        arxiv_id = url.replace("https://radar.ai/migrated/", "")
        arxiv_id_clean = arxiv_id.replace("_", ".")
        # 判定是否是标准的 arXiv ID 或旧版的类别/ID形式
        if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", arxiv_id_clean) or "/" in arxiv_id_clean:
            return f"https://arxiv.org/abs/{arxiv_id_clean}"
    return url

def get_api_call_count_summary():
    """按 API provider 聚合本日、本周、本月调用次数。"""
    providers = ["deepseek", "dashscope", "google", "exa", "firecrawl"]
    summary = {p: {"today": 0, "week": 0, "month": 0} for p in providers}
    try:
        conn = get_db_connection()
        rows = conn.execute("""
            SELECT
                api_provider,
                SUM(CASE WHEN created_at >= datetime('now', 'start of day') THEN 1 ELSE 0 END) AS today_count,
                SUM(CASE WHEN created_at >= datetime('now', '-7 days') THEN 1 ELSE 0 END) AS week_count,
                SUM(CASE WHEN created_at >= datetime('now', 'start of month') THEN 1 ELSE 0 END) AS month_count
            FROM quota_ledger
            GROUP BY api_provider
        """).fetchall()
        conn.close()
        for row in rows:
            provider = row["api_provider"]
            summary.setdefault(provider, {"today": 0, "week": 0, "month": 0})
            summary[provider]["today"] = int(row["today_count"] or 0)
            summary[provider]["week"] = int(row["week_count"] or 0)
            summary[provider]["month"] = int(row["month_count"] or 0)
    except Exception as e:
        print(f"读取 API 调用次数统计失败: {e}")
    return summary

if "unanalyzed_papers" not in st.session_state:
    st.session_state["unanalyzed_papers"] = []

if "active_view_paper_id" not in st.session_state:
    st.session_state["active_view_paper_id"] = None

if "search_keyword" not in st.session_state:
    st.session_state["search_keyword"] = ""

# 初始化后台数据库状态变化监测器 (在学术雷达或后台定时扫描抓取到新文献或 AI 报告解析完成时，自动触发当前 UI 页面的热重载刷新)
if "db_monitor_started" not in st.session_state:
    st.session_state["db_monitor_started"] = True
    
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    from streamlit.runtime import get_instance as get_runtime_instance
    import time
    
    ctx = get_script_run_ctx()
    if ctx:
        session_id = ctx.session_id
        
        def monitor_db_changes():
            from core.database import get_db_connection
            def get_db_state():
                conn = get_db_connection()
                try:
                    papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
                    summaries = conn.execute("SELECT COUNT(*) FROM ai_summaries WHERE dialectical_analysis IS NOT NULL").fetchone()[0]
                    return papers, summaries
                except Exception:
                    return 0, 0
                finally:
                    conn.close()
            
            last_papers, last_summaries = get_db_state()
            
            while True:
                time.sleep(3)  # 每 3 秒轮询一次数据库状态
                
                # 检查 Streamlit runtime 实例及当前会话是否依然活跃，如果不活跃（如关闭了页面）则安全退出线程
                try:
                    rt = get_runtime_instance()
                    if not rt or not rt.is_active_session(session_id):
                        break
                except Exception:
                    break
                    
                current_papers, current_summaries = get_db_state()
                if current_papers != last_papers or current_summaries != last_summaries:
                    last_papers, last_summaries = current_papers, current_summaries
                    # 触发当前 session 的原生重绘刷新
                    try:
                        session_info = rt._session_mgr.get_active_session_info(session_id)
                        if session_info and session_info.session:
                            session_info.session.request_rerun(None)
                    except Exception as e:
                        print(f"⏰ [自动刷新异常] 无法触发 UI 刷新: {e}")
                        
        t = threading.Thread(target=monitor_db_changes, daemon=True)
        t.start()
        print(f"⏰ [UI监控激活] 已针对会话 {session_id} 启动后台数据库变化自适应刷新监控器。")

st.set_page_config(page_title="🪐 Infrastructure AI Radar Hub", layout="wide")

# 注入 CSS 消除 Streamlit 默认的顶部巨大空白并隐藏空置头部栏，提供防裁切的响应式自适应布局
st.markdown("""
    <style>
        /* 仅将头部页眉背景设为透明，避免遮挡，同时确保最左侧的侧边栏收缩/展开控制按钮完美可见与正常操作 */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }
        /* 动态温和地缩减主体容器顶部外边距，适应不同显示器分辨率与屏幕缩放 */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        /* 移除 h1 大标题的任何负外边距以完全防范文字裁切 */
        h1 {
            margin-top: 0rem !important;
        }
        /* 调整 Tab 标签页头的字体大小与样式，使功能导航更加大气易读 */
        button[data-baseweb="tab"] {
            font-size: 1.15rem !important;
            font-weight: 600 !important;
        }
        button[data-baseweb="tab"] p {
            font-size: 1.15rem !important;
            font-weight: 600 !important;
        }
        /* 统一缩小多列布局中的操作按钮尺寸，保持极其精致的高端观感，并确保不折行 */
        div[data-testid="column"] button {
            font-size: 0.85rem !important;
            padding: 0.25rem 0.5rem !important;
            min-height: 2.1rem !important;
            line-height: 1.2 !important;
            white-space: nowrap !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🪐 AI 基础设施与软硬件协同 —— 个人智能论文知识库")

# 侧边栏：全局诊断与状态中心
st.sidebar.title("📡 全局诊断与状态中心")
st.sidebar.markdown("---")

# 📊 大仓资产看板
st.sidebar.subheader("📊 大仓资产看板")

try:
    conn = get_db_connection()
    total_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    analyzed_papers = conn.execute("SELECT COUNT(*) FROM ai_summaries WHERE dialectical_analysis IS NOT NULL AND dialectical_analysis != ''").fetchone()[0]
    conn.close()
except Exception as e:
    total_papers = 0
    analyzed_papers = 0

coverage = (analyzed_papers / total_papers * 100.0) if total_papers > 0 else 0.0

col_metric1, col_metric2 = st.sidebar.columns(2)
with col_metric1:
    st.metric("已收录文献", f"{total_papers} 篇")
with col_metric2:
    st.metric("已解构报告", f"{analyzed_papers} 篇")
st.sidebar.metric("大仓解构率", f"{coverage:.1f}%")

st.sidebar.markdown("---")

# 🔌 系统诊断与状态
st.sidebar.subheader("🔌 系统诊断与状态")

try:
    from core.api_clients import check_embedding_service_health_sync, check_rerank_service_health_sync
except ImportError:
    from core.api_clients import LocalComputeKernelClient
    check_embedding_service_health_sync = LocalComputeKernelClient.check_service_health_sync
    check_rerank_service_health_sync = LocalComputeKernelClient.check_rerank_service_health_sync

from core.ingestion import get_pending_vectorization_documents, batch_process_pending_vectorization_sync

is_embed_ready = check_embedding_service_health_sync()
is_rerank_ready = check_rerank_service_health_sync()

pending_vec_docs = get_pending_vectorization_documents()
pending_vec_count = len(pending_vec_docs)

st.sidebar.caption(f"🎯 Rerank 引擎 (8082): {'🟢 就绪 (Ready)' if is_rerank_ready else '🟡 离线/未启动 (已自动跳过)'}")

if is_embed_ready:
    st.sidebar.caption("🧬 Embedding 引擎 (8081): 🟢 就绪 (Ready)")
    if pending_vec_count > 0:
        st.sidebar.warning(f"🟢 **Embedding 服务已就绪！** 检测到 `{pending_vec_count}` 篇待向量化积压文档。")
        if st.sidebar.button("⚡ 启动待入库队列处理", type="primary", key="side_run_pending_sync", width="stretch"):
            with st.spinner("正在为待处理队列补全切片与向量特征..."):
                res_sync = batch_process_pending_vectorization_sync()
                st.sidebar.success(f"🎉 补偿处理成功！完成 {res_sync.get('processed_count', 0)} 篇。")
                st.toast("🟢 待向量化队列自愈完成！")
                st.rerun()
else:
    st.sidebar.caption("🧬 Embedding 引擎 (8081): 🟡 离线/未启动")
    if pending_vec_count > 0:
        st.sidebar.warning(f"⚠️ **Embedding 未就绪** | 待处理积压: `{pending_vec_count}` 篇 (已被安全挂起存储)")

# 读取开机默认大脑设置并匹配选项索引
default_model_id = get_default_model()
model_keys = list(api_models.keys())
global_settings_sidebar = get_global_settings()
detailed_analysis_model_id = global_settings_sidebar.get("detailed_analysis_model_id", default_model_id)
abstract_relevance_model_id = global_settings_sidebar.get("abstract_relevance_model_id", default_model_id)
default_index = model_keys.index(detailed_analysis_model_id) if detailed_analysis_model_id in model_keys else (model_keys.index(default_model_id) if default_model_id in model_keys else 0)

selected_brain_key = st.sidebar.selectbox(
    "论文精细化分析大脑",
    options=model_keys,
    index=default_index,
    format_func=lambda x: api_models[x].get("name", x),
    key="active_reading_brain_sidebar"
)
selected_analysis_model_id = selected_brain_key
selected_relevance_model_id = abstract_relevance_model_id if abstract_relevance_model_id in model_keys else selected_analysis_model_id
selected_online_search_model_id = global_settings_sidebar.get("search_model_id", selected_analysis_model_id)
if selected_online_search_model_id not in model_keys:
    selected_online_search_model_id = selected_analysis_model_id

# 守护调度线程状态
is_scheduler_running = any(t.name == "RadarSchedulerDaemon" for t in threading.enumerate())
scheduler_status_html = (
    "<span style='color: green; font-weight: bold;'>🟢 运行中</span>" 
    if is_scheduler_running 
    else "<span style='color: red; font-weight: bold;'>🔴 未启动</span>"
)

# 物理大仓目录状态
from core.library_scanner import LIBRARY_DIR
folder_exists = os.path.exists(LIBRARY_DIR)
folder_writable = os.access(LIBRARY_DIR, os.W_OK) if folder_exists else False
if folder_exists and folder_writable:
    folder_status_html = "<span style='color: green; font-weight: bold;'>🟢 正常</span>"
else:
    folder_status_html = "<span style='color: red; font-weight: bold;'>🔴 异常</span>"

st.sidebar.markdown(f"""
<div style='font-size: 0.95rem; line-height: 1.8; color: #1F2937;'>
    ⏳ <b>守护调度状态</b>: {scheduler_status_html}<br>
    📂 <b>物理大仓状态</b>: {folder_status_html}
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 API 调用次数")
api_call_counts = get_api_call_count_summary()
provider_labels = {
    "deepseek": "DeepSeek",
    "dashscope": "DashScope",
    "google": "Google",
    "exa": "Exa",
    "firecrawl": "Firecrawl"
}
st.sidebar.caption("统计口径：`quota_ledger` 中已登记的 API 调用流水；本周为最近 7 天。")
for provider, label in provider_labels.items():
    item = api_call_counts.get(provider, {"today": 0, "week": 0, "month": 0})
    st.sidebar.markdown(
        f"""
        <div style='font-size: 0.88rem; line-height: 1.55; padding: 4px 0;'>
            <b>{label}</b><br>
            今日 <code>{item['today']}</code> 次 · 本周 <code>{item['week']}</code> 次 · 本月 <code>{item['month']}</code> 次
        </div>
        """,
        unsafe_allow_html=True
    )


# 主界面：六重选项卡分流
tab_library, tab_model_search, tab_local_search, tab_scheduler, tab_briefings, tab_global_config = st.tabs([
    "📂 本地沉淀文献大仓", 
    "🔍 AI 联网学术探测",
    "🔎 本地大仓检索",
    "⏰ 智能定时扫描与解构调度", 
    "🌐 AI 24h雷达与技术洞察", 
    "⚙️ 全局系统配置"
])

with tab_library:
    # 📡 学术雷达漏斗探测与大仓维护
    st.markdown("### 📡 论文雷达漏斗探测与大仓维护")
    
    # 提前获取未解构的文献信息以决定补全按钮的内容
    unanalyzed_list = st.session_state.get("unanalyzed_papers", [])
    if not unanalyzed_list:
        unanalyzed_list = get_unanalyzed_papers()
        st.session_state["unanalyzed_papers"] = unanalyzed_list
        
    global_settings = get_global_settings()
    max_workers = global_settings.get("max_concurrent_analysis", 2)
    max_batch = global_settings.get("max_papers_per_batch", 3)
    
    papers_to_process = unanalyzed_list[:max_batch]
    total_papers = len(papers_to_process)

    col_topic_sel, col_limit_sel, col_scan_btn, col_sync_btn, col_batch_btn = st.columns([2.0, 0.7, 0.75, 0.75, 0.8])
    with col_topic_sel:
        selected_topic_key = st.selectbox(
            "选择技术演进方向",
            options=list(TOPIC_REGISTRY.keys()),
            format_func=lambda x: TOPIC_REGISTRY[x]["name"],
            key="library_scan_topic_selector"
        )
    with col_limit_sel:
        search_limit = st.selectbox(
            "探测数量",
            options=[10, 15, 20, 25, 30],
            index=1, # 15
            key="library_scan_limit_selector",
            help="设定本次雷达扫描探测的文献最大数量上限。"
        )
    with col_scan_btn:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        scan_triggered = st.button("🚀 触发雷达", key="library_scan_btn", width="stretch")
    with col_sync_btn:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        sync_triggered = st.button("🔄 同步大仓", key="library_sync_btn", width="stretch", help="扫描并同步本地手动下载的 PDF 文件，更新大仓索引")
    with col_batch_btn:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        batch_label = f"🤖 并发补全 ({total_papers})" if total_papers > 0 else "🤖 无待补全"
        batch_disabled = (total_papers == 0)
        batch_completer = st.button(batch_label, key="library_batch_complete_btn", width="stretch", disabled=batch_disabled, help="并发解析已入库但尚未生成报告的文献")

    # 反馈信息容器
    status_container = st.container()
    
    if scan_triggered:
        with status_container:
            topic = TOPIC_REGISTRY[selected_topic_key]
            new_items = []
            used_engine = "多源漏斗管道"
            
            with st.spinner("正在启动双阶段漏斗扫描探测..."):
                try:
                    new_items, used_engine = execute_two_stage_funnel_search(
                        topic_name=topic["name"],
                        query_string=topic["mapping_query"],
                        target_limit=search_limit,
                        model_id=selected_relevance_model_id
                    )
                except Exception as e:
                    st.error(f"❌ 漏斗检索发生异常故障: {e}")
                    
            if new_items:
                st.success(f"🎉 【{used_engine}】成功抓取并仲裁沉淀 {len(new_items)} 篇黄金文献！")
                has_error = False
                for item in new_items:
                    brain_name = api_models[selected_analysis_model_id].get("name", selected_analysis_model_id)
                    with st.spinner(f"🤖 正在激活 {brain_name} 全景解构: {item['title'][:30]}..."):
                        res = analyze_and_store_paper(item["paper_id"], item["pdf_path"], item["title"], model_id=selected_analysis_model_id)
                        if res.startswith("❌"):
                            st.error(res)
                            has_error = True
                if not has_error:
                    st.rerun()
            else:
                st.info("📭 探测完毕，大仓内当前方向在近期无更替。")

    if sync_triggered:
        with status_container:
            with st.spinner("正在扫描 storage/library 并更新本地索引..."):
                added = sync_local_library()
                unanalyzed = get_unanalyzed_papers()
                
                if added > 0:
                    st.success(f"🎉 物理大仓同步成功！新发现 {added} 篇本地 PDF 文件并自动入库登记。")
                else:
                    st.info("📂 物理同步完毕，未发现新增加的物理 PDF 文件。")
                    
                if unanalyzed:
                    st.warning(f"⏳ 诊断：库中当前共有 {len(unanalyzed)} 篇文献尚未生成 AI 剖析报告。")
                    st.session_state["unanalyzed_papers"] = unanalyzed
                else:
                    st.success("🟢 诊断：库内所有文献均拥有完美的 AI 辩证剖析报告！")
                    st.session_state["unanalyzed_papers"] = []
                st.rerun()

    if batch_completer and total_papers > 0:
        with status_container:
            progress_bar = st.progress(0.0)
            has_any_error = False
            
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            status_text = st.empty()
            error_container = st.empty()
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(analyze_and_store_paper, paper["paper_id"], paper["pdf_path"], paper["title"], model_id=selected_analysis_model_id): paper
                    for paper in papers_to_process
                }
                
                for idx, future in enumerate(as_completed(futures)):
                    paper = futures[future]
                    brain_name = api_models[selected_analysis_model_id].get("name", selected_analysis_model_id)
                    status_text.caption(f"[{idx+1}/{total_papers}] 并发完成: {paper['title'][:15]}...")
                    
                    try:
                        res = future.result()
                        if res.startswith("❌"):
                            error_container.error(f"❌ 《{paper['title'][:10]}》剖析失败: {res}")
                            has_any_error = True
                    except Exception as e:
                        error_container.error(f"❌ 《{paper['title'][:10]}》触发异常: {e}")
                        has_any_error = True
                        
                    progress_bar.progress((idx + 1) / total_papers)
                    
            if not has_any_error:
                st.success("🎉 一键并发剖析成功！所有学术剖析报告已补齐！")
                st.session_state["unanalyzed_papers"] = []
                st.rerun()

    st.markdown("---")

    # 0. 全局数据装载与检索过滤 (位于最顶层以保持结构规整与数据一致)
    col_filter1, col_filter2 = st.columns([2.5, 1])
    with col_filter1:
        search_keyword = st.text_input(
            "🔍 全文搜索大模型分析报告", 
            value=st.session_state.get("search_keyword", ""),
            placeholder="输入关键词进行过滤，清空可浏览全部大仓...",
            key="library_search_input"
        )
        st.session_state["search_keyword"] = search_keyword.strip()
    with col_filter2:
        analysis_filter = st.selectbox(
            "🧠 AI 解析状态筛选",
            options=["全部文献", "🟢 已完成 AI 解析", "⏳ 尚未完成 AI 解析"],
            index=0,
            key="library_analysis_filter"
        )
    
    conn = get_db_connection()
    if st.session_state["search_keyword"]:
        # 全文搜索逻辑
        query = """
            SELECT p.*, s.dialectical_analysis, s.model_name, s.updated_at 
            FROM papers p
            INNER JOIN ai_summaries s ON p.paper_id = s.paper_id
            WHERE s.dialectical_analysis IS NOT NULL AND s.dialectical_analysis != ''
        """
        all_papers = conn.execute(query).fetchall()
        conn.close()
        
        # 计算匹配频次
        keyword_lower = st.session_state["search_keyword"].lower()
        matched_results = []
        for paper in all_papers:
            analysis_text = paper['dialectical_analysis'] or ""
            match_count = analysis_text.lower().count(keyword_lower)
            if match_count > 0:
                matched_results.append({
                    "paper": paper,
                    "match_count": match_count
                })
        # 按频次降序
        matched_results.sort(key=lambda x: x["match_count"], reverse=True)
        top_results = matched_results[:10]
        papers_to_show = [r["paper"] for r in top_results]
    else:
        # 默认无搜索词展示所有已入库的文献
        query = """
            SELECT p.*, s.dialectical_analysis, s.model_name, s.updated_at 
            FROM papers p
            LEFT JOIN ai_summaries s ON p.paper_id = s.paper_id
            ORDER BY p.created_at DESC
        """
        papers_to_show = conn.execute(query).fetchall()
        conn.close()
        
    # 应用 AI 解析状态的二次过滤
    filtered_papers = []
    for p in papers_to_show:
        is_analyzed = bool(p["dialectical_analysis"] and p["dialectical_analysis"].strip())
        if analysis_filter == "🟢 已完成 AI 解析" and not is_analyzed:
            continue
        if analysis_filter == "⏳ 尚未完成 AI 解析" and is_analyzed:
            continue
        filtered_papers.append(p)
    papers_to_show = filtered_papers
        
    paper_ids = [p["paper_id"] for p in papers_to_show]
    st.session_state["library_paper_ids"] = paper_ids
    
    # 选中项维护
    active_paper_id = st.session_state.get("active_view_paper_id")
    if active_paper_id not in paper_ids and paper_ids:
        active_paper_id = paper_ids[0]
        st.session_state["active_view_paper_id"] = active_paper_id
        
    # 读取当前选中的论文实体
    paper = None
    if active_paper_id:
        conn = get_db_connection()
        paper = conn.execute("""
            SELECT p.*, s.dialectical_analysis, s.model_name, s.updated_at 
            FROM papers p
            LEFT JOIN ai_summaries s ON p.paper_id = s.paper_id
            WHERE p.paper_id = ?
        """, (active_paper_id,)).fetchone()
        conn.close()

    # ---------------- 1. 第一区域：贯穿页面的窄区（工具按钮栏） ----------------
    if paper:
        paper = dict(paper)
        try:
            curr_idx = paper_ids.index(active_paper_id)
        except ValueError:
            curr_idx = -1
            
        with st.container(border=True):
            btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns([1, 1, 2.2, 2.2, 0.8])
            with btn_col1:
                prev_disabled = (curr_idx <= 0)
                if st.button("⏮️ 上一篇", key="prev_paper_btn", width="stretch", disabled=prev_disabled):
                    st.session_state["active_view_paper_id"] = paper_ids[curr_idx - 1]
                    st.rerun()
            with btn_col2:
                next_disabled = (curr_idx == len(paper_ids) - 1 or curr_idx == -1)
                if st.button("下一篇 ⏭️", key="next_paper_btn", width="stretch", disabled=next_disabled):
                    st.session_state["active_view_paper_id"] = paper_ids[curr_idx + 1]
                    st.rerun()
            with btn_col3:
                has_valid_analysis = paper.get('dialectical_analysis') and not paper['dialectical_analysis'].startswith("❌")
                if has_valid_analysis:
                    st.download_button(
                        label="📥 一键导出 Markdown 报告",
                        data=paper['dialectical_analysis'],
                        file_name=f"{paper['title']}_AI学术解构报告.md",
                        mime="text/markdown",
                        key=f"export_detail_{paper['paper_id']}",
                        width="stretch"
                    )
                else:
                    brain_name = api_models[selected_analysis_model_id].get("name", selected_analysis_model_id)
                    btn_label = f"🔄 重新激活 {brain_name} 解构" if paper['dialectical_analysis'] else f"🤖 激活 {brain_name} 解构"
                    if st.button(btn_label, key=f"detail_activate_{paper['paper_id']}", width="stretch", type="primary"):
                        with st.spinner("正在解构剖析中..."):
                            analysis_text = analyze_and_store_paper(paper['paper_id'], paper['pdf_path'], paper['title'], model_id=selected_analysis_model_id)
                            if analysis_text.startswith("❌"):
                                st.error(analysis_text)
                            else:
                                st.rerun()
            with btn_col4:
                st.markdown(f"<div style='padding-top: 6px; font-size: 0.9rem; color: #4B5563; font-weight: bold; text-align: right;'>📖 进度: {curr_idx + 1}/{len(paper_ids)} 篇 | 🧠 剖析: {paper.get('model_name') or '待激活'}</div>", unsafe_allow_html=True)
            with btn_col5:
                with st.popover("🗑️", help="删除文献元数据", use_container_width=True):
                    st.markdown("**🗑️ 彻底删除文献元数据**")
                    st.caption("系统将从本地数据库中永久删除该文献的登记元数据及生成的 AI 全景解构报告（不删除本地 PDF 物理文件）。此操作不可逆，请谨慎操作。")
                    if st.button("🔥 确认删除", key=f"delete_paper_confirm_{paper['paper_id']}", use_container_width=True, type="primary"):
                        delete_paper_metadata(paper['paper_id'])
                        if "active_view_paper_id" in st.session_state:
                            del st.session_state["active_view_paper_id"]
                        st.toast("🗑️ 文献元数据与解析报告已成功删除！")
                        st.rerun()
    else:
        st.info("💡 暂无正在阅读的文献数据。")

    # ---------------- 2. 第二区域：紧贴上一个区域的下面（状态显示区） ----------------
    if paper:
        # 核验本地 PDF 关联状态
        resolved_pdf = resolve_pdf_path(paper['pdf_path']) if paper['pdf_path'] else ""
        if resolved_pdf and os.path.exists(resolved_pdf):
            pdf_status = "<span style='color: green; font-weight: bold;'>🟢 本地 PDF 已安全关联</span>"
        else:
            pdf_status = "<span style='color: red; font-weight: bold;'>🔴 本地 PDF 物理文件缺失</span>"
            
        with st.container(border=True):
            st.markdown(f"""
                <div style='font-size: 0.95rem; line-height: 1.5; color: #1F2937;'>
                    📖 <b>当前阅读</b>：《{paper['title']}》 &nbsp;|&nbsp; 
                    🏷️ <b>顶会/期刊</b>：<code>{paper['venue'] or '顶会/未标注'}</code> &nbsp;|&nbsp; 
                    📅 <b>年份</b>：<code>{paper['year'] or '未知'}</code> &nbsp;|&nbsp; 
                    📈 <b>引用</b>：<code>{paper['citations'] or 0}</code> &nbsp;|&nbsp; 
                    📎 <b>关联状态</b>：{pdf_status}
                </div>
            """, unsafe_allow_html=True)

    # ---------------- 3. 下面区域：左右两半对称滚动布局（高度一致） ----------------
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1.3])
    
    with col_left:
        st.markdown("##### 📂 文献大仓导航")
        if not papers_to_show:
            st.info("📭 无匹配文献数据。")
        else:
            # 独立滚动的左侧列表框 (高度设为 600px，与右侧框体完美绝对一致)
            with st.container(height=600):
                for p in papers_to_show:
                    p_id = p["paper_id"]
                    is_selected = (p_id == active_paper_id)
                    
                    card_emoji = "📖" if is_selected else "📄"
                    if p["dialectical_analysis"]:
                        if p["dialectical_analysis"].startswith("❌"):
                            ai_status = "🔴"
                        else:
                            ai_status = "🟢"
                    else:
                        ai_status = "⏳"
                    
                    card_title = f"{card_emoji} {p['title']}"
                    add_time = format_utc_to_local(p['created_at'])
                    parse_time = format_utc_to_local(p['updated_at']) if p['dialectical_analysis'] else "尚未解析"
                    card_meta = f"{ai_status} [{p['venue'] or '顶会'}] {p['year']} | 📈 引用: {p['citations']} | 📅 添加时间: {add_time} | 🕒 解析时间: {parse_time}"
                    
                    # 渲染为小卡片样式
                    with st.container(border=True):
                        st.markdown(f"**{card_title}**")
                        st.caption(card_meta)
                        
                        # 搜索模式高亮
                        if st.session_state["search_keyword"] and p["dialectical_analysis"]:
                            snippet = extract_snippet_with_highlight(p['dialectical_analysis'], st.session_state["search_keyword"])
                            st.markdown(f"<div style='font-size: 0.85rem; color: #555; background: #f0f2f6; padding: 4px; border-radius: 4px; margin-bottom: 6px;'>🔍 {snippet}</div>", unsafe_allow_html=True)
                            
                        btn_label = "👉 正在阅读" if is_selected else "📖 极速阅读解构报告"
                        if st.button(btn_label, key=f"select_btn_{p_id}", width="stretch", type="primary" if is_selected else "secondary"):
                            st.session_state["active_view_paper_id"] = p_id
                            st.rerun()

    with col_right:
        st.markdown("##### 💡 首席科学家 AI 辩证剖析报告")
        if paper:
            paper = dict(paper)
            # 独立滚动的右侧报告内容框 (高度设为 600px，与左侧框体完美绝对一致)
            with st.container(height=600):
                st.markdown(f"### 📘 《{paper['title']}》")
                st.markdown("---")
                
                # 作者团队与物理文件信息
                st.markdown(f"**👥 作者团队**: {paper['authors'] or '未知团队'}")
                st.markdown(f"**📝 物理文件**: `{os.path.basename(paper['pdf_path']) if paper['pdf_path'] else '未关联'}`")
                
                # 格式化并渲染元数据信息 (来源、解析模型、添加时间、解析时间)
                source_val = paper.get('source_engine', '')
                if source_val == 'manual':
                    source_desc = "✍️ 手动添加"
                elif source_val == 'arxiv':
                    source_desc = "🤖 自动从 arXiv 添加"
                elif source_val == 'semantic_scholar':
                    source_desc = "🤖 自动从 Semantic Scholar 添加"
                elif source_val == 'model_web_search':
                    source_desc = "🔍 从 AI 联网学术探测添加"
                else:
                    source_desc = f"📁 {source_val}" if source_val else "📁 其他来源"

                model_desc = paper.get('model_name')
                if not model_desc or not paper.get('dialectical_analysis'):
                    model_desc = "⏳ 尚未进行 AI 解析"
                    parse_time_desc = "⏳ 尚未进行 AI 解析"
                else:
                    model_desc = f"🧠 {model_desc}"
                    parse_time_desc = format_utc_to_local(paper.get('updated_at'))

                add_time_desc = format_utc_to_local(paper.get('created_at'))
                
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 12px; border-radius: 6px; margin-bottom: 15px; border: 1px solid #e9ecef; font-size: 0.9rem; color: #374151;">
                    <table style="width: 100%; border: none; border-collapse: collapse;">
                        <tr style="border: none;">
                            <td style="border: none; padding: 4px 0; width: 50%;"><b>📥 文档来源</b>: {source_desc}</td>
                            <td style="border: none; padding: 4px 0; width: 50%;"><b>⚙️ 解析模型</b>: {model_desc}</td>
                        </tr>
                        <tr style="border: none;">
                            <td style="border: none; padding: 4px 0; width: 50%;"><b>📅 添加时间</b>: {add_time_desc}</td>
                            <td style="border: none; padding: 4px 0; width: 50%;"><b>🕒 解析时间</b>: {parse_time_desc}</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Abstract (摘要)**:")
                st.info(paper['abstract'] or "暂无摘要描述。")
                
                st.markdown("---")
                
                st.markdown("##### 💡 首席科学家 AI 剖析报告正文")
                if paper['dialectical_analysis']:
                    if paper['dialectical_analysis'].startswith("❌"):
                        st.error(paper['dialectical_analysis'])
                    else:
                        st.markdown(paper['dialectical_analysis'])
                else:
                    st.warning("⏳ 暂无该论文的 AI 深度解构。")
        else:
            with st.container(height=600):
                st.markdown("<h3 style='text-align: center; color: #4B5563; padding-top: 150px;'>🪐 个人学术大仓阅读器</h3>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; color: #6B7280;'>请点选左侧大仓导航中的文献卡片以加载解构报告</p>", unsafe_allow_html=True)

with tab_model_search:
    st.subheader("🔍 AI 大脑联网学术探测")
    st.markdown("该模块已升级为 **Pipeline B 一体化学术探测通路**。系统将执行 FTS5 + LanceDB 双路并行检索，当置信度未击穿门槛时，自动唤醒 Exa 探针精准打捞并 2PC 安全落库，最终由选定的 AI 大脑进行事实问答合成，保证 100% 真实引用。")
    
    # 初始化会话状态以存储搜索结果
    if "model_search_results_v2" not in st.session_state:
        st.session_state["model_search_results_v2"] = None
    if "model_search_query_used" not in st.session_state:
        st.session_state["model_search_query_used"] = ""

    # 配置区
    with st.container(border=True):
        col_query_in, col_filter, col_web, col_penetrate = st.columns([2.2, 1.1, 1.2, 1.2])
        with col_query_in:
            model_query = st.text_input(
                "输入您关心的技术关键词/提问 Query", 
                value="", 
                placeholder="例如: CXL 3.0 cache coherence, vLLM KV cache optimization...",
                key="model_search_query_input",
                label_visibility="collapsed"
            )
        with col_filter:
            target_filter = st.selectbox(
                "打捞数据源类型",
                options=["arxiv_paper", "ext_blog"],
                format_func=lambda x: {
                    "arxiv_paper": "📚 学术论文",
                    "ext_blog": "🌐 技术博客/网页"
                }.get(x, x),
                index=0,
                key="model_search_filter_selector"
            )
        with col_web:
            allow_web_search = st.checkbox(
                "🌐 允许联网打捞",
                value=True,
                help="关闭后仅使用本地知识大仓进行检索与解答，绝对不消耗任何联网抓取与外部 API 配额"
            )
        with col_penetrate:
            force_penetrate = st.checkbox(
                "🔥 强制穿透外部打捞",
                value=False,
                disabled=not allow_web_search,
                help="即使本地大仓存在高置信度匹配，依然强制发起 Exa 外部联网打捞（需要开启允许联网打捞）"
            )

        trigger_search = st.button("🚀 启动 AI 联网学术探测", type="primary", width="stretch")

    if trigger_search:
        if not model_query.strip():
            st.warning("⚠️ 请输入有效的提问 Query。")
        else:
              with st.status("🚀 正在启动双路混合检索与 Pipeline B 打捞合成流水线...", expanded=True) as status:
                  import asyncio
                  from core.search_engine import execute_unified_studio_search_flow
                  
                  def update_progress(text):
                      status.write(text)
                      status.update(label=text)
                  
                  try:
                      res = asyncio.run(execute_unified_studio_search_flow(
                          query=model_query.strip(),
                          filter_type=target_filter,
                          force_penetrate=force_penetrate,
                          allow_web_search=allow_web_search,
                          model_id=selected_online_search_model_id,
                          on_progress=update_progress
                      ))
                      
                      if res.get("status") == "success":
                          st.session_state["model_search_results_v2"] = {
                              "answer": res.get("answer", ""),
                              "evidences": res.get("evidences", []),
                              "routing_path": res.get("routing_path", ""),
                              "query": model_query.strip(),
                              "model_id": selected_online_search_model_id
                          }
                          st.session_state["model_search_query_used"] = model_query.strip()
                          status.write("🟢 AI 大脑对账与解答合成完毕！")
                          status.update(label="🟢 联网学术探测及问答合成成功！", state="complete", expanded=False)
                          st.toast("🟢 联网学术探测及问答合成成功！")
                          st.rerun()
                      else:
                          status.write(f"🔴 探测失败: {res.get('error', '未知错误')}")
                          status.update(label=f"🔴 探测失败: {res.get('error', '未知错误')}", state="error", expanded=True)
                  except Exception as e:
                      status.write(f"❌ 运行探测流发生异常: {e}")
                      status.update(label=f"❌ 运行探测流发生异常: {e}", state="error", expanded=True)


    # 显示结果
    search_res = st.session_state.get("model_search_results_v2")
    if search_res is not None:
        st.markdown("---")
        
        # 1. 判定返回的是 V2 格式 (dict) 还是 V1 格式 (list)
        if isinstance(search_res, dict):
            # V2 格式：单栏渲染问答与文献
            st.markdown(f"##### 🧠 AI 大脑辩证答复 — `{search_res['query']}`")
            
            # 展现路由路径诊断
            path = search_res.get("routing_path", "")
            if path == "local_cache_hit":
                st.success("🎯 **检索路径诊断**：本地大仓高置信度精确命中！执行 0 成本本地解答合成。")
            elif path == "local_only":
                st.info("🔒 **检索路径诊断**：联网检索已被选项关闭，限定仅使用本地大仓（FTS5 + LanceDB 向量）进行检索与解答合成。")
            elif path == "exa_penetrate_funnel":
                st.warning("🔍 **检索路径诊断**：本地大仓未完全覆盖，或者开启了强制穿透。已启动 Exa 神经网络打捞，并将最新成果 2PC 沉淀落库。")
            else:
                st.info(f"ℹ️ **检索路径诊断**：{path}")
            
            # 用 Glassmorphism 样式容器渲染 AI 答复
            st.markdown(f"""
            <div style='background: rgba(255, 255, 255, 0.7); 
                        backdrop-filter: blur(10px); 
                        border: 1px solid rgba(229, 231, 235, 0.5); 
                        border-radius: 12px; 
                        padding: 20px; 
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
                        margin-bottom: 20px;'>
                <div style='font-size: 1rem; color: #1F2937; line-height: 1.7;'>
                    {search_res['answer']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### 📚 事实证据与引用文献")
            
            evidences = search_res.get("evidences", [])
            if not evidences:
                st.info("💡 该答复未绑定任何具体的本地证据卡片。")
            else:
                for idx, ev in enumerate(evidences):
                    with st.container(border=True):
                        title_str = ev.get("title", "无标题")
                        url_str = format_evidence_url(ev.get("canonical_url", "#"))
                        doc_id = ev.get("doc_id", "unknown")
                        hybrid_score = ev.get("hybrid_score", 0.0)
                        
                        st.markdown(f"**[{idx+1}] [{doc_id}]** [{title_str}]({url_str})")
                        st.caption(f"📍 章节/位置: `{ev.get('section_path', 'N/A')}` &nbsp;|&nbsp; 📄 页码: `p.{ev.get('page_number', '1')}` &nbsp;|&nbsp; 📈 得分: `{hybrid_score:.4f}`")
                        st.markdown(f"<div style='font-size: 0.88rem; color: #4B5563; background-color: #F9FAFB; padding: 8px; border-radius: 6px; border-left: 3px solid #6366F1;'>{ev.get('text', '')[:300]}...</div>", unsafe_allow_html=True)

        # 显示最新生成的 AI 剖析报告入口 (直观查看解析结果)
        if st.session_state.get("last_imported_paper_ids"):
            st.markdown("---")
            st.markdown("#### 📘 最新生成的 AI 剖析报告")
            st.caption("以下为本次下载任务最新生成的 AI 全景解构报告。您可以直接展开查看，或将其设为当前阅读文献以在【📂 本地沉淀文献大仓】中进行对比。")
            
            conn = get_db_connection()
            imported_papers = []
            for pid in st.session_state["last_imported_paper_ids"]:
                p = conn.execute("""
                    SELECT p.*, s.dialectical_analysis, s.model_name, s.updated_at 
                    FROM papers p
                    LEFT JOIN ai_summaries s ON p.paper_id = s.paper_id
                    WHERE p.paper_id = ?
                """, (pid,)).fetchone()
                if p:
                    imported_papers.append(p)
            conn.close()
            
            for ip in imported_papers:
                with st.container(border=True):
                    col_title, col_action = st.columns([2.5, 1.2])
                    with col_title:
                        st.markdown(f"**📖 《{ip['title']}》**")
                        st.caption(f"📅 年份: `{ip['year'] or '未知'}` | 🏷️ 顶会/期刊: `{ip['venue'] or '未知'}` | 🧠 剖析大脑: `{ip['model_name'] or '未知'}`")
                    with col_action:
                        if st.button("📂 设为当前阅读文献", key=f"set_active_imported_{ip['paper_id']}", use_container_width=True):
                            st.session_state["active_view_paper_id"] = ip["paper_id"]
                            if "search_keyword" in st.session_state:
                                st.session_state["search_keyword"] = ""
                            st.toast(f"👉 已成功将 《{ip['title'][:15]}...》 设为当前阅读文献！请切换到上方【📂 本地沉淀文献大仓】选项卡阅读。")
                            
                    with st.expander("👁️ 直接在此查看 AI 全景剖析报告正文"):
                        if ip["dialectical_analysis"]:
                            st.markdown(ip["dialectical_analysis"])
                        else:
                            st.warning("⏳ 报告正文正在生成或写入中，请稍后刷新。")

    # 历史归档列表
    st.markdown("---")
    st.markdown("### 🗄️ 历史学术检索归档大仓")
    
    archives = get_search_archives()
    if not archives:
        st.info("💡 暂无历史检索归档记录。")
    else:
        for arc in archives:
            with st.container(border=True):
                col_arc_info, col_arc_btn1, col_arc_btn2 = st.columns([3, 1, 1])
                with col_arc_info:
                    st.markdown(f"**🔍 技术主题**: `{arc['query']}`")
                    st.caption(f"📅 归档时间: `{arc['archived_at']}` &nbsp;|&nbsp; 🆔 归档编号: `{arc['archive_id']}`")
                with col_arc_btn1:
                    if st.button("📂 载入查看", key=f"load_arc_{arc['archive_id']}", width="stretch"):
                        import json
                        try:
                            parsed_res = json.loads(arc["results_json"])
                            if isinstance(parsed_res, dict) and "candidates" in parsed_res:
                                st.session_state["local_search_results"] = parsed_res
                                st.session_state["local_search_query_used"] = arc["query"]
                                if "model_search_results_v2" in st.session_state:
                                    del st.session_state["model_search_results_v2"]
                            elif isinstance(parsed_res, dict) and "answer" in parsed_res:
                                st.session_state["model_search_results_v2"] = parsed_res
                                st.session_state["model_search_query_used"] = arc["query"]
                            else:
                                st.session_state["model_search_results_v2"] = parsed_res
                                st.session_state["model_search_query_used"] = arc["query"]
                            
                            if "last_imported_paper_ids" in st.session_state:
                                del st.session_state["last_imported_paper_ids"]
                            st.toast("🟢 成功载入历史检索归档数据！")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 载入归档失败: {e}")
                with col_arc_btn2:
                    if st.button("🗑️ 移除归档", key=f"del_arc_{arc['archive_id']}", width="stretch"):
                        delete_search_archive(arc["archive_id"])
                        st.toast("🗑️ 归档已成功删除")
                        st.rerun()

with tab_local_search:
    st.subheader("🔎 本地文献大仓双路检索")
    st.markdown("该模块专用于**纯本地大仓资产检索与事实问答**。系统将仅在本地数据湖（SQLite + LanceDB）中执行 FTS5 和向量混合检索，经 Rerank 后由 AI 进行事实解说合成，**100% 物理隔绝互联网请求，0 商业打捞开销**。")
    
    if "local_search_results" not in st.session_state:
        st.session_state["local_search_results"] = None
    if "local_search_query_used" not in st.session_state:
        st.session_state["local_search_query_used"] = ""

    # 配置区
    with st.container(border=True):
        col_local_query, col_local_filter = st.columns([3, 2])
        with col_local_query:
            local_query = st.text_input(
                "输入本地检索与提问 Query", 
                value="", 
                placeholder="例如: vLLM KV Cache, CXL memory numa-aware...",
                key="local_search_query_input_text"
            )
        with col_local_filter:
            local_filters = st.multiselect(
                "筛选本地数据源",
                options=["arxiv_paper", "local_pdf", "ext_blog", "daily_brief", "weekly_insight"],
                default=["arxiv_paper", "local_pdf", "ext_blog", "daily_brief"],
                format_func=lambda x: {
                    "arxiv_paper": "📚 学术论文",
                    "local_pdf": "📂 本地大仓 PDF",
                    "ext_blog": "🌐 技术博客/网页",
                    "daily_brief": "📰 每日快讯",
                    "weekly_insight": "📊 每周洞察周报"
                }.get(x, x),
                key="local_search_filter_multiselect"
            )

        trigger_local_search = st.button("🔎 启动本地大仓混合检索", type="primary", width="stretch")

    if trigger_local_search:
        if not local_query.strip():
            st.warning("⚠️ 请输入有效的检索 Query。")
        elif not local_filters:
            st.warning("⚠️ 请至少选择一个本地数据源。")
        else:
            with st.spinner("正在启动本地大仓双路混合检索与重排..."):
                import asyncio
                from core.search_engine import api_search
                try:
                    res = asyncio.run(api_search(
                        query=local_query.strip(),
                        filter_type=local_filters,
                        routing_mode="retrieve_rerank",
                        model_id=selected_online_search_model_id
                    ))
                    
                    if res.get("status") == "success":
                        st.session_state["local_search_results"] = {
                            "candidates": res.get("candidates", []),
                            "query": local_query.strip()
                        }
                        st.session_state["local_search_query_used"] = local_query.strip()
                        st.toast("🟢 本地大仓检索与重排成功！")
                    else:
                        st.error(f"🔴 检索失败: {res.get('error', '未知错误')}")
                except Exception as e:
                    st.error(f"❌ 运行本地大仓检索发生异常: {e}")

    # 显示本地检索结果
    l_res = st.session_state.get("local_search_results")
    if l_res is not None:
        st.markdown("---")
        st.markdown(f"##### 📚 本地大仓检索候选结果 — `{l_res['query']}`")
        
        candidates = l_res.get("candidates", [])
        if not candidates:
            st.info("💡 未检索到任何匹配的本地大仓候选数据。")
        else:
            for idx, ev in enumerate(candidates):
                with st.container(border=True):
                    title_str = ev.get("title", "无标题")
                    url_str = format_evidence_url(ev.get("canonical_url", "#"))
                    doc_id = ev.get("doc_id", "unknown")
                    hybrid_score = ev.get("hybrid_score", 0.0)
                    rerank_score = ev.get("rerank_score", 0.0)
                    stype = ev.get("source_type", "unknown")
                    
                    stype_name = {
                        "arxiv_paper": "📚 学术论文",
                        "local_pdf": "📂 本地大仓 PDF",
                        "ext_blog": "🌐 技术博客/网页",
                        "daily_brief": "📰 每日快讯",
                        "weekly_insight": "📊 每周洞察周报"
                    }.get(stype, stype)
                    
                    badge_html = f"<span style='background-color:#E0F2FE; color:#0369A1; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:bold; margin-right:8px;'>{stype_name}</span>"
                    
                    st.markdown(f"**[{idx+1}]** {badge_html} **[{doc_id}]** [{title_str}]({url_str})", unsafe_allow_html=True)
                    st.caption(f"📍 章节/位置: `{ev.get('section_path', 'N/A')}` &nbsp;|&nbsp; 📄 页码: `p.{ev.get('page_number', '1')}` &nbsp;|&nbsp; 📈 混合检索得分: `{hybrid_score:.4f}` &nbsp;|&nbsp; 🔄 重排得分: `{rerank_score:.4f}`")
                    st.markdown(f"<div style='font-size: 0.9rem; color: #374151; background-color: #F9FAFB; padding: 12px; border-radius: 8px; border-left: 4px solid #10B981; line-height: 1.6;'>{ev.get('text', '')}</div>", unsafe_allow_html=True)
                    
            # 归档按钮
            st.markdown("---")
            if st.button("🗄️ 归档本次本地检索历史", width="stretch"):
                import json
                import uuid
                archive_id = f"arc_{uuid.uuid4().hex[:8]}"
                results_json = json.dumps(l_res, ensure_ascii=False)
                insert_search_archive(archive_id, l_res["query"], results_json)
                st.success("🎉 本次本地检索历史已成功归档到本地数据库！")
                st.rerun()

with tab_scheduler:
    st.subheader("⏰ 智能定时扫描与解构调度")
    st.markdown("此模块支持对本地学术大仓与线上论文雷达进行定时自动化维护：在到达指定时间后，自动**执行本地物理同步与 AI 补全**，或者自动启动**线上雷达探测、双阶段摘要过滤、物理下载与并发深度解析**。")
    
    col_add, col_list = st.columns([1, 1.2])
    
    with col_add:
        st.markdown("### ➕ 新建定时扫描任务")
        
        # 1. 任务目标分流
        task_goal = st.selectbox(
            "选择任务目标 (Goal)",
            options=["local_scan", "online_search"],
            format_func=lambda x: "📂 物理大仓全盘扫描与分析" if x == "local_scan" else "🚀 线上雷达自动探测、下载与分析",
            key="scheduler_task_goal_selector"
        )
        
        # 2. 动态展现线上探测参数
        selected_topic_key = None
        scheduled_search_limit = 15
        if task_goal == "online_search":
            selected_topic_key = st.selectbox(
                "选择定时探测技术方向",
                options=list(TOPIC_REGISTRY.keys()),
                format_func=lambda x: TOPIC_REGISTRY[x]["name"],
                key="scheduler_topic_selector"
            )
            scheduled_search_limit = st.slider(
                "单次探测文献数量", 
                min_value=10, 
                max_value=30, 
                value=15,
                key="scheduler_search_limit_selector"
            )
            
        task_type = st.radio("任务周期", ["单次定时扫描", "每日重复扫描"])
        
        # 选择执行任务的大脑
        task_model = st.selectbox(
            "任务执行 AI 大脑",
            options=model_keys,
            format_func=lambda x: api_models[x].get("name", x),
            key="scheduler_task_model_selector"
        )
        
        if task_type == "单次定时扫描":
            st.caption("设定特定的未来时间执行一次扫描或探测分析：")
            
            from datetime import date
            import datetime
            d = st.date_input("设定任务日期", min_value=date.today())
            t = st.time_input("设定任务时刻", value=datetime.time(12, 0))
            
            scheduled_time = f"{d.strftime('%Y-%m-%d')} {t.strftime('%H:%M')}"
            
            st.info(f"💡 任务计划于 `{scheduled_time}` 执行")
            
            if st.button("➕ 创建单次预约扫描/探测"):
                add_scheduler_task(
                    task_type="one_shot", 
                    scheduled_time=scheduled_time, 
                    model_id=task_model,
                    task_goal=task_goal,
                    topic_key=selected_topic_key,
                    search_limit=scheduled_search_limit
                )
                st.success(f"🎉 成功预约单次任务！时间：{scheduled_time}")
                st.rerun()
                
        else: # 每日重复扫描
            st.caption("设定每天在特定时刻自动运行扫描或探测分析：")
            import datetime
            t = st.time_input("每日运行时间点", value=datetime.time(12, 0))
            scheduled_time = t.strftime("%H:%M")
            
            st.info(f"💡 任务计划每日在 `{scheduled_time}` 自动运行")
            
            if st.button("➕ 创建每日重复扫描/探测"):
                add_scheduler_task(
                    task_type="daily", 
                    scheduled_time=scheduled_time, 
                    model_id=task_model,
                    task_goal=task_goal,
                    topic_key=selected_topic_key,
                    search_limit=scheduled_search_limit
                )
                st.success(f"🎉 成功创建每日定时任务！每日时间：{scheduled_time}")
                st.rerun()
                
    with col_list:
        st.markdown("### 📋 运行中定时任务列表")
        
        active_tasks = get_active_tasks()
        if not active_tasks:
            st.info("💡 当前尚无任何待执行的定时扫描任务。")
        else:
            for task in active_tasks:
                task_id = task["task_id"]
                t_type = "单次定时" if task["task_type"] == "one_shot" else "每日重复"
                t_time = task["scheduled_time"]
                t_model_name = api_models.get(task["model_id"], {}).get("name", task["model_id"])
                
                t_goal = task.get("task_goal", "local_scan")
                if t_goal == "local_scan":
                    t_goal_desc = "📂 本地大仓全盘物理扫描与 AI 分析"
                else:
                    topic_name = TOPIC_REGISTRY.get(task.get("topic_key"), {}).get("name", task.get("topic_key"))
                    t_goal_desc = f"🚀 线上雷达自动探测与解构 (方向: `{topic_name}`, 限制: `{task.get('search_limit', 15)}` 篇)"
                
                # 展现每一个定时任务卡片
                with st.container(border=True):
                    st.markdown(f"**【{t_type}】** — ⏰ 运行时间：`{t_time}`")
                    st.caption(f"🎯 任务目标：**{t_goal_desc}**")
                    st.caption(f"🧠 执行大脑：`{t_model_name}` | 创建于：{task['created_at']}")
                    
                    if st.button("🗑️ 取消并删除", key=f"del_task_{task_id}"):
                        delete_scheduler_task(task_id)
                        st.success("🗑️ 任务已成功取消并移除！")
                        st.rerun()

with tab_briefings:
    st.subheader("🌐 AI 24小时雷达与技术洞察")
    st.markdown("该板块支持利用 Google Gemini 强联网搜索引擎，实时检索并辩证剖析全球 AI 与大语言模型领域的最新硬核技术进展。**此模块拥有独立的 API 模型及调度配置，与文献大仓完全隔离互不影响。**")
    
    from core.briefing_manager import load_briefing_config, save_briefing_config, test_briefing_api_connection, generate_daily_briefing_manually, generate_weekly_insight_manually, list_archived_reports, get_gemini_api_key
    
    br_config = load_briefing_config()
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    col_control, col_viewer = st.columns([1.0, 2.2])
    
    with col_control:
        st.markdown("### 🚀 自动探测与手动收割")
        
        # 立即抓取按钮
        if st.button("📰 立即抓取今日简报 (TOP 10)", width="stretch", help="立即检索过去24小时并生成硬核科技简报"):
            with st.spinner("🚀 正在强联网检索并进行第一性原理剖析中..."):
                success, result = generate_daily_briefing_manually()
                if success:
                    st.success("🎉 今日简报成功抓取并落盘归档！")
                    st.rerun()
                else:
                    st.error(f"❌ 抓取失败: {result}")
                    
        if st.button("🔍 立即抓取每周技术深入洞察", width="stretch", help="立即深入剖析过去一周底层技术亮点"):
            with st.spinner("🚀 正在强联网检索底物理突破并进行冷酷批判中..."):
                success, result = generate_weekly_insight_manually()
                if success:
                    st.success("🎉 每周技术洞察白皮书成功生成并落盘归档！")
                    st.rerun()
                else:
                    st.error(f"❌ 抓取失败: {result}")
                    
        st.markdown("---")
        
        # 将专属 AI 大脑与调度参数隐藏在折叠 Expander 中，避免占用空间
        with st.expander("⚙️ 专属 AI 大脑与定时调度配置 (通常仅设置一次)", expanded=False):
            st.markdown("**🧠 专属 AI 大脑配置**")
            raw_key = br_config.get("gemini_api_key", "")
            masked_key = st.text_input(
                "Gemini API Key (独立于大仓配置)",
                value=raw_key,
                type="password",
                placeholder="若留空则自动读取系统 GEMINI_API_KEY",
                help="输入专门用于联网简报分析的 Gemini API 密钥"
            )
            
            selected_model = st.selectbox(
                "强联网分析大脑",
                options=["gemini-2.5-flash", "gemini-2.5-pro"],
                index=["gemini-2.5-flash", "gemini-2.5-pro"].index(br_config.get("model_name", "gemini-2.5-flash")) if br_config.get("model_name", "gemini-2.5-flash") in ["gemini-2.5-flash", "gemini-2.5-pro"] else 0,
                help="使用 gemini-2.5-flash 或 gemini-2.5-pro 进行快速联网分析与技术简报。"
            )
            
            proxy_input = st.text_input(
                "API 代理服务器地址 (可选)",
                value=br_config.get("proxy", ""),
                placeholder="例如: http://127.0.0.1:7890",
                help="如果后台任务需要科学上网，请输入代理服务器地址，如 http://127.0.0.1:7890"
            )
            
            # 测试连通性按钮
            if st.button("⚡ 测试简报大脑连通性", key="test_briefing_api_btn", width="stretch"):
                resolved_key = masked_key.strip() if masked_key.strip() else get_env_var("GEMINI_API_KEY", "").strip()
                if not resolved_key:
                    st.error("🔴 连通性测试失败！未配置 API Key，且未检测到系统环境变量。")
                else:
                    with st.spinner("正在向 Google Gemini 发送强联网诊断数据..."):
                        success, message, latency = test_briefing_api_connection(resolved_key, selected_model, proxy_input.strip())
                        if success:
                            st.success(f"🟢 **测试通过！**\n\n- 响应延时: `{latency}s`\n- {message}")
                        else:
                            st.error(f"🔴 **连通性测试失败！**\n\n{message}")
                            
            st.markdown("---")
            st.markdown("**⏰ 自动定时扫描调度**")
            
            daily_time_str = st.text_input(
                "每日简报时间",
                value=br_config.get("daily_briefing_time", "09:00"),
                help="格式 HH:MM，如 09:00"
            )
            weekly_time_str = st.text_input(
                "每周洞察时间",
                value=br_config.get("weekly_insight_time", "10:00"),
                help="格式 HH:MM，如 10:00"
            )
            weekly_day = st.selectbox(
                "每周洞察运行日",
                options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(br_config.get("weekly_insight_day", "Monday")),
                help="选择每周执行深入技术洞察报告的星期几"
            )
            auto_scheduled = st.toggle(
                "启用自动定时守护",
                value=br_config.get("auto_scheduled", True),
                help="开启后，后台轮询线程会在设定的时刻自动执行强联网抓取"
            )
            
            # 保存独立配置按钮
            if st.button("💾 保存专属配置与调度设定", key="save_briefing_config_btn", width="stretch"):
                # 时间格式校验
                time_pattern = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
                if not time_pattern.match(daily_time_str.strip()) or not time_pattern.match(weekly_time_str.strip()):
                    st.error("❌ 格式错误！每日简报或每周洞察的时间格式必须为 HH:MM。")
                else:
                    updated_br_config = {
                        "gemini_api_key": masked_key.strip(),
                        "model_name": selected_model,
                        "daily_briefing_time": daily_time_str.strip(),
                        "weekly_insight_time": weekly_time_str.strip(),
                        "weekly_insight_day": weekly_day,
                        "auto_scheduled": auto_scheduled,
                        "proxy": proxy_input.strip()
                    }
                    if save_briefing_config(updated_br_config):
                        st.success("🎉 简报专属配置与调度设置保存成功！已即时热重载生效。")
                        st.rerun()
                    else:
                        st.error("❌ 保存配置失败，请检查写入权限。")
                        
    with col_viewer:
        st.markdown("### 📂 强联网 AI 报告历史归档与分类阅读")
        
        # 获取报告列表
        archived_reports = list_archived_reports()
        
        sub_tab_daily, sub_tab_weekly = st.tabs(["📰 每日 AI 进展简报 (TOP 10)", "🔍 每周 AI 技术深入洞察"])
        
        with sub_tab_daily:
            daily_reports = [r for r in archived_reports if r["category"] == "每日简报"]
            if not daily_reports:
                st.info("💡 暂无已归档的每日 AI 进展简报。请先在左侧点击“立即抓取今日简报 (TOP 10)”！")
            else:
                col_ym, col_wk = st.columns([1, 1])
                with col_ym:
                    d_yms = sorted(list(set(r["year_month"] for r in daily_reports)), reverse=True)
                    d_selected_ym = st.selectbox("归档年月", d_yms, key="daily_ym_selector")
                with col_wk:
                    d_ym_filtered = [r for r in daily_reports if r["year_month"] == d_selected_ym]
                    d_wks = sorted(list(set(r["week"] for r in d_ym_filtered)), reverse=True)
                    d_selected_wk = st.selectbox("归档周数", d_wks, key="daily_wk_selector")
                
                final_daily_reports = [r for r in d_ym_filtered if r["week"] == d_selected_wk]
                if not final_daily_reports:
                    st.info("💡 该周下暂无已归档的简报。")
                else:
                    d_titles = [r["title"] for r in final_daily_reports]
                    d_selected_title = st.selectbox("选择历史简报", d_titles, key="daily_report_title_selector")
                    
                    d_rep = [r for r in final_daily_reports if r["title"] == d_selected_title][0]
                    d_path = os.path.join(PROJECT_ROOT, d_rep["path"])
                    
                    if os.path.exists(d_path):
                        try:
                            with open(d_path, "r", encoding="utf-8") as f_d:
                                content = f_d.read()
                            with st.container(border=True):
                                st.markdown(f"## {d_selected_title}")
                                st.caption(f"📝 相对物理路径: `{d_rep['path']}` | 🕒 归档时间: `{datetime.datetime.fromtimestamp(d_rep['mtime']).strftime('%Y-%m-%d %H:%M:%S')}`")
                                st.markdown("---")
                                st.markdown(content)
                        except Exception as ex:
                            st.error(f"❌ 读取简报文件失败: {ex}")
                    else:
                        st.error("❌ 简报物理文件不存在。")
                        
        with sub_tab_weekly:
            weekly_reports = [r for r in archived_reports if r["category"] == "每周洞察报告"]
            if not weekly_reports:
                st.info("💡 暂无已归档的每周 AI 技术深入洞察。请先在左侧点击“立即抓取每周技术深入洞察”！")
            else:
                col_ym, col_wk = st.columns([1, 1])
                with col_ym:
                    w_yms = sorted(list(set(r["year_month"] for r in weekly_reports)), reverse=True)
                    w_selected_ym = st.selectbox("归档年月", w_yms, key="weekly_ym_selector")
                with col_wk:
                    w_ym_filtered = [r for r in weekly_reports if r["year_month"] == w_selected_ym]
                    w_wks = sorted(list(set(r["week"] for r in w_ym_filtered)), reverse=True)
                    w_selected_wk = st.selectbox("归档周数", w_wks, key="weekly_wk_selector")
                
                final_weekly_reports = [r for r in w_ym_filtered if r["week"] == w_selected_wk]
                if not final_weekly_reports:
                    st.info("💡 该周下暂无已归档的洞察报告。")
                else:
                    w_titles = [r["title"] for r in final_weekly_reports]
                    w_selected_title = st.selectbox("选择历史洞察报告", w_titles, key="weekly_report_title_selector")
                    
                    w_rep = [r for r in final_weekly_reports if r["title"] == w_selected_title][0]
                    w_path = os.path.join(PROJECT_ROOT, w_rep["path"])
                    
                    if os.path.exists(w_path):
                        try:
                            with open(w_path, "r", encoding="utf-8") as f_w:
                                content = f_w.read()
                            with st.container(border=True):
                                st.markdown(f"## {w_selected_title}")
                                st.caption(f"📝 相对物理路径: `{w_rep['path']}` | 🕒 归档时间: `{datetime.datetime.fromtimestamp(w_rep['mtime']).strftime('%Y-%m-%d %H:%M:%S')}`")
                                st.markdown("---")
                                st.markdown(content)
                        except Exception as ex:
                            st.error(f"❌ 读取每周洞察文件失败: {ex}")
                    else:
                        st.error("❌ 每周洞察物理文件不存在。")

with tab_global_config:
    st.subheader("⚙️ 全局系统配置中心")
    st.markdown("此板块允许您配置全局学术大脑的解析控制参数、管理/编辑底层 LLM 模型提供商，以及定义开机自启动选项。")
    
    col_settings, col_providers = st.columns([1, 1.2])
    
    with col_settings:
        st.markdown("### 📊 全局解析控制参数")
        
        # 读取当前的全局配置
        current_settings = get_global_settings()
        
        max_concurrent = st.number_input(
            "发送给LLM解析的最大并发数量",
            min_value=1,
            max_value=10,
            value=int(current_settings.get("max_concurrent_analysis", 2)),
            help="当批量补全文献剖析报告时，同时运行的最大后台并发线程数量。"
        )
        
        max_batch_papers = st.number_input(
            "单次发送给LLM解析的最大论文并发数量 (批次上限)",
            min_value=1,
            max_value=20,
            value=int(current_settings.get("max_papers_per_batch", 3)),
            help="限制单次自动或手动批量补全时，送入大模型处理的最大论文数量，避免额度超出。"
        )
        
        granularity_opts = ["summary", "detailed"]
        current_granularity = current_settings.get("analysis_granularity", "summary")
        granularity_idx = granularity_opts.index(current_granularity) if current_granularity in granularity_opts else 0
        
        selected_granularity = st.selectbox(
            "论文解析精细度 (System Prompt 模版)",
            options=granularity_opts,
            index=granularity_idx,
            format_func=lambda x: "概要 (Summary - 快速学术解构)" if x == "summary" else "完整 (Detailed - 异构计算硬件深度剖析)",
            help="概要一档提供精炼辩证摘要；完整一档提供针对微架构、总线带宽、Host 内核与异构算力的超级硬核解构白皮书。"
        )

        st.markdown("##### 🪙 商业 API 开销限制 (硬熔断配额)")
        
        daily_budget = st.number_input(
            "每日 API 消费限额 (美元)",
            min_value=0.1,
            max_value=100.0,
            value=float(current_settings.get("daily_budget", 2.0)),
            step=0.1,
            help="限制每日商业付费 API 调用总额，超出此额度将自动熔断付费接口。"
        )
        
        weekly_budget = st.number_input(
            "每周 API 消费限额 (美元)",
            min_value=0.5,
            max_value=500.0,
            value=float(current_settings.get("weekly_budget", 10.0)),
            step=0.5,
            help="限制每周商业付费 API 调用总额，超出此额度将自动熔断付费接口。"
        )
        
        st.markdown("---")
        st.markdown("### 📦 V2 向量数据库与混合索引库管理")
        from core.library_scanner import get_papers_pending_v2_ingestion
        from core.ingestion import ingest_pdf_to_v2_sync
        from core.database import resolve_pdf_path
        
        pending_papers = get_papers_pending_v2_ingestion()
        pending_count = len(pending_papers)
        
        if pending_count > 0:
            st.warning(f"⚠️ **检测到 {pending_count} 篇历史存量文献未构建 V2 向量与 FTS5 混合索引库**。建议一键同步以启用高阶 RAG 搜索。")
            with st.expander(f"📋 查看待同步的 {pending_count} 篇文献清单"):
                for idx_p, p in enumerate(pending_papers[:10]):
                    st.caption(f"{idx_p+1}. `{p['paper_id']}` — {p['title']}")
                if pending_count > 10:
                    st.caption(f"...等共 {pending_count} 篇")
                    
            col_b1, col_b2 = st.columns([1.2, 1])
            with col_b1:
                run_sync = st.button("⚡ 一键无缝构建 V2 检索大仓", type="primary", key="ui_run_v2_sync_btn", width="stretch")
            with col_b2:
                skip_embed = st.checkbox("⚡ 快速构建（跳过 Embedding）", value=False, key="ui_skip_embed_chk", help="开启此选项跳过向量提取，但依然可秒级构建 FTS5 全文索引")
                
            if run_sync:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                success_count = 0
                failed_count = 0
                
                for idx_p, paper in enumerate(pending_papers):
                    status_text.markdown(f"⏳ **正在构建 [{idx_p+1}/{pending_count}]**: `{paper['title'][:35]}...`")
                    p_id = paper["paper_id"]
                    p_title = paper["title"]
                    p_path = resolve_pdf_path(paper.get("pdf_path", ""))
                    p_authors = paper.get("authors", "手动导入 (Local Import)") or "手动导入 (Local Import)"
                    p_summary = paper.get("dialectical_analysis", "") or ""
                    
                    if p_path and os.path.exists(p_path):
                        try:
                            ok = ingest_pdf_to_v2_sync(
                                doc_id=p_id,
                                title=p_title,
                                pdf_path=p_path,
                                source_type="local_pdf",
                                authors=p_authors,
                                ai_summary=p_summary
                            )
                            if ok:
                                success_count += 1
                            else:
                                failed_count += 1
                        except Exception as e:
                            failed_count += 1
                    else:
                        failed_count += 1
                        
                    progress_bar.progress((idx_p + 1) / pending_count)
                    
                status_text.empty()
                st.success(f"🎉 V2 索引库同步构建完成！成功: {success_count} 篇 | 失败: {failed_count} 篇。")
                st.toast("🟢 V2 混合检索索引库全量构建完成！")
                st.rerun()
        else:
            st.success("🟢 **所有存量论文与 AI 剖析报告均已注入 V2 向量数据库及 FTS5 索引库**！物理大仓保持 100% 同步。")
            
        st.markdown("---")
        st.markdown("### 🤖 系统AI脑区分配与兼容性诊断")
        
        model_keys = list(api_models.keys())
        
        # 1. 论文摘要相关性分析大脑
        default_model_id = get_default_model()
        current_relevance_brain = current_settings.get("abstract_relevance_model_id", default_model_id)
        relevance_index = model_keys.index(current_relevance_brain) if current_relevance_brain in model_keys else 0

        selected_relevance_brain = st.selectbox(
            "🧪 1. 论文摘要相关性分析大脑",
            options=model_keys,
            index=relevance_index,
            format_func=lambda x: api_models[x].get("name", x),
            key="config_abstract_relevance_brain",
            help="用于双阶段漏斗第二阶段：读取候选论文 Title / Abstract，判断是否与当前技术方向强相关。"
        )

        relevance_cfg = api_models.get(selected_relevance_brain, {})
        relevance_provider = relevance_cfg.get("provider", "")
        if relevance_provider in ["gemini", "openai_compatible"]:
            st.markdown("<p style='color: green; font-size: 0.88rem; margin-top: -10px; margin-bottom: 15px;'>🟢 兼容 (摘要语义仲裁可用)</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: red; font-size: 0.88rem; margin-top: -10px; margin-bottom: 15px;'>🔴 不兼容 (未知 API 协议)</p>", unsafe_allow_html=True)

        # 2. 论文精细化分析大脑
        current_detailed_brain = current_settings.get("detailed_analysis_model_id", default_model_id)
        detailed_index = model_keys.index(current_detailed_brain) if current_detailed_brain in model_keys else (model_keys.index(default_model_id) if default_model_id in model_keys else 0)
        
        col_read_sel, col_test_btn = st.columns([2.5, 1.2])
        with col_read_sel:
            selected_active_reading_brain = st.selectbox(
                "🧠 2. 论文精细化分析大脑",
                options=model_keys,
                index=detailed_index,
                format_func=lambda x: api_models[x].get("name", x),
                key="config_active_reading_brain",
                help="用于本地 PDF 全文精读、生成辩证解构报告和写入本地沉淀文献大仓。"
            )
        with col_test_btn:
            st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
            test_triggered = st.button("⚡ 测试连通性", key="config_test_brain_btn", width="stretch")
            
        read_cfg = api_models.get(selected_active_reading_brain, {})
        read_provider = read_cfg.get("provider", "")
        if read_provider in ["gemini", "openai_compatible"]:
            st.markdown("<p style='color: green; font-size: 0.88rem; margin-top: -10px; margin-bottom: 15px;'>🟢 兼容 (提供标准 Chat Completion 服务)</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: red; font-size: 0.88rem; margin-top: -10px; margin-bottom: 15px;'>🔴 不兼容 (未知 API 协议)</p>", unsafe_allow_html=True)
            
        if test_triggered:
            with st.spinner("正在发送诊断数据以验证 API 端点连通性..."):
                from core.ai_analyst import test_api_connection
                success, message, latency = test_api_connection(selected_active_reading_brain)
                if success:
                    st.success(f"🟢 **测试通过！**\n\n- 响应延迟: `{latency}s`\n- {message}")
                else:
                    st.error(f"🔴 **连通性测试失败！**\n\n{message}")
                    
        # 3. AI 联网学术探测大脑
        current_search_brain = current_settings.get("search_model_id", "")
        search_index = model_keys.index(current_search_brain) if current_search_brain in model_keys else 0
        selected_search_brain = st.selectbox(
            "🔍 3. AI 联网学术探测大脑",
            options=model_keys,
            index=search_index,
            format_func=lambda x: api_models[x].get("name", x),
            key="config_search_brain",
            help="进行联网学术搜索时使用的模型。仅支持百炼兼容模式的 Responses 联网接口。"
        )
        
        from core.detection import model_supports_web_search
        search_cfg = api_models.get(selected_search_brain, {})
        if model_supports_web_search(search_cfg):
            st.markdown("<p style='color: green; font-size: 0.88rem; margin-top: -10px; margin-bottom: 15px;'>🟢 兼容 (支持 responses 终结点并开启联网搜索)</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: red; font-size: 0.88rem; margin-top: -10px; margin-bottom: 15px;'>🔴 不兼容 (不支持 responses 终结点，请配置百炼兼容模式 Responses API)</p>", unsafe_allow_html=True)
            
        # 4. 24小时雷达简报大脑
        br_config = load_briefing_config()
        current_briefing_brain = br_config.get("model_name", "")
        # briefing_config.json 中的 model_name 可能是接口模型 ID，我们需要匹配 api_models 中的 ID
        briefing_index = 0
        for idx, k in enumerate(model_keys):
            if api_models[k].get("model") == current_briefing_brain or k == current_briefing_brain:
                briefing_index = idx
                break
                
        selected_briefing_brain = st.selectbox(
            "🌐 4. 24小时雷达简报大脑",
            options=model_keys,
            index=briefing_index,
            format_func=lambda x: api_models[x].get("name", x),
            key="config_briefing_brain",
            help="24小时自动简报与洞察引擎所用的大脑。必须配置为原生 Gemini 模型以启用 Google Search Grounding 功能。"
        )
        
        briefing_cfg = api_models.get(selected_briefing_brain, {})
        briefing_provider = briefing_cfg.get("provider", "")
        if briefing_provider == "gemini":
            st.markdown("<p style='color: green; font-size: 0.88rem; margin-top: -10px; margin-bottom: 25px;'>🟢 兼容 (原生支持 Google Search Grounding 网关接入)</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: red; font-size: 0.88rem; margin-top: -10px; margin-bottom: 25px;'>🔴 不兼容 (简报引擎依赖 Google Search Grounding，当前非 Gemini 模型将不可用)</p>", unsafe_allow_html=True)
            
        # 统一保存按钮
        if st.button("💾 保存全局控制参数与大脑分配", key="save_global_all_btn", width="stretch"):
            updated_settings = {
                "max_concurrent_analysis": max_concurrent,
                "max_papers_per_batch": max_batch_papers,
                "analysis_granularity": selected_granularity,
                "abstract_relevance_model_id": selected_relevance_brain,
                "detailed_analysis_model_id": selected_active_reading_brain,
                "search_model_id": selected_search_brain,
                "daily_budget": daily_budget,
                "weekly_budget": weekly_budget
            }
            
            success_all = True
            
            # 保存全局设置
            if not update_global_settings(updated_settings):
                success_all = False
                
            # 保存默认阅读大脑
            if not set_default_model(selected_active_reading_brain):
                success_all = False
                
            # 保存简报大脑
            br_config["model_name"] = briefing_cfg.get("model", selected_briefing_brain)
            if briefing_provider == "gemini":
                gemini_key = briefing_cfg.get("api_key", "").strip()
                if not gemini_key:
                    env_var = briefing_cfg.get("api_key_env", "")
                    if env_var:
                        gemini_key = get_env_var(env_var, "").strip()
                if gemini_key:
                    br_config["gemini_api_key"] = gemini_key
            
            if not save_briefing_config(br_config):
                success_all = False
                
            if success_all:
                st.success("🎉 全局控制参数、大脑角色分配与 API 连通关系已成功保存！")
                st.rerun()
            else:
                st.error("❌ 部分参数保存失败，请检查配置文件写入权限。")
                
    with col_providers:
        st.markdown("### 🔌 大模型提供商及 API 管理")
        
        # 1. 列表渲染当前模型，并提供选择编辑或添加
        edit_modes = ["新建模型提供商"] + list(api_models.keys())
        selected_edit_model = st.selectbox(
            "选择要编辑或配置的模型",
            options=edit_modes,
            index=0
        )
        
        st.markdown("---")
        
        if selected_edit_model == "新建模型提供商":
            st.markdown("**➕ 注册新的 API 大脑**")
            new_id = st.text_input("模型唯一标识 ID (如: qwen-max)", placeholder="仅限小写字母 and 中划线")
            new_name = st.text_input("显示名称 (如: Qwen Max (通义千问))")
            new_provider = st.selectbox("API 驱动类型 (Provider)", ["openai_compatible", "gemini"])
            new_model_name = st.text_input("接口模型 ID (Model Name, 如: qwen-max)", placeholder="对应的 API 官方模型名")
            new_api_key = st.text_input("API Key (为空则自动读取环境变量)", type="password")
            new_env = st.text_input("API Key 对应的环境变量名 (如: QWEN_API_KEY)")
            new_url = st.text_input("API 终结点 Endpoint URL (OpenAI 兼容类型必填)", placeholder="如: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
            
            # 默认 JSON 参数模板
            default_json_template = """{
    "extra_body": {
        "enable_thinking": true
    }
}"""
            
            # 自定义请求参数输入框 (选填，JSON格式)
            new_custom_params_str = st.text_area(
                "自定义额外 API 参数 (JSON 格式，选填。每个选项建议占一行，字符串/键名须双引号，布尔值须为小写的 true/false)", 
                value=default_json_template,
                help="如果您的非标准 API 终结点需要额外的参数，可以在此指定。请用标准的 JSON 格式，每一对配置选项最好放在单独的一行，以便于阅读和管理。"
            )
            
            # 实时进行 JSON 格式的就地检测与状态反馈
            is_new_json_valid = True
            new_custom_params = {}
            if new_custom_params_str.strip():
                try:
                    new_custom_params = json.loads(new_custom_params_str.strip())
                    if not isinstance(new_custom_params, dict):
                        st.warning("⚠️ 自定义额外 API 参数必须是一个以 {} 包围的 JSON 对象。")
                        is_new_json_valid = False
                    else:
                        st.success("🟢 自定义 API 参数 JSON 格式校验通过")
                except Exception as je:
                    st.warning(f"⚠️ JSON 格式不正确 (键/字符串须双引号，布尔须小写，例如使用 true 而非 Python 的 True): {je}")
                    is_new_json_valid = False
            
            if st.button("➕ 确认注册并保存"):
                if not is_new_json_valid:
                    st.error("❌ 自定义额外 API 参数 JSON 格式校验失败，请修改后再保存。")
                elif not new_id or not new_name or not new_model_name:
                    st.error("❌ 请填齐模型唯一标识 ID、显示名称与接口模型 ID。")
                elif new_provider == "openai_compatible" and not new_url:
                    st.error("❌ OpenAI 兼容类型必填 API 终结点 Endpoint URL。")
                else:
                    if update_model_config(new_id.strip(), new_name.strip(), new_provider, new_model_name.strip(), new_api_key.strip(), new_url.strip(), new_env.strip(), custom_params=new_custom_params):
                        st.success(f"🎉 成功注册大模型提供商: `{new_name}`！")
                        st.rerun()
        else:
            # 编辑已有模型配置
            cfg = api_models[selected_edit_model]
            st.markdown(f"**📝 编辑模型：`{cfg.get('name')}`**")
            
            edit_name = st.text_input("显示名称", value=cfg.get("name", ""))
            edit_provider = st.selectbox("API 驱动类型 (Provider)", ["openai_compatible", "gemini"], index=["openai_compatible", "gemini"].index(cfg.get("provider", "openai_compatible")))
            edit_model_name = st.text_input("接口模型 ID (Model Name)", value=cfg.get("model", ""))
            
            # 显示密文框，只在用户手动修改时提交
            edit_api_key = st.text_input("API Key (如果已配置，留空则保持原配置)", type="password", placeholder="已加密保存")
            
            # 读取已保存的 api_key
            original_api_key = cfg.get("api_key", "")
            
            edit_env = st.text_input("API Key 对应的环境变量名", value=cfg.get("api_key_env", ""))
            edit_url = st.text_input("API 终结点 Endpoint URL", value=cfg.get("url", ""))
            
            # 获取已保存的 custom_params
            saved_custom_params = cfg.get("custom_params", {})
            
            # 默认 JSON 参数模板
            default_json_template = """{
    "extra_body": {
        "enable_thinking": true
    }
}"""
            saved_custom_params_str = json.dumps(saved_custom_params, ensure_ascii=False, indent=4) if saved_custom_params else default_json_template
            
            # 自定义请求参数输入框 (选填，JSON格式)
            edit_custom_params_str = st.text_area(
                "自定义额外 API 参数 (JSON 格式，选填。每个选项建议占一行，字符串/键名须双引号，布尔值须为小写的 true/false)", 
                value=saved_custom_params_str,
                help="如果您的非标准 API 终结点需要额外的参数，可以在此指定。请用标准的 JSON 格式，每一对配置选项最好放在单独的一行，以便于阅读和管理。"
            )
            
            # 实时进行 JSON 格式的就地检测与状态反馈
            is_edit_json_valid = True
            edit_custom_params = {}
            if edit_custom_params_str.strip():
                try:
                    edit_custom_params = json.loads(edit_custom_params_str.strip())
                    if not isinstance(edit_custom_params, dict):
                        st.warning("⚠️ 自定义额外 API 参数必须是一个以 {} 包围 of JSON 对象。")
                        is_edit_json_valid = False
                    else:
                        st.success("🟢 自定义 API 参数 JSON 格式校验通过")
                except Exception as je:
                    st.warning(f"⚠️ JSON 格式不正确 (键/字符串须双引号，布尔须小写，例如使用 true 而非 Python 的 True): {je}")
                    is_edit_json_valid = False
            
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("💾 保存模型修改"):
                    if not is_edit_json_valid:
                        st.error("❌ 自定义额外 API 参数 JSON 格式校验失败，请修改后再保存。")
                    else:
                        # 如果密文框为空，使用原始已保存的 Key
                        final_key = edit_api_key.strip() if edit_api_key.strip() else original_api_key
                        if update_model_config(selected_edit_model, edit_name.strip(), edit_provider, edit_model_name.strip(), final_key, edit_url.strip(), edit_env.strip(), custom_params=edit_custom_params):
                            st.success("🎉 模型配置修改已成功保存！")
                            st.rerun()
            with col_btn2:
                if st.button("🗑️ 彻底删除该模型配置"):
                    if delete_model_config(selected_edit_model):
                        st.success(f"🗑️ 模型 {selected_edit_model} 已从大仓中安全注销并移除！")
                        st.rerun()

    # --- 📊 API 计费审计与滑动配额对账明细 ---
    st.markdown("---")
    st.markdown("### 📊 API 计费审计与滑动配额对账明细")
    
    # 获取当前的费用统计
    try:
        conn = get_db_connection()
        # 1. 计算今日累计开销
        day_cost = conn.execute("SELECT SUM(cost_usd) FROM quota_ledger WHERE created_at >= datetime('now', 'start of day')").fetchone()[0] or 0.0
        # 2. 计算本周累计开销
        week_cost = conn.execute("SELECT SUM(cost_usd) FROM quota_ledger WHERE created_at >= datetime('now', '-7 days')").fetchone()[0] or 0.0
        
        # 3. 按提供商汇总
        provider_summary = conn.execute("""
            SELECT api_provider, 
                   SUM(cost_usd) as total_cost, 
                   SUM(tokens_in) as total_in, 
                   SUM(tokens_out) as total_out, 
                   SUM(credits_spent) as total_credits
            FROM quota_ledger 
            GROUP BY api_provider
        """).fetchall()
        
        # 4. 获取最近 15 条流水账单
        recent_ledgers = conn.execute("""
            SELECT api_provider, model_name, api_metric, tokens_in, tokens_out, credits_spent, cost_usd, created_at, request_payload_summary 
            FROM quota_ledger 
            ORDER BY created_at DESC 
            LIMIT 15
        """).fetchall()
        
        conn.close()
    except Exception as e:
        day_cost = 0.0
        week_cost = 0.0
        provider_summary = []
        recent_ledgers = []
        st.error(f"无法读取计费明细: {e}")
        
    # 展示今日/本周配额
    col_day_budget, col_week_budget = st.columns(2)
    
    with col_day_budget:
        day_limit = float(current_settings.get("daily_budget", 2.0))
        day_pct = min(100.0, (day_cost / day_limit * 100.0)) if day_limit > 0 else 0.0
        st.metric(
            "今日累计 API 消费", 
            f"${day_cost:.4f}", 
            delta=f"配额: ${day_limit:.2f}",
            delta_color="off"
        )
        st.progress(day_pct / 100.0, text=f"今日配额已使用 {day_pct:.1f}%")
        
    with col_week_budget:
        week_limit = float(current_settings.get("weekly_budget", 10.0))
        week_pct = min(100.0, (week_cost / week_limit * 100.0)) if week_limit > 0 else 0.0
        st.metric(
            "本周 (滑动7天) 累计 API 消费", 
            f"${week_cost:.4f}", 
            delta=f"配额: ${week_limit:.2f}",
            delta_color="off"
        )
        st.progress(week_pct / 100.0, text=f"本周配额已使用 {week_pct:.1f}%")
        
    st.markdown("##### 🔌 各 API 渠道开销汇总")
    if not provider_summary:
        st.info("💡 暂无任何计费记录。")
    else:
        summary_cols = st.columns(len(provider_summary))
        for idx, row in enumerate(provider_summary):
            with summary_cols[idx]:
                with st.container(border=True):
                    st.markdown(f"**{row['api_provider'].upper()}**")
                    st.markdown(f"**累计开销**: `${row['total_cost']:.4f}`")
                    if row['total_credits'] > 0:
                        st.caption(f"消耗积分: `{row['total_credits']}`")
                    else:
                        st.caption(f"Tokens: `输入 {row['total_in']} / 输出 {row['total_out']}`")
                        
    with st.expander("📝 查看最近 15 笔 API 对账明细", expanded=False):
        if not recent_ledgers:
            st.info("💡 无账单流水。")
        else:
            # 格式化表格显示
            import pandas as pd
            table_data = []
            for row in recent_ledgers:
                table_data.append({
                    "时间": row["created_at"],
                    "提供商": row["api_provider"],
                    "模型/服务": row["model_name"],
                    "指标": row["api_metric"],
                    "输入 Tokens": row["tokens_in"],
                    "输出 Tokens": row["tokens_out"],
                    "消耗积分": row["credits_spent"],
                    "花费 (USD)": f"${row['cost_usd']:.4f}",
                    "请求描述": row["request_payload_summary"]
                })
            df = pd.DataFrame(table_data)
            st.dataframe(df, width="stretch", hide_index=True)
