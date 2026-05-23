# -*- coding: utf-8 -*-
import streamlit as st
import os
import re
from core.database import init_db, get_db_connection, resolve_pdf_path
from core.engine_semantic import execute_semantic_search
from core.engine_arxiv import execute_arxiv_search
from core.ai_analyst import analyze_and_store_paper, test_api_connection
from core.config_loader import load_api_config, get_default_model, set_default_model, get_global_settings, update_global_settings, update_model_config, delete_model_config
from core.library_scanner import sync_local_library, get_unanalyzed_papers
from core.scheduler import start_scheduler, add_scheduler_task, delete_scheduler_task, get_active_tasks
from core.funnel_search import execute_two_stage_funnel_search
from config.research_topics import TOPIC_REGISTRY

# 初始化本地数据库和 API 配置大仓，启动后台定时扫描服务
init_db()
api_models = load_api_config()
start_scheduler()

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

if "unanalyzed_papers" not in st.session_state:
    st.session_state["unanalyzed_papers"] = []

if "active_view_paper_id" not in st.session_state:
    st.session_state["active_view_paper_id"] = None

if "search_keyword" not in st.session_state:
    st.session_state["search_keyword"] = ""

st.set_page_config(page_title="🪐 Infrastructure AI Radar Hub", layout="wide")
st.title("🪐 AI 基础设施与软硬件协同 —— 个人智能论文知识库")

# 侧边栏：控制平面触发器
st.sidebar.header("📡 论文雷达探测控制台")
selected_topic_key = st.sidebar.selectbox(
    "1. 选择技术演进方向",
    options=list(TOPIC_REGISTRY.keys()),
    format_func=lambda x: TOPIC_REGISTRY[x]["name"]
)

search_limit = st.sidebar.slider(
    "2. 本次探测文献数量", 
    min_value=10, 
    max_value=30, 
    value=15,
    help="设定本次雷达扫描探测的文献最大数量上限。漏斗架构会从大吞吐拉取的初审文献中，为您精选出该数量的黄金论文执行高精度物理下载与解析。"
)

# 读取开机默认大脑设置并匹配选项索引
default_model_id = get_default_model()
model_keys = list(api_models.keys())
default_index = model_keys.index(default_model_id) if default_model_id in model_keys else 0

selected_brain_key = st.sidebar.selectbox(
    "3. 选择首席科学家 AI 大脑",
    options=model_keys,
    index=default_index,
    format_func=lambda x: api_models[x].get("name", x)
)

# API 连通性诊断测试仪
if st.sidebar.button(
    "⚡ 测试当前 AI 大脑连通性",
    help="向当前选定的大模型提供商接口发送一份诊断请求，以测试 API Key、Endpoint 终结点和网络连通性是否正常，并展示实时的响应延迟。"
):
    with st.sidebar.spinner("正在发送诊断数据以验证 API 端点连通性..."):
        success, message, latency = test_api_connection(selected_brain_key)
        if success:
            st.sidebar.success(f"🟢 **测试通过！**\n\n- 响应延迟: `{latency}s`\n- {message}")
        else:
            st.sidebar.error(f"🔴 **连通性测试失败！**\n\n{message}")


if st.sidebar.button(
    "🚀 触发雷达扫描（多源增量拉取）",
    help="启动双阶段智能漏斗，根据选定的演进方向拉取论文摘要，并调用 AI 首席科学家大脑进行摘要仲裁初审，精准抓取黄金文献自动落盘入库并生成解构报告。"
):
    topic = TOPIC_REGISTRY[selected_topic_key]
    new_items = []
    used_engine = "多源漏斗管道"
    
    with st.spinner("正在启动双阶段漏斗架构（宽进初审 + 大脑仲裁 + 精确收割）..."):
        try:
            new_items, used_engine = execute_two_stage_funnel_search(
                topic_name=topic["name"],
                query_string=topic["mapping_query"],
                target_limit=search_limit,
                model_id=selected_brain_key
            )
        except Exception as e:
            st.sidebar.error(f"❌ 漏斗检索发生异常故障: {e}")
            
    if new_items:
        st.sidebar.success(f"🎉 【{used_engine}】成功抓取并仲裁沉淀 {len(new_items)} 篇黄金文献！")
        # 自动联动大模型分析大脑
        has_error = False
        for item in new_items:
            brain_name = api_models[selected_brain_key].get("name", selected_brain_key)
            with st.spinner(f"🤖 正在激活 {brain_name} 全景解构: {item['title'][:30]}..."):
                res = analyze_and_store_paper(item["paper_id"], item["pdf_path"], item["title"], model_id=selected_brain_key)
                if res.startswith("❌"):
                    st.sidebar.error(res)
                    has_error = True
        if not has_error:
            st.rerun()
    else:
        st.sidebar.info("📭 探测完毕，大仓内当前方向在近期无更替。")

# 侧边栏：大仓同步与默认设置
st.sidebar.markdown("---")
st.sidebar.header("⚙️ 大仓同步与系统配置")

# 1. 设定开机默认大脑
new_default_brain = st.sidebar.selectbox(
    "系统开机默认大脑",
    options=model_keys,
    index=default_index,
    format_func=lambda x: api_models[x].get("name", x),
    key="sys_default_brain_selector"
)
if new_default_brain != default_model_id:
    if st.sidebar.button(
        "💾 保存默认大脑设置",
        help="将当前选中的大模型大脑设定为系统启动时的默认首选大脑，下次打开应用时无需再次手动选择。"
    ):
        if set_default_model(new_default_brain):
            st.sidebar.success(f"💾 默认大脑成功变更为 `{api_models[new_default_brain].get('name')}`，下次启动将首选此配置！")
        else:
            st.sidebar.error("❌ 默认大脑设置保存失败，请检查 api_config.json 写权限。")

# 2. 一键同步与诊断物理大仓
if st.sidebar.button(
    "🔄 一键同步物理大仓并诊断",
    help="全盘扫描 storage/library 物理目录中的 PDF 文件，自动识别新下载的文件入库，并诊断未生成 AI 全景报告的缺失论文，及时更新会话状态。"
):
    with st.sidebar.spinner("正在扫描 storage/library 并更新本地索引..."):
        # 扫描并同步本地手动下载的 PDF
        added = sync_local_library()
        # 诊断未剖析的文献
        unanalyzed = get_unanalyzed_papers()
        
        if added > 0:
            st.sidebar.success(f"🎉 物理大仓同步成功！新发现 {added} 篇本地 PDF 文件并自动入库登记。")
        else:
            st.sidebar.info("📂 物理同步完毕，未发现新增加的物理 PDF 文件。")
            
        if unanalyzed:
            st.sidebar.warning(f"⏳ 诊断：库中当前共有 {len(unanalyzed)} 篇文献尚未生成 AI 剖析报告。")
            st.session_state["unanalyzed_papers"] = unanalyzed
        else:
            st.sidebar.success("🟢 诊断：库内所有文献均拥有完美的 AI 辩证剖析报告！")
            st.session_state["unanalyzed_papers"] = []

# 3. 一键补全缺失的 AI 解构报告
if st.session_state.get("unanalyzed_papers"):
    # 从全局配置中读取限制
    global_settings = get_global_settings()
    max_workers = global_settings.get("max_concurrent_analysis", 2)
    max_batch = global_settings.get("max_papers_per_batch", 3)
    
    # 限制每批次补全的最大文件数量
    papers_to_process = st.session_state["unanalyzed_papers"]
    papers_to_process = papers_to_process[:max_batch]
    total_papers = len(papers_to_process)
    
    st.sidebar.markdown(f"**⚡ 未解构文献快捷补全 (本批次上限: {max_batch} 篇)：**")
    if st.sidebar.button(
        f"🤖 一键并发补全 {total_papers} 篇",
        help="利用多线程线程池，并发调用当前设置的首席科学家 AI 大脑，为物理大仓中所有未解析的 PDF 论文批量补全生成学术全景辩证解构报告。"
    ):
        progress_bar = st.sidebar.progress(0.0)
        has_any_error = False
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        status_text = st.sidebar.empty()
        error_container = st.sidebar.empty()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(analyze_and_store_paper, paper["paper_id"], paper["pdf_path"], paper["title"], model_id=selected_brain_key): paper
                for paper in papers_to_process
            }
            
            for idx, future in enumerate(as_completed(futures)):
                paper = futures[future]
                brain_name = api_models[selected_brain_key].get("name", selected_brain_key)
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
            st.sidebar.success("🎉 一键并发剖析成功！所有缺失报告已补齐！")
            st.session_state["unanalyzed_papers"] = []
            st.rerun()


# 主界面：三重选项卡分流
tab_library, tab_scheduler, tab_global_config = st.tabs(["📂 本地沉淀文献大仓", "⏰ 智能定时扫描与解构调度", "⚙️ 全局系统配置"])

with tab_library:
    # 视图路由：详情页 vs 列表页
    active_paper_id = st.session_state.get("active_view_paper_id")
    
    if active_paper_id:
        # --- 详情页视图 (高级全景解构详情页) ---
        conn = get_db_connection()
        paper = conn.execute("""
            SELECT p.*, s.dialectical_analysis, s.model_name 
            FROM papers p
            LEFT JOIN ai_summaries s ON p.paper_id = s.paper_id
            WHERE p.paper_id = ?
        """, (active_paper_id,)).fetchone()
        conn.close()
        
        if paper:
            # 顶部返回按钮与标题
            col_back, col_title = st.columns([1.5, 6])
            with col_back:
                if st.button("⬅️ 返回结果列表", key="back_btn_top", use_container_width=True):
                    st.session_state["active_view_paper_id"] = None
                    st.rerun()
            with col_title:
                st.markdown(f"### 📖 论文解构报告: 《{paper['title']}》")
                
            st.markdown("---")
            
            # 双栏精美布局
            col_meta, col_report = st.columns([1, 1.2])
            
            with col_meta:
                st.subheader("📋 论文原厂元数据")
                st.markdown(f"**🏷️ 顶会/期刊**: `{paper['venue'] or '顶会/未标注'}`")
                st.markdown(f"**📅 发表年份**: `{paper['year'] or '未知'}`")
                st.markdown(f"**📈 引用频次**: `{paper['citations'] or 0}` 次")
                st.markdown(f"**👥 作者团队**: {paper['authors'] or '未知'}")
                st.markdown(f"**📝 物理文件**: `{os.path.basename(paper['pdf_path']) if paper['pdf_path'] else '未关联'}`")
                
                # 本地 PDF 校验状态
                resolved_pdf = resolve_pdf_path(paper['pdf_path']) if paper['pdf_path'] else ""
                if resolved_pdf and os.path.exists(resolved_pdf):
                    st.success("💾 本地 PDF 文件已安全关联")
                else:
                    st.error("❌ 本地物理文件缺失")
                    
                st.markdown("**Abstract (摘要)**:")
                st.info(paper['abstract'] or "暂无摘要描述。")
                
            with col_report:
                st.subheader("💡 首席科学家 AI 辩证剖析报告")
                if paper['dialectical_analysis']:
                    st.caption(f"🧠 驱动智能大脑: `{paper['model_name'] or '未知'}`")
                    with st.container(border=True):
                        st.markdown(paper['dialectical_analysis'])
                else:
                    st.warning("⏳ 暂无该论文的 AI 深度解构。")
                    brain_name = api_models[selected_brain_key].get("name", selected_brain_key)
                    if st.button(f"🤖 立即激活 {brain_name} 技术解构", key=f"detail_activate_{paper['paper_id']}"):
                        with st.spinner(f"正在使用 {brain_name} 深度剖析论文中..."):
                            analysis_text = analyze_and_store_paper(paper['paper_id'], paper['pdf_path'], paper['title'], model_id=selected_brain_key)
                            if analysis_text.startswith("❌"):
                                st.error(analysis_text)
                            else:
                                st.rerun()
            
            st.markdown("---")
            if st.button("⬅️ 返回结果列表", key="back_btn_bottom"):
                st.session_state["active_view_paper_id"] = None
                st.rerun()
        else:
            st.error("❌ 未找到指定的论文，可能已被移除。")
            if st.button("⬅️ 返回列表"):
                st.session_state["active_view_paper_id"] = None
                st.rerun()
                
    else:
        # --- 列表页视图 ---
        st.subheader("📂 个人智能学术论文大仓")
        
        # 增加搜索框与状态同步
        search_keyword = st.text_input(
            "🔍 全文搜索大模型论文分析报告 (输入关键字直接筛选最相关的 TOP 10 黄金文献)", 
            value=st.session_state.get("search_keyword", ""),
            placeholder="例如: GPU, FlashAttention, 异构计算, 拓扑等..."
        )
        st.session_state["search_keyword"] = search_keyword.strip()
        
        conn = get_db_connection()
        if st.session_state["search_keyword"]:
            # 全文搜索逻辑：仅针对已完成 AI 论文分析的 PDF 论文进行关键词匹配
            query = """
                SELECT p.*, s.dialectical_analysis, s.model_name 
                FROM papers p
                INNER JOIN ai_summaries s ON p.paper_id = s.paper_id
                WHERE s.dialectical_analysis IS NOT NULL AND s.dialectical_analysis != ''
            """
            all_papers = conn.execute(query).fetchall()
            conn.close()
            
            # 计算关键字匹配频次
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
                    
            # 按匹配频次降书/降序排序
            matched_results.sort(key=lambda x: x["match_count"], reverse=True)
            top_results = matched_results[:10]
            
            if not top_results:
                st.warning(f"📭 未在大仓中检索到包含关键字 `{st.session_state['search_keyword']}` 的 AI 论文剖析报告。建议更换关键字，或先下载论文并激活 AI 技术解构！")
            else:
                st.success(f"🎯 成功检索到 `{len(top_results)}` 篇相关黄金文献（已为您按匹配相关度以降序排列，最多展示 TOP 10）：")
                
                # 渲染搜索结果列表
                for idx, result in enumerate(top_results):
                    paper = result["paper"]
                    count = result["match_count"]
                    
                    with st.container(border=True):
                        st.markdown(f"##### 📄 {idx + 1}. [{paper['venue'] or '顶会'}] {paper['title']} ({paper['year']})")
                        st.caption(f"👥 作者团队: {paper['authors'] or '未知'} | 📈 引用: {paper['citations']} | 🎯 匹配频次: **{count} 次关键字匹配**")
                        
                        # 提取高亮上下文片段
                        highlighted_snippet = extract_snippet_with_highlight(paper['dialectical_analysis'], st.session_state["search_keyword"])
                        st.markdown(f"**🔍 匹配上下文摘要**:\n{highlighted_snippet}", unsafe_allow_html=True)
                        
                        col_action, _ = st.columns([1.8, 5])
                        with col_action:
                            if st.button("📖 查看 AI 论文解构报告", key=f"view_report_btn_{paper['paper_id']}", use_container_width=True):
                                st.session_state["active_view_paper_id"] = paper["paper_id"]
                                st.rerun()
        else:
            # 默认的无搜索词时，降序展示所有已下载/入库的文献
            query = """
                SELECT p.*, s.dialectical_analysis, s.model_name 
                FROM papers p
                LEFT JOIN ai_summaries s ON p.paper_id = s.paper_id
                ORDER BY p.created_at DESC
            """
            papers_list = conn.execute(query).fetchall()
            conn.close()
            
            if not papers_list:
                st.info("💡 目前本地大仓空空如也，请在左侧选择方向并点击【触发雷达扫描】！")
            else:
                # 渲染常规的论文列表网格（原 chronological list）
                for paper in papers_list:
                    pdf_basename = os.path.basename(paper['pdf_path']) if paper['pdf_path'] else "未关联"
                    with st.expander(f"📄 [{paper['venue'] or '顶会'}] {paper['title']} ({paper['year']}) — 📈 引用: {paper['citations']} 📝 关联路径: {pdf_basename}"):
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("📋 论文原厂元数据")
                            st.caption(f"**作者团队**: {paper['authors']}")
                            st.markdown(f"**Abstract (摘要)**:\n{paper['abstract'] or '暂无描述'}")
                            
                            # 本地打开物理文件按钮
                            resolved_pdf = resolve_pdf_path(paper['pdf_path']) if paper['pdf_path'] else ""
                            if resolved_pdf and os.path.exists(resolved_pdf):
                                st.success("💾 本地 PDF 文件已安全关联")
                            else:
                                st.error("❌ 本地物理文件缺失")
                                
                        with col2:
                            st.subheader("💡 首席科学家 AI 辩证剖析报告")
                            if paper['dialectical_analysis']:
                                st.caption(f"驱动大脑: `{paper['model_name']}`")
                                st.markdown(paper['dialectical_analysis'])
                            else:
                                st.warning("⏳ 暂无该论文的 AI 深度解构。")
                                brain_name = api_models[selected_brain_key].get("name", selected_brain_key)
                                if st.button(f"🤖 立即激活 {brain_name} 技术解构", key=paper['paper_id']):
                                    with st.spinner(f"正在使用 {brain_name} 深度剖析论文中..."):
                                        analysis_text = analyze_and_store_paper(paper['paper_id'], paper['pdf_path'], paper['title'], model_id=selected_brain_key)
                                        if analysis_text.startswith("❌"):
                                            st.error(analysis_text)
                                        else:
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
        
        if st.button("💾 保存全局控制参数"):
            updated_settings = {
                "max_concurrent_analysis": max_concurrent,
                "max_papers_per_batch": max_batch_papers,
                "analysis_granularity": selected_granularity
            }
            if update_global_settings(updated_settings):
                st.success("🎉 全局解析控制参数已成功保存并即时生效！")
                st.rerun()
            else:
                st.error("❌ 全局配置保存失败，请检查 api_config.json 权限。")
                
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
            new_id = st.text_input("模型唯一标识 ID (如: qwen-max)", placeholder="仅限小写字母和中划线")
            new_name = st.text_input("显示名称 (如: Qwen Max (通义千问))")
            new_provider = st.selectbox("API 驱动类型 (Provider)", ["openai_compatible", "gemini"])
            new_model_name = st.text_input("接口模型 ID (Model Name, 如: qwen-max)", placeholder="对应的 API 官方模型名")
            new_api_key = st.text_input("API Key (为空则自动读取环境变量)", type="password")
            new_env = st.text_input("API Key 对应的环境变量名 (如: QWEN_API_KEY)")
            new_url = st.text_input("API 终结点 Endpoint URL (OpenAI 兼容类型必填)", placeholder="如: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
            
            if st.button("➕ 确认注册并保存"):
                if not new_id or not new_name or not new_model_name:
                    st.error("❌ 请填齐模型唯一标识 ID、显示名称与接口模型 ID。")
                elif new_provider == "openai_compatible" and not new_url:
                    st.error("❌ OpenAI 兼容类型必填 API 终结点 Endpoint URL。")
                else:
                    if update_model_config(new_id.strip(), new_name.strip(), new_provider, new_model_name.strip(), new_api_key.strip(), new_url.strip(), new_env.strip()):
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
            
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("💾 保存模型修改"):
                    # 如果密文框为空，使用原始已保存的 Key
                    final_key = edit_api_key.strip() if edit_api_key.strip() else original_api_key
                    if update_model_config(selected_edit_model, edit_name.strip(), edit_provider, edit_model_name.strip(), final_key, edit_url.strip(), edit_env.strip()):
                        st.success("🎉 模型配置修改已成功保存！")
                        st.rerun()
            with col_btn2:
                if st.button("🗑️ 彻底删除该模型配置"):
                    if delete_model_config(selected_edit_model):
                        st.success(f"🗑️ 模型 {selected_edit_model} 已从大仓中安全注销并移除！")
                        st.rerun()