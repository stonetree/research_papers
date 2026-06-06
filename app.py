# -*- coding: utf-8 -*-
import streamlit as st
import os
import re
import threading
from core.database import init_db, get_db_connection, resolve_pdf_path, insert_search_archive, get_search_archives, delete_search_archive
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

# 读取开机默认大脑设置并匹配选项索引
default_model_id = get_default_model()
model_keys = list(api_models.keys())
default_index = model_keys.index(default_model_id) if default_model_id in model_keys else 0

selected_brain_key = st.sidebar.selectbox(
    "首席科学家 AI 大脑",
    options=model_keys,
    index=default_index,
    format_func=lambda x: api_models[x].get("name", x),
    key="active_reading_brain_sidebar"
)

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


# 主界面：五重选项卡分流
tab_library, tab_model_search, tab_scheduler, tab_briefings, tab_global_config = st.tabs([
    "📂 本地沉淀文献大仓", 
    "🔍 AI 联网学术探测",
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
        scan_triggered = st.button("🚀 触发雷达", key="library_scan_btn", use_container_width=True)
    with col_sync_btn:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        sync_triggered = st.button("🔄 同步大仓", key="library_sync_btn", use_container_width=True, help="扫描并同步本地手动下载的 PDF 文件，更新大仓索引")
    with col_batch_btn:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        batch_label = f"🤖 并发补全 ({total_papers})" if total_papers > 0 else "🤖 无待补全"
        batch_disabled = (total_papers == 0)
        batch_completer = st.button(batch_label, key="library_batch_complete_btn", use_container_width=True, disabled=batch_disabled, help="并发解析已入库但尚未生成报告的文献")

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
                        model_id=selected_brain_key
                    )
                except Exception as e:
                    st.error(f"❌ 漏斗检索发生异常故障: {e}")
                    
            if new_items:
                st.success(f"🎉 【{used_engine}】成功抓取并仲裁沉淀 {len(new_items)} 篇黄金文献！")
                has_error = False
                for item in new_items:
                    brain_name = api_models[selected_brain_key].get("name", selected_brain_key)
                    with st.spinner(f"🤖 正在激活 {brain_name} 全景解构: {item['title'][:30]}..."):
                        res = analyze_and_store_paper(item["paper_id"], item["pdf_path"], item["title"], model_id=selected_brain_key)
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
                st.success("🎉 一键并发剖析成功！所有学术剖析报告已补齐！")
                st.session_state["unanalyzed_papers"] = []
                st.rerun()

    st.markdown("---")

    # 0. 全局数据装载与检索过滤 (位于最顶层以保持结构规整与数据一致)
    search_keyword = st.text_input(
        "🔍 全文搜索大模型分析报告", 
        value=st.session_state.get("search_keyword", ""),
        placeholder="输入关键词进行过滤，清空可浏览全部大仓...",
        key="library_search_input"
    )
    st.session_state["search_keyword"] = search_keyword.strip()
    
    conn = get_db_connection()
    if st.session_state["search_keyword"]:
        # 全文搜索逻辑
        query = """
            SELECT p.*, s.dialectical_analysis, s.model_name 
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
            SELECT p.*, s.dialectical_analysis, s.model_name 
            FROM papers p
            LEFT JOIN ai_summaries s ON p.paper_id = s.paper_id
            ORDER BY p.created_at DESC
        """
        papers_to_show = conn.execute(query).fetchall()
        conn.close()
        
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
            SELECT p.*, s.dialectical_analysis, s.model_name 
            FROM papers p
            LEFT JOIN ai_summaries s ON p.paper_id = s.paper_id
            WHERE p.paper_id = ?
        """, (active_paper_id,)).fetchone()
        conn.close()

    # ---------------- 1. 第一区域：贯穿页面的窄区（工具按钮栏） ----------------
    if paper:
        try:
            curr_idx = paper_ids.index(active_paper_id)
        except ValueError:
            curr_idx = -1
            
        with st.container(border=True):
            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 1, 2, 2.5])
            with btn_col1:
                prev_disabled = (curr_idx <= 0)
                if st.button("⏮️ 上一篇", key="prev_paper_btn", use_container_width=True, disabled=prev_disabled):
                    st.session_state["active_view_paper_id"] = paper_ids[curr_idx - 1]
                    st.rerun()
            with btn_col2:
                next_disabled = (curr_idx == len(paper_ids) - 1 or curr_idx == -1)
                if st.button("下一篇 ⏭️", key="next_paper_btn", use_container_width=True, disabled=next_disabled):
                    st.session_state["active_view_paper_id"] = paper_ids[curr_idx + 1]
                    st.rerun()
            with btn_col3:
                if paper['dialectical_analysis']:
                    st.download_button(
                        label="📥 一键导出 Markdown 报告",
                        data=paper['dialectical_analysis'],
                        file_name=f"{paper['title']}_AI学术解构报告.md",
                        mime="text/markdown",
                        key=f"export_detail_{paper['paper_id']}",
                        use_container_width=True
                    )
                else:
                    brain_name = api_models[selected_brain_key].get("name", selected_brain_key)
                    if st.button(f"🤖 激活 {brain_name} 解构", key=f"detail_activate_{paper['paper_id']}", use_container_width=True, type="primary"):
                        with st.spinner("正在解构剖析中..."):
                            analysis_text = analyze_and_store_paper(paper['paper_id'], paper['pdf_path'], paper['title'], model_id=selected_brain_key)
                            if analysis_text.startswith("❌"):
                                st.error(analysis_text)
                            else:
                                st.rerun()
            with btn_col4:
                st.markdown(f"<div style='padding-top: 6px; font-size: 0.95rem; color: #4B5563; font-weight: bold; text-align: right;'>📖 进度: {curr_idx + 1}/{len(paper_ids)} 篇 | 🧠 剖析大脑: {paper['model_name'] or '待激活'}</div>", unsafe_allow_html=True)
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
                    ai_status = "🟢" if p["dialectical_analysis"] else "⏳"
                    
                    card_title = f"{card_emoji} {p['title']}"
                    card_meta = f"{ai_status} [{p['venue'] or '顶会'}] {p['year']} | 📈 引用: {p['citations']}"
                    
                    # 渲染为小卡片样式
                    with st.container(border=True):
                        st.markdown(f"**{card_title}**")
                        st.caption(card_meta)
                        
                        # 搜索模式高亮
                        if st.session_state["search_keyword"] and p["dialectical_analysis"]:
                            snippet = extract_snippet_with_highlight(p['dialectical_analysis'], st.session_state["search_keyword"])
                            st.markdown(f"<div style='font-size: 0.85rem; color: #555; background: #f0f2f6; padding: 4px; border-radius: 4px; margin-bottom: 6px;'>🔍 {snippet}</div>", unsafe_allow_html=True)
                            
                        btn_label = "👉 正在阅读" if is_selected else "📖 极速阅读解构报告"
                        if st.button(btn_label, key=f"select_btn_{p_id}", use_container_width=True, type="primary" if is_selected else "secondary"):
                            st.session_state["active_view_paper_id"] = p_id
                            st.rerun()

    with col_right:
        st.markdown("##### 💡 首席科学家 AI 辩证剖析报告")
        if paper:
            # 独立滚动的右侧报告内容框 (高度设为 600px，与左侧框体完美绝对一致)
            with st.container(height=600):
                st.markdown(f"### 📘 《{paper['title']}》")
                st.markdown("---")
                
                # 作者团队与物理文件信息
                st.markdown(f"**👥 作者团队**: {paper['authors'] or '未知团队'}")
                st.markdown(f"**📝 物理文件**: `{os.path.basename(paper['pdf_path']) if paper['pdf_path'] else '未关联'}`")
                st.markdown("**Abstract (摘要)**:")
                st.info(paper['abstract'] or "暂无摘要描述。")
                
                st.markdown("---")
                
                st.markdown("##### 💡 首席科学家 AI 剖析报告正文")
                if paper['dialectical_analysis']:
                    st.markdown(paper['dialectical_analysis'])
                else:
                    st.warning("⏳ 暂无该论文的 AI 深度解构。")
        else:
            with st.container(height=600):
                st.markdown("<h3 style='text-align: center; color: #4B5563; padding-top: 150px;'>🪐 个人学术大仓阅读器</h3>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; color: #6B7280;'>请点选左侧大仓导航中的文献卡片以加载解构报告</p>", unsafe_allow_html=True)

with tab_model_search:
    st.subheader("🔍 AI 大脑联网学术探测")
    st.markdown("该模块直接将您的科研检索词发送给当前选定的 AI 大脑，触发大模型的内置联网搜索功能（Bailian Responses API），联网发掘最前沿、高质量 of 学术成果，并提供即时分析报告与链接。")
    
    # 自动检索可用的联网探测大脑
    search_capable_models = get_search_capable_models(api_models)
    
    if not search_capable_models:
        st.warning("⚠️ **联网搜索探测功能当前不可用**")
        st.info(
            "**原因**：未在您的模型配置中检测到任何支持内置联网搜索功能（Bailian Responses API）的模型。\n\n"
            "**💡 解决方案**：\n"
            "1. 请前往右侧的 **“⚙️ 全局系统配置”** 选项卡；\n"
            "2. 新增或编辑现有模型，确保其服务提供商为 `openai_compatible` 或 `deepseek`，且接口的 Endpoint URL 配置为百炼的 Responses 终结点（例如 `https://dashscope.aliyuncs.com/compatible-mode/v1/responses`）。"
        )
        
        # 禁用输入面板
        with st.container(border=True):
            st.text_input(
                "输入您关心的技术关键词/搜索 Query", 
                value="", 
                placeholder="探测功能已锁定，请先配置支持联网搜索的模型...",
                key="model_search_query_input_disabled",
                disabled=True,
                label_visibility="collapsed"
            )
            st.button("🚀 启动 AI 联网搜索", type="primary", use_container_width=True, disabled=True)
    else:
        # 读取配置的联网搜索大脑
        global_settings = get_global_settings()
        configured_search_model = global_settings.get("search_model_id", "")
        
        # 确定实际使用的模型ID
        if configured_search_model in search_capable_models:
            active_search_model_id = configured_search_model
        else:
            active_search_model_id = list(search_capable_models.keys())[0]
            
        active_search_model_name = search_capable_models[active_search_model_id].get("name", active_search_model_id)
        
        st.markdown(f"🎯 **当前联网搜索大脑**：`{active_search_model_name}`")
        
        # 初始化会话状态以存储搜索结果
        if "model_search_results" not in st.session_state:
            st.session_state["model_search_results"] = None
        if "model_search_query_used" not in st.session_state:
            st.session_state["model_search_query_used"] = ""

        # 联网搜索面板
        with st.container(border=True):
            col_query_in, col_query_btn = st.columns([4, 1])
            with col_query_in:
                model_query = st.text_input(
                    "输入您关心的技术关键词/搜索 Query", 
                    value="", 
                    placeholder="例如: CXL 3.0 cache coherence, vLLM KV cache optimization...",
                    key="model_search_query_input",
                    label_visibility="collapsed"
                )
            with col_query_btn:
                trigger_search = st.button("🚀 启动 AI 联网搜索", type="primary", use_container_width=True)

        if trigger_search:
            if not model_query.strip():
                st.warning("⚠️ 请输入有效的搜索 Query。")
            else:
                with st.spinner(f"正在通过 {active_search_model_name} 联网检索 {model_query.strip()}..."):
                    success, result = model_web_search(model_query.strip(), active_search_model_id)
                    if success:
                        st.session_state["model_search_results"] = result
                        st.session_state["model_search_query_used"] = model_query.strip()
                        st.toast("🟢 检索完成！")
                    else:
                        st.error(f"🔴 检索失败: {result}")

    # 显示检索结果
    if st.session_state.get("model_search_results") is not None:
        st.markdown("---")
        res_col_left, res_col_right = st.columns([1.8, 1])
        
        selected_indices = []
        with res_col_left:
            st.markdown(f"##### 📡 联网搜索结果 — `{st.session_state['model_search_query_used']}`")
            
            for idx, p in enumerate(st.session_state["model_search_results"]):
                with st.container(border=True):
                    sel = st.checkbox(
                        f"**{p.get('title', '无标题')}**", 
                        value=True, 
                        key=f"model_search_sel_{idx}"
                    )
                    if sel:
                        selected_indices.append(idx)
                    st.markdown(f"**👥 作者团队**: {p.get('authors', '未知')} &nbsp;|&nbsp; **📅 年份/会议**: {p.get('year_venue', '未知')}")
                    st.info(f"创新点简述: {p.get('summary', '无')}")
                    st.markdown(f"🔗 [可访问链接/PDF下载地址]({p.get('url', '#')})")
                    
        with res_col_right:
            st.markdown("##### ⚙️ 探测结果控制台")
            
            # 1. 归档历史功能
            with st.container(border=True):
                st.markdown("**🗄️ 搜索归档管理**")
                st.caption("将本次搜索返回的论文列表以当前时间戳记录归档，以便日后重新载入查看或批量下载。")
                if st.button("🗄️ 归档本次检索历史", use_container_width=True):
                    import json
                    import uuid
                    import datetime
                    archive_id = f"arc_{uuid.uuid4().hex[:8]}"
                    results_json = json.dumps(st.session_state["model_search_results"], ensure_ascii=False)
                    insert_search_archive(archive_id, st.session_state["model_search_query_used"], results_json)
                    st.success("🎉 本次检索历史成功归档到本地数据库！")
                    st.rerun()
                    
            # 2. 批量物理下载与 AI 解构功能
            with st.container(border=True):
                st.markdown("**📥 批量下载与 AI 全景剖析**")
                st.caption(f"系统将对左侧勾选的 **{len(selected_indices)}** 篇文献进行物理 PDF 抓取，完成自动登记，并调用 AI 首席科学家完成力作报告的生成。")
                
                if st.button("📥 开始下载并生成解构报告", type="primary", use_container_width=True):
                    if not selected_indices:
                        st.warning("⚠️ 请至少勾选一篇论文进行操作。")
                    else:
                        progress_bar = st.progress(0.0)
                        status_text = st.empty()
                        
                        success_count = 0
                        for i, idx in enumerate(selected_indices):
                            paper_info = st.session_state["model_search_results"][idx]
                            status_text.caption(f"正在处理 [{i+1}/{len(selected_indices)}]: {paper_info.get('title', '')[:20]}...")
                            
                            success, msg = download_and_import_paper(paper_info, selected_brain_key)
                            if success:
                                success_count += 1
                                st.toast(f"✅ {paper_info.get('title')[:15]} 导入成功")
                            else:
                                st.error(f"❌ 导入失败 《{paper_info.get('title')[:15]}》: {msg}")
                                
                            progress_bar.progress((i + 1) / len(selected_indices))
                            
                        status_text.empty()
                        if success_count > 0:
                            st.success(f"🎉 成功完成 {success_count} 篇黄金文献的下载、入库与大模型剖析！")
                            # 清除已有的未分析文献缓存
                            if "unanalyzed_papers" in st.session_state:
                                del st.session_state["unanalyzed_papers"]
                            st.rerun()

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
                    if st.button("📂 载入查看", key=f"load_arc_{arc['archive_id']}", use_container_width=True):
                        import json
                        st.session_state["model_search_results"] = json.loads(arc["results_json"])
                        st.session_state["model_search_query_used"] = arc["query"]
                        st.rerun()
                with col_arc_btn2:
                    if st.button("🗑️ 移除归档", key=f"del_arc_{arc['archive_id']}", use_container_width=True):
                        delete_search_archive(arc["archive_id"])
                        st.toast("🗑️ 归档已成功删除")
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
        if st.button("📰 立即抓取今日简报 (TOP 10)", use_container_width=True, help="立即检索过去24小时并生成硬核科技简报"):
            with st.spinner("🚀 正在强联网检索并进行第一性原理剖析中..."):
                success, result = generate_daily_briefing_manually()
                if success:
                    st.success("🎉 今日简报成功抓取并落盘归档！")
                    st.rerun()
                else:
                    st.error(f"❌ 抓取失败: {result}")
                    
        if st.button("🔍 立即抓取每周技术深入洞察", use_container_width=True, help="立即深入剖析过去一周底层技术亮点"):
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
                options=["gemini-2.5-flash", "gemini-3.5-flash"],
                index=["gemini-2.5-flash", "gemini-3.5-flash"].index(br_config.get("model_name", "gemini-2.5-flash")) if br_config.get("model_name", "gemini-2.5-flash") in ["gemini-2.5-flash", "gemini-3.5-flash"] else 0,
                help="使用 gemini-2.5-flash 或最新的 gemini-3.5-flash 进行快速联网分析与技术简报。"
            )
            
            # 测试连通性按钮
            if st.button("⚡ 测试简报大脑连通性", key="test_briefing_api_btn", use_container_width=True):
                resolved_key = masked_key.strip() if masked_key.strip() else os.environ.get("GEMINI_API_KEY", "").strip()
                if not resolved_key:
                    st.error("🔴 连通性测试失败！未配置 API Key，且未检测到系统环境变量。")
                else:
                    with st.spinner("正在向 Google Gemini 发送强联网诊断数据..."):
                        success, message, latency = test_briefing_api_connection(resolved_key, selected_model)
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
            if st.button("💾 保存专属配置与调度设定", key="save_briefing_config_btn", use_container_width=True):
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
                        "auto_scheduled": auto_scheduled
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
        
        st.markdown("---")
        st.markdown("### 🤖 系统AI脑区分配与兼容性诊断")
        
        model_keys = list(api_models.keys())
        
        # 1. 文献阅读与解构大脑
        default_model_id = get_default_model()
        default_index = model_keys.index(default_model_id) if default_model_id in model_keys else 0
        
        col_read_sel, col_test_btn = st.columns([2.5, 1.2])
        with col_read_sel:
            selected_active_reading_brain = st.selectbox(
                "🧠 1. 文献阅读与解构大脑",
                options=model_keys,
                index=default_index,
                format_func=lambda x: api_models[x].get("name", x),
                key="config_active_reading_brain",
                help="系统首选的文献深度阅读与辩证解构大模型大脑。"
            )
        with col_test_btn:
            st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
            test_triggered = st.button("⚡ 测试连通性", key="config_test_brain_btn", use_container_width=True)
            
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
                    
        # 2. AI 联网学术探测大脑
        current_search_brain = current_settings.get("search_model_id", "")
        search_index = model_keys.index(current_search_brain) if current_search_brain in model_keys else 0
        selected_search_brain = st.selectbox(
            "🔍 2. AI 联网学术探测大脑",
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
            
        # 3. 24小时雷达简报大脑
        br_config = load_briefing_config()
        current_briefing_brain = br_config.get("model_name", "")
        # briefing_config.json 中的 model_name 可能是接口模型 ID，我们需要匹配 api_models 中的 ID
        briefing_index = 0
        for idx, k in enumerate(model_keys):
            if api_models[k].get("model") == current_briefing_brain or k == current_briefing_brain:
                briefing_index = idx
                break
                
        selected_briefing_brain = st.selectbox(
            "🌐 3. 24小时雷达简报大脑",
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
        if st.button("💾 保存全局控制参数与大脑分配", key="save_global_all_btn", use_container_width=True):
            updated_settings = {
                "max_concurrent_analysis": max_concurrent,
                "max_papers_per_batch": max_batch_papers,
                "analysis_granularity": selected_granularity,
                "search_model_id": selected_search_brain
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
                        gemini_key = os.environ.get(env_var, "").strip()
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