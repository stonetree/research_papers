# -*- coding: utf-8 -*-
from google import genai
from google.genai import types
import os
import requests
from .database import save_ai_summary
from .config_loader import get_model_config, get_global_settings

def _audit_api_call(provider, model_name, api_url="", request_label="llm_request"):
    """记录同步 LLM 调用次数，供侧边栏日/周/月 API 计数展示。"""
    try:
        from .database import get_db_connection
        provider_norm = (provider or "").lower()
        model_norm = (model_name or "").lower()
        url_norm = (api_url or "").lower()
        if provider_norm == "gemini":
            api_provider = "google"
        elif "dashscope" in url_norm or "qwen" in model_norm:
            api_provider = "dashscope"
        elif "deepseek" in url_norm or "deepseek" in model_norm or provider_norm == "deepseek":
            api_provider = "deepseek"
        else:
            api_provider = "deepseek"

        conn = get_db_connection()
        conn.execute(
            """
            INSERT INTO quota_ledger (
                api_provider, model_name, api_metric, pricing_rule_id,
                cost_usd, request_payload_summary
            ) VALUES (?, ?, ?, 0, 0.0, ?)
            """,
            (api_provider, model_name or "unknown-model", "request_count", request_label)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ API 调用次数审计写入失败: {e}")

def make_llm_request(api_url, api_key, model_name, messages, temperature=0.1, max_tokens=None, timeout=3600, custom_params=None):
    is_responses_api = "/responses" in api_url
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Connection": "close"
    }
    
    if is_responses_api:
        payload = {
            "model": model_name,
            "input": messages,
            "tools": [{"type": "web_search"}]
        }
    else:
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
    # 动态融合用户指定的自定义额外参数
    if custom_params and isinstance(custom_params, dict):
        for k, v in custom_params.items():
            if k in payload and isinstance(payload[k], dict) and isinstance(v, dict):
                # 递归融合子字典 (例如：融合 extra_body)
                payload[k] = payload[k].copy()
                payload[k].update(v)
            else:
                payload[k] = v
            
    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    return response

def parse_llm_response(response, is_responses_api):
    if response.status_code != 200:
        return None, f"HTTP {response.status_code}: {response.text}"
    
    try:
        res_json = response.json()
        if is_responses_api:
            content_text = ""
            for block in res_json.get("output", []):
                if block.get("type") == "message" and "content" in block:
                    for item in block.get("content", []):
                        if item.get("type") == "output_text" and "text" in item:
                            content_text += item["text"]
                        elif "text" in item:
                            content_text += item["text"]
            if content_text:
                return content_text, None
            # fallback
            for block in res_json.get("output", []):
                if "content" in block:
                    for item in block.get("content", []):
                        if "text" in item:
                            return item["text"], None
            return None, f"在响应数据中未找到 message 文本。原始响应: {response.text}"
        else:
            # chat completions structure: res_json["choices"][0]["message"]["content"]
            content = res_json["choices"][0]["message"]["content"]
            return content, None
    except Exception as e:
        return None, f"解析响应 JSON 失败: {e}. 原始响应: {response.text}"


def clean_json_string(text):
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    return text.strip()

def extract_text_from_pdf(pdf_path):
    """从 PDF 文件中提取文本（适用于非原生多模态大语言模型，如 DeepSeek）"""
    print(f"📄 [PDF 文本提取] 开始读取本地 PDF 物理文件: {pdf_path}")
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        total_pages = len(reader.pages)
        print(f"📄 [PDF 文本提取] 检测到 PDF 共 {total_pages} 页。正在逐页提取文本...")
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            text += page_text
            if (i + 1) % 5 == 0 or (i + 1) == total_pages:
                print(f"⏳ [PDF 文本提取] 已成功处理并提取 {i+1}/{total_pages} 页...")
        print(f"📄 [PDF 文本提取完成] 成功读取所有页面，共提取 {len(text)} 字符。")
        return text
    except Exception as e:
        print(f"❌ [PDF 文本提取失败] 读取/解析 PDF 发生异常: {e}")
        return ""

def analyze_and_store_paper(paper_id, pdf_path, title, model_id="deepseek-v4"):
    from .database import resolve_pdf_path, get_db_connection
    print(f"📂 [分析准备] 开始对论文《{title}》(ID: {paper_id}) 发起 AI 大脑分析...")
    # 从配置文件中获取对应的模型配置
    cfg = get_model_config(model_id)
    if not cfg:
        err_msg = f"❌ [分析准备失败] 未在 API 配置文件中找到模型标识为 '{model_id}' 的配置。"
        print(err_msg)
        return err_msg
        
    provider = cfg.get("provider", "openai_compatible")
    model_name = cfg.get("model", model_id)
    api_key = cfg.get("resolved_api_key", "").strip()
    api_url = cfg.get("url", "").strip()
    display_name = cfg.get("name", model_id)

    pdf_path = resolve_pdf_path(pdf_path)
    print(f"📁 [PDF 路径解析] 物理 PDF 路径已解析为: {pdf_path}")
    if not os.path.exists(pdf_path):
        err_msg = "❌ 本地物理 PDF 文件丢失。"
        save_ai_summary(paper_id, f"{display_name} ({model_name})", err_msg)
        return err_msg
        
    # 查询当前文献是否是手动导入的，同时读取 authors 备用
    is_manual = False
    paper_authors = "手动导入 (Local Import)"
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT source_engine, authors FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        if row:
            if row["source_engine"] == "manual":
                is_manual = True
            paper_authors = row["authors"] or "手动导入 (Local Import)"
    except Exception as e:
        print(f"查询 source_engine 异常: {e}")
    finally:
        conn.close()
    
    # 读取全局配置以确定解析精细度及对应的 System Prompt
    global_settings = get_global_settings()
    granularity = global_settings.get("analysis_granularity", "summary")
    
    if granularity == "summary":
        system_instruction = (
            "你是一个极其严谨、具有挑剔眼光的 AI 首席科学家与顶尖系统架构师。\n"
            "用户向你提供了一篇学术论文的完整原装 PDF（包含所有原始文字、数学公式、系统拓扑图、消融实验折线图和数据表格）。\n"
            "请严格遵循第一性原理，将文字论述与图表进行交叉事实校验，快速完成一份辩证的技术概要。\n\n"
            "请严格包含以下模块并输出为标准的 Markdown 格式，避免空泛的修辞：\n\n"
            "1. 【论文背景与核心问题】：\n"
            "   - 论文提出的背景是什么？\n"
            "   - 论文试图解决什么核心问题？\n"
            "2. 【技术手段与亮点归纳】：\n"
            "   - 论文采用了哪些具体的技术手段？\n"
            "   - 请简要归纳这些技术手段的核心亮点与创新点。\n"
            "3. 【实验效果与问题解决度】：\n"
            "   - 最后的实验或测试效果如何？是否完整、彻底地解决了提出的问题？\n"
            "4. 【本质技术痛点】：用极其精炼的一句话，点明该系统或算法从本质上攻克了什么物理世界、工程底层或存储层次结构的漏洞？\n"
            "5. 【核心架构与机理】：它提出了什么新颖的机制（如控制流、内存虚拟化、硬件互联协议等）？请结合文本中提到的关键模块名称简述其逻辑机理。\n"
            "6. 【冷酷批判（重点）】：请利用你的长上下文事实审视能力，一针见血地找出作者可能刻意隐藏或回避的致命短板是什么（例如实验基线过老、极度依赖特定Prompt、或者总线时延压倒了算法收益）？\n"
            "7. 【行业工程落地价值】：如果我们要将该设计引入实际的 AI Agent 或大模型推理生产系统，它有什么具体的参考价值？"
        )
    else:
        system_instruction = (
            "# 角色定义\n"
            "你是一个享誉国际的异构计算首席科学家、高级计算机体系结构专家与操作系统内核技术总监。你拥有深厚的软硬件协同设计（Hardware-Software Co-design）功底，对微架构（Microarchitecture）、高速互联总线（CXL/NVLink）、现代操作系统内核、以及大模型推理基础设施（如 vLLM, SGLang）的底层物理实现有极为深刻的第一性原理认知。\n\n"
            "# 任务目标\n"
            "用户向你交付了一篇关于 AI 基础设施与 Agentic AI 软硬件交叉领域的完整原装多模态 PDF 论文。请不要顺从作者的描述与自夸，必须开启最大强度的严苛科学批判和技术拆解，从以下指定的五个硬核高维特征空间进行深度全景剖析，并输出为一份极具洞察力的 Markdown 技术白皮书。\n\n"
            "---\n\n"
            "## 📋 深度解构规范（必须严格包含以下所有模块）\n\n"
            "### 0. 📖 【论文背景、技术路线与效果评估】\n"
            "分析论文的出发点与系统宏观结论：\n"
            "- **论文提出的背景**：详细阐述论文是在什么产业、技术或学术背景下提出的？当时的瓶颈是什么？\n"
            "- **解决的痛点问题**：论文具体锁定了什么痛点问题并加以解决？\n"
            "- **技术手段与架构路线**：论文采用了哪些关键技术手段来支撑其设计？\n"
            "- **技术手段亮点归纳**：简要归纳这些技术手段的核心亮点与创新贡献。\n"
            "- **最终效果与解决度评估**：最后的实验效果如何？其技术手段是否完整、彻底地解决了论文最初提出的问题？\n\n"
            "### 1. 🔬 【微架构级物理开销与访存拓扑分析】\n"
            "请彻底剥离算法外壳，透视其底层的硬件物理本质：\n"
            "- **数据流向与边界**：结合论文中的系统架构图或控制流图，详细绘制出数据在计算与存储单元（GPU HBM <-> NVLink <-> Host CPU DDR5 <-> CXL.mem <-> NVMe SSD）之间的精确移动轨迹。\n"
            "- **总线与带宽饱和度**：深入分析该方法在处理超长上下文（如百万级 KV Cache）时，对 PCIe 总线（如 PCIe 5.0/6.0）、CXL 链路或 NVLink 带宽 of 物理压迫。它是否会引发严重的硬件总线阻塞？\n"
            "- **硬件级时延隐藏**：审查其是否触发了硬件缺页异常（Page Fault）？它如何平衡计算边界（Compute-bound）与访存边界（Memory-bound）？是否通过异步双缓冲、重叠（Overlap）通信与计算或指令集预取（Prefetching）来隐蔽搬运时延。\n\n"
            "### 2. 💻 【HOST 侧软件生态、虚拟化与内核原语重构】\n"
            "分析该项研究在宿主操作系统（Host OS）侧的工程切入点：\n"
            "- **内核态与用户态原语**：该框架是纯粹运行在用户态的动态调度，还是深入到了内核态（如利用 eBPF、特定内核驱动、自定义系统调用 System Calls）？它如何管理不连续的物理内存块？\n"
            "- **内存页表管理（Paging）**：若涉及类似于 PagedAttention 的机制，分析其在 Host 侧物理内存中的常驻虚拟内存分配策略。它是否涉及固定内存（Pinned Memory/Page-locked Memory）？其分散-聚集 DMA（Scatter-Gather DMA）的开销是否具有统计学上的物理可行性？\n"
            "- **安全沙箱与隔离边界**：若涉及 Agent 对 Host systems 的操控（Computer Use），深入分析其在 Host 侧构建的环境防线。它是依赖于轻量级虚拟化（如 Firecracker microVM, gVisor）还是传统的 Container 隔离？对 Host 系统产生的额外虚拟化穿透延迟（Hypervisor Latency）是多少？\n\n"
            "### 3. 🧠 【CPU 在异构计算场景中的关键作用与生态位定性】\n"
            "从第一性原理出发，重新评估 Host CPU 在该架构中的角色演进：\n"
            "- **从“纯控制面”到“混合计算面”**：在这篇论文的设计中，Host CPU 仅仅充当传统慢速搬运的“指挥官（控制面）”，还是深度参与了数据计算（计算面）？\n"
            "- **现代 CPU 指令集硬件红利**：论文是否充分压榨了最新 Host CPU 架构的硬件基础设施潜力？例如：\n"
                      r"  - 是否利用了 **Intel AMX / AVX-512** 或 **ARM SVE/SVE2（如鲲鹏架构）** 的高性能矢量/矩阵指令集，在 Host 侧原地执行 $KV\ Cache$ 的高性能量化与解量化（INT4/FP4/FP8）？\n"
            "  - 是否利用了特定的现代 ARM 特性来加速地址转译或内存屏障？\n"
            r"- **算力抢占与生存空间（生态位）**：当 CPU 满载执行 $KV\ Cache$ 压缩、内存置换或 Agent 的沙箱安全审计时，其对 Host 服务器其他常驻进程（如 OS 任务调度、网络 IO 驱动）的算力抢占效应如何？在真实的工业生产集群中，它处于什么生态位？\n\n"
            "### 4. ⚖️ 【科学批判：消融实验去伪存真与落地壁垒】\n"
            "请站在绝对中立、严苛批判的视角，挑剔地审视论文的硬伤：\n"
            "- **实验水分审计**：其对比的基线（Baselines）是否故意选择了过时的软件版本（如拿最新优化去对比未开启 PagedAttention 的早起 baseline）？其测试数据集是否属于“精挑细选的理想封闭场景（Cherry-picked）”？\n"
            "- **消融实验（Ablation Study）深度解密**：拆解消融实验图表，指出哪一个硬件参数或软件 Trick 才是该系统得以维系的“生命线”？一旦去除该特定的 Trick，其宣称的性能红利是否会发生断崖式暴跌（Cliff Effect）？\n"
            "- **边际效应与工程代价**：该方案为了提升 10% 的吞吐量，是否引入了过于冗余、复杂的软硬件堆栈与拓扑复杂度（Over-engineering）？\n"
            "- **真实硬件验证度**：该论文是在**真实的物理实体硬件拓扑（CXL 2.0/3.0 刀片服务器、物理 NVLink 节点）**上跑出来的硬核数据，还是仅仅基于**架构级仿真器（如 Gem5, NVMain, SimPoints）**跑出来的理想化数学数字？\n\n"
            "-----------\n"
            "# 约束条件\n"
            "- 你的所有结论必须完全尊崇科学事实和逻辑机理，绝对禁止复述作者带有夸张色彩 of 结论。\n"
            "- 如果论文中缺失某些关键实验或未披露核心微架构开销，必须在报告中明确指出该论文的【信息缺失与黑盒疑点】。"
        )

    if provider == "gemini":
        if not api_key:
            err_msg = f"❌ [分析准备失败] 运行环境中缺失 API Key (未在 api_config.json 设置且未在 {cfg.get('api_key_env', 'GEMINI_API_KEY')} 中找到)，Gemini 分析终止。"
            print(err_msg)
            return err_msg
            
        client = genai.Client(api_key=api_key)
        
        print(f"🤖 深度模型激活 [{display_name}]：正在剖析 《{title}》...")
        try:
            print(f"📤 [Gemini 文件上传] 正在将物理 PDF 上传至 Gemini API (多模态原生输入)...")
            uploaded_file = client.files.upload(file=pdf_path)
            print(f"📡 [Gemini 文件上传] 物理 PDF 已成功上传至 Gemini 云端，文件 ID: {uploaded_file.name}")
            while uploaded_file.state.name == "PROCESSING":
                print(f"⏳ [Gemini 预处理] 正在云端对 PDF 进行安全及多模态预处理，等待 2 秒...")
                import time; time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)
                
            print(f"✅ [Gemini 预处理完成] 文件状态已就绪 (ID: {uploaded_file.name})。")
            
            # 如果是手动添加的文献，先通过 Gemini 提炼出真实论文标题并更新数据库关联
            if is_manual:
                try:
                    print(f"🔍 [Gemini 标题提取] 手动导入的文献，正在发送标题提取请求...")
                    title_response = client.models.generate_content(
                        model=model_name,
                        contents=[uploaded_file, "请直接给出这篇论文的官方英文或中文真实标题，不需要任何其他解释、前缀、双引号或标点。只返回标题本身即可。"],
                        config=types.GenerateContentConfig(temperature=0.0)
                    )
                    extracted_title = title_response.text.strip().replace('"', '').replace("'", "").replace("`", "")
                    if extracted_title and len(extracted_title) > 3 and not extracted_title.startswith("❌"):
                        conn = get_db_connection()
                        conn.execute("UPDATE papers SET title = ? WHERE paper_id = ?", (extracted_title, paper_id))
                        conn.commit()
                        conn.close()
                        print(f"✅ 成功提取并关联论文真实标题: {extracted_title}")
                except Exception as e:
                    print(f"⚠️ 提取论文真实标题失败: {e}")
  
            print(f"🚀 [Gemini 请求发送] 正在向 {display_name} ({model_name}) 发起多模态学术解构请求，等待大模型生成报告中...")
            response = client.models.generate_content(
                model=model_name,
                contents=[uploaded_file, f"请全面解构此论文: {title}"],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1
                )
            )
            print(f"📥 [Gemini 响应接收] 成功接收大模型分析结果 (长度 {len(response.text) if response.text else 0} 字符)。")
            _audit_api_call("gemini", model_name, request_label="paper_detailed_analysis")
            
            client.files.delete(name=uploaded_file.name)
            print(f"🧹 [Gemini 临时文件清理] 已删除云端临时文件: {uploaded_file.name}")
            analysis_result = response.text
            save_ai_summary(paper_id, f"{display_name} ({model_name})", analysis_result)
            print(f"✅ [Gemini 联合解构成功] 《{title}》解析完成，报告已成功落盘入库！")

            # 注入 V2 检索层（幂等安全，失败不影响主流程）
            try:
                from .ingestion import ingest_pdf_to_v2_sync
                # 读取最新 title（可能已由 is_manual 分支更新）
                conn2 = get_db_connection()
                _row2 = conn2.execute("SELECT title FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
                conn2.close()
                _latest_title = _row2["title"] if _row2 else title
                ingest_pdf_to_v2_sync(
                    doc_id=paper_id,
                    title=_latest_title,
                    pdf_path=pdf_path,
                    source_type="local_pdf",
                    authors=paper_authors,
                    ai_summary=analysis_result
                )
            except Exception as _v2_err:
                print(f"⚠️ V2 摄取注入失败（不影响主流程）: {_v2_err}")

            return analysis_result
            
        except Exception as e:
            err_msg = f"❌ Gemini 联合解构失败: {e}"
            print(err_msg)
            save_ai_summary(paper_id, f"{display_name} ({model_name})", err_msg)
            return err_msg
    elif provider == "openai_compatible" or provider == "deepseek":
        if not api_key:
            err_msg = f"❌ [分析准备失败] 运行环境中缺失 API Key (未在 api_config.json 中设置，且未能在环境变量中读取)，分析终止。"
            print(err_msg)
            return err_msg
            
        if not api_url:
            err_msg = f"❌ [分析准备失败] OpenAI 兼容提供商需要配置有效的 'url' 终结点地址。"
            print(err_msg)
            return err_msg
            
        print(f"🤖 深度模型激活 [{display_name}]：正在提取并剖析 《{title}》...")
        try:
            paper_text = extract_text_from_pdf(pdf_path)
            if not paper_text:
                err_msg = "❌ [PDF 文本提取失败] 提取文本为空，无法进行 non-multimodal 分析。"
                print(err_msg)
                save_ai_summary(paper_id, f"{display_name} ({model_name})", err_msg)
                return err_msg
                
            # 限制论文文本长度，防止超大 HTTP 负载导致 MTU 分片与 SSL 握手断开 (Bad Record MAC)
            max_char_limit = 60000
            if len(paper_text) > max_char_limit:
                paper_text = paper_text[:max_char_limit] + "\n\n[...部分过长附录/参考文献文本已由系统安全截断以提升传输稳定性...]"
                print(f"✂️ [PDF 文本裁切] 文本长度超过 {max_char_limit}，已自动裁切为 {len(paper_text)} 字符以保证 API 传输稳定性。")
 
            # 如果是手动添加的文献，先通过 OpenAI/DeepSeek 接口提炼出真实论文标题并更新数据库关联
            if is_manual:
                try:
                    title_messages = [
                        {"role": "system", "content": "你是一个学术助手。请从给出的论文文本片段中提取出这篇论文的官方真实标题。只返回标题本身，不要有任何多余的解释、前缀、双引号或标点。"},
                        {"role": "user", "content": f"提取以下论文开头的标题：\n\n{paper_text[:3000]}"}
                    ]
                    print(f"🔍 [{display_name} 标题提取] 手动导入的文献，正在请求提取真实官方标题...")
                    t_response = make_llm_request(api_url, api_key, model_name, title_messages, temperature=0.0, timeout=3600, custom_params=cfg.get("custom_params"))
                    content, err = parse_llm_response(t_response, "/responses" in api_url)
                    if not err and content:
                        extracted_title = content.strip().replace('"', '').replace("'", "").replace("`", "")
                        if extracted_title and len(extracted_title) > 3:
                            conn = get_db_connection()
                            conn.execute("UPDATE papers SET title = ? WHERE paper_id = ?", (extracted_title, paper_id))
                            conn.commit()
                            conn.close()
                            print(f"✅ 成功提取并关联论文真实标题: {extracted_title}")
                except Exception as e:
                    print(f"⚠️ 提取论文真实标题失败: {e}")
 
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"以下是学术论文《{title}》的完整文本内容，请全面进行辩证客观解构：\n\n{paper_text}"}
            ]
            
            # 带指数退避的鲁棒性重试机制，抗 SSL 抖动
            import time
            max_retries = 3
            response = None
            last_err = None
            is_responses_api = "/responses" in api_url
            
            print(f"🚀 [LLM 请求发送] 正在向 {display_name} ({model_name}) 发起学术解构请求...")
            for attempt in range(max_retries):
                try:
                    print(f"⏳ [LLM 请求尝试 {attempt+1}/{max_retries}] 正在向 API 发送学术分析请求，等待大模型响应 (通常需要 20-60 秒，请耐心等待)...")
                    response = make_llm_request(api_url, api_key, model_name, messages, temperature=0.1, timeout=3600, custom_params=cfg.get("custom_params"))
                    if response.status_code == 200:
                        print(f"📥 [LLM 响应接收] 成功收到大模型 HTTP 200 响应。")
                        _audit_api_call(provider, model_name, api_url, "paper_detailed_analysis")
                        break
                    else:
                        print(f"⚠️ LLM 请求尝试 {attempt+1} 失败 (HTTP {response.status_code}): {response.text}")
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                    last_err = e
                    print(f"⚠️ LLM 请求尝试 {attempt+1} 触发网络/SSL抖动: {e}")
                    if attempt < max_retries - 1:
                        print(f"⏳ 正在进行指数退避，等待 {2 * (attempt + 1)} 秒后重试...")
                        time.sleep(2 * (attempt + 1))  # 指数退避退缩
                except Exception as e:
                    last_err = e
                    print(f"⚠️ LLM 请求尝试 {attempt+1} 触发未知异常: {e}")
                    if attempt < max_retries - 1:
                        print(f"⏳ 正在等待 1 秒后重试...")
                        time.sleep(1)
 
            if response is not None and response.status_code == 200:
                print(f"📝 [LLM 响应解析] 正在解析大模型返回的学术报告...")
                content, err = parse_llm_response(response, is_responses_api)
                if not err and content:
                    save_ai_summary(paper_id, f"{display_name} ({model_name})", content)
                    print(f"✅ [{display_name} 联合解构成功] 《{title}》解析完成，报告已成功落盘入库！")

                    # 注入 V2 检索层（幂等安全，失败不影响主流程）
                    try:
                        from .ingestion import ingest_pdf_to_v2_sync
                        # 读取最新 title（可能已由 is_manual 分支更新）
                        conn2 = get_db_connection()
                        _row2 = conn2.execute("SELECT title FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
                        conn2.close()
                        _latest_title = _row2["title"] if _row2 else title
                        ingest_pdf_to_v2_sync(
                            doc_id=paper_id,
                            title=_latest_title,
                            pdf_path=pdf_path,
                            source_type="local_pdf",
                            authors=paper_authors,
                            ai_summary=content
                        )
                    except Exception as _v2_err:
                        print(f"⚠️ V2 摄取注入失败（不影响主流程）: {_v2_err}")

                    return content
                else:
                    err_msg = f"❌ [{display_name} 联合解构失败] 解析响应失败: {err}"
                    print(err_msg)
                    save_ai_summary(paper_id, f"{display_name} ({model_name})", err_msg)
                    return err_msg
            elif response is not None:
                err_msg = f"❌ [{display_name} 联合解构失败] API 请求最终失败 (HTTP {response.status_code}): {response.text}"
                print(err_msg)
                save_ai_summary(paper_id, f"{display_name} ({model_name})", err_msg)
                return err_msg
            else:
                err_msg = f"❌ [{display_name} 联合解构失败] 物理连接与 SSL 握手最终失败: {last_err}"
                print(err_msg)
                save_ai_summary(paper_id, f"{display_name} ({model_name})", err_msg)
                return err_msg
                
        except Exception as e:
            err_msg = f"❌ {display_name} 联合解构失败: {e}"
            print(err_msg)
            save_ai_summary(paper_id, f"{display_name} ({model_name})", err_msg)
            return err_msg
            
    else:
        return f"❌ 未知的 AI 分析大脑提供商: {provider}"


def test_api_connection(model_id):
    """测试指定模型配置的连通性，并返回 (success, message, latency_seconds)"""
    import time
    cfg = get_model_config(model_id)
    if not cfg:
        return False, f"未在 API 配置文件中找到模型标识为 '{model_id}' 的配置。", 0
        
    provider = cfg.get("provider", "openai_compatible")
    model_name = cfg.get("model", model_id)
    api_key = cfg.get("resolved_api_key", "").strip()
    api_url = cfg.get("url", "").strip()
    display_name = cfg.get("name", model_id)
    
    if not api_key:
        return False, f"未配置 API Key (请在 api_config.json 中设置，或配置对应的环境变量 {cfg.get('api_key_env', '')})。", 0
        
    start_time = time.time()
    
    if provider == "gemini":
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents="Hello, connection check! Please reply exactly with 'OK' in 1 word.",
            )
            latency = time.time() - start_time
            reply = response.text.strip() if response.text else "空响应"
            return True, f"连通成功！模型响应: '{reply}'", round(latency, 2)
        except Exception as e:
            return False, f"Gemini API 联通失败: {e}", 0
            
    elif provider == "openai_compatible" or provider == "deepseek":
        if not api_url:
            return False, "未配置 API URL (对于 OpenAI 兼容提供商必填)。", 0
            
        try:
            messages = [
                {"role": "user", "content": "Hello, connection check! Please reply exactly with 'OK' in 1 word."}
            ]
            is_responses_api = "/responses" in api_url
            response = make_llm_request(api_url, api_key, model_name, messages, max_tokens=2048, timeout=15, custom_params=cfg.get("custom_params"))
            latency = time.time() - start_time
            if response.status_code == 200:
                content, err = parse_llm_response(response, is_responses_api)
                if not err and content:
                    reply = content.strip()
                    return True, f"连通成功！模型响应: '{reply}'", round(latency, 2)
                else:
                    return False, f"解析响应错误: {err}", 0
            else:
                return False, f"API 响应错误 (HTTP {response.status_code}): {response.text}", 0
        except Exception as e:
            return False, f"网络请求失败: {e}", 0
            
    else:
        return False, f"未知的 AI 提供商类型: {provider}", 0

def arbitrate_papers(candidates, topic_name, model_id):
    """大模型闪电初审：从论文候选列表中筛选出最符合选定技术主题的黄金论文 ID 列表"""
    import json
    cfg = get_model_config(model_id)
    if not cfg:
        print(f"❌ 仲裁失败：未找到模型 {model_id} 配置。")
        return []
        
    provider = cfg.get("provider", "openai_compatible")
    model_name = cfg.get("model", model_id)
    api_key = cfg.get("resolved_api_key", "").strip()
    api_url = cfg.get("url", "").strip()
    
    if not api_key:
        print("❌ 仲裁失败：未配置 API Key。")
        return []

    # 格式化候选论文
    candidates_text = ""
    for i, c in enumerate(candidates):
        abstract_snippet = c.get('abstract', '暂无摘要')[:250] if c.get('abstract') else '暂无摘要'
        candidates_text += f"ID: {c['paper_id']}\n标题: {c['title']}\n摘要: {abstract_snippet}...\n---\n"
        
    system_instruction = (
        "你是一个极其敏锐、具有深厚软硬件系统底层底蕴的 AI 首席科学家。\n"
        "用户为你提供了一个当前关心的硬核技术主题，以及一组候选论文的标题和摘要列表。\n"
        "请从底层软硬件基础设施（操作系统、内核、编译器、互联芯片、硬件架构）的第一性原理出发，严格挑选出与当前技术主题真正强相关的论文，剔除泛泛而谈、蹭热度或不相关的论文。\n"
        "你的回复必须仅仅是一个有效的 JSON 数组，包含你筛选出的最相关的论文 ID 列表（限制在 5 篇以内）。\n"
        "不要包含任何解释性文字或 markdown 代码块，直接返回标准 JSON 字符串，例如：\n"
        "[\"id1\", \"id2\"]"
    )
    
    user_prompt = (
        f"技术主题：{topic_name}\n"
        f"候选论文列表：\n{candidates_text}\n"
        "请严格进行语义过滤，仅返回与主题高度相关的论文 ID 的 JSON 数组。"
    )


    if provider == "gemini":
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1
                )
            )
            _audit_api_call("gemini", model_name, request_label="paper_abstract_relevance")
            text = clean_json_string(response.text)
            return json.loads(text)
        except Exception as e:
            print(f"❌ Gemini 仲裁异常: {e}")
            return []
            
    elif provider == "openai_compatible" or provider == "deepseek":
        if not api_url:
            return []
        try:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ]
            is_responses_api = "/responses" in api_url
            response = make_llm_request(api_url, api_key, model_name, messages, temperature=0.1, timeout=3600, custom_params=cfg.get("custom_params"))
            if response.status_code == 200:
                _audit_api_call(provider, model_name, api_url, "paper_abstract_relevance")
                content, err = parse_llm_response(response, is_responses_api)
                if not err and content:
                    text = clean_json_string(content)
                    return json.loads(text)
                else:
                    print(f"❌ 解析仲裁响应失败: {err}")
                    return []
            else:
                print(f"❌ DeepSeek 仲裁失败 (HTTP {response.status_code}): {response.text}")
                return []
        except Exception as e:
            print(f"❌ DeepSeek 仲裁异常: {e}")
            return []
            
    return []

def model_web_search(query_string, model_id):
    """
    通过 AI 联网学术探测大脑查找最新学术论文，返回 (success, papers_list)。
    双路径架构支持：
    - 路径 A: API Endpoint 包含 /responses 或原生支持 Grounding 联网工具；
    - 路径 B: 标准通用 Chat 端口模型，框架自动调用 Exa 预抓取全网学术文献 highlights 切片并注入 Prompt 上下文。
    """
    cfg = get_model_config(model_id)
    if not cfg:
        return False, f"未在 API 配置文件中找到模型标识为 '{model_id}' 的配置。"
        
    provider = cfg.get("provider", "openai_compatible")
    model_name = cfg.get("model", model_id)
    api_key = cfg.get("resolved_api_key", "").strip()
    api_url = cfg.get("url", "").strip()
    
    if not api_key:
        return False, "未配置 API Key。"
        
    if provider != "gemini" and not api_url:
        return False, "未配置 API URL。"

    is_native_response = bool(api_url and "/responses" in api_url)
    
    exa_context_text = ""
    if not is_native_response:
        # === 路径 B 触发：通用 Chat 端口，启动 Exa 神经网络预先全网搜集学术文献 ===
        print(f"🌐 [路径 B 触发] 当前配置为标准 Chat 端口模型 ({model_name})，框架自动调用 Exa 神经网络进行前沿打捞...")
        try:
            from core.api_clients import ExaApiClient
            exa = ExaApiClient()
            include_domains = [
                "arxiv.org", "semanticscholar.org", "biorxiv.org", "medrxiv.org",
                "pubmed.ncbi.nlm.nih.gov", "researchgate.net", "ieeexplore.ieee.org",
                "dl.acm.org", "link.springer.com", "sciencedirect.com", "nature.com"
            ]
            import asyncio
            import concurrent.futures
            async def _do_exa():
                return await exa.search_and_extract_highlights(
                    query_string,
                    num_results=6,
                    include_domains=include_domains,
                    category="research paper"
                )
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    exa_res = pool.submit(asyncio.run, _do_exa()).result()
            else:
                exa_res = asyncio.run(_do_exa())

            results = exa_res.get("results", [])
            for idx, r in enumerate(results):
                exa_context_text += f"[全网打捞文献片段 {idx+1}]\n"
                exa_context_text += f"- 标题: {r.get('title', 'Unknown Title')}\n"
                exa_context_text += f"- direct URL: {r.get('url', '')}\n"
                exa_context_text += f"- 核心亮点: {r.get('highlights') or r.get('text', '')[:450]}\n\n"
            print(f"🟢 [Exa 打捞完成] 成功搜集到 {len(results)} 条实效文献数据切片，注入 Chat 大模型 Prompt。")
        except Exception as exa_e:
            print(f"⚠️ [Exa 预打捞警告] 启发式打捞发生非阻断异常 ({exa_e})，直接将 Query 发送给大模型。")

    system_instruction = (
        "你是一个极其严谨的 AI 首席科学家与顶尖科研检索专家。\n"
        "请结合提供的【全网实时打捞学术文献上下文】以及最前沿学术研讨，总结推荐 5-8 篇关于用户指定主题的最相关、最高质量的学术论文（来自于 OSDI, SOSP, ASPLOS, ISCA, VLDB, arXiv 等）。\n"
        "请必须返回 5-8 篇最相关的论文，且必须仅以一个标准的 JSON 数组格式输出，不要包含任何额外的解释性文字、前缀或后缀。\n"
        "JSON 数组中的每个对象必须包含以下字段：\n"
        "1. \"title\": 论文标题\n"
        "2. \"authors\": 作者团队\n"
        "3. \"year_venue\": 发表年份与会议/期刊名称 (例如 'ASPLOS 2025' 或 'arXiv 2026')\n"
        "4. \"summary\": 核心技术创新点与贡献简述\n"
        "5. \"url\": 论文的 PDF 下载链接或官方访问链接 (若上下文中有 direct URL 请优先原样引用)\n"
        "不要返回其他任何非 JSON 的文本！直接返回一个标准的 JSON 数组。"
    )
    
    user_prompt = f"请检索最新、高质量的学术文献，技术主题是：{query_string}\n\n"
    if exa_context_text:
        user_prompt += f"【全网实时打捞学术文献上下文】:\n{exa_context_text}"

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = make_llm_request(api_url, api_key, model_name, messages, temperature=0.1, timeout=3600, custom_params=cfg.get("custom_params"))
        if response.status_code == 200:
            content, err = parse_llm_response(response, is_native_response)
            if not err and content:
                import json
                cleaned = clean_json_string(content)
                try:
                    papers_list = json.loads(cleaned)
                    if isinstance(papers_list, list):
                        return True, papers_list
                    else:
                        return False, f"模型未返回一个列表。原始文本: {content}"
                except Exception as je:
                    return False, f"解析 JSON 列表失败: {je}. 原始文本: {content}"
            else:
                return False, f"解析模型响应失败: {err}"
        else:
            return False, f"API 请求失败 (HTTP {response.status_code}): {response.text}"
    except Exception as e:
        return False, f"模型检索过程中发生异常: {e}"
