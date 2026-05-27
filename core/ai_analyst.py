# -*- coding: utf-8 -*-
from google import genai
from google.genai import types
import os
import requests
from .database import save_ai_summary
from .config_loader import get_model_config, get_global_settings

def extract_text_from_pdf(pdf_path):
    """从 PDF 文件中提取文本（适用于非原生多模态大语言模型，如 DeepSeek）"""
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"提取 PDF 文本失败: {e}")
        return ""

def analyze_and_store_paper(paper_id, pdf_path, title, model_id="deepseek-v4"):
    from .database import resolve_pdf_path, get_db_connection
    pdf_path = resolve_pdf_path(pdf_path)
    if not os.path.exists(pdf_path):
        return "❌ 本地物理 PDF 文件丢失。"
        
    # 查询当前文献是否是手动导入的
    is_manual = False
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT source_engine FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        if row and row["source_engine"] == "manual":
            is_manual = True
    except Exception as e:
        print(f"查询 source_engine 异常: {e}")
    finally:
        conn.close()
        
    # 从配置文件中获取对应的模型配置
    cfg = get_model_config(model_id)
    if not cfg:
        return f"❌ 未在 API 配置文件中找到模型标识为 '{model_id}' 的配置。"
        
    provider = cfg.get("provider", "openai_compatible")
    model_name = cfg.get("model", model_id)
    api_key = cfg.get("resolved_api_key", "").strip()
    api_url = cfg.get("url", "").strip()
    display_name = cfg.get("name", model_id)
    
    # 读取全局配置以确定解析精细度及对应的 System Prompt
    global_settings = get_global_settings()
    granularity = global_settings.get("analysis_granularity", "summary")
    
    if granularity == "summary":
        system_instruction = (
            "你是一个极其严谨、具有挑剔眼光的 AI 首席科学家与顶尖系统架构师。\n"
            "用户向你提供了一篇学术论文的完整原装 PDF（包含所有原始文字、数学公式、系统拓扑图、消融实验折线图和数据表格）。\n"
            "请严格遵循第一性原理，将文字论述与图表进行交叉事实校验，快速完成一份辩证的技术概要。\n\n"
            "请严格包含以下模块并输出为标准的 Markdown 格式，避免空泛的修辞：\n\n"
            "1. 【本质技术痛点】：用极其精炼的一句话，点明该系统或算法从本质上攻克了什么物理世界、工程底层或存储层次结构的漏洞？\n"
            "2. 【核心架构与机理】：它提出了什么新颖的机制（如控制流、内存虚拟化、硬件互联协议等）？请结合文本中提到的关键模块名称简述其逻辑机理。\n"
            "3. 【冷酷批判（重点）】：请利用你的长上下文事实审视能力，一针见血地找出作者可能刻意隐藏或回避的致命短板是什么（例如实验基线过老、极度依赖特定Prompt、或者总线时延压倒了算法收益）？\n"
            "4. 【行业工程落地价值】：如果我们要将该设计引入实际的 AI Agent 或大模型推理生产系统，它有什么具体的参考价值？"
        )
    else:
        system_instruction = (
            "# 角色定义\n"
            "你是一个享誉国际的异构计算首席科学家、高级计算机体系结构专家与操作系统内核技术总监。你拥有深厚的软硬件协同设计（Hardware-Software Co-design）功底，对微架构（Microarchitecture）、高速互联总线（CXL/NVLink）、现代操作系统内核、以及大模型推理基础设施（如 vLLM, SGLang）的底层物理实现有极为深刻的第一性原理认知。\n\n"
            "# 任务目标\n"
            "用户向你交付了一篇关于 AI 基础设施与 Agentic AI 软硬件交叉领域的完整原装多模态 PDF 论文。请不要顺从作者的描述与自夸，必须开启最大强度的严苛科学批判和技术拆解，从以下指定的四个硬核高维特征空间进行深度全景剖析，并输出为一份极具洞察力的 Markdown 技术白皮书。\n\n"
            "---\n\n"
            "## 📋 深度解构规范（必须严格包含以下所有模块）\n\n"
            "### 1. 🔬 【微架构级物理开销与访存拓扑分析】\n"
            "请彻底剥离算法外壳，透视其底层的硬件物理本质：\n"
            "- **数据流向与边界**：结合论文中的系统架构图或控制流图，详细绘制出数据在计算与存储单元（GPU HBM <-> NVLink <-> Host CPU DDR5 <-> CXL.mem <-> NVMe SSD）之间的精确移动轨迹。\n"
            "- **总线与带宽饱和度**：深入分析该方法在处理超长上下文（如百万级 KV Cache）时，对 PCIe 总线（如 PCIe 5.0/6.0）、CXL 链路或 NVLink 带宽的物理压迫。它是否会引发严重的硬件总线阻塞？\n"
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
            "  - 是否利用了 **Intel AMX / AVX-512** 或 **ARM SVE/SVE2（如鲲鹏架构）** 的高性能矢量/矩阵指令集，在 Host 侧原地执行 $KV\\ Cache$ 的高性能量化与解量化（INT4/FP4/FP8）？\n"
            "  - 是否利用了特定的现代 ARM 特性来加速地址转译或内存屏障？\n"
            "- **算力抢占与生存空间（生态位）**：当 CPU 满载执行 $KV\\ Cache$ 压缩、内存置换或 Agent 的沙箱安全审计时，其对 Host 服务器其他常驻进程（如 OS 任务调度、网络 IO 驱动）的算力抢占效应如何？在真实的工业生产集群中，它处于什么生态位？\n\n"
            "### 4. ⚖️ 【科学批判：消融实验去伪存真与落地壁垒】\n"
            "请站在绝对中立、严苛批判的视角，挑剔地审视论文的硬伤：\n"
            "- **实验水分审计**：其对比的基线（Baselines）是否故意选择了过时的软件版本（如拿最新优化去对比未开启 PagedAttention 的早起 baseline）？其测试数据集是否属于“精挑细选的理想封闭场景（Cherry-picked）”？\n"
            "- **消融实验（Ablation Study）深度解密**：拆解消融实验图表，指出哪一个硬件参数或软件 Trick 才是该系统得以维系的“生命线”？一旦去除该特定的 Trick，其宣称的性能红利是否会发生断崖式暴跌（Cliff Effect）？\n"
            "- **边际效应与工程代价**：该方案为了提升 10% 的吞吐量，是否引入了过于冗余、复杂的软硬件堆栈与拓扑复杂度（Over-engineering）？\n"
            "- **真实硬件验证度**：该论文是在**真实的物理实体硬件拓扑（CXL 2.0/3.0 刀片服务器、物理 NVLink 节点）**上跑出来的硬核数据，还是仅仅基于**架构级仿真器（如 Gem5, NVMain, SimPoints）**跑出来的理想化数学数字？\n\n"
            "---\n"
            "# 约束条件\n"
            "- 你的所有结论必须完全尊崇科学事实和逻辑机理，绝对禁止复述作者带有夸张色彩 of 结论。\n"
            "- 如果论文中缺失某些关键实验或未披露核心微架构开销，必须在报告中明确指出该论文的【信息缺失与黑盒疑点】。"
        )

    if provider == "gemini":
        if not api_key:
            return f"❌ 运行环境中缺失 API Key (未在 api_config.json 设置且未在 {cfg.get('api_key_env', 'GEMINI_API_KEY')} 中找到)，Gemini 分析终止。"
            
        client = genai.Client(api_key=api_key)
        
        print(f"🤖 深度模型激活 [{display_name}]：正在剖析 {title}...")
        try:
            uploaded_file = client.files.upload(file=pdf_path)
            while uploaded_file.state.name == "PROCESSING":
                import time; time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)
                
            # 如果是手动添加的文献，先通过 Gemini 提炼出真实论文标题并更新数据库关联
            if is_manual:
                try:
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

            response = client.models.generate_content(
                model=model_name,
                contents=[uploaded_file, f"请全面解构此论文: {title}"],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1
                )
            )
            
            client.files.delete(name=uploaded_file.name)
            save_ai_summary(paper_id, f"{display_name} ({model_name})", response.text)
            return response.text
            
        except Exception as e:
            return f"❌ Gemini 联合解构失败: {e}"

    elif provider == "openai_compatible" or provider == "deepseek":
        if not api_key:
            return f"❌ 运行环境中缺失 API Key (未在 api_config.json 中设置，且未能在环境变量中读取)，分析终止。"
            
        if not api_url:
            return f"❌ OpenAI 兼容提供商需要配置有效的 'url' 终结点地址。"
            
        print(f"🤖 深度模型激活 [{display_name}]：正在提取并剖析 {title}...")
        try:
            paper_text = extract_text_from_pdf(pdf_path)
            if not paper_text:
                return "❌ PDF 文本提取失败，无法进行非多模态分析。"
                
            # 限制论文文本长度，防止超大 HTTP 负载导致 MTU 分片与 SSL 握手断开 (Bad Record MAC)
            # 60,000 字符 (~1.5万词) 已足够完美覆盖论文的核心引言、架构、算法和实验，跳过冗长的参考文献
            max_char_limit = 60000
            if len(paper_text) > max_char_limit:
                paper_text = paper_text[:max_char_limit] + "\n\n[...部分过长附录/参考文献文本已由系统安全截断以提升传输稳定性...]"

            # 如果是手动添加的文献，先通过 OpenAI/DeepSeek 接口提炼出真实论文标题并更新数据库关联
            if is_manual:
                try:
                    title_payload = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": "你是一个学术助手。请从给出的论文文本片段中提取出这篇论文的官方真实标题。只返回标题本身，不要有任何多余的解释、前缀、双引号或标点。"},
                            {"role": "user", "content": f"提取以下论文开头的标题：\n\n{paper_text[:3000]}"}
                        ],
                        "temperature": 0.0
                    }
                    t_response = requests.post(api_url, headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Connection": "close"
                    }, json=title_payload, timeout=30)
                    if t_response.status_code == 200:
                        t_json = t_response.json()
                        extracted_title = t_json["choices"][0]["message"]["content"].strip().replace('"', '').replace("'", "").replace("`", "")
                        if extracted_title and len(extracted_title) > 3:
                            conn = get_db_connection()
                            conn.execute("UPDATE papers SET title = ? WHERE paper_id = ?", (extracted_title, paper_id))
                            conn.commit()
                            conn.close()
                            print(f"✅ 成功提取并关联论文真实标题: {extracted_title}")
                except Exception as e:
                    print(f"⚠️ 提取论文真实标题失败: {e}")

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Connection": "close"
            }
            
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"以下是学术论文《{title}》的完整文本内容，请全面进行辩证客观解构：\n\n{paper_text}"}
                ],
                "temperature": 0.1
            }
            
            # 带指数退避的鲁棒性重试机制，抗 SSL 抖动
            import time
            max_retries = 3
            response = None
            last_err = None
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(api_url, headers=headers, json=payload, timeout=90)
                    if response.status_code == 200:
                        break
                    else:
                        print(f"⚠️ DeepSeek 请求尝试 {attempt+1} 失败 (HTTP {response.status_code})")
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                    last_err = e
                    print(f"⚠️ DeepSeek 请求尝试 {attempt+1} 触发网络/SSL抖动: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))  # 指数退避退缩
                except Exception as e:
                    last_err = e
                    print(f"⚠️ DeepSeek 请求尝试 {attempt+1} 触发未知异常: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)

            if response is not None and response.status_code == 200:
                result_json = response.json()
                analysis_text = result_json["choices"][0]["message"]["content"]
                save_ai_summary(paper_id, f"{display_name} ({model_name})", analysis_text)
                return analysis_text
            elif response is not None:
                return f"❌ API 请求失败 (HTTP {response.status_code}): {response.text}"
            else:
                return f"❌ 联合解构失败 (网络与SSL握手在多次尝试后均断开): {last_err}"
                
        except Exception as e:
            return f"❌ {display_name} 联合解构失败: {e}"
            
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
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Connection": "close"
            }
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "Hello, connection check! Please reply exactly with 'OK' in 1 word."}
                ],
                "max_tokens": 2048
            }
            response = requests.post(api_url, headers=headers, json=payload, timeout=15)
            latency = time.time() - start_time
            if response.status_code == 200:
                result_json = response.json()
                reply = result_json["choices"][0]["message"]["content"].strip()
                return True, f"连通成功！模型响应: '{reply}'", round(latency, 2)
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

    def clean_json_string(text):
        text = text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
        return text.strip()

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
            text = clean_json_string(response.text)
            return json.loads(text)
        except Exception as e:
            print(f"❌ Gemini 仲裁异常: {e}")
            return []
            
    elif provider == "openai_compatible" or provider == "deepseek":
        if not api_url:
            return []
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Connection": "close"
            }
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1
            }
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                text = clean_json_string(response.json()["choices"][0]["message"]["content"])
                return json.loads(text)
            else:
                print(f"❌ DeepSeek 仲裁失败 (HTTP {response.status_code}): {response.text}")
                return []
        except Exception as e:
            print(f"❌ DeepSeek 仲裁异常: {e}")
            return []
            
    return []