# -*- coding: utf-8 -*-
"""
控制平面配置：定义宏观技术方向到微观硬核检索原语的语义映射字典
"""

TOPIC_REGISTRY = {
    "1": {
        "name": "Host侧操作系统级框架与AIOS (OS Agent / Kernel Execution)",
        "mapping_query": '("AIOS" OR "LLM-based OS" OR "Operating System Agent" OR "system-level agent") AND ("architecture" OR "framework")'
    },
    "2": {
        "name": "Host侧安全隔离、沙箱与底层监控 (Sandbox / eBPF / Isolation)",
        "mapping_query": '("LLM Agent" OR "Agentic AI") AND ("sandbox" OR "secure execution" OR "eBPF" OR "isolation" OR "microvm")'
    },
    "3": {
        "name": "Host侧动作生成与环境对齐算法 (Computer Use / CLI / Action Space)",
        "mapping_query": '("computer use" OR "OS environment" OR "GUI-to-CLI") AND ("action generation" OR "bash command") AND "Agent"'
    },
    "4": {
        "name": "KV Cache 宿主侧跨层级卸载与调度 (KV Cache Offloading / Swapping)",
        "mapping_query": '("KV cache offloading" OR "KV cache swapping" OR "hierarchical KV cache") AND ("PCIe" OR "Host memory" OR "NVMe")'
    },
    "5": {
        "name": "Host CPU 侧硬件指令集加速与量化 (SIMD / AVX-512 / SVE / Quantization)",
        "mapping_query": '("KV cache quantization" OR "KV cache compression") AND ("CPU" OR "AVX" OR "SVE" OR "SIMD") AND "inference"'
    },
    "6": {
        "name": "CXL 尖端互联与硬件级一致性内存池 (CXL.mem / Memory Pooling)",
        "mapping_query": '("CXL" OR "CXL.mem" OR "Compute Express Link") AND ("KV cache" OR "KVcache") AND ("offloading" OR "memory pool")'
    },
    "7": {
        "name": "NVLink、统一内存与硬件级缺页异常优化 (NVIDIA UVM / Page Fault)",
        "mapping_query": '("Unified Virtual Memory" OR "UVM" OR "NVLink") AND ("KV cache" OR "PagedAttention") AND ("page fault" OR "zero-copy")'
    }
}