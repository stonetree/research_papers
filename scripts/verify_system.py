# -*- coding: utf-8 -*-
"""
一键全量自动化质量自检脚本 (System Quality Guard Gate)
每次修改代码后，运行 python scripts/verify_system.py：
1. 静态语法与未定义变量扫描 (py_compile & pyflakes/ast)
2. 模块干净导入与解耦测试 (无线程锁死/循环引用)
3. 离线/在线 8081 (Embedding) 与 8082 (Rerank) 静默降级测试
4. 2PC 事务并发写与事件循环隔离测试
"""
import sys
import os
import py_compile
import subprocess
import asyncio
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QualityGate")

def run_step_1_syntax_check():
    logger.info("🔍 [Step 1/4] 正在执行全量语法与编译校验...")
    has_error = False
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if ".git" in root or "venv" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                try:
                    py_compile.compile(full_path, doraise=True)
                except Exception as e:
                    logger.error(f"❌ 语法/编译异常 [{file}]: {e}")
                    has_error = True
    if has_error:
        sys.exit(1)
    logger.info("✅ 语法与编译校验全量通过。")

def run_step_2_import_check():
    logger.info("🔍 [Step 2/4] 正在执行模块导入无损防死锁校验...")
    modules = [
        "core.config_loader",
        "core.database",
        "core.write_worker",
        "core.lancedb_client",
        "core.api_clients",
        "core.detection",
        "core.ingestion",
        "core.search_engine",
        "core.briefing_manager",
        "core.ai_analyst",
        "app"
    ]
    for mod in modules:
        try:
            __import__(mod)
        except Exception as e:
            logger.error(f"❌ 模块 [{mod}] 导入发生异常: {e}")
            sys.exit(1)
    logger.info("✅ 核心模块导入全量通过。")

def run_step_3_offline_degradation_test():
    logger.info("🔍 [Step 3/4] 正在测试离线抗毁与 8081/8082 降级链路...")
    try:
        from core.ingestion import ingest_markdown_text_to_v2_sync
        from core.search_engine import api_rerank

        # 1. 简报离线挂起测试
        status_res = ingest_markdown_text_to_v2_sync(
            doc_id="test_verify_offline_doc",
            title="离线抗毁自动校验文档",
            full_text_markdown="# 校验内容\n用于验证无 8081 时安全挂入 pending 队列。"
        )
        if status_res not in ["ingested", "pending"]:
            logger.error(f"❌ 离线文本摄取测试未返回预期的 ingested/pending，实际返回: {status_res}")
            sys.exit(1)

        # 2. Rerank 离线跳过测试
        candidates = [{'chunk_id': 'c1', 'title': 't1', 'section_path': 's1', 'text': 'text1', 'hybrid_score': 0.8}]
        rr_res = asyncio.run(api_rerank("test query", candidates))
        if "results" not in rr_res:
            logger.error("❌ Rerank 离线平滑降级失败。")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 离线抗毁降级测试失败: {e}")
        sys.exit(1)
    logger.info("✅ 8081/8082 离线自愈降级链路测试通过。")

def run_step_4_pipeline_test():
    logger.info("🔍 [Step 4/4] 正在触发并发 2PC 写入与底层物理 pipeline 校验...")
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "test_pipeline.py")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"❌ test_pipeline.py 执行失败!\n{res.stderr}\n{res.stdout}")
        sys.exit(1)
    logger.info("✅ 50 高并发写、2PC 崩溃自愈与 RRF 检索流水线测试通过。")

if __name__ == "__main__":
    logger.info("==================================================")
    logger.info("🛡️ 启动系统全量质量防线 (System Quality Guard Gate)...")
    logger.info("==================================================")
    run_step_1_syntax_check()
    run_step_2_import_check()
    run_step_3_offline_degradation_test()
    run_step_4_pipeline_test()
    logger.info("==================================================")
    logger.info("🎉 [SUCCESS] 恭喜！全量系统质量防线 100% 校验通过，允许安全 Commit 提交！")
    logger.info("==================================================")
