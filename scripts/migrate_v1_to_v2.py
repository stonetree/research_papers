# -*- coding: utf-8 -*-
"""
scripts/migrate_v1_to_v2.py
===========================
存量数据迁移工具：将所有已在 V1（papers + ai_summaries）中沉淀的文献批量注入 V2
检索层（documents + chunks + search_chunks + LanceDB）。

迁移完成后，无需再次手动执行：新流程中，每次 AI 解构完成后会自动注入 V2。

使用方法：
    python scripts/migrate_v1_to_v2.py             # 正式执行迁移
    python scripts/migrate_v1_to_v2.py --dry-run    # 仅统计，不写入
    python scripts/migrate_v1_to_v2.py --batch-size 5  # 每次处理 5 篇
    python scripts/migrate_v1_to_v2.py --skip-embedding # 跳过向量化（FTS5 仍可用）
"""

import sys
import os
import argparse
import logging
import time

# 将项目根目录加入 sys.path，使 core 包可以被导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MigrateV1toV2")


def parse_args():
    parser = argparse.ArgumentParser(description="将 V1 存量数据批量迁移到 V2 检索层")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅统计待迁移数量，不执行写入")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="每批处理的文献数量（默认: 10）")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="跳过向量化（写入零向量），FTS5 文本检索仍可用；适合 Embedding 服务离线时使用")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="每篇之间的休眠时间（秒），防止 Embedding 服务过载（默认: 0.5s）")
    return parser.parse_args()


def get_pending_papers():
    """查询所有在 V1 中已有 AI 解构报告但尚未完成 V2 摄取的论文"""
    from core.library_scanner import get_papers_pending_v2_ingestion
    return get_papers_pending_v2_ingestion()


def resolve_pdf_path(raw_path: str) -> str:
    """解析相对路径或绝对路径，返回绝对路径"""
    from core.database import resolve_pdf_path as _resolve
    return _resolve(raw_path)


def migrate_single_paper(paper: dict, skip_embedding: bool = False) -> str:
    """
    迁移单篇论文到 V2。
    返回: 'success' | 'skipped' | 'failed:<reason>'
    """
    from core.ingestion import ingest_pdf_to_v2_sync

    paper_id = paper["paper_id"]
    title = paper["title"]
    raw_path = paper.get("pdf_path", "")
    authors = paper.get("authors", "手动导入 (Local Import)") or "手动导入 (Local Import)"
    ai_summary = paper.get("dialectical_analysis", "") or ""

    # 解析物理 PDF 路径
    pdf_path = resolve_pdf_path(raw_path) if raw_path else ""
    if not pdf_path or not os.path.exists(pdf_path):
        return f"failed:PDF 文件不存在 ({pdf_path})"

    try:
        success = ingest_pdf_to_v2_sync(
            doc_id=paper_id,
            title=title,
            pdf_path=pdf_path,
            source_type="local_pdf",
            authors=authors,
            ai_summary=ai_summary
        )
        return "success" if success else "failed:2PC 摄取返回 False"
    except Exception as e:
        return f"failed:{str(e)}"


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("📦 V1 → V2 存量数据迁移工具启动")
    logger.info(f"   dry-run:        {args.dry_run}")
    logger.info(f"   batch-size:     {args.batch_size}")
    logger.info(f"   skip-embedding: {args.skip_embedding}")
    logger.info(f"   sleep:          {args.sleep}s")
    logger.info("=" * 60)

    # 查询待迁移列表
    logger.info("🔍 正在查询待迁移的 V1 存量文献...")
    pending = get_pending_papers()
    total = len(pending)
    logger.info(f"📊 共发现 {total} 篇文献待迁移（已有 AI 解构报告但尚未完成 V2 摄取）。")

    if total == 0:
        logger.info("✅ 所有存量文献已完成 V2 摄取，无需迁移。")
        return

    if args.dry_run:
        logger.info("🔍 [DRY-RUN 模式] 以下文献将被迁移：")
        for i, p in enumerate(pending):
            logger.info(f"  [{i+1:3d}] {p['paper_id']} | v2_status={p.get('v2_status','?')} | {p['title'][:60]}")
        logger.info(f"\n📊 总计: {total} 篇（仅统计，未写入）")
        return

    # 正式批量迁移
    success_count = 0
    skip_count = 0
    fail_count = 0
    fail_list = []

    for batch_start in range(0, total, args.batch_size):
        batch = pending[batch_start: batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (total + args.batch_size - 1) // args.batch_size
        logger.info(f"\n📦 处理第 {batch_num}/{total_batches} 批（{len(batch)} 篇）...")

        for paper in batch:
            pid = paper["paper_id"]
            title_short = paper["title"][:50]
            logger.info(f"  ➡️  迁移中: [{pid}] {title_short}")

            result = migrate_single_paper(paper, skip_embedding=args.skip_embedding)

            if result == "success":
                logger.info(f"  ✅ 成功: {pid}")
                success_count += 1
            elif result == "skipped":
                logger.info(f"  ⏭️  跳过: {pid}")
                skip_count += 1
            else:
                reason = result.replace("failed:", "")
                logger.warning(f"  ❌ 失败: {pid} — {reason}")
                fail_count += 1
                fail_list.append({"paper_id": pid, "title": paper["title"], "reason": reason})

            if args.sleep > 0:
                time.sleep(args.sleep)

    # 最终报告
    logger.info("\n" + "=" * 60)
    logger.info("📊 迁移完成报告")
    logger.info(f"   总计:  {total} 篇")
    logger.info(f"   ✅ 成功: {success_count} 篇")
    logger.info(f"   ⏭️  跳过: {skip_count} 篇")
    logger.info(f"   ❌ 失败: {fail_count} 篇")

    if fail_list:
        logger.warning("\n⚠️ 失败清单：")
        for item in fail_list:
            logger.warning(f"  - [{item['paper_id']}] {item['title'][:50]}: {item['reason']}")

    logger.info("=" * 60)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
