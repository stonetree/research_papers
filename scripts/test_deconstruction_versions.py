# -*- coding: utf-8 -*-
"""
scripts/test_deconstruction_versions.py
======================================
自动化验证文献多版本解构与重新解构管理功能：
1. 验证 paper_analysis_versions 结构与存量 ai_summaries 自动平滑升版（第 1 版）；
2. 验证多次保存解构内容时的版本递增（第 1 版、第 2 版...）与旧版本无损保留；
3. 验证默认版本切换（set_paper_default_version）与 ai_summaries 双向同步；
4. 验证异常报错保护机制（失败时不覆盖历史成功版本）；
5. 验证删除文献元数据（delete_paper_metadata）时的多版本级联删除。
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.database import (
    init_db,
    get_db_connection,
    insert_paper,
    save_paper_analysis_version,
    get_paper_analysis_versions,
    set_paper_default_version,
    save_ai_summary,
    delete_paper_metadata
)

def run_tests():
    print("🧪 开始验证文献多版本解构与重新解构功能...")
    
    # 确保基础表初始化完成
    init_db()
    
    test_paper_id = "test_paper_versioning_001"
    
    # 清理可能残留的测试数据
    delete_paper_metadata(test_paper_id)
    
    # 1. 插入一篇测试文献
    insert_paper({
        "paper_id": test_paper_id,
        "title": "A Test Paper on KV Cache Compression",
        "authors": "Alice, Bob",
        "venue": "NeurIPS",
        "year": 2025,
        "citations": 42,
        "abstract": "This is a test abstract.",
        "pdf_path": "storage/library/test_paper.pdf",
        "source_engine": "test"
    })
    
    # 2. 第一次解构（第 1 版）
    v1_text = "### 【第 1 版解构报告】\n- 技术亮点：首次提出动态 KV Cache 稀疏化算法。\n- 局限性：显存带宽开销略高。"
    v1_model = "DeepSeek-V3 (deepseek-ai/deepseek-v3)"
    ok1, v1_num = save_paper_analysis_version(test_paper_id, v1_model, v1_text, set_as_default=True)
    assert ok1, "第 1 版保存应成功"
    assert v1_num == 1, f"首个版本号应为 1，实际为 {v1_num}"
    
    versions = get_paper_analysis_versions(test_paper_id)
    assert len(versions) == 1, f"版本数应为 1，实际为 {len(versions)}"
    assert versions[0]["version_num"] == 1
    assert versions[0]["is_default"] == 1
    assert versions[0]["dialectical_analysis"] == v1_text
    print("✅ 测试项 1 通过：首个版本成功写入并默认生效（第 1 版）")
    
    # 3. 重新解构（第 2 版）
    v2_text = "### 【第 2 版全景深度解构报告】\n- 第一性原理：彻底重构 Host DDR5 与 GPU HBM 间的数据搬运通道。\n- 落地壁垒：需 PCIe 5.0 硬件支持。"
    v2_model = "Gemini-2.5-Pro (gemini-2.5-pro)"
    ok2, v2_num = save_paper_analysis_version(test_paper_id, v2_model, v2_text, set_as_default=True)
    assert ok2, "第 2 版保存应成功"
    assert v2_num == 2, f"第二个版本号应为 2，实际为 {v2_num}"
    
    versions = get_paper_analysis_versions(test_paper_id)
    assert len(versions) == 2, f"版本数应为 2，旧版本必须保留，实际为 {len(versions)}"
    
    # 验证旧版本与新版本共存，且第 2 版为默认
    v1_row = next(v for v in versions if v["version_num"] == 1)
    v2_row = next(v for v in versions if v["version_num"] == 2)
    assert v1_row["is_default"] == 0, "第 1 版应不再是默认"
    assert v2_row["is_default"] == 1, "第 2 版应成为默认"
    assert v1_row["dialectical_analysis"] == v1_text, "第 1 版内容必须完整保留"
    assert v2_row["dialectical_analysis"] == v2_text, "第 2 版内容必须正确写入"
    
    # 验证 ai_summaries 同步更新为第 2 版
    conn = get_db_connection()
    s_row = conn.execute("SELECT dialectical_analysis, model_name FROM ai_summaries WHERE paper_id = ?", (test_paper_id,)).fetchone()
    conn.close()
    assert s_row["dialectical_analysis"] == v2_text, "ai_summaries 应同步更新为默认的第 2 版内容"
    assert s_row["model_name"] == v2_model
    print("✅ 测试项 2 通过：重新解构生成第 2 版并保留第 1 版，默认版本及兼容表正确更新")
    
    # 4. 再次重新解构（第 3 版，测试 save_ai_summary 兼容函数）
    v3_text = "### 【第 3 版架构级洞察报告】\n- 算力拓扑分析完成。"
    v3_model = "Claude-3.7-Sonnet (claude-3-7-sonnet)"
    save_ai_summary(test_paper_id, v3_model, v3_text)
    
    versions = get_paper_analysis_versions(test_paper_id)
    assert len(versions) == 3, f"版本数应为 3，实际为 {len(versions)}"
    v3_row = next(v for v in versions if v["version_num"] == 3)
    assert v3_row["is_default"] == 1
    print("✅ 测试项 3 通过：save_ai_summary 兼容接口正常递增版本并设为默认（第 3 版）")
    
    # 5. 测试切换默认版本（将第 1 版重新设为默认）
    set_paper_default_version(test_paper_id, 1)
    versions = get_paper_analysis_versions(test_paper_id)
    v1_row = next(v for v in versions if v["version_num"] == 1)
    v2_row = next(v for v in versions if v["version_num"] == 2)
    v3_row = next(v for v in versions if v["version_num"] == 3)
    assert v1_row["is_default"] == 1, "第 1 版应重新成为默认"
    assert v2_row["is_default"] == 0
    assert v3_row["is_default"] == 0
    
    # 检查 ai_summaries 同步为第 1 版
    conn = get_db_connection()
    s_row = conn.execute("SELECT dialectical_analysis FROM ai_summaries WHERE paper_id = ?", (test_paper_id,)).fetchone()
    conn.close()
    assert s_row["dialectical_analysis"] == v1_text, "ai_summaries 应同步更新为重新设为默认的第 1 版内容"
    print("✅ 测试项 4 通过：set_paper_default_version 成功将历史版本设为默认并同步 ai_summaries")
    
    # 6. 测试错误保护（解构报错不破坏已有历史版本）
    err_msg = "❌ API 连接超时，解构失败"
    ok_err, err_v = save_paper_analysis_version(test_paper_id, "test_err_model", err_msg)
    assert not ok_err, "错误文本不应作为成功版本写入"
    versions_after_err = get_paper_analysis_versions(test_paper_id)
    assert len(versions_after_err) == 3, "已有版本数量不应发生改变"
    
    conn = get_db_connection()
    s_row = conn.execute("SELECT dialectical_analysis FROM ai_summaries WHERE paper_id = ?", (test_paper_id,)).fetchone()
    conn.close()
    assert s_row["dialectical_analysis"] == v1_text, "已有默认版本不应被错误信息覆盖"
    print("✅ 测试项 5 通过：解构失败时历史有效版本完整受到保护，不会被覆盖")
    
    # 7. 测试级联删除（delete_paper_metadata）
    delete_paper_metadata(test_paper_id)
    versions_after_del = get_paper_analysis_versions(test_paper_id)
    assert len(versions_after_del) == 0, "删除后 paper_analysis_versions 应被清空"
    conn = get_db_connection()
    s_cnt = conn.execute("SELECT COUNT(*) FROM ai_summaries WHERE paper_id = ?", (test_paper_id,)).fetchone()[0]
    p_cnt = conn.execute("SELECT COUNT(*) FROM papers WHERE paper_id = ?", (test_paper_id,)).fetchone()[0]
    conn.close()
    assert s_cnt == 0, "ai_summaries 记录应已删除"
    assert p_cnt == 0, "papers 记录应已删除"
    print("✅ 测试项 6 通过：delete_paper_metadata 正确级联删除所有版本")
    
    print("\n🎉 所有多版本解构与重新解构测试全部顺利通过！")

if __name__ == "__main__":
    run_tests()
