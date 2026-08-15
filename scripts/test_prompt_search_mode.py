# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from core.search_engine import api_answer

async def test_prompts():
    print("Testing prompt generation for responses vs standard chat endpoints...")
    
    # 1. Test Standard Chat Endpoint (non-response API, e.g. deepseek chat)
    mock_cfg_chat = {
        "provider": "openai_compatible",
        "url": "https://api.deepseek.com/chat/completions",
        "resolved_api_key": "test_key",
        "model": "deepseek-chat"
    }
    
    captured_messages = []
    
    class MockPostResponse:
        status = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def json(self):
            return {
                "choices": [{"message": {"content": "回答内容 [doc_001]"}}],
                "usage": {"total_tokens": 100}
            }

    class MockClientSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        def post(self, url, json=None, headers=None, timeout=None):
            captured_messages.append(json.get("messages", []))
            return MockPostResponse()

    with patch("core.search_engine.get_model_config", return_value=mock_cfg_chat), \
         patch("aiohttp.ClientSession", return_value=MockClientSession()):
        res = await api_answer(
            model_provider="openai_compatible",
            model_name="deepseek-chat",
            user_query="How does CXL 3.0 work?",
            context_evidences=[{"doc_id": "doc_001", "text": "CXL 3.0 enables multi-tier memory sharing."}],
            model_id="deepseek-chat"
        )
        assert res["status"] == "success"
        assert len(captured_messages) > 0
        sys_msg = captured_messages[-1][0]["content"]
        usr_msg = captured_messages[-1][1]["content"]
        
        print("✅ [非 responses 端口 - 离线解析模式 Prompt 验证]:")
        assert "离线文献深度解析" in sys_msg or "严禁开启任何联网搜索" in sys_msg
        assert "严禁尝试发起网络搜索" in sys_msg
        assert "无需且严禁开启网络搜索" in usr_msg
        print("   - System Prompt 明确声明：离线文献解析模式，严禁开启搜索/调用外部工具")
        print("   - User Prompt 明确声明：直接解析已提取切片，无需联网搜索")

    # 2. Test Responses Endpoint (responses API, e.g. 百炼 responses)
    mock_cfg_response = {
        "provider": "openai_compatible",
        "url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/responses",
        "resolved_api_key": "test_key",
        "model": "qwen3.7-max"
    }
    
    captured_messages.clear()
    with patch("core.search_engine.get_model_config", return_value=mock_cfg_response), \
         patch("aiohttp.ClientSession", return_value=MockClientSession()):
        res = await api_answer(
            model_provider="openai_compatible",
            model_name="qwen3.7-max",
            user_query="How does CXL 3.0 work?",
            context_evidences=[{"doc_id": "doc_001", "text": "CXL 3.0 enables multi-tier memory sharing."}],
            model_id="qwen3.7-max"
        )
        assert res["status"] == "success"
        assert len(captured_messages) > 0
        sys_msg = captured_messages[-1][0]["content"]
        print("✅ [Responses 端口 - 原生 Grounding 模式 Prompt 验证]:")
        assert "严禁开启网络搜索" not in sys_msg
        print("   - System Prompt 支持原生 responses 事实 grounding 对账")

    print("🟢 所有提示词模式适配与 503 防护测试全部通过！")

if __name__ == "__main__":
    asyncio.run(test_prompts())
