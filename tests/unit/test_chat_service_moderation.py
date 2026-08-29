"""Phase4FM Section16 (items 8/9/10/18): ChatService統合レベルのモデレーション
unit tests。dispatch/RAG/LLM呼び出しがHARD_BLOCK_INPUTで確実に0回になること、
streamingパスでブロック対象コンテンツが漏れないことを、フェイクの呼び出し回数
計測で検証する(実モデル・実DBは使わない)。"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pachislot_ai.services.chat_service as chat_service_module
from pachislot_ai.core.config import SYSTEM_PROMPT_PATH
from pachislot_ai.llm.base import ChatCompletionResult, ChatMessage, LLMProvider
from pachislot_ai.rag.context_builder import RagContext
from pachislot_ai.services.chat_service import ChatService


class _CountingLLMProvider(LLMProvider):
    def __init__(self, response_text: str = "safe response") -> None:
        self.chat_call_count = 0
        self.chat_stream_call_count = 0
        self._response_text = response_text

    async def chat(self, messages, *, max_tokens=None, temperature=None) -> ChatCompletionResult:  # noqa: ANN001
        self.chat_call_count += 1
        return ChatCompletionResult(content=self._response_text, prompt_tokens=1, completion_tokens=2)

    async def chat_stream(self, messages, *, max_tokens=None, temperature=None) -> AsyncIterator[str]:  # noqa: ANN001
        self.chat_stream_call_count += 1
        for token in self._response_text.split(" "):
            yield token + " "

    async def health_check(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "fake-model"


class _CountingRagPipeline:
    def __init__(self) -> None:
        self.build_context_call_count = 0

    def build_context(self, query: str, *, machine_id: str | None = None) -> RagContext:
        self.build_context_call_count += 1
        return RagContext(
            prompt_text="dummy", structured_source_ids=[], structured_sources=[],
            chunk_sources=[], is_empty=False,
        )


def _counting_dispatch_wrapper(monkeypatch):
    calls = {"count": 0}
    original = chat_service_module.dispatch

    def wrapper(query):
        calls["count"] += 1
        return original(query)

    monkeypatch.setattr(chat_service_module, "dispatch", wrapper)
    return calls


# 8. input block prevents dispatch
async def test_input_block_prevents_dispatch(monkeypatch):
    calls = _counting_dispatch_wrapper(monkeypatch)
    llm = _CountingLLMProvider()
    rag = _CountingRagPipeline()
    service = ChatService(llm, SYSTEM_PROMPT_PATH, rag)

    result = await service.chat([ChatMessage(role="user", content="TEST_BLOCK_INPUT_A")])

    assert calls["count"] == 0
    assert result.content != "TEST_BLOCK_INPUT_A"


# 9. input block prevents RAG
async def test_input_block_prevents_rag():
    llm = _CountingLLMProvider()
    rag = _CountingRagPipeline()
    service = ChatService(llm, SYSTEM_PROMPT_PATH, rag)

    await service.chat([ChatMessage(role="user", content="TEST_BLOCK_INPUT_A")])

    assert rag.build_context_call_count == 0


# 10. input block prevents LLM generation
async def test_input_block_prevents_llm_generation():
    llm = _CountingLLMProvider()
    rag = _CountingRagPipeline()
    service = ChatService(llm, SYSTEM_PROMPT_PATH, rag)

    await service.chat([ChatMessage(role="user", content="TEST_BLOCK_INPUT_A")])

    assert llm.chat_call_count == 0


async def test_input_block_prevents_llm_generation_streaming():
    llm = _CountingLLMProvider()
    rag = _CountingRagPipeline()
    service = ChatService(llm, SYSTEM_PROMPT_PATH, rag)

    deltas = [d async for d in service.chat_stream([ChatMessage(role="user", content="TEST_BLOCK_INPUT_A")])]

    assert llm.chat_stream_call_count == 0
    assert rag.build_context_call_count == 0
    assert len(deltas) == 1
    assert "TEST_BLOCK_INPUT_A" not in deltas[0]


# safe input still reaches dispatch/RAG/LLM normally (control case, proves the
# above 0-counts are due to blocking, not a broken wiring)
async def test_safe_input_reaches_dispatch_rag_llm(monkeypatch):
    calls = _counting_dispatch_wrapper(monkeypatch)
    llm = _CountingLLMProvider()
    rag = _CountingRagPipeline()
    service = ChatService(llm, SYSTEM_PROMPT_PATH, rag)

    await service.chat([ChatMessage(role="user", content="天井は何ゲームですか")])

    assert calls["count"] >= 1
    assert rag.build_context_call_count == 1
    assert llm.chat_call_count == 1


# 18. streaming path does not leak blocked content (出力側でブロックされる場合、
#     バッファリングされた全文がそのままclientへ流れないことを確認する)
async def test_streaming_output_block_does_not_leak_content():
    llm = _CountingLLMProvider(response_text="TEST_BLOCK_OUTPUT_A")
    service = ChatService(llm, SYSTEM_PROMPT_PATH, None)

    deltas = [d async for d in service.chat_stream([ChatMessage(role="user", content="こんにちは")])]

    assert len(deltas) == 1  # バッファリングされ、1回のdeltaとしてのみ送出される
    assert "TEST_BLOCK_OUTPUT_A" not in deltas[0]


async def test_streaming_safe_output_passes_through():
    llm = _CountingLLMProvider(response_text="こんにちは、元気だよ！")
    service = ChatService(llm, SYSTEM_PROMPT_PATH, None)

    deltas = [d async for d in service.chat_stream([ChatMessage(role="user", content="こんにちは")])]

    assert len(deltas) == 1
    assert deltas[0] == "こんにちは、元気だよ！ "


async def test_chat_output_block_replaces_content_non_streaming():
    llm = _CountingLLMProvider(response_text="これはTEST_BLOCK_OUTPUT_Aを含みます")
    service = ChatService(llm, SYSTEM_PROMPT_PATH, None)

    result = await service.chat([ChatMessage(role="user", content="こんにちは")])

    assert "TEST_BLOCK_OUTPUT_A" not in result.content
