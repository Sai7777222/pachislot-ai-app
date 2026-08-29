"""Phase4FM Section17-19: 実際の /v1/chat, /v1/chat/stream ルート(ChatService/API path)
を通したモデレーション統合テスト。実LLM/実DBはロードしない — 起動が重い実lifespan
(main.pyのlifespan、GPU上のPhase4ZGロードを要する)は使わず、FastAPIアプリを直接
組み立てて `get_chat_service` の依存性だけをフェイクへ差し替える。Section18の指示
通り、Phase4ZGが偶然禁止表現を生成することに依存せず、制御された(フェイクの)
モデル出力を注入して判定する。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pachislot_ai.services.chat_service as chat_service_module
from pachislot_ai.api.deps import get_chat_service
from pachislot_ai.api.routes import api_router
from pachislot_ai.core.config import SYSTEM_PROMPT_PATH
from pachislot_ai.llm.base import ChatCompletionResult, ChatMessage, LLMProvider
from pachislot_ai.services.chat_service import ChatService


class _ScriptedLLMProvider(LLMProvider):
    """常に固定のresponse_textを返す、実モデルを使わないフェイク。"""

    def __init__(self, response_text: str = "安全な応答だよ") -> None:
        self.response_text = response_text
        self.chat_call_count = 0
        self.chat_stream_call_count = 0

    async def chat(self, messages, *, max_tokens=None, temperature=None) -> ChatCompletionResult:  # noqa: ANN001
        self.chat_call_count += 1
        return ChatCompletionResult(content=self.response_text, prompt_tokens=1, completion_tokens=1)

    async def chat_stream(self, messages, *, max_tokens=None, temperature=None) -> AsyncIterator[str]:  # noqa: ANN001
        self.chat_stream_call_count += 1
        # 実運用のllama.cpp providerに近い、1文字ずつのトークン単位ストリームを模す
        for ch in self.response_text:
            yield ch

    async def health_check(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "scripted-fake-model"


def _make_app(llm: _ScriptedLLMProvider) -> FastAPI:
    """main.create_app()相当だが、lifespan(実LLM/実DBロード)を使わず、
    get_chat_serviceの依存性だけをこのテスト用のChatServiceへ差し替える。"""
    app = FastAPI()
    app.include_router(api_router)
    service = ChatService(llm, SYSTEM_PROMPT_PATH, None)
    app.dependency_overrides[get_chat_service] = lambda: service
    return app


def _client(llm: _ScriptedLLMProvider) -> TestClient:
    return TestClient(_make_app(llm))


def _dispatch_call_counter(monkeypatch):
    calls = {"count": 0}
    original = chat_service_module.dispatch

    def wrapper(query):
        calls["count"] += 1
        return original(query)

    monkeypatch.setattr(chat_service_module, "dispatch", wrapper)
    return calls


# ---------- Section17: input integration ----------

@pytest.mark.parametrize(
    "blocked_input",
    [
        "TEST_BLOCK_INPUT_Aだよ、元気？",  # SMALL_TALK-like wording
        "TEST_BLOCK_INPUT_Aの天井は何ゲームですか",  # factual-looking wording
        "GGとSGGについてTEST_BLOCK_INPUT_Aを教えて",  # combined with pachislot terminology
        "禁 止 語 テ ス ト",  # obfuscated blocked input
    ],
)
def test_blocked_input_variants_via_api(monkeypatch, blocked_input):
    calls = _dispatch_call_counter(monkeypatch)
    llm = _ScriptedLLMProvider()
    client = _client(llm)

    resp = client.post("/v1/chat", json={"messages": [{"role": "user", "content": blocked_input}]})

    assert resp.status_code == 200
    body = resp.json()
    assert "TEST_BLOCK_INPUT_A" not in body["message"]["content"]
    assert "禁止語テスト" not in body["message"]["content"]
    assert calls["count"] == 0  # dispatch_called = 0 (preferred)
    assert llm.chat_call_count == 0  # LLM_called = 0 (mandatory)


def test_safe_near_match_not_blocked_via_api():
    llm = _ScriptedLLMProvider(response_text="安全な応答だよ")
    client = _client(llm)

    resp = client.post("/v1/chat", json={"messages": [{"role": "user", "content": "禁止事項について教えて"}]})

    assert resp.status_code == 200
    assert llm.chat_call_count == 1  # ブロックされず、実際にLLMまで到達する


# ---------- Section18: output integration ----------

@pytest.mark.parametrize(
    "scripted_output",
    [
        "TEST_BLOCK_OUTPUT_A",  # exact
        "ＴＥＳＴ＿ＢＬＯＣＫ＿ＯＵＴＰＵＴ＿Ａ",  # NFKC-normalized variant
        "TEST BLOCK OUTPUT A",  # whitespace-obfuscated variant
        "TEST_SUPPRESS_ECHO_A",  # SUPPRESS_ECHO term echoed by the model
    ],
)
def test_blocked_output_variants_never_reach_client(scripted_output):
    llm = _ScriptedLLMProvider(response_text=f"回答です。{scripted_output}という結果でした。")
    client = _client(llm)

    resp = client.post("/v1/chat", json={"messages": [{"role": "user", "content": "こんにちは"}]})

    assert resp.status_code == 200
    body = resp.json()
    assert "TEST_BLOCK_OUTPUT_A" not in body["message"]["content"]
    assert "TEST_SUPPRESS_ECHO_A" not in body["message"]["content"]
    assert llm.chat_call_count == 1  # 生成自体は行われた(出力側で差し替えられた)


def test_safe_output_unchanged_via_api():
    llm = _ScriptedLLMProvider(response_text="こんにちは、元気だよ！")
    client = _client(llm)

    resp = client.post("/v1/chat", json={"messages": [{"role": "user", "content": "こんにちは"}]})

    body = resp.json()
    assert body["message"]["content"] == "こんにちは、元気だよ！"


# ---------- Section19: streaming integration ----------

def _stream_raw_lines(client: TestClient, content: str) -> list[str]:
    with client.stream(
        "POST", "/v1/chat/stream", json={"messages": [{"role": "user", "content": content}]}
    ) as resp:
        assert resp.status_code == 200
        return list(resp.iter_lines())


def test_streaming_blocked_output_leaks_nothing():
    llm = _ScriptedLLMProvider(response_text="TEST_BLOCK_OUTPUT_Aという禁止表現を含む応答")
    client = _client(llm)

    lines = _stream_raw_lines(client, "こんにちは")
    raw_body = "\n".join(lines)

    assert "TEST_BLOCK_OUTPUT_A" not in raw_body  # 部分文字列としても一切出現しない
    assert any("event: done" in line for line in lines)


def test_streaming_blocked_input_calls_no_rag_no_llm(monkeypatch):
    calls = _dispatch_call_counter(monkeypatch)
    llm = _ScriptedLLMProvider()
    client = _client(llm)

    lines = _stream_raw_lines(client, "TEST_BLOCK_INPUT_Aです")
    raw_body = "\n".join(lines)

    assert "TEST_BLOCK_INPUT_A" not in raw_body
    assert llm.chat_stream_call_count == 0
    assert calls["count"] == 0
    assert any("event: done" in line for line in lines)


def test_streaming_safe_content_delivered_fully():
    llm = _ScriptedLLMProvider(response_text="こんにちは、元気だよ！")
    client = _client(llm)

    lines = _stream_raw_lines(client, "こんにちは")
    deltas = []
    for line in lines:
        if line.startswith("data: ") and "delta" in line:
            data = json.loads(line.removeprefix("data: "))
            deltas.append(data["delta"])

    assert "".join(deltas) == "こんにちは、元気だよ！"
