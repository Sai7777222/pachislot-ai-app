"""API 統合テスト。実際に LLM をロードして /v1/health, /v1/chat, /v1/chat/stream を叩く。"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.llm


def test_health(client: TestClient) -> None:
    resp = client.get("/v1/health")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm"]["loaded"] is True
    assert body["llm"]["gpu_offload_supported"] is True


def test_chat(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat",
        json={
            "messages": [{"role": "user", "content": "こんにちは、簡潔に挨拶を返してください。"}],
            "max_tokens": 64,
        },
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["message"]["role"] == "assistant"
    assert body["message"]["content"].strip() != ""
    assert body["usage"]["completion_tokens"] > 0


def test_chat_validation_error_on_empty_messages(client: TestClient) -> None:
    resp = client.post("/v1/chat", json={"messages": []})
    assert resp.status_code == 422


def test_chat_stream(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/chat/stream",
        json={
            "messages": [{"role": "user", "content": "1から3まで数えてください。"}],
            "max_tokens": 64,
        },
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events: list[str] = []
        deltas: list[str] = []
        for line in resp.iter_lines():
            if not line:
                continue
            events.append(line)
            if line.startswith("data: ") and "delta" in line:
                data = json.loads(line.removeprefix("data: "))
                deltas.append(data["delta"])

    assert any("event: done" in e for e in events)
    assert len(deltas) > 0
    assert "".join(deltas).strip() != ""
