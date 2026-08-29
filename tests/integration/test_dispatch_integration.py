"""Phase4FC3: production dispatchが実際のChatService/API経路で正しくRAG context
注入をスキップ/実行することを確認する結合テスト(実LLM使用)。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pachislot_ai.core.config import get_settings

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not get_settings().vector_db_path.exists(),
        reason="vector_db not built (run scripts/build_index.py first)",
    ),
]


def _ask(client: TestClient, question: str, *, max_tokens=100):
    payload = {"messages": [{"role": "user", "content": question}], "max_tokens": max_tokens}
    resp = client.post("/v1/chat", json=payload)
    assert resp.status_code == 200
    return resp.json()


def test_small_talk_gets_no_rag_sources(client: TestClient) -> None:
    body = _ask(client, "おはよう！")
    assert body["sources"]["chunk_sources"] == []
    assert body["sources"]["structured_sources"] == []


def test_identity_question_gets_no_rag_sources(client: TestClient) -> None:
    body = _ask(client, "君の名前は？")
    assert body["sources"]["chunk_sources"] == []
    assert body["sources"]["structured_sources"] == []


def test_ood_factual_gets_no_rag_sources(client: TestClient) -> None:
    body = _ask(client, "今日の東京の天気を教えて")
    assert body["sources"]["chunk_sources"] == []
    assert body["sources"]["structured_sources"] == []


def test_pachislot_factual_keyword_still_gets_rag_sources(client: TestClient) -> None:
    body = _ask(client, "天井は何ゲームですか")
    assert body["sources"]["structured_sources"] != [] or body["sources"]["chunk_sources"] != []


def test_ambiguous_machine_specific_query_still_gets_rag_sources(client: TestClient) -> None:
    # dispatch()自体はUNKNOWNになるが、Policy C3的に既存のRAG pipelineへ委譲され、
    # 実evidenceがある場合は通常通りsourcesが返る(GG/SGGは機種固有名詞のため
    # dispatchの一般語彙リストには一致しない設計)。
    body = _ask(client, "GGとSGGの違いを初心者向けに説明して")
    assert body["sources"]["chunk_sources"] != [] or body["sources"]["structured_sources"] != []
