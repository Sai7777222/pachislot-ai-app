"""Phase 3: RAG統合チャットの結合テスト (実LLM + 実Embedding + 実DB使用)。

要件の「テスト」節にある9項目に対応する。GPU/LLMを使うため `llm` マーカー付き。
"""

from __future__ import annotations

import json

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

MACHINE_ID = "smart_million_god_kamigami_no_kiseki"


def _ask(client: TestClient, question: str, *, machine_id: str | None = None, max_tokens=150):
    payload = {"messages": [{"role": "user", "content": question}], "max_tokens": max_tokens}
    if machine_id is not None:
        payload["machine_id"] = machine_id
    resp = client.post("/v1/chat", json=payload)
    assert resp.status_code == 200
    return resp.json()


# 1. 設定6の機械割
def test_setting6_payout_rate_from_structured_db(client: TestClient) -> None:
    body = _ask(client, "設定6の機械割は？")
    assert "114.6" in body["message"]["content"]
    assert len(body["sources"]["structured_sources"]) > 0


# 2. 設定6の初当り
def test_setting6_hit_rate_from_structured_db(client: TestClient) -> None:
    body = _ask(client, "設定6の初当りは？")
    assert "1/295" in body["message"]["content"]
    assert len(body["sources"]["structured_sources"]) > 0


# 3. 天井
def test_ceiling_info_from_structured_db(client: TestClient) -> None:
    body = _ask(client, "天井は何ゲームですか？", max_tokens=200)
    content = body["message"]["content"]
    assert "1480" in content
    assert len(body["sources"]["structured_sources"]) > 0


# 4. ガイアベルの確率
def test_gaia_bell_probability_from_structured_db(client: TestClient) -> None:
    body = _ask(client, "ガイアベルの確率は？")
    assert "1/37.6" in body["message"]["content"]
    assert len(body["sources"]["structured_sources"]) > 0


# 5. Z-ZONEって何？ (RAG文章、かつ機種名のハルシネーションが無いこと)
def test_zzone_explanation_from_rag_without_wrong_machine_name(client: TestClient) -> None:
    body = _ask(client, "Z-ZONEって何？", max_tokens=200)
    content = body["message"]["content"]
    assert "Z-ZONE" in content
    assert "ミリオンゴッド" in content  # 対象機種が正しく明示されている
    assert "モンスターハンター" not in content  # 既知のハルシネーションが再発していない
    assert len(body["sources"]["chunk_sources"]) > 0


# 6. GGとSGGの違い (RAG文章)
def test_gg_sgg_difference_from_rag(client: TestClient) -> None:
    body = _ask(client, "GGとSGGの違いは？", max_tokens=250)
    content = body["message"]["content"]
    assert "GG" in content
    assert "SGG" in content
    assert len(body["sources"]["chunk_sources"]) > 0


# 7. 未登録情報は推測しない
def test_unregistered_info_is_not_fabricated(client: TestClient) -> None:
    body = _ask(
        client,
        "このミリオンゴッドのプレミアム演出「銀河系ボーナス」の発生率を教えてください。",
        max_tokens=200,
    )
    content = body["message"]["content"]
    assert "登録データにありません" in content
    assert len(body["sources"]["structured_sources"]) == 0
    # 数字を創作していないこと (簡易チェック: 分数/パーセント表記が出ていない)
    assert "1/" not in content


# 8. source情報を内部的に追跡できる
def test_sources_are_traceable(client: TestClient) -> None:
    body = _ask(client, "設定6の機械割は？")
    structured = body["sources"]["structured_sources"]
    assert structured
    assert structured[0]["url"].startswith("file:///")
    assert "data_source_type" in structured[0]

    chunks = body["sources"]["chunk_sources"]
    if chunks:
        assert chunks[0]["source_url"].startswith("file:///")
        assert "chunk_id" in chunks[0]


# machine_id 指定時はその機種を優先検索する
def test_explicit_machine_id_is_used(client: TestClient) -> None:
    body = _ask(client, "設定6の初当りは？", machine_id=MACHINE_ID)
    assert "1/295" in body["message"]["content"]


def test_unknown_machine_id_does_not_fabricate(client: TestClient) -> None:
    body = _ask(client, "設定6の初当りは？", machine_id="no_such_machine_xyz")
    assert len(body["sources"]["structured_sources"]) == 0


# 9. ストリーミングでもRAG回答が正常に流れる
def test_chat_stream_includes_sources_event_and_correct_answer(client: TestClient) -> None:
    question = "設定6の機械割とガイアベルの確率を教えてください。"
    payload = {
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 150,
    }
    with client.stream("POST", "/v1/chat/stream", json=payload) as resp:
        assert resp.status_code == 200
        events: list[str] = []
        deltas: list[str] = []
        sources_payload = None
        for line in resp.iter_lines():
            if not line:
                continue
            events.append(line)
            if line.startswith("data: ") and "delta" in line:
                deltas.append(json.loads(line.removeprefix("data: "))["delta"])

        for i, line in enumerate(events):
            if line == "event: sources":
                sources_payload = json.loads(events[i + 1].removeprefix("data: "))
                break

    assert any("event: done" in e for e in events)
    assert sources_payload is not None
    assert len(sources_payload["structured_sources"]) > 0

    full_text = "".join(deltas)
    assert "114.6" in full_text
    assert "1/37.6" in full_text
