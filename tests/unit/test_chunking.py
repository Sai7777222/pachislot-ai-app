"""チャンク化ロジックの単体テスト (Embedding/LLM不要、高速)。"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from pachislot_ai.rag.chunking import chunk_rag_document, chunk_text


def test_short_text_is_a_single_chunk() -> None:
    chunks = chunk_text("短い文章です。", max_chars=500)
    assert len(chunks) == 1
    assert chunks[0].text == "短い文章です。"


def test_empty_text_produces_no_chunks() -> None:
    assert chunk_text("", max_chars=500) == []
    assert chunk_text("   ", max_chars=500) == []


def test_long_text_is_split_by_paragraph() -> None:
    para_a = "あ" * 300
    para_b = "い" * 300
    text = f"{para_a}\n{para_b}"
    chunks = chunk_text(text, max_chars=400)
    assert len(chunks) == 2
    assert chunks[0].text == para_a
    assert chunks[1].text == para_b
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_paragraphs_are_merged_until_max_chars() -> None:
    paragraphs = ["短文A。", "短文B。", "短文C。"]
    text = "\n".join(paragraphs)
    chunks = chunk_text(text, max_chars=500)
    # 十分小さいので1チャンクにまとまる (意味のまとまりを保ちつつ効率化)
    assert len(chunks) == 1
    assert "短文A" in chunks[0].text
    assert "短文C" in chunks[0].text


def test_single_long_paragraph_falls_back_to_sentence_split() -> None:
    sentence = "これは一文です。" * 20  # 「。」区切りの長い1段落
    chunks = chunk_text(sentence, max_chars=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 50


def _fake_doc(**overrides):
    base = dict(
        doc_id="m1::overview::test",
        machine_id="m1",
        category="overview",
        title="テスト文書",
        body_text="本文です。",
        source_url="file:///dummy.xlsx",
        source_label="dummy",
        data_source_type="unknown",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        review_status="unverified",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_chunk_rag_document_keeps_provenance_on_every_chunk() -> None:
    doc = _fake_doc(body_text="あ" * 300 + "\n" + "い" * 300)
    records = chunk_rag_document(doc, max_chars=400)
    assert len(records) == 2
    for r in records:
        assert r["doc_id"] == "m1::overview::test"
        assert r["machine_id"] == "m1"
        assert r["source_url"] == "file:///dummy.xlsx"
        assert r["data_source_type"] == "unknown"
        assert r["chunk_id"].startswith("m1::overview::test::chunk")


def test_chunk_rag_document_single_chunk_for_short_body() -> None:
    doc = _fake_doc(body_text="短い解説文です。")
    records = chunk_rag_document(doc, max_chars=500)
    assert len(records) == 1
    assert records[0]["chunk_index"] == 0
    assert records[0]["chunk_count"] == 1
