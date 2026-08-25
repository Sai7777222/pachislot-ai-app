"""Embedding + Vector DB (Chroma) の検索テスト。GPUは使わないがEmbeddingモデルの
ロードを伴うため `embedding` マーカーを付ける (デフォルトの `pytest` でも実行されるが
数秒〜十数秒かかる)。
"""

from __future__ import annotations

import pytest

from pachislot_ai.core.config import get_settings
from pachislot_ai.rag.embedder import Embedder
from pachislot_ai.rag.retriever import Retriever
from pachislot_ai.rag.vector_store import VectorStore

MACHINE_ID = "smart_million_god_kamigami_no_kiseki"

pytestmark = [
    pytest.mark.embedding,
    pytest.mark.skipif(
        not get_settings().vector_db_path.exists(),
        reason="vector_db not built (run scripts/build_index.py first)",
    ),
]


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    settings = get_settings()
    embedder = Embedder(settings.embedding_model_path, device=settings.embedding_device)
    store = VectorStore(settings.vector_db_path, settings.vector_db_collection)
    return Retriever(embedder, store, default_top_k=6)


def test_vector_db_has_chunks_indexed(retriever: Retriever) -> None:
    assert retriever._store.count() > 0  # noqa: SLF001


def test_search_returns_relevant_chunk_for_zzone(retriever: Retriever) -> None:
    results = retriever.search("Z-ZONEって何？", top_k=5)
    assert results
    joined_titles = " ".join(r.title for r in results)
    joined_texts = " ".join(r.text for r in results)
    assert "Z-ZONE" in joined_titles or "Z-ZONE" in joined_texts


def test_search_respects_machine_id_filter(retriever: Retriever) -> None:
    results = retriever.search("打ち方を教えて", machine_id=MACHINE_ID, top_k=5)
    assert results
    assert all(r.machine_id == MACHINE_ID for r in results)


def test_search_with_nonexistent_machine_id_returns_empty(retriever: Retriever) -> None:
    results = retriever.search("打ち方を教えて", machine_id="no_such_machine", top_k=5)
    assert results == []


def test_search_results_are_deduplicated(retriever: Retriever) -> None:
    results = retriever.search("設定示唆について教えて", top_k=10)
    texts = [r.text.strip() for r in results]
    assert len(texts) == len(set(texts))


def test_search_results_carry_source_info(retriever: Retriever) -> None:
    results = retriever.search("打ち方を教えて", top_k=3)
    assert results
    for r in results:
        assert r.source_url.startswith("file:///")
        assert r.doc_id
        assert r.chunk_id
