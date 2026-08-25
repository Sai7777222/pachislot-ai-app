"""Embedder + VectorStore を組み合わせたシンプルな検索。

top-k 取得、machine_id / category フィルタ、重複チャンクの除去のみを行う
(BM25・リランカー・エージェント的な複数回検索は Phase 3 の対象外)。
"""

from __future__ import annotations

from dataclasses import dataclass

from pachislot_ai.rag.embedder import Embedder
from pachislot_ai.rag.vector_store import VectorStore


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    doc_id: str
    machine_id: str
    category: str
    title: str
    source_url: str
    source_label: str | None
    data_source_type: str
    score: float


class Retriever:
    def __init__(
        self, embedder: Embedder, vector_store: VectorStore, *, default_top_k: int = 6
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._default_top_k = default_top_k

    def search(
        self,
        query: str,
        *,
        machine_id: str | None = None,
        category: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        k = top_k or self._default_top_k
        query_embedding = self._embedder.embed_query(query)
        results = self._store.query(
            query_embedding, top_k=k, machine_id=machine_id, category=category
        )

        deduped: list[RetrievedChunk] = []
        seen_texts: set[str] = set()
        for r in results:
            normalized = r.text.strip()
            if not normalized or normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            m = r.metadata
            deduped.append(
                RetrievedChunk(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    doc_id=str(m.get("doc_id", "")),
                    machine_id=str(m.get("machine_id", "")),
                    category=str(m.get("category", "")),
                    title=str(m.get("title", "")),
                    source_url=str(m.get("source_url", "")),
                    source_label=(m.get("source_label") or None),
                    data_source_type=str(m.get("data_source_type", "")),
                    score=1.0 - r.distance,
                )
            )
        return deduped
