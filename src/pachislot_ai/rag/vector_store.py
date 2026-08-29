"""ローカル Vector DB (Chroma) ラッパー。

DESIGN.md の方針どおり、まずはローカル用に Chroma を使う
(`D:\\AI\\cache\\vector_db` に永続化。将来クラウド移行時は Qdrant 等に差し替え可能)。
このモジュールはネットワークへ一切アクセスしない (`PersistentClient` はローカル
ファイルのみを使い、Chroma のテレメトリも明示的に無効化する)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk_id: str
    text: str
    metadata: dict
    distance: float


def _sanitize_metadata(metadata: dict) -> dict:
    """Chroma のメタデータは str/int/float/bool のみ許容 (None不可) のため変換する。"""
    sanitized = {}
    for key, value in metadata.items():
        sanitized[key] = "" if value is None else value
    return sanitized


class VectorStore:
    def __init__(self, path: Path, collection_name: str) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            _sanitize_metadata({k: v for k, v in c.items() if k not in ("chunk_id", "text")})
            for c in chunks
        ]
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def delete_by_machine(self, machine_id: str) -> None:
        self._collection.delete(where={"machine_id": machine_id})

    def count(self) -> int:
        return self._collection.count()

    def get_all(self, *, machine_id: str | None = None) -> list[SearchResult]:
        """title supplemental retrieval (Phase4FX/4FY) 用に、embeddingスコアを介さず
        全チャンク(または指定machine_idのチャンク)をそのまま取得する。新しい検索器では
        なく、既存Chromaコレクションに対する単純な一括取得(distance/scoreは持たない)。"""
        if self.count() == 0:
            return []
        where = {"machine_id": machine_id} if machine_id else None
        result = self._collection.get(where=where, include=["documents", "metadatas"])
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        return [
            SearchResult(chunk_id=i, text=d, metadata=m, distance=0.0)
            for i, d, m in zip(ids, documents, metadatas, strict=True)
        ]

    def query(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 6,
        machine_id: str | None = None,
        category: str | None = None,
    ) -> list[SearchResult]:
        where = None
        conditions = []
        if machine_id:
            conditions.append({"machine_id": machine_id})
        if category:
            conditions.append({"category": category})
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        # Chroma は空コレクションに対する query でエラーになることがあるため防御する
        if self.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(self.count(), 1)),
            where=where,
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            SearchResult(chunk_id=i, text=d, metadata=m, distance=dist)
            for i, d, m, dist in zip(ids, documents, metadatas, distances, strict=True)
        ]
