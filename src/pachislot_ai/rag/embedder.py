"""Embedding モデル (multilingual-e5-base) のラッパー。

e5 系モデルは "query: " / "passage: " のプレフィックスを付けることで検索性能が
向上する設計になっている (公式カード記載の慣習)。文書側は "passage: "、
検索クエリ側は "query: " を付与する。

常にローカルのモデルディレクトリを直接指定してロードし、`HF_HUB_OFFLINE` /
`TRANSFORMERS_OFFLINE` を設定することで、万一のネットワークアクセスも防ぐ。
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class Embedder:
    def __init__(self, model_path: Path, *, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        if not model_path.is_dir():
            raise FileNotFoundError(f"Embedding model directory not found: {model_path}")

        self._model = SentenceTransformer(str(model_path), device=device)
        get_dim = getattr(
            self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension
        )
        self.dimension = get_dim()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        vectors = self._model.encode(
            prefixed, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            f"query: {text}", normalize_embeddings=True, convert_to_numpy=True
        )
        return vector.tolist()
