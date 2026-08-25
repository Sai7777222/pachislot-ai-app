"""Phase 4E: 既存17問 (structured DB/RAG) のRAGコンテキストを事前計算する。

既存のRagPipeline (structured.db / rag_store.db / Vector DB、いずれも読み取り専用)
をそのまま使い、17問それぞれについてLLMへ渡されるコンテキスト文字列を取得して
JSONへ保存する。これにより、QLoRA venv側 (transformers/peft) では
sentence-transformers/chromadbを別途インストールせずに、
「本番と同じRAGコンテキスト」を使ってQwenベース/LoRA適用後の生成比較ができる。

このスクリプトは本体venv (.venv) で実行する。DB/RAGは一切変更しない。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from compare_llms import MACHINE_ID, QUESTIONS  # noqa: E402

from pachislot_ai.core.config import RAG_CONTEXT_PROMPT_PATH, get_settings  # noqa: E402
from pachislot_ai.data.db import create_structured_engine  # noqa: E402
from pachislot_ai.rag.embedder import Embedder  # noqa: E402
from pachislot_ai.rag.pipeline import RagPipeline  # noqa: E402
from pachislot_ai.rag.retriever import Retriever  # noqa: E402
from pachislot_ai.rag.vector_store import VectorStore  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "structured_rag_17q_context.json"


def main() -> int:
    settings = get_settings()
    embedder = Embedder(settings.embedding_model_path, device=settings.embedding_device)
    vector_store = VectorStore(settings.vector_db_path, settings.vector_db_collection)
    retriever = Retriever(embedder, vector_store, default_top_k=settings.rag_top_k)
    structured_engine = create_structured_engine(settings.structured_db_path)
    rag_pipeline = RagPipeline(
        retriever, structured_engine, RAG_CONTEXT_PROMPT_PATH, top_k=settings.rag_top_k
    )

    results = []
    for q in QUESTIONS:
        ctx = rag_pipeline.build_context(q["text"], machine_id=MACHINE_ID)
        results.append(
            {
                "id": q["id"],
                "group": q["group"],
                "question": q["text"],
                "rag_context_text": ctx.prompt_text,
                "structured_sources": len(ctx.structured_sources),
                "chunk_sources": len(ctx.chunk_sources),
            }
        )
        print(f"{q['id']}: context_chars={len(ctx.prompt_text)}")

    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} precomputed contexts -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
