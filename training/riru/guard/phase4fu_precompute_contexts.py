"""Phase4FU: 独立GT用の30 probe全件についてread-only retrievalを実行し、
retrieved contextをJSONに保存する(generation前、GT作成のための素材)。"""
from __future__ import annotations
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "riru"))
REPORTS_DIR = PROJECT_ROOT / "training" / "riru" / "reports"

from pachislot_ai.core.config import Settings  # noqa: E402
from pachislot_ai.rag.embedder import Embedder  # noqa: E402
from pachislot_ai.rag.vector_store import VectorStore  # noqa: E402
from pachislot_ai.rag.retriever import Retriever  # noqa: E402
from eval.phase4fu_probes import ALL_PROBES  # noqa: E402


def main():
    settings = Settings()
    embedder = Embedder(settings.embedding_model_path, device=settings.embedding_device)
    vector_store = VectorStore(settings.vector_db_path, settings.vector_db_collection)
    retriever = Retriever(embedder, vector_store, default_top_k=settings.rag_top_k)

    out = []
    for p in ALL_PROBES:
        chunks = retriever.search(p["prompt"], top_k=settings.rag_top_k)
        retrieved = [
            {
                "chunk_id": c.chunk_id, "text": c.text, "doc_id": c.doc_id,
                "machine_id": c.machine_id, "category": c.category, "title": c.title,
                "data_source_type": c.data_source_type, "score": round(c.score, 4),
            }
            for c in chunks
        ]
        out.append({
            "id": p["id"], "category": p["category"], "prompt": p["prompt"],
            "source": p["source"], "retrieved_chunks": retrieved,
        })
        print(f"{p['id']}: top_score={retrieved[0]['score'] if retrieved else None} n_chunks={len(retrieved)}")

    out_path = REPORTS_DIR / "phase4fu_precomputed_contexts.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path} ({len(out)} probes)")


if __name__ == "__main__":
    main()
