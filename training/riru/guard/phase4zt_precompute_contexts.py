"""Phase4ZT: UNKNOWN77件+Q6についてretrieval(read-only, .venv側)を事前計算する。
GPU generation(.venv-qlora側)はこのJSONを読むだけで、retrieverを直接importしない。"""
from __future__ import annotations
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
REPORTS_DIR = PROJECT_ROOT / "training" / "riru" / "reports"

from pachislot_ai.core.config import Settings  # noqa: E402
from pachislot_ai.rag.embedder import Embedder  # noqa: E402
from pachislot_ai.rag.vector_store import VectorStore  # noqa: E402
from pachislot_ai.rag.retriever import Retriever  # noqa: E402

Q6_QUERY = "GGとSGGの違いを初心者向けに説明して"


def render_context(chunks) -> str:
    return "\n".join(f"[{c.title}] {c.text}" for c in chunks)


def main():
    settings = Settings()
    embedder = Embedder(settings.embedding_model_path, device=settings.embedding_device)
    vector_store = VectorStore(settings.vector_db_path, settings.vector_db_collection)
    retriever = Retriever(embedder, vector_store, default_top_k=settings.rag_top_k)

    d = json.loads((REPORTS_DIR / "phase4zr_dispatch_results.json").read_text(encoding="utf-8"))
    unk_rows = [r for r in d["rows"] if r["dispatched_mode"] == "UNKNOWN"]

    contexts = {}
    q6_chunks = retriever.search(Q6_QUERY, top_k=6)
    contexts["ZS-Q6"] = {"prompt": Q6_QUERY, "context": render_context(q6_chunks),
                          "titles": [c.title for c in q6_chunks], "texts": [c.text for c in q6_chunks]}
    for r in unk_rows:
        chunks = retriever.search(r["prompt"], top_k=6)
        contexts[r["probe_id"]] = {"prompt": r["prompt"], "expected_mode": r["expected_mode"],
                                    "context": render_context(chunks),
                                    "titles": [c.title for c in chunks], "texts": [c.text for c in chunks]}

    out_path = REPORTS_DIR / "phase4zt_precomputed_contexts.json"
    out_path.write_text(json.dumps(contexts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"precomputed {len(contexts)} contexts -> {out_path}")


if __name__ == "__main__":
    main()
