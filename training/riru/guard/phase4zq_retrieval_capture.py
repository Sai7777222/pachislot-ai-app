"""Phase4ZQ Stage A: 既存retrieverをREAD-ONLYで使い、全probeについてraw retrieval
signalを取得する。generation(Phase4ZG呼び出し)は行わない。新しい類似度計算器も
追加しない -- 既存Retriever.search()がそのまま返すscoreのみを使う。"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
REPORTS_DIR = PROJECT_ROOT / "training" / "riru" / "reports"

from pachislot_ai.core.config import Settings  # noqa: E402
from pachislot_ai.rag.embedder import Embedder  # noqa: E402
from pachislot_ai.rag.vector_store import VectorStore  # noqa: E402
from pachislot_ai.rag.retriever import Retriever  # noqa: E402


def build_retriever() -> Retriever:
    settings = Settings()
    embedder = Embedder(settings.embedding_model_path, device=settings.embedding_device)
    vector_store = VectorStore(settings.vector_db_path, settings.vector_db_collection)
    return Retriever(embedder, vector_store, default_top_k=settings.rag_top_k)


def main():
    gt = json.loads((REPORTS_DIR / "phase4zq_ground_truth.json").read_text(encoding="utf-8"))
    retriever = build_retriever()

    rows = []
    t0_all = time.perf_counter()
    for i, row in enumerate(gt["rows"]):
        t0 = time.perf_counter()
        chunks = retriever.search(row["prompt"], top_k=10)  # 少し広めにtop_kを取り、Stage D分析に使う
        dt = time.perf_counter() - t0
        scores = [c.score for c in chunks]
        rows.append({
            "probe_id": row["probe_id"], "expected_mode": row["expected_mode"], "query": row["prompt"],
            "retrieval_called": True, "result_count": len(chunks),
            "top_k_scores": scores,
            "top1_score": scores[0] if scores else None,
            "score_gap_top1_top2": (scores[0] - scores[1]) if len(scores) >= 2 else None,
            "retrieved_chunk_ids": [c.chunk_id for c in chunks],
            "retrieved_titles": [c.title for c in chunks],
            "retrieved_machine_ids": [c.machine_id for c in chunks],
            "retrieved_categories": [c.category for c in chunks],
            "retrieved_source_types": [c.data_source_type for c in chunks],
            "context_lengths": [len(c.text) for c in chunks],
            "empty": len(chunks) == 0,
            "retrieval_time_sec": dt,
        })
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(gt['rows'])} done ({time.perf_counter()-t0_all:.1f}s)", flush=True)

    out = {
        "purpose": "Stage A: raw retrieval signal(score/hit数/chunk metadata等)を全probeについて捕捉。"
                   "generationは呼んでいない。新規similarity計算器は追加していない。",
        "n_total": len(rows), "total_time_sec": time.perf_counter() - t0_all,
        "rows": rows,
    }
    out_path = REPORTS_DIR / "phase4zq_retrieval_trace.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE {len(rows)} probes, {time.perf_counter()-t0_all:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
