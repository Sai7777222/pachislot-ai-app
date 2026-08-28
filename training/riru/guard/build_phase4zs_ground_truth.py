"""Phase4ZS Section3-4: 独立ground truth構築。retrieval結果(read-only、generation前)
のみを根拠にし、モデル生成結果は一切参照しない。RULE EVAL-001~004準拠。"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "riru" / "eval"))
REPORTS_DIR = PROJECT_ROOT / "training" / "riru" / "reports"

from pachislot_ai.core.config import Settings  # noqa: E402
from pachislot_ai.rag.embedder import Embedder  # noqa: E402
from pachislot_ai.rag.vector_store import VectorStore  # noqa: E402
from pachislot_ai.rag.retriever import Retriever  # noqa: E402
from phase4zf_rag_stress_eval import load_rag_probe_pool  # noqa: E402
from phase4zs_fabrication_probes import ALL_PROBES as ZS_NEW  # noqa: E402

NUMERIC_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?パーセント|1/\d+|\d+G(?:\D|$)|\d+枚")


def build_rag50():
    pool = load_rag_probe_pool()
    by_id = {p["id"]: p for p in pool}
    must_have = {"P02", "P04", "LC-08", "Q11", "Q15", "Q17", "AD-04"} | {f"Q{i}" for i in range(1, 18)}
    selected_ids = [pid for pid in must_have if pid in by_id]
    remaining = [p["id"] for p in pool if p["id"] not in selected_ids]
    for pid in remaining:
        if len(selected_ids) >= 50:
            break
        selected_ids.append(pid)
    return [{"probe_id": f"RAG50-{pid}", "prompt": by_id[pid]["question"], "source": "rag50", "original_id": pid}
            for pid in selected_ids]


def main():
    settings = Settings()
    embedder = Embedder(settings.embedding_model_path, device=settings.embedding_device)
    vector_store = VectorStore(settings.vector_db_path, settings.vector_db_collection)
    retriever = Retriever(embedder, vector_store, default_top_k=settings.rag_top_k)

    all_probes = build_rag50() + [
        {"probe_id": p["id"], "prompt": p["prompt"], "source": "zs_new", "style": p["style"]} for p in ZS_NEW
    ]

    rows = []
    for p in all_probes:
        chunks = retriever.search(p["prompt"], top_k=settings.rag_top_k)
        texts = [c.text for c in chunks]
        joined = "\n".join(texts)
        numerics = NUMERIC_PATTERN.findall(joined)
        category = "A_numeric_present" if numerics else "B_no_numeric_in_context"
        rows.append({
            "probe_id": p["probe_id"], "prompt": p["prompt"], "source": p["source"],
            "style": p.get("style"), "original_id": p.get("original_id"),
            "numeric_category": category,
            "numerics_found_in_context": list(dict.fromkeys(numerics)),
            "retrieved_chunk_ids": [c.chunk_id for c in chunks],
            "retrieved_titles": [c.title for c in chunks],
            "annotation_source": "read_only_retrieval_before_any_generation",
            "frozen": True,
        })

    n_a = sum(1 for r in rows if r["numeric_category"] == "A_numeric_present")
    out = {
        "purpose": "Phase4ZS Section3-4: RAG50(50件)+新規Q6型probe(20件)=70件の独立ground truth。"
                   "retrieval結果(generation前)のみを根拠に、各probeの検索context内に数値が"
                   "存在するか否か(category A/B)を機械的に判定した。C(一部数値のみ)/D-H(質問様式)は"
                   "各probe定義時のstyleタグ、または後続stageでの目視判定に委ねる(自動判定の限界を"
                   "誠実に認める)。",
        "total": len(rows), "category_a_numeric_present": n_a, "category_b_no_numeric": len(rows) - n_a,
        "rows": rows,
    }
    out_path = REPORTS_DIR / "phase4zs_ground_truth.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    import hashlib
    h = hashlib.sha256(out_path.read_bytes()).hexdigest()
    (REPORTS_DIR / "phase4zs_ground_truth_hash.txt").write_text(
        f"sha256: {h}\nfile: phase4zs_ground_truth.json\nfrozen_before_generation: true\n"
        f"total_rows: {len(rows)}\n", encoding="utf-8")
    print(f"total={len(rows)} category_A(numeric_present)={n_a} category_B(no_numeric)={len(rows)-n_a}")
    print(f"sha256={h}")


if __name__ == "__main__":
    main()
