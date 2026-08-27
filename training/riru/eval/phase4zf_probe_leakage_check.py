"""Phase 4ZF Section11: 新規probe(phase4zf_stress_probes, phase4zf_ood_probes)の
既存training datasetとの重複検査。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
REPORTS_DIR = TRAINING_ROOT / "reports"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(TRAINING_ROOT))


def normalize(t: str) -> str:
    t = re.sub(r"\s+", "", t)
    t = t.replace("！", "").replace("？", "").replace("、", "").replace("。", "")
    return t.lower()


def char_bigrams(text: str) -> set[str]:
    t = re.sub(r"\s+", "", text)
    return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) >= 2 else {t}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> int:
    from phase4zf_stress_probes import ALL_PROBES as STRESS_PROBES
    from phase4zf_ood_probes import ALL_PROBES as OOD_PROBES
    from phase4ze_identity_margin_source_data import ALL_RECORDS as ZE_TRAIN_RECORDS

    new_probes = [{"id": p["id"], "text": p["prompt"]} for p in STRESS_PROBES + OOD_PROBES]
    train_texts = [{"id": r["id"], "text": r["user"]} for r in ZE_TRAIN_RECORDS]

    candidate_path = TRAINING_ROOT / "processed" / "riru_phase4ze_identity_margin_candidate.jsonl"
    all_train_user_texts = []
    with open(candidate_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                all_train_user_texts.append({"id": rec["metadata"].get("index", "?"),
                                              "text": rec["messages"][0]["content"]})

    exact_dup = []
    normalized_dup = []
    high_sim = []
    for p in new_probes:
        p_norm = normalize(p["text"])
        p_bi = char_bigrams(p["text"])
        for t in all_train_user_texts:
            if p["text"] == t["text"]:
                exact_dup.append({"probe": p["id"], "train_id": t["id"], "text": p["text"][:60]})
            elif p_norm == normalize(t["text"]):
                normalized_dup.append({"probe": p["id"], "train_id": t["id"], "text": p["text"][:60]})
            else:
                sim = jaccard(p_bi, char_bigrams(t["text"]))
                if sim >= 0.85:
                    high_sim.append({"probe": p["id"], "train_id": t["id"], "similarity": round(sim, 3),
                                      "probe_text": p["text"][:60], "train_text": t["text"][:60]})

    result = {
        "n_new_probes": len(new_probes),
        "n_train_records_checked": len(all_train_user_texts),
        "exact_duplicate": exact_dup,
        "normalized_exact_duplicate": normalized_dup,
        "high_similarity_pairs_jaccard_ge_0.85": high_sim,
        "n_exact_duplicate": len(exact_dup),
        "n_normalized_duplicate": len(normalized_dup),
        "n_high_similarity": len(high_sim),
        "verdict": "CLEAN" if not (exact_dup or normalized_dup or high_sim) else "REVIEW_NEEDED",
    }
    out_path = REPORTS_DIR / "phase4zf_probe_leakage_analysis.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
