"""Phase4ZP Stage A: router accuracy評価。frozen ground truthと比較(RULE EVAL-002準拠)。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import Counter

GUARD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GUARD_DIR))
REPORTS_DIR = GUARD_DIR.parent / "reports"

from phase4zp_router import route, SMALL_TALK, PACHISLOT_FACTUAL, PACHISLOT_CONVERSATIONAL, OOD_FACTUAL  # noqa: E402

MODES = [SMALL_TALK, PACHISLOT_FACTUAL, PACHISLOT_CONVERSATIONAL, OOD_FACTUAL]


def main():
    gt = json.loads((REPORTS_DIR / "phase4zp_router_ground_truth.json").read_text(encoding="utf-8"))
    rows = gt["rows"]

    confusion = {a: {b: 0 for b in MODES} for a in MODES}
    per_mode_correct = Counter()
    per_mode_total = Counter()
    misroutes = []
    dangerous_misroutes = []  # PACHISLOT_FACTUAL -> anything else (loses RAG grounding)

    eval_rows = []
    for row in rows:
        expected = row["expected_mode"]
        r = route(row["prompt"])
        predicted = r.mode
        confusion[expected][predicted] += 1
        per_mode_total[expected] += 1
        correct = predicted == expected
        if correct:
            per_mode_correct[expected] += 1
        else:
            misroutes.append({"probe_id": row["probe_id"], "prompt": row["prompt"],
                               "expected": expected, "predicted": predicted, "matched_rule": r.matched_rule,
                               "matched_keyword": r.matched_keyword})
            if expected == PACHISLOT_FACTUAL:
                dangerous_misroutes.append({"probe_id": row["probe_id"], "prompt": row["prompt"],
                                             "predicted": predicted})
        eval_rows.append({"probe_id": row["probe_id"], "prompt": row["prompt"], "expected_mode": expected,
                           "predicted_mode": predicted, "correct": correct, "matched_rule": r.matched_rule})

    total = len(rows)
    total_correct = sum(per_mode_correct.values())
    overall_accuracy = total_correct / total

    out = {
        "purpose": "Stage A: router accuracy評価。router自身の予測をground truthとして使わず、"
                   "独立assetのphase4zp_router_ground_truth.jsonと比較する(RULE EVAL-001準拠)。",
        "total_probes": total, "overall_accuracy": overall_accuracy,
        "per_mode_accuracy": {m: {"correct": per_mode_correct[m], "total": per_mode_total[m],
                                    "accuracy": per_mode_correct[m] / per_mode_total[m]} for m in MODES},
        "confusion_matrix": confusion,
        "misroute_count": len(misroutes), "misroutes": misroutes,
        "dangerous_pachislot_factual_misroute_count": len(dangerous_misroutes),
        "dangerous_pachislot_factual_misroutes": dangerous_misroutes,
        "target_overall_accuracy": 0.95,
        "target_dangerous_misroute": 0,
        "gate_result": {
            "overall_accuracy_met": overall_accuracy >= 0.95,
            "dangerous_misroute_met": len(dangerous_misroutes) == 0,
        },
        "rows": eval_rows,
    }
    (REPORTS_DIR / "phase4zp_router_confusion_matrix.json").write_text(
        json.dumps(confusion, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "phase4zp_router_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"overall_accuracy={overall_accuracy:.4f} ({total_correct}/{total})")
    for m in MODES:
        print(f"  {m}: {per_mode_correct[m]}/{per_mode_total[m]}")
    print(f"dangerous PACHISLOT_FACTUAL misroutes: {len(dangerous_misroutes)}")
    print(f"Saved -> phase4zp_router_eval.json / phase4zp_router_confusion_matrix.json")


if __name__ == "__main__":
    main()
