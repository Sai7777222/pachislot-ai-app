# -*- coding: utf-8 -*-
"""Phase4FW Stage D評価: deterministic verifierをGTの129 atomic claim全件に対して実行し、
precision/recall/TP/FP/TN/FNを計測する。unsafe claim(UNSUPPORTED/MISATTRIBUTED) = positive。"""
from __future__ import annotations
import json
import sys
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parent
REPORTS_DIR = GUARD_DIR.parent / "reports"
sys.path.insert(0, str(GUARD_DIR))
from phase4fw_deterministic_verifier import deterministic_verify  # noqa: E402

UNSAFE_STATUSES = {"UNSUPPORTED", "MISATTRIBUTED"}
SAFE_STATUSES = {"SUPPORTED", "NON_FACTUAL"}


def main():
    gt = json.loads((REPORTS_DIR / "phase4fw_ground_truth.json").read_text(encoding="utf-8"))
    targets = json.loads((REPORTS_DIR / "phase4fw_target_responses.json").read_text(encoding="utf-8"))
    ctx_by_id = {t["id"]: t["context"] for t in targets}

    rows = []
    tp = fp = tn = fn = 0
    ambiguous_excluded = 0
    for resp_id, claims in gt["claims_by_response"].items():
        context = ctx_by_id[resp_id]
        for i, cl in enumerate(claims):
            result = deterministic_verify(cl["text"], cl["subject"], context)
            gt_status = cl["status"]
            pred_unsafe = result["status"] in UNSAFE_STATUSES
            row = {"response_id": resp_id, "claim_idx": i, "claim_text": cl["text"], "subject": cl["subject"],
                   "gt_status": gt_status, "gt_claim_type": cl["claim_type"],
                   "predicted_status": result["status"], "predicted_grounding": result["grounding"],
                   "predicted_reason": result["reason"]}
            if gt_status == "AMBIGUOUS":
                ambiguous_excluded += 1
                row["eval_bucket"] = "EXCLUDED_AMBIGUOUS"
            elif gt_status in UNSAFE_STATUSES and pred_unsafe:
                tp += 1
                row["eval_bucket"] = "TP"
            elif gt_status in UNSAFE_STATUSES and not pred_unsafe:
                fn += 1
                row["eval_bucket"] = "FN"
            elif gt_status in SAFE_STATUSES and pred_unsafe:
                fp += 1
                row["eval_bucket"] = "FP"
            else:
                tn += 1
                row["eval_bucket"] = "TN"
            rows.append(row)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    out = {
        "purpose": "Stage D: deterministic verifier(entity grounding + evidence binding + numeric/symbol exact match)をGT全129 claimに対して評価。unsafe claim(UNSUPPORTED/MISATTRIBUTED)=positiveとする。",
        "summary": {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "ambiguous_excluded": ambiguous_excluded,
                     "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)},
        "rows": rows,
    }
    (REPORTS_DIR / "phase4fw_deterministic_verifier.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn} ambiguous_excluded={ambiguous_excluded}")
    print(f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

    # FNの内訳(見逃した危険claim)を特に表示
    fns = [r for r in rows if r["eval_bucket"] == "FN"]
    print(f"\n=== FN (見逃したunsafe claim) {len(fns)}件 ===")
    for r in fns:
        print(f"  [{r['response_id']}] subject={r['subject']!r} gt={r['gt_status']} pred={r['predicted_status']}")


if __name__ == "__main__":
    main()
