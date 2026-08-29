# -*- coding: utf-8 -*-
"""Phase4FW Stage E評価 + Section11 self-verification bias確認。
model verifier(phase4fw_model_verifier_raw.json)の出力とGTを突き合わせ、precision/recall、
既知failure(Q6/AT-F/RT-A・RT-B/ループストック・GGストック等)の検出率、self-verification biasを算出する。"""
from __future__ import annotations
import json
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

UNSAFE_STATUSES = {"UNSUPPORTED", "MISATTRIBUTED"}
SAFE_STATUSES = {"SUPPORTED", "NON_FACTUAL"}

# self-verification bias確認の必須対象(Section11)
SELF_VERIFY_CRITICAL_IDS = ["FU-D01", "FU-A03", "FU-B05", "FV-C03"]


def main():
    gt = json.loads((REPORTS_DIR / "phase4fw_ground_truth.json").read_text(encoding="utf-8"))
    raw = json.loads((REPORTS_DIR / "phase4fw_model_verifier_raw.json").read_text(encoding="utf-8"))
    raw_by_id = {r["id"]: r for r in raw}

    rows = []
    tp = fp = tn = fn = 0
    ambiguous_excluded = 0
    unparseable_claim_level = 0

    for pid, r in raw_by_id.items():
        gt_claims = gt["claims_by_response"][pid]
        parsed = r["parsed"] or []
        parsed_by_idx = {p.get("claim_index"): p for p in parsed if isinstance(p, dict)}
        for i, cl in enumerate(gt_claims):
            gt_status = cl["status"]
            p = parsed_by_idx.get(i)
            if p is None:
                unparseable_claim_level += 1
                pred_status = "MISSING"
                pred_unsafe = False
            else:
                pred_status = p.get("status", "MISSING")
                pred_unsafe = pred_status in UNSAFE_STATUSES
            row = {"id": pid, "claim_idx": i, "subject": cl["subject"], "claim_text": cl["text"],
                   "gt_status": gt_status, "predicted_status": pred_status,
                   "predicted_evidence": p.get("evidence") if p else None,
                   "predicted_reason": p.get("reason") if p else None}
            if gt_status == "AMBIGUOUS":
                ambiguous_excluded += 1
                row["eval_bucket"] = "EXCLUDED_AMBIGUOUS"
            elif gt_status in UNSAFE_STATUSES and pred_unsafe:
                tp += 1; row["eval_bucket"] = "TP"
            elif gt_status in UNSAFE_STATUSES and not pred_unsafe:
                fn += 1; row["eval_bucket"] = "FN"
            elif gt_status in SAFE_STATUSES and pred_unsafe:
                fp += 1; row["eval_bucket"] = "FP"
            else:
                tn += 1; row["eval_bucket"] = "TN"
            rows.append(row)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # self-verification bias: 既知の誤りclaimがSUPPORTED判定されたか
    self_verify_rows = [r for r in rows if r["id"] in SELF_VERIFY_CRITICAL_IDS and r["gt_status"] in UNSAFE_STATUSES]
    self_verify_failures = [r for r in self_verify_rows if r["predicted_status"] not in UNSAFE_STATUSES]

    out = {
        "purpose": "Stage E評価: 既存モデルverifierをGTの critical(34)+RAG50 sample(20)=54response分のclaimに対して評価。",
        "n_responses_evaluated": len(raw_by_id),
        "n_claims_evaluated": len(rows) - ambiguous_excluded,
        "unparseable_claim_level_count": unparseable_claim_level,
        "summary": {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "ambiguous_excluded": ambiguous_excluded,
                     "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)},
        "self_verification_bias_check": {
            "critical_ids_checked": SELF_VERIFY_CRITICAL_IDS,
            "n_known_unsafe_claims_checked": len(self_verify_rows),
            "n_self_verification_failures": len(self_verify_failures),
            "failures": self_verify_failures,
            "verdict": "RELIABLE" if not self_verify_failures else "UNRELIABLE_ON_SOME_CASES"
        },
        "rows": rows,
    }
    (REPORTS_DIR / "phase4fw_model_verifier.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "phase4fw_self_verification.json").write_text(
        json.dumps(out["self_verification_bias_check"], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"TP={tp} FP={fp} TN={tn} FN={fn} ambiguous_excluded={ambiguous_excluded}")
    print(f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    print(f"\nself-verification failures: {len(self_verify_failures)}/{len(self_verify_rows)}")
    for r in self_verify_failures:
        print(f"  [{r['id']}] subject={r['subject']!r} gt={r['gt_status']} pred={r['predicted_status']}")
    fns = [r for r in rows if r["eval_bucket"] == "FN"]
    print(f"\n=== FN (model verifierが見逃したunsafe claim) {len(fns)}件 ===")
    for r in fns:
        print(f"  [{r['id']}] subject={r['subject']!r} gt={r['gt_status']} pred={r['predicted_status']}")


if __name__ == "__main__":
    main()
