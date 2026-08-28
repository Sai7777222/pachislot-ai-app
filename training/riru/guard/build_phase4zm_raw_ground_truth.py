"""Phase4ZM補助asset: RAW段階(guard適用前のPhase4ZG生成そのもの)の安全性に
関する独立ground truth。phase4zm_holdout_ground_truth_v1.json(FINAL段階の
ground truth)とは目的が異なる: guardのdetection性能(1回目のvalidator呼び出し
がraw文をunsafeと正しく判定できるか)を測るには、raw自体が本当にunsafeだった
かどうかのラベルが別途必要なため、[[phase4zl_detection_metrics]]の手動分析結果
(39件の真にunsafeなraw、うち3件はStage C/Eで確認済みの旧validator false positive)
を独立assetとして固定する。"""
from __future__ import annotations
import json
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

FINAL_GT = json.loads((REPORTS_DIR / "phase4zm_holdout_ground_truth_v1.json").read_text(encoding="utf-8"))
RAW_RESULTS = json.loads((REPORTS_DIR / "phase4zl_new_holdout_100_raw_results.json").read_text(encoding="utf-8"))

KNOWN_OLD_VALIDATOR_FALSE_POSITIVES = {("ZL-G02", 1), ("ZL-H02", 1), ("ZL-I07", 1)}


def main():
    final_unsafe = {(r["probe_id"], r["turn_id"]) for r in FINAL_GT["rows"] if r["expected_identity_violation"]}
    raw_unsafe = set(final_unsafe)
    for pid, r in RAW_RESULTS.items():
        for t in r["turns"]:
            key = (pid, t["turn"])
            if t["stage"] in ("regenerated_pass", "fallback") and key not in KNOWN_OLD_VALIDATOR_FALSE_POSITIVES:
                raw_unsafe.add(key)

    rows = []
    for pid, r in RAW_RESULTS.items():
        for t in r["turns"]:
            key = (pid, t["turn"])
            rows.append({
                "probe_id": pid, "turn_id": t["turn"], "category": r["category"],
                "expected_raw_unsafe": key in raw_unsafe,
                "annotation_source": "derived_from_phase4zl_detection_metrics_manual_analysis_2026-08-28",
                "frozen": True,
            })
    out = {
        "purpose": "RAW(guard適用前)段階のindependent ground truth。detection(TP/FP/TN/FN)の"
                   "算出にはこちらを使う。FINAL(guard適用後)段階のground truthは"
                   "phase4zm_holdout_ground_truth_v1.jsonを使う。",
        "expected_raw_unsafe_count": len(raw_unsafe),
        "denominator": len(rows),
        "rows": rows,
    }
    out_path = REPORTS_DIR / "phase4zm_holdout_raw_ground_truth_v1.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"expected_raw_unsafe = {len(raw_unsafe)}/{len(rows)}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
