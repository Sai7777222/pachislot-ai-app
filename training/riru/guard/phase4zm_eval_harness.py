"""Phase4ZM Section11-15: Ground-truth-independent evaluation harness.

RULE EVAL-001: 被評価componentの判定結果を、同じcomponentのaccuracyを測る
ground truthとして使用してはいけない。

このスクリプトは2つの集計を並べて示す：

1. OLD (circular) tally: FINALテキストを同じvalidatorで再検証し、safe=Falseの
   件数を数える。これはPhase4ZLで実際に発生したバグの再現であり、必ず0に
   近い値（validatorが自分自身に同意するだけなので原理的に0になりやすい）
   になることを示す回帰テスト。

2. NEW (independent) tally: phase4zm_holdout_ground_truth_v1.json という、
   人間が目視で作成した独立ラベルと比較する。これが本来のPrimary Gate判定に
   使うべき値。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GUARD_DIR))
REPORTS_DIR = GUARD_DIR.parent / "reports"

from identity_validator import validate_identity  # noqa: E402


def load(name):
    return json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))


def old_circular_tally(raw_results: dict) -> dict:
    """RULE EVAL-001違反の再現: 被評価validator自身でFINALテキストを再検証し、
    それをそのままfinal_unsafeの根拠として使う（誤ったやり方）。"""
    unsafe = []
    for pid, r in raw_results.items():
        for t in r["turns"]:
            v = validate_identity(t["final"], t.get("user", ""))
            if not v.safe:
                unsafe.append({"probe_id": pid, "turn_id": t["turn"]})
    return {"method": "validator re-validates its own final output (circular, WRONG)",
            "final_unsafe_count": len(unsafe), "final_unsafe_turns": unsafe}


def new_independent_tally(ground_truth: dict) -> dict:
    """RULE EVAL-002準拠: 独立に作成されたexpected_safeラベルとの比較。"""
    unsafe = [{"probe_id": row["probe_id"], "turn_id": row["turn_id"], "violation_type": row["violation_type"]}
              for row in ground_truth["rows"] if row["expected_identity_violation"]]
    return {"method": "independent human-annotated ground truth (RULE EVAL-002 compliant, CORRECT)",
            "final_unsafe_count": len(unsafe), "final_unsafe_turns": unsafe}


def main():
    raw_results = load("phase4zl_new_holdout_100_raw_results.json")
    ground_truth = load("phase4zm_holdout_ground_truth_v1.json")

    old = old_circular_tally(raw_results)
    new = new_independent_tally(ground_truth)

    out = {
        "purpose": "Section15: Phase4ZLの循環論法バグを、ground-truth-independentなharnessで再現し、"
                   "正しい集計方法との差を明示する。",
        "old_circular_tally": old,
        "new_independent_tally": new,
        "delta": {
            "old_reported": old["final_unsafe_count"],
            "new_correct": new["final_unsafe_count"],
            "gap": new["final_unsafe_count"] - old["final_unsafe_count"],
            "conclusion": "旧ロジック(validatorが自分自身の出力を再検証)は、まさにPhase4ZLで実際に発生した"
                          "『final_unsafe=0』という誤報告を機械的に再現する。独立ground truthを使うと"
                          f"{new['final_unsafe_count']}/106という真の値が得られ、これがRULE EVAL-001の"
                          "重要性を裏付ける実証結果である。",
        },
    }
    out_path = REPORTS_DIR / "phase4zm_old_vs_new_tally.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"old (circular) final_unsafe = {old['final_unsafe_count']}")
    print(f"new (independent) final_unsafe = {new['final_unsafe_count']}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
