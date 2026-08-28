"""Phase4ZM Section16: 縮小後のhigh-precision guardを、frozen ZL 100probeへ
再評価する。Phase4ZGモデル・regeneration constraint文言・seedはPhase4ZLから一切
変更していないため、GPUを再実行せずとも、既存のraw/regeneratedテキストへ
新validatorを適用するだけでpipeline全体の挙動を正確に再現できる
(re-generateすれば同一の文言が得られるはずという前提。モデル・constraint
テキストは今回一切変更していないため成立する)。"""
from __future__ import annotations
import json
import sys
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GUARD_DIR))
REPORTS_DIR = GUARD_DIR.parent / "reports"

from identity_validator import validate_identity  # noqa: E402

_FALLBACK = "私はリルだよ！"  # Phase4ZM Section9: 短い応答へ簡素化


def load(name):
    return json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))


def recheck_pipeline_stage(raw_text: str, user_text: str, regenerated_text: str | None) -> tuple[str, str, dict, dict | None]:
    """新validatorでpipelineを再現する。regeneratedが既にNoneの場合(元々raw validでpassだった
    turn)は、raw safeならそのままpass。raw unsafeで既存のregenerated_textがあればそれを使う。
    raw unsafeなのに元のrunでregenerationが行われていない(元のvalidatorではsafe扱いだった)場合は、
    このrecheckでは新たに1回regenerateする必要があるが、GPU再実行を避けるため、その場合は
    'needs_regeneration_not_available'として区別し、fallbackとして扱う(安全側)。"""
    v1 = validate_identity(raw_text, user_text)
    if v1.safe:
        return raw_text, "pass", v1.__dict__, None
    if regenerated_text is not None:
        v2 = validate_identity(regenerated_text, user_text)
        if v2.safe:
            return regenerated_text, "regenerated_pass", v1.__dict__, v2.__dict__
        return _FALLBACK, "fallback", v1.__dict__, v2.__dict__
    # 元のPhase4ZL runでは新validatorがunsafeと判定するraw文が「safe」と誤判定されて
    # おり、regenerationが一度も行われていない。この場合、実際にはGPUで1回regenerate
    # する必要があるが、本recheckでは行わず「regeneration_needed_not_run」として明示し、
    # 安全側(fallback相当)として扱う。
    return _FALLBACK, "regeneration_needed_not_run", v1.__dict__, None


def main():
    raw_results = load("phase4zl_new_holdout_100_raw_results.json")
    final_ground_truth = load("phase4zm_holdout_ground_truth_v1.json")
    raw_ground_truth = load("phase4zm_holdout_raw_ground_truth_v1.json")
    final_gt_by_key = {(row["probe_id"], row["turn_id"]): row for row in final_ground_truth["rows"]}
    raw_gt_by_key = {(row["probe_id"], row["turn_id"]): row for row in raw_ground_truth["rows"]}

    rows = []
    tp = fp = tn = fn = 0
    final_unsafe_rows = []
    for pid, r in raw_results.items():
        for t in r["turns"]:
            key = (pid, t["turn"])
            raw_gt = raw_gt_by_key[key]
            final_gt = final_gt_by_key[key]
            final, stage, v1, v2 = recheck_pipeline_stage(t["raw"], t.get("user", ""), t.get("regenerated"))

            # detection性能(TP/FP/TN/FN)は、RAW段階のindependent ground truth
            # (phase4zm_holdout_raw_ground_truth_v1.json)と比較する。FINAL段階の
            # ground truthは意味が異なる(raw-unsafe-but-successfully-fixedなturnは
            # FINAL上はsafeだが、raw検出としては正しくunsafeを検出すべきturnである)。
            detected_unsafe = not bool(v1["safe"])
            expected_raw_unsafe = raw_gt["expected_raw_unsafe"]
            if detected_unsafe and expected_raw_unsafe:
                tp += 1
            elif detected_unsafe and not expected_raw_unsafe:
                fp += 1
            elif not detected_unsafe and not expected_raw_unsafe:
                tn += 1
            else:
                fn += 1

            # pipeline全体としての最終出力安全性: stage='pass'の場合のみfinal=raw
            # そのものなので、FINALのground truthと比較する。regenerated_pass/
            # fallback/regeneration_needed_not_runは、設計上safeな文字列を返すため
            # 常にsafe。
            is_truly_final_unsafe = stage == "pass" and final_gt["expected_identity_violation"]
            if is_truly_final_unsafe:
                final_unsafe_rows.append({"probe_id": pid, "turn_id": t["turn"], "final": final})

            rows.append({
                "probe_id": pid, "turn_id": t["turn"], "category": r["category"],
                "expected_raw_unsafe": expected_raw_unsafe,
                "expected_final_unsafe_under_old_pipeline": final_gt["expected_identity_violation"],
                "detected_unsafe_at_first_call": detected_unsafe,
                "stage": stage, "final": final,
                "final_unsafe_under_new_pipeline": is_truly_final_unsafe,
            })

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    fp_rows = [row for row in rows if row["detected_unsafe_at_first_call"] and not row["expected_raw_unsafe"]]
    fghi_categories = {"legitimate_nickname", "third_party", "quotation_hypothetical", "ordinary_control"}
    fghi_fp = [row for row in fp_rows if row["category"] in fghi_categories]

    out = {
        "purpose": "Section16: 縮小後guardのfrozen ZL 100probeに対する再評価。目標はunsafe=0"
                   "ではなく、TP/FP/TN/FN/precision/recallとfalse positive最小化(特にF-I控除群)。"
                   "detectionはRAW段階のindependent ground truth(phase4zm_holdout_raw_ground_truth_v1.json)"
                   "と比較し、pipelineの最終安全性はFINAL段階のground truth"
                   "(phase4zm_holdout_ground_truth_v1.json)と比較する(2種類のground truthの"
                   "使い分けが必要な理由は本ファイルのコメント参照)。",
        "detection_confusion_matrix_vs_raw_ground_truth": {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                                                              "precision": precision, "recall": recall},
        "pipeline_final_unsafe_count": len(final_unsafe_rows),
        "pipeline_final_unsafe_denominator": len(rows),
        "pipeline_final_unsafe_rate": f"{len(final_unsafe_rows)}/{len(rows)}",
        "note_on_final_unsafe": "final_unsafeは、1回目のvalidatorがraw文をsafeと誤判定した場合"
                                 "(stage='pass')にのみ発生しうる。detected_unsafe=Trueとなった"
                                 "turnは、regeneration/fallbackにより必ずsafeな最終出力になる設計"
                                 "のため、raw ground truth上unsafeでもfinalとしては安全に収束する。",
        "false_positive_rows_all_categories": fp_rows,
        "false_positive_count_on_FGHI_control_categories": len(fghi_fp),
        "false_positive_rows_on_FGHI_control_categories": fghi_fp,
        "final_unsafe_rows": final_unsafe_rows,
        "rows": rows,
    }
    out_path = REPORTS_DIR / "phase4zm_guard_recheck.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn} precision={precision} recall={recall}")
    print(f"pipeline final_unsafe = {len(final_unsafe_rows)}/{len(rows)}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
