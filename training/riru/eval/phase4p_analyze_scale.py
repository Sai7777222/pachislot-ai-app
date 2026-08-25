"""Phase 4P: o_proj-only scale sweep結果の集約分析。

各scaleについてQ3/P01/P02/P04/Q9/Q11/E36の主要指標をまとめ、
sweet spot判定 (17節の基準A〜K) を機械的に評価する。
"""

from __future__ import annotations

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EVAL_DIR.parents[0] / "reports"

# Phase4O reference (read-only, not overwritten)
PHASE4O_RESULTS = EVAL_DIR / "phase4o_comprehensive_results.json"


def main() -> int:
    data = json.loads((EVAL_DIR / "phase4p_oproj_scale_results.json").read_text(encoding="utf-8"))
    by_scale = data["by_scale"]

    criteria_rows = {}
    summary_rows = {}
    for scale_str, rec in by_scale.items():
        q3_avg = rec["q3_sampled_avg"]
        all3_game = rec["q3_all3_gamecount_seeds"]
        all3_pct = rec["q3_all3_pct_seeds"]
        p01 = rec["p01"]["avg_recall"]
        p02 = rec["p02"]["avg_recall"]
        p04 = rec["p04"]["avg_recall"]
        q9_halluc = rec["q9_calc_hallucination_seeds"]
        q11_yamedoki = rec["q11_yamedoki_seeds"]
        q11_strategy = rec["q11_strategy_seeds"]
        q11_causal = rec["q11_causal_fabrication_seeds"]
        e36_wrong = rec["e36_wrong_name_seeds"]
        e36_placeholder = rec["e36_placeholder_seeds"]

        persona_lens = [len(v["text"]) for v in rec["persona_extra"].values()]
        persona_lens.append(rec["e36"]["42"]["length"])
        avg_persona_len = round(sum(persona_lens) / len(persona_lens), 1)

        summary_rows[scale_str] = {
            "q3_greedy_recall": rec["q3_greedy"]["recall_pct"],
            "q3_sampled_avg": q3_avg,
            "q3_sampled_min": rec["q3_sampled_min"],
            "q3_sampled_max": rec["q3_sampled_max"],
            "q3_all3_gamecount_seeds": all3_game,
            "q3_all3_pct_seeds": all3_pct,
            "q3_any_pct_only_seeds": rec["q3_any_pct_only_seeds"],
            "q3_ceiling_reach_seeds": rec["q3_ceiling_reach_seeds"],
            "q3_loopstock_seeds": rec["q3_loopstock_seeds"],
            "p01_avg_recall": p01,
            "p02_avg_recall": p02,
            "p04_avg_recall": p04,
            "q9_calc_hallucination_seeds": q9_halluc,
            "q11_yamedoki_seeds": q11_yamedoki,
            "q11_strategy_seeds": q11_strategy,
            "q11_causal_fabrication_seeds": q11_causal,
            "e36_wrong_name_seeds": e36_wrong,
            "e36_placeholder_seeds": e36_placeholder,
            "e36_correct_name_seeds": rec["e36_correct_name_seeds"],
            "avg_persona_len_chars": avg_persona_len,
            "q3_length_greedy": rec["q3_greedy"]["length"],
        }

        # --- 17節 sweet spot 基準 A-K ---
        crit = {
            "A_q3_recall_ge80": q3_avg >= 80.0,
            "B_all3_gamecount_5of5": all3_game == 5,
            "C_pct_ge4of5": all3_pct >= 4,
            "D_p01p02_not_worse_than_v4": p01 >= 50.0 and p02 >= 40.0,
            "E_p04_not_much_worse": p04 >= 66.7,
            "F_q9_clean": q9_halluc == 0,
            "G_q11_advice_clean": (q11_yamedoki == 0 and q11_strategy == 0),
            "H_q11_causal_clean": q11_causal == 0,
            "I_e36_no_wrong_name": e36_wrong == 0,
            "J_e36_no_placeholder": e36_placeholder == 0,
        }
        crit["all_pass"] = all(crit.values())
        criteria_rows[scale_str] = crit

    result = {"summary": summary_rows, "sweet_spot_criteria": criteria_rows}
    sweet_spots = [s for s, c in criteria_rows.items() if c["all_pass"]]
    result["sweet_spot_candidates"] = sweet_spots

    # Pareto-style single-axis "best" picks (for when no full sweet spot exists)
    scales_sorted_by_recall = sorted(
        summary_rows.items(), key=lambda kv: kv[1]["q3_sampled_avg"], reverse=True
    )
    scales_sorted_by_halluc = sorted(
        summary_rows.items(),
        key=lambda kv: (
            kv[1]["q9_calc_hallucination_seeds"]
            + kv[1]["q11_yamedoki_seeds"]
            + kv[1]["q11_strategy_seeds"]
            + kv[1]["q11_causal_fabrication_seeds"]
        ),
    )
    scales_sorted_by_persona = sorted(
        summary_rows.items(),
        key=lambda kv: (kv[1]["e36_wrong_name_seeds"] + kv[1]["e36_placeholder_seeds"]),
    )
    result["pareto_picks"] = {
        "recall_first": scales_sorted_by_recall[0][0],
        "hallucination_min": scales_sorted_by_halluc[0][0],
        "persona_first": scales_sorted_by_persona[0][0],
    }

    out_path = REPORTS_DIR / "phase4p_oproj_scale_analysis.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(f"sweet_spot_candidates: {sweet_spots}")
    print(json.dumps(result["pareto_picks"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
