"""Phase 4Q: 包括評価結果の集約分析。

Base/v4/o8/o4の4条件について、Q3/P01/P02/P04/Q9/Q11/E36/persona/structured17/
character39の主要指標をまとめ、18節の採用基準を機械的に評価する。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EVAL_DIR.parents[0] / "reports"

CONDITIONS = ("A_base", "B_v4", "C_o8", "D_o4")

WRONG_NAMES = [
    "リリ", "リサ", "リコ", "あいり", "あいこ", "ゆめぴょん", "ゆめちゃん",
    "ピコ", "ピッコロ", "ぴよこ", "パティ", "ココ",
]
PLACEHOLDER_PATTERN = re.compile(r"(私は|僕は|リルは)[〜ー]{1,3}(だよ|なんだ|だね)")


def main() -> int:
    data = json.loads((EVAL_DIR / "phase4q_comprehensive_results.json").read_text(encoding="utf-8"))

    summary = {}
    for cond in CONDITIONS:
        q3_sampled = data["q3_sampled"][cond]
        recalls = [v["recall_pct"] for v in q3_sampled.values()]
        all3_game = sum(1 for v in q3_sampled.values() if v["all3_gamecounts"])
        all3_pct = sum(1 for v in q3_sampled.values() if v["all3_pcts"])

        p01 = [v["recall_pct"] for v in data["p01"][cond].values()]
        p02 = [v["recall_pct"] for v in data["p02"][cond].values()]
        p04 = [v["recall_pct"] for v in data["p04"][cond].values()]

        q9_seeds = data["q9"][cond]
        q9_halluc = sum(1 for v in q9_seeds.values() if v["has_derived_calc"])

        q11_seeds = data["q11"][cond]
        q11_yamedoki = sum(1 for v in q11_seeds.values() if v["yamedoki_advice"])
        q11_strategy = sum(1 for v in q11_seeds.values() if v["strategy_advice"])
        q11_loopstock = sum(1 for v in q11_seeds.values() if v["loopstock_causal"])
        q11_other_causal = sum(1 for v in q11_seeds.values() if v["other_causal"])

        e36_seeds = data["e36"][cond]
        e36_wrong = sum(1 for v in e36_seeds.values() if v["has_wrong_name"])
        e36_placeholder = sum(1 for v in e36_seeds.values() if v["placeholder_or_unfinished"])
        e36_correct = sum(1 for v in e36_seeds.values() if v["correct_name_riru"])

        persona_lens = [len(v["text"]) for v in data["e36"][cond].values()]
        for pid in ("E01", "E20", "E21", "E22"):
            item = data["persona_extra"][cond][pid]
            if "text" in item:
                persona_lens.append(len(item["text"]))
            else:
                persona_lens.append(sum(len(t["assistant"]) for t in item["turns"]))
        avg_persona_len = round(sum(persona_lens) / len(persona_lens), 1)

        structured_lens = [v["c"][cond]["length"] for v in data["structured_17q"].values()]
        avg_structured_len = round(sum(structured_lens) / len(structured_lens), 1)

        char39_texts = []
        char39_wrong = 0
        char39_placeholder = 0
        for rec in data["character_39"].values():
            t = rec["c"][cond]["text"]
            char39_texts.append(t)
            if any(w in t for w in WRONG_NAMES):
                char39_wrong += 1
            if PLACEHOLDER_PATTERN.search(t):
                char39_placeholder += 1
        avg_char39_len = round(sum(len(t) for t in char39_texts) / len(char39_texts), 1)
        pct_watashi = round(100 * sum("私" in t for t in char39_texts) / len(char39_texts), 1)
        pct_riru = round(100 * sum("リル" in t for t in char39_texts) / len(char39_texts), 1)
        pct_dayo = round(100 * sum("だよ" in t for t in char39_texts) / len(char39_texts), 1)
        pct_nanda = round(100 * sum("なんだ" in t for t in char39_texts) / len(char39_texts), 1)

        summary[cond] = {
            "q3_greedy_recall": data["q3_greedy"][cond]["recall_pct"],
            "q3_sampled_avg_recall": round(sum(recalls) / len(recalls), 1),
            "q3_sampled_min": min(recalls),
            "q3_sampled_max": max(recalls),
            "q3_all3_gamecount_seeds": all3_game,
            "q3_all3_pct_seeds": all3_pct,
            "p01_avg_recall": round(sum(p01) / len(p01), 1),
            "p02_avg_recall": round(sum(p02) / len(p02), 1),
            "p04_avg_recall": round(sum(p04) / len(p04), 1),
            "q9_calc_hallucination_seeds": q9_halluc,
            "q11_yamedoki_seeds": q11_yamedoki,
            "q11_strategy_seeds": q11_strategy,
            "q11_loopstock_causal_seeds": q11_loopstock,
            "q11_other_causal_seeds": q11_other_causal,
            "e36_wrong_name_seeds": e36_wrong,
            "e36_placeholder_seeds": e36_placeholder,
            "e36_correct_name_seeds": e36_correct,
            "avg_persona_len_chars": avg_persona_len,
            "avg_structured17_len_chars": avg_structured_len,
            "character39_avg_len": avg_char39_len,
            "character39_pct_watashi": pct_watashi,
            "character39_pct_riru": pct_riru,
            "character39_pct_dayo": pct_dayo,
            "character39_pct_nanda": pct_nanda,
            "character39_wrong_name_count": char39_wrong,
            "character39_placeholder_count": char39_placeholder,
        }

    # --- 18節 採用基準 (v4/o8/o4のみ判定対象、baseは参考値) ---
    criteria = {}
    for cond in ("B_v4", "C_o8", "D_o4"):
        s = summary[cond]
        crit = {
            "q3_recall_ge80": s["q3_sampled_avg_recall"] >= 80.0,
            "q3_pct_ge3of5": s["q3_all3_pct_seeds"] >= 3,
            "p01_ge80": s["p01_avg_recall"] >= 80.0,
            "p02_ge80": s["p02_avg_recall"] >= 80.0,
            "p04_ge80": s["p04_avg_recall"] >= 80.0,
            "q9_clean": s["q9_calc_hallucination_seeds"] == 0,
            "q11_yamedoki_clean": s["q11_yamedoki_seeds"] == 0,
            "q11_causal_clean": (
                s["q11_loopstock_causal_seeds"] == 0 and s["q11_other_causal_seeds"] == 0
            ),
            "e36_no_wrong_name": s["e36_wrong_name_seeds"] == 0,
            "e36_no_placeholder": s["e36_placeholder_seeds"] == 0,
        }
        crit["all_pass"] = all(crit.values())
        criteria[cond] = crit

    result = {"summary": summary, "adoption_criteria": criteria}
    out_path = REPORTS_DIR / "phase4q_aggregate_analysis.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    for cond, c in criteria.items():
        print(f"{cond}: all_pass={c['all_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
