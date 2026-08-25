"""Phase 4N: scale sweep結果の集約分析 (v4/v2)。

各scaleについて:
  - Q3 greedy: 各key factの有無
  - Q3 sampled: 5 seed平均recall、%出現seed数、game count(510G/1000G/1480G)出現seed数
  - persona screen 14問: E36のプレースホルダー再現有無、簡易文字数統計
を1つのJSON/テキストにまとめる。E36含むJapaneseテキストはUTF-8ファイルへ出力し、
Bashコンソールへは印字しない (cp932文字化け回避)。
"""

from __future__ import annotations

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent

INCOMPLETE_PREDICATE_MARKERS = ["私は〜〜だよ", "私は〜、", "僕は〜〜", "リルは〜〜"]


def analyze_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for scale, rec in data.items():
        greedy = rec["q3_greedy"]
        sampled = rec["q3_sampled"]
        n_seeds = len(sampled)
        pct_hits = sum(
            1 for v in sampled.values() if v["has_15.2pct"] or v["has_20.3pct"] or v["has_64.5pct"]
        )
        gamecount_hits = sum(
            1 for v in sampled.values() if v["has_510G"] or v["has_1000G"] or v["has_1480G"]
        )
        all3_gamecount_hits = sum(
            1 for v in sampled.values() if v["has_510G"] and v["has_1000G"] and v["has_1480G"]
        )
        avg_recall = rec["q3_avg_recall_sampled"]
        lengths = [v["length"] for v in sampled.values()]

        persona = rec["persona_screen"]
        e36_text = persona.get("E36", {}).get("text", "")
        e36_placeholder = any(m in e36_text for m in INCOMPLETE_PREDICATE_MARKERS)
        persona_lengths = {k: v.get("length", len(v["text"])) for k, v in persona.items()}
        avg_persona_len = sum(len(v["text"]) for v in persona.values()) / len(persona)

        out[scale] = {
            "n_lora_layers_matched": rec["n_lora_layers_matched"],
            "q3_greedy_recall_pct": greedy["recall_pct"],
            "q3_greedy_key_facts_found": greedy["key_facts_found"],
            "q3_sampled_avg_recall_pct": avg_recall,
            "q3_sampled_seeds_with_any_pct": f"{pct_hits}/{n_seeds}",
            "q3_sampled_seeds_with_any_gamecount": f"{gamecount_hits}/{n_seeds}",
            "q3_sampled_seeds_with_all3_gamecounts": f"{all3_gamecount_hits}/{n_seeds}",
            "q3_sampled_length_min_max": [min(lengths), max(lengths)],
            "q3_top_logit_token_1": rec["q3_first_token_top_logits"]["top_tokens"][0],
            "q3_top_logit_prob_1": rec["q3_first_token_top_logits"]["top_probs"][0],
            "e36_placeholder_present": e36_placeholder,
            "avg_persona_response_length": round(avg_persona_len, 1),
            "persona_lengths_by_id": persona_lengths,
        }
    return out


def main() -> int:
    v4_summary = analyze_file(EVAL_DIR / "phase4n_scale_results_v4.json")
    v2_summary = analyze_file(EVAL_DIR / "phase4n_scale_results_v2.json")

    combined = {"v4": v4_summary, "v2": v2_summary}
    out_path = EVAL_DIR / "phase4n_scale_summary.json"
    out_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")

    # UTF-8 human-readable text dump for review (avoid cp932 console issues)
    lines = []
    for adapter_name, summary in combined.items():
        lines.append(f"=== {adapter_name} scale sweep ===")
        for scale, s in summary.items():
            lines.append(
                f"scale={scale}: greedy_recall={s['q3_greedy_recall_pct']}% "
                f"facts={s['q3_greedy_key_facts_found']} | "
                f"sampled_avg_recall={s['q3_sampled_avg_recall_pct']}% "
                f"pct_seeds={s['q3_sampled_seeds_with_any_pct']} "
                f"gamecount_seeds={s['q3_sampled_seeds_with_any_gamecount']} "
                f"all3_seeds={s['q3_sampled_seeds_with_all3_gamecounts']} | "
                f"top1_tok='{s['q3_top_logit_token_1']}'({s['q3_top_logit_prob_1']}) | "
                f"E36_placeholder={s['e36_placeholder_present']} "
                f"avg_persona_len={s['avg_persona_response_length']}"
            )
        lines.append("")
    text_path = EVAL_DIR.parents[0] / "reports" / "_phase4n_scale_summary_utf8.txt"
    text_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved -> {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
