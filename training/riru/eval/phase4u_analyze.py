"""Phase 4U: ratio_high_identity 結果の集約分析。

Phase4S/4Tの既存v4/ratio-high結果を読み取り専用で参照しつつ、新規生成した
identity結果を同一の分類・採点ロジックで評価し、3-way比較を行う。
identity intrusion (聞かれていないのに毎回自己紹介するような過剰適合) も検査する。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EVAL_DIR.parents[0] / "reports"

sys.path.insert(0, str(EVAL_DIR))
from phase4t_probes import P04_PROBES  # noqa: E402
from phase4u_reclassify_naming import classify_generation  # noqa: E402

Q9_CALC_PATTERN = re.compile(r"約\s*\d+(\.\d+)?\s*(倍|ポイント)")
Q11_YAMEDOKI_PATTERN = re.compile(r"ヤメ時|一旦ヤメ|止めるのが|ヤメる")
Q11_STRATEGY_PATTERN = re.compile(r"おすすめ|べきです|べきだ|戦略|コツ")
Q11_LOOPSTOCK_CAUSAL_PATTERN = re.compile(r"ループストック.{0,15}(ほど|により|によって)")
Q11_OTHER_CAUSAL_PATTERN = re.compile(r"(可能性が高くなり|なりやすく|傾向にあり).{0,10}(ため|から)")
Q3_KEY_FACTS = ["510G", "1000G", "1480G", "15.2%", "20.3%", "64.5%"]


def normalize(t: str) -> str:
    return t.replace("％", "%").replace(",", "").replace("ゲーム", "G")


def score_p04_text(
    text: str, required: list[str], optional: list[str], irrelevant: list[str]
) -> dict:
    text_n = normalize(text)
    req_found = [f for f in required if f in text or normalize(f) in text_n]
    return {
        "required_recall_pct": round(len(req_found) / len(required) * 100, 1) if required else None,
        "direct_answer_correct": len(req_found) == len(required),
    }


def q3_recall(text: str) -> float:
    found = [k for k in Q3_KEY_FACTS if k in text]
    return round(len(found) / len(Q3_KEY_FACTS) * 100, 1)


def analyze_naming(results: dict) -> dict:
    counts = {c: 0 for c in "ABCDEFG"}
    total = 0
    for pid, rec in results["naming_probes_identity"].items():
        texts = [rec["greedy"]] + list(rec["sampled"].values())
        for t in texts:
            cat = classify_generation(t)
            counts[cat] += 1
            total += 1
    return {
        "total": total,
        "counts": counts,
        "rates_pct": {k: round(100 * v / total, 1) for k, v in counts.items()},
        "genuine_wrong_name_rate_pct": round(100 * counts["A"] / total, 1),
        "correct_name_rate_pct": round(100 * counts["E"] / total, 1),
        "no_name_rate_pct": round(100 * counts["B"] / total, 1),
        "generic_role_only_rate_pct": round(100 * counts["D"] / total, 1),
        "placeholder_rate_pct": round(100 * counts["C"] / total, 1),
    }


def analyze_e36_e02(results: dict) -> dict:
    out = {}
    for key in ("e36_extended", "e02_extended"):
        out[key] = {}
        for cond, seeds in results[key].items():
            counts = {c: 0 for c in "ABCDEFG"}
            for t in seeds.values():
                counts[classify_generation(t)] += 1
            total = len(seeds)
            out[key][cond] = {
                "total": total,
                "counts": counts,
                "genuine_wrong_name_rate_pct": round(100 * counts["A"] / total, 1),
                "correct_name_rate_pct": round(100 * counts["E"] / total, 1),
                "placeholder_rate_pct": round(100 * counts["C"] / total, 1),
            }
    return out


def analyze_p04_type(results: dict) -> dict:
    probe_by_id = {p["id"]: p for p in P04_PROBES}
    per_probe = {}
    all_recalls = []
    for pid, rec in results["p04_type_probes_identity"].items():
        probe = probe_by_id[pid]
        scores = [
            score_p04_text(
                t, probe["required_facts"], probe["optional_facts"], probe["irrelevant_facts"]
            )
            for t in rec["sampled"].values()
        ]
        avg_recall = round(sum(s["required_recall_pct"] for s in scores) / len(scores), 1)
        per_probe[pid] = {"category": probe["category"], "avg_required_recall_pct": avg_recall}
        all_recalls.append(avg_recall)
    return {
        "per_probe": per_probe,
        "overall_mean_required_recall_pct": round(sum(all_recalls) / len(all_recalls), 1),
    }


def analyze_regression(results: dict) -> dict:
    cond = "D_identity"
    q3_sampled_recalls = [q3_recall(t) for t in results["q3_sampled"][cond].values()]
    q3_greedy_recall = q3_recall(results["q3_greedy"][cond])

    holdout_path = EVAL_DIR / "phase4i_holdout_omission_v2.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    p_items = {r["id"]: r for r in holdout}

    def holdout_recall(pid_key: str, holdout_id: str) -> float:
        item = p_items[holdout_id]
        texts = results[pid_key][cond].values()
        recalls = []
        for t in texts:
            found = [f for f in item["key_facts"] if f in t]
            recalls.append(round(len(found) / len(item["key_facts"]) * 100, 1))
        return round(sum(recalls) / len(recalls), 1)

    p04_required_only = [
        score_p04_text(t, ["16.7%"], ["96.8%", "113.5%"], [])
        for t in results["p04"][cond].values()
    ]
    p04_required_recall = round(
        sum(s["required_recall_pct"] for s in p04_required_only) / len(p04_required_only), 1
    )

    q9_halluc = sum(1 for t in results["q9"][cond].values() if Q9_CALC_PATTERN.search(t))
    q11_texts = list(results["q11"][cond].values())
    q11_yamedoki = sum(1 for t in q11_texts if Q11_YAMEDOKI_PATTERN.search(t))
    q11_strategy = sum(1 for t in q11_texts if Q11_STRATEGY_PATTERN.search(t))
    q11_loopstock = sum(1 for t in q11_texts if Q11_LOOPSTOCK_CAUSAL_PATTERN.search(t))
    q11_other = sum(1 for t in q11_texts if Q11_OTHER_CAUSAL_PATTERN.search(t))

    structured_lens = [
        v["length"] if "length" in v else len(v["text"])
        for v in results["structured_17q"].values()
    ]
    char39_lens = [len(v["text"]) for v in results["character_39"].values()]

    # identity intrusion: 名前を聞いていない質問(structured17 + Q3/P01/P02/P04/Q9/Q11)で
    # 「リル」という自己紹介的な語が不要に出現していないか
    non_naming_texts = (
        [results["q3_greedy"][cond]] + list(results["q3_sampled"][cond].values())
        + [t for pid in ("p01", "p02", "p04") for t in results[pid][cond].values()]
        + list(results["q9"][cond].values()) + list(results["q11"][cond].values())
        + [v["text"] for v in results["structured_17q"].values()]
    )
    intrusion_count = sum(1 for t in non_naming_texts if "リル" in t)

    return {
        "q3_greedy_recall_pct": q3_greedy_recall,
        "q3_sampled_avg_recall_pct": round(sum(q3_sampled_recalls) / len(q3_sampled_recalls), 1),
        "q3_sampled_min": min(q3_sampled_recalls),
        "q3_sampled_max": max(q3_sampled_recalls),
        "p01_context_all_fact_recall_pct": holdout_recall("p01", "P01"),
        "p02_context_all_fact_recall_pct": holdout_recall("p02", "P02"),
        "p04_context_all_fact_recall_pct_OLD_METRIC": holdout_recall("p04", "P04"),
        "p04_required_fact_recall_pct_NEW_METRIC": p04_required_recall,
        "q9_calc_hallucination_seeds": q9_halluc,
        "q11_yamedoki_seeds": q11_yamedoki,
        "q11_strategy_seeds": q11_strategy,
        "q11_loopstock_causal_seeds": q11_loopstock,
        "q11_other_causal_seeds": q11_other,
        "avg_structured17_len_chars": round(sum(structured_lens) / len(structured_lens), 1),
        "avg_character39_len_chars": round(sum(char39_lens) / len(char39_lens), 1),
        "identity_intrusion_count_out_of": {
            "count": intrusion_count, "total_non_naming_generations": len(non_naming_texts),
        },
    }


def main() -> int:
    results = json.loads(
        (EVAL_DIR / "phase4u_comprehensive_results.json").read_text(encoding="utf-8")
    )

    naming = analyze_naming(results)
    e36_e02 = analyze_e36_e02(results)
    p04_type = analyze_p04_type(results)
    regression = analyze_regression(results)

    # Phase4T/4S既存v4/high参照値 (読み取り専用で再利用)
    phase4t_naming = json.loads(
        (REPORTS_DIR / "phase4u_naming_reclassification.json").read_text(encoding="utf-8")
    )["summary"]
    phase4t_p04 = json.loads(
        (REPORTS_DIR / "phase4t_p04_analysis.json").read_text(encoding="utf-8")
    )["overall"]

    out = {
        "identity_naming": naming,
        "reference_v4_high_naming": phase4t_naming,
        "e36_e02_extended": e36_e02,
        "identity_p04_type_probes": p04_type,
        "reference_v4_high_base_p04_type_overall": phase4t_p04,
        "identity_regression": regression,
    }
    (REPORTS_DIR / "phase4u_evaluation_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
