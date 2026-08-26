"""Phase 4T: P04型probe・naming probe・E36拡張結果の集約分析。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EVAL_DIR.parents[0] / "reports"

sys.path.insert(0, str(EVAL_DIR))
from phase4t_probes import NAMING_PROBES, P04_PROBES  # noqa: E402
from phase4t_wrongname_detector import classify_naming  # noqa: E402

P04_CONDITIONS = ("A_base", "B_v4", "C_high")
NAMING_CONDITIONS = ("B_v4", "C_high")


def normalize(t: str) -> str:
    return t.replace("％", "%").replace(",", "").replace("ゲーム", "G")


def score_p04_text(
    text: str, required: list[str], optional: list[str], irrelevant: list[str]
) -> dict:
    text_n = normalize(text)
    req_found = [f for f in required if f in text or normalize(f) in text_n]
    opt_found = [f for f in optional if f in text or normalize(f) in text_n]
    irr_found = [f for f in irrelevant if f in text]
    return {
        "required_found": req_found,
        "required_recall_pct": round(len(req_found) / len(required) * 100, 1) if required else None,
        "optional_found": opt_found,
        "optional_inclusion_pct": (
            round(len(opt_found) / len(optional) * 100, 1) if optional else None
        ),
        "irrelevant_found": irr_found,
        "direct_answer_correct": len(req_found) == len(required),
    }


def analyze_p04(results: dict) -> dict:
    probe_by_id = {p["id"]: p for p in P04_PROBES}
    per_probe = {}
    by_category: dict[str, dict[str, list]] = {}

    for pid, rec in results["p04_probes"].items():
        probe = probe_by_id[pid]
        cat = probe["category"]
        per_probe[pid] = {"category": cat, "question": probe["question"], "conditions": {}}
        for cond in P04_CONDITIONS:
            cond_data = rec["conditions"][cond]
            greedy_score = score_p04_text(
                cond_data["greedy"], probe["required_facts"], probe["optional_facts"],
                probe["irrelevant_facts"],
            )
            sampled_scores = {
                seed: score_p04_text(
                    text, probe["required_facts"], probe["optional_facts"],
                    probe["irrelevant_facts"],
                )
                for seed, text in cond_data["sampled"].items()
            }
            n_seeds = len(sampled_scores)
            avg_req_recall = round(
                sum(s["required_recall_pct"] for s in sampled_scores.values()) / n_seeds, 1
            )
            direct_correct_rate = round(
                100 * sum(s["direct_answer_correct"] for s in sampled_scores.values())
                / len(sampled_scores), 1
            )
            opt_vals = [
                s["optional_inclusion_pct"]
                for s in sampled_scores.values()
                if s["optional_inclusion_pct"] is not None
            ]
            avg_opt_inclusion = round(sum(opt_vals) / len(opt_vals), 1) if opt_vals else None
            irr_leak_rate = round(
                100 * sum(1 for s in sampled_scores.values() if s["irrelevant_found"])
                / len(sampled_scores), 1
            )
            per_probe[pid]["conditions"][cond] = {
                "greedy": greedy_score,
                "sampled_avg_required_recall_pct": avg_req_recall,
                "sampled_direct_answer_correct_rate_pct": direct_correct_rate,
                "sampled_avg_optional_inclusion_pct": avg_opt_inclusion,
                "sampled_irrelevant_leak_rate_pct": irr_leak_rate,
            }
            by_category.setdefault(cat, {}).setdefault(cond, []).append(
                {
                    "required_recall": avg_req_recall,
                    "direct_correct": direct_correct_rate,
                    "optional_inclusion": avg_opt_inclusion,
                }
            )

    category_summary = {}
    for cat, cond_map in by_category.items():
        category_summary[cat] = {}
        for cond, vals in cond_map.items():
            rr = [v["required_recall"] for v in vals]
            dc = [v["direct_correct"] for v in vals]
            oi = [v["optional_inclusion"] for v in vals if v["optional_inclusion"] is not None]
            category_summary[cat][cond] = {
                "n_probes": len(vals),
                "mean_required_recall_pct": round(sum(rr) / len(rr), 1),
                "mean_direct_answer_correct_pct": round(sum(dc) / len(dc), 1),
                "mean_optional_inclusion_pct": round(sum(oi) / len(oi), 1) if oi else None,
            }

    overall = {}
    for cond in P04_CONDITIONS:
        rr = [
            per_probe[pid]["conditions"][cond]["sampled_avg_required_recall_pct"]
            for pid in per_probe
        ]
        dc = [
            per_probe[pid]["conditions"][cond]["sampled_direct_answer_correct_rate_pct"]
            for pid in per_probe
        ]
        overall[cond] = {
            "mean_required_recall_pct": round(sum(rr) / len(rr), 1),
            "mean_direct_answer_correct_pct": round(sum(dc) / len(dc), 1),
        }

    return {"per_probe": per_probe, "by_category": category_summary, "overall": overall}


def analyze_naming(results: dict) -> dict:
    per_probe = {}
    wrong_flags = {c: 0 for c in NAMING_CONDITIONS}
    total_gens = {c: 0 for c in NAMING_CONDITIONS}
    correct_name_count = {c: 0 for c in NAMING_CONDITIONS}
    placeholder_count = {c: 0 for c in NAMING_CONDITIONS}
    review_candidates_all: dict[str, list] = {c: [] for c in NAMING_CONDITIONS}

    for probe in NAMING_PROBES:
        pid = probe["id"]
        rec = results["naming_probes"][pid]
        per_probe[pid] = {"prompt": probe["prompt"], "conditions": {}}
        for cond in NAMING_CONDITIONS:
            cond_data = rec["conditions"][cond]
            all_texts = [cond_data["greedy"]] + list(cond_data["sampled"].values())
            classified = [classify_naming(t) for t in all_texts]
            for c in classified:
                total_gens[cond] += 1
                if c["correct_name_used"]:
                    correct_name_count[cond] += 1
                if c["has_review_required"]:
                    wrong_flags[cond] += 1
                    for cand in c["review_required_candidates"]:
                        review_candidates_all[cond].append(
                            {"probe": pid, "candidate": cand, "text": c["text"]}
                        )
                if c["placeholder_or_unfinished"]:
                    placeholder_count[cond] += 1
            per_probe[pid]["conditions"][cond] = {
                "n_gens": len(classified),
                "n_correct_name": sum(c["correct_name_used"] for c in classified),
                "n_review_required": sum(c["has_review_required"] for c in classified),
                "n_placeholder": sum(c["placeholder_or_unfinished"] for c in classified),
            }

    summary = {}
    for cond in NAMING_CONDITIONS:
        summary[cond] = {
            "total_generations": total_gens[cond],
            "correct_name_rate_pct": round(100 * correct_name_count[cond] / total_gens[cond], 1),
            "review_required_rate_pct": round(100 * wrong_flags[cond] / total_gens[cond], 1),
            "placeholder_rate_pct": round(100 * placeholder_count[cond] / total_gens[cond], 1),
            "n_review_required_instances": wrong_flags[cond],
        }

    return {
        "per_probe": per_probe, "summary": summary,
        "review_candidates": review_candidates_all,
    }


def analyze_e36_extended(results: dict) -> dict:
    out = {}
    for cond in NAMING_CONDITIONS:
        texts = list(results["e36_extended"][cond].values())
        classified = [classify_naming(t) for t in texts]
        out[cond] = {
            "n": len(texts),
            "n_correct_name": sum(c["correct_name_used"] for c in classified),
            "n_review_required": sum(c["has_review_required"] for c in classified),
            "n_placeholder": sum(c["placeholder_or_unfinished"] for c in classified),
            "review_required_details": [
                {"text": c["text"], "candidates": c["review_required_candidates"]}
                for c in classified
                if c["has_review_required"]
            ],
        }
    return out


def main() -> int:
    results_path = EVAL_DIR / "phase4t_comprehensive_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))

    p04_analysis = analyze_p04(results)
    naming_analysis = analyze_naming(results)
    e36_analysis = analyze_e36_extended(results)

    (REPORTS_DIR / "phase4t_p04_analysis.json").write_text(
        json.dumps(p04_analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "phase4t_naming_analysis.json").write_text(
        json.dumps(naming_analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "phase4t_e36_extended_analysis.json").write_text(
        json.dumps(e36_analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== P04 overall ===")
    print(json.dumps(p04_analysis["overall"], ensure_ascii=False, indent=2))
    print("=== P04 by category ===")
    print(json.dumps(p04_analysis["by_category"], ensure_ascii=False, indent=2))
    print("=== naming summary ===")
    print(json.dumps(naming_analysis["summary"], ensure_ascii=False, indent=2))
    print("=== E36 extended ===")
    e36_brief = {
        k: {kk: vv for kk, vv in v.items() if kk != "review_required_details"}
        for k, v in e36_analysis.items()
    }
    print(json.dumps(e36_brief, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
