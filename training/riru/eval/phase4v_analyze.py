"""Phase 4V: Broad-Question Completeness 診断結果の集約分析。

required_fact_recall / complete_answer_rate を条件別・カテゴリ別・fact種別に
算出し、ratio-high vs ratio-high-identity の paired比較(win/tie/loss)を行う。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EVAL_DIR.parents[0] / "reports"

sys.path.insert(0, str(EVAL_DIR))
from phase4v_probes import PROBES  # noqa: E402

CONDITIONS = ("A_base", "B_v4", "C_high", "D_identity")
BROAD_CATEGORIES = {"broad_topic", "overview", "explain", "tell_me_all"}
NARROW_CATEGORIES = {"specific_complete", "narrow_control"}

PCT_PATTERN = re.compile(r"\d+(\.\d+)?%")


def normalize(t: str) -> str:
    return t.replace("％", "%").replace(",", "").replace("ゲーム", "G")


def fact_type(fact: str) -> str:
    if PCT_PATTERN.fullmatch(fact.strip()):
        return "percentage"
    if re.search(r"\d", fact):
        return "numeric"
    return "categorical"


def score_text(text: str, required: list[str]) -> dict:
    text_n = normalize(text)
    found = [f for f in required if f in text or normalize(f) in text_n]
    recall = round(len(found) / len(required) * 100, 1) if required else None
    return {"found": found, "recall_pct": recall, "complete": len(found) == len(required)}


def main() -> int:
    results = json.loads(
        (EVAL_DIR / "phase4v_comprehensive_results.json").read_text(encoding="utf-8")
    )
    probe_by_id = {p["id"]: p for p in PROBES}

    per_probe: dict = {}
    scores_by_cond: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
    scores_by_cond_cat: dict[str, dict[str, list[dict]]] = {c: {} for c in CONDITIONS}
    scores_by_cond_facttype: dict[str, dict[str, list[dict]]] = {c: {} for c in CONDITIONS}
    paired = []  # ratio-high vs identity, per probe per seed

    for pid, rec in results.items():
        probe = probe_by_id[pid]
        required = probe["required_facts"]
        cat = probe["category"]
        broad_or_narrow = "broad" if cat in BROAD_CATEGORIES else "narrow"
        per_probe[pid] = {"category": cat, "family": probe["family"], "conditions": {}}

        for cond in CONDITIONS:
            cond_data = rec["conditions"][cond]
            greedy_score = score_text(cond_data["greedy"], required)
            sampled_scores = {
                seed: score_text(t, required) for seed, t in cond_data["sampled"].items()
            }
            avg_recall = round(
                sum(s["recall_pct"] for s in sampled_scores.values()) / len(sampled_scores), 1
            )
            complete_rate = round(
                100 * sum(s["complete"] for s in sampled_scores.values()) / len(sampled_scores), 1
            )
            per_probe[pid]["conditions"][cond] = {
                "greedy_recall_pct": greedy_score["recall_pct"],
                "sampled_avg_recall_pct": avg_recall,
                "complete_answer_rate_pct": complete_rate,
            }

            for s in sampled_scores.values():
                entry = {"probe": pid, "category": cat, "broad_or_narrow": broad_or_narrow, **s}
                scores_by_cond[cond].append(entry)
                scores_by_cond_cat[cond].setdefault(cat, []).append(entry)
                for f in required:
                    ft = fact_type(f)
                    scores_by_cond_facttype[cond].setdefault(ft, []).append(
                        {"probe": pid, "fact": f, "retained": f in entry["found"]}
                    )

            if cond == "C_high":
                high_sampled = sampled_scores
            if cond == "D_identity":
                identity_sampled = sampled_scores

        for seed in high_sampled:
            h = high_sampled[seed]["recall_pct"]
            i = identity_sampled[seed]["recall_pct"]
            if i > h:
                outcome = "win"
            elif i < h:
                outcome = "loss"
            else:
                outcome = "tie"
            paired.append(
                {
                    "probe": pid, "seed": seed, "category": cat, "broad_or_narrow": broad_or_narrow,
                    "high_recall": h, "identity_recall": i, "delta": round(i - h, 1),
                    "outcome": outcome,
                }
            )

    def summarize(entries: list[dict]) -> dict:
        recalls = [e["recall_pct"] for e in entries]
        completes = [e["complete"] for e in entries]
        n = len(recalls)
        recalls_sorted = sorted(recalls)
        median = (
            recalls_sorted[n // 2]
            if n % 2 == 1
            else (recalls_sorted[n // 2 - 1] + recalls_sorted[n // 2]) / 2
        )
        return {
            "n": n,
            "mean_recall_pct": round(sum(recalls) / n, 1),
            "median_recall_pct": round(median, 1),
            "complete_answer_rate_pct": round(100 * sum(completes) / n, 1),
        }

    overall = {c: summarize(scores_by_cond[c]) for c in CONDITIONS}
    by_category = {
        c: {cat: summarize(v) for cat, v in scores_by_cond_cat[c].items()} for c in CONDITIONS
    }
    broad_narrow = {}
    for c in CONDITIONS:
        broad_entries = [e for e in scores_by_cond[c] if e["broad_or_narrow"] == "broad"]
        narrow_entries = [e for e in scores_by_cond[c] if e["broad_or_narrow"] == "narrow"]
        broad_narrow[c] = {
            "broad": summarize(broad_entries), "narrow": summarize(narrow_entries),
        }
    by_facttype = {}
    for c in CONDITIONS:
        by_facttype[c] = {}
        for ft, entries in scores_by_cond_facttype[c].items():
            retained = sum(e["retained"] for e in entries)
            by_facttype[c][ft] = {
                "n_facts": len(entries),
                "retention_pct": round(100 * retained / len(entries), 1),
            }

    win = sum(1 for p in paired if p["outcome"] == "win")
    tie = sum(1 for p in paired if p["outcome"] == "tie")
    loss = sum(1 for p in paired if p["outcome"] == "loss")
    deltas = [p["delta"] for p in paired]
    broad_deltas = [p["delta"] for p in paired if p["broad_or_narrow"] == "broad"]
    narrow_deltas = [p["delta"] for p in paired if p["broad_or_narrow"] == "narrow"]

    paired_summary = {
        "n_pairs": len(paired),
        "win_high_worse": win, "tie": tie, "loss_identity_worse": loss,
        "mean_delta_identity_minus_high": round(sum(deltas) / len(deltas), 2),
        "mean_broad_delta": (
            round(sum(broad_deltas) / len(broad_deltas), 2) if broad_deltas else None
        ),
        "mean_narrow_delta": (
            round(sum(narrow_deltas) / len(narrow_deltas), 2) if narrow_deltas else None
        ),
    }

    out = {
        "overall": overall,
        "by_category": by_category,
        "broad_vs_narrow": broad_narrow,
        "by_fact_type": by_facttype,
        "paired_high_vs_identity": paired_summary,
        "per_probe": per_probe,
    }
    (REPORTS_DIR / "phase4v_broad_question_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "phase4v_paired_analysis.json").write_text(
        json.dumps({"summary": paired_summary, "pairs": paired}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(json.dumps(broad_narrow, ensure_ascii=False, indent=2))
    print(json.dumps(paired_summary, ensure_ascii=False, indent=2))
    print(json.dumps(by_facttype, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
