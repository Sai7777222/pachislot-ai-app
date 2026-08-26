"""Phase 4W: Final Candidate Gate 結果の集約分析。

新規seed再現性(Q3/P01/P02/Q9/Q11/E36/E02)、新規Q9/Q11型probe、naming
stress、adversarial、conflicting、long-context、broad-completeness再確認の
各結果を採点し、Section18の最終Gate基準に照らして判定材料を出力する。
目視確認が必要な項目は別ファイルにダンプする(このスクリプト自体は判定しない)。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EVAL_DIR.parents[0] / "reports"

sys.path.insert(0, str(EVAL_DIR))
from phase4u_reclassify_naming import classify_generation  # noqa: E402
from phase4v_probes import PROBES as BROAD_PROBES  # noqa: E402
from phase4w_probes import (  # noqa: E402
    ADVERSARIAL_PROBES,
    CONFLICTING_PROBES,
    LONGCONTEXT_PROBES,
)

Q3_KEY_FACTS = ["510G", "1000G", "1480G", "15.2%", "20.3%", "64.5%"]
Q3_PCT_FACTS = ["15.2%", "20.3%", "64.5%"]

BROAD_CATEGORIES = {"broad_topic", "overview", "explain", "tell_me_all"}
Q11_STRATEGY_PATTERN = re.compile(r"おすすめ|べきです|べきだ|戦略|コツ|投資額")
Q9_UNGROUNDED_CALC_PATTERN = re.compile(r"約\s*\d+(\.\d+)?\s*(倍|ポイント|%\s*(高い|低い|差))")
NO_INFO_PATTERN = re.compile(
    r"わから|不明|情報が(ない|ありません)|記載がない|載っていない|分かりかね|お答えできません|データがない"
)
NUM_PATTERN = re.compile(r"\d+(\.\d+)?\s*(%|枚|G|倍|回)")


def normalize(t: str) -> str:
    return t.replace("％", "%").replace(",", "").replace("ゲーム", "G")


def recall_pct(text: str, required: list[str]) -> float:
    text_n = normalize(text)
    found = [f for f in required if f in text or normalize(f) in text_n]
    return round(len(found) / len(required) * 100, 1) if required else 100.0


def all_texts(rec: dict) -> list[str]:
    out = []
    if "greedy" in rec:
        out.append(rec["greedy"])
    out.extend(rec.get("sampled", {}).values())
    return out


def analyze_newseed_facts(results: dict) -> dict:
    out = {}
    q3_texts = all_texts(results["q3_newseed"])
    q3_recalls = [recall_pct(t, Q3_KEY_FACTS) for t in q3_texts]
    q3_pct_recalls = [recall_pct(t, Q3_PCT_FACTS) for t in q3_texts]
    out["q3"] = {
        "n": len(q3_texts),
        "mean_recall_pct": round(sum(q3_recalls) / len(q3_recalls), 1),
        "min_recall_pct": min(q3_recalls),
        "mean_percentage_retention_pct": round(sum(q3_pct_recalls) / len(q3_pct_recalls), 1),
        "all_100": all(r == 100.0 for r in q3_recalls),
    }

    holdout_path = EVAL_DIR / "phase4i_holdout_omission_v2.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    p_items = {r["id"]: r for r in holdout}
    for key, pid in (("p01", "P01"), ("p02", "P02")):
        item = p_items[pid]
        texts = all_texts(results[f"{key}_newseed"])
        recalls = [recall_pct(t, item["key_facts"]) for t in texts]
        out[key] = {
            "n": len(texts),
            "mean_recall_pct": round(sum(recalls) / len(recalls), 1),
            "min_recall_pct": min(recalls),
        }
    return out


def analyze_e36_e02_newseed(results: dict) -> dict:
    out = {}
    for key in ("e36_newseed", "e02_newseed"):
        counts = {c: 0 for c in "ABCDEFG"}
        for t in results[key].values():
            counts[classify_generation(t)] += 1
        total = len(results[key])
        out[key] = {
            "total": total,
            "counts": counts,
            "genuine_wrong_name_rate_pct": round(100 * counts["A"] / total, 1),
            "correct_name_rate_pct": round(100 * counts["E"] / total, 1),
            "placeholder_rate_pct": round(100 * counts["C"] / total, 1),
        }
    return out


def analyze_naming_stress(results: dict) -> dict:
    counts = {c: 0 for c in "ABCDEFG"}
    per_probe = {}
    review_required = []
    total = 0
    for pid, rec in results["naming_stress"].items():
        probe_counts = {c: 0 for c in "ABCDEFG"}
        for seed_label, t in [("greedy", rec.get("greedy"))] + list(rec.get("sampled", {}).items()):
            if t is None:
                continue
            cat = classify_generation(t)
            counts[cat] += 1
            probe_counts[cat] += 1
            total += 1
            if cat == "A":
                review_required.append({"probe": pid, "seed": seed_label, "text": t})
        per_probe[pid] = probe_counts
    return {
        "total": total,
        "counts": counts,
        "genuine_wrong_name_rate_pct": round(100 * counts["A"] / total, 1),
        "correct_name_rate_pct": round(100 * counts["E"] / total, 1),
        "placeholder_rate_pct": round(100 * counts["C"] / total, 1),
        "per_probe": per_probe,
        "review_required_A": review_required,
    }


def analyze_adversarial(results: dict) -> dict:
    probe_by_id = {p["id"]: p for p in ADVERSARIAL_PROBES}
    per_probe = {}
    flagged = []
    total = 0
    for pid, rec in results["adversarial"].items():
        probe = probe_by_id[pid]
        probe_flags = 0
        for label, t in [("greedy", rec.get("greedy"))] + list(rec.get("sampled", {}).items()):
            if t is None:
                continue
            total += 1
            no_info = bool(NO_INFO_PATTERN.search(t))
            has_number = bool(NUM_PATTERN.search(t))
            suspect = (not no_info) and has_number
            if suspect:
                probe_flags += 1
                flagged.append(
                    {"probe": pid, "seed": label, "question": probe["question"], "text": t}
                )
        per_probe[pid] = {"flagged": probe_flags}
    return {
        "total": total,
        "flagged_count": len(flagged),
        "flagged_rate_pct": round(100 * len(flagged) / total, 1) if total else 0.0,
        "per_probe": per_probe,
        "review_required": flagged,
    }


def analyze_conflicting(results: dict) -> dict:
    probe_by_id = {p["id"]: p for p in CONFLICTING_PROBES}
    per_probe = {}
    wrong = []
    total = 0
    for pid, rec in results["conflicting"].items():
        probe = probe_by_id[pid]
        expected = probe["expected"]
        correct_n = 0
        n = 0
        for label, t in [("greedy", rec.get("greedy"))] + list(rec.get("sampled", {}).items()):
            if t is None:
                continue
            n += 1
            total += 1
            t_n = normalize(t)
            ok = expected in t or normalize(expected) in t_n
            if ok:
                correct_n += 1
            else:
                wrong.append({"probe": pid, "seed": label, "expected": expected, "text": t})
        per_probe[pid] = {"correct_rate_pct": round(100 * correct_n / n, 1)}
    return {
        "total": total,
        "wrong_count": len(wrong),
        "correct_rate_pct": round(100 * (total - len(wrong)) / total, 1) if total else 0.0,
        "per_probe": per_probe,
        "review_required": wrong,
    }


def analyze_longcontext(results: dict) -> dict:
    probe_by_id = {p["id"]: p for p in LONGCONTEXT_PROBES}
    per_probe = {}
    all_recalls = []
    leakage = []
    for pid, rec in results["longcontext"].items():
        probe = probe_by_id[pid]
        required = probe["required_facts"]
        irrelevant = probe["irrelevant_facts"]
        texts = all_texts(rec)
        recalls = [recall_pct(t, required) for t in texts]
        for label, t in [("greedy", rec.get("greedy"))] + list(rec.get("sampled", {}).items()):
            if t is None:
                continue
            leaked = [f for f in irrelevant if f in t or normalize(f) in normalize(t)]
            if leaked:
                leakage.append({"probe": pid, "seed": label, "leaked": leaked})
        per_probe[pid] = {
            "mean_recall_pct": round(sum(recalls) / len(recalls), 1),
            "min_recall_pct": min(recalls),
        }
        all_recalls.extend(recalls)
    return {
        "overall_mean_recall_pct": round(sum(all_recalls) / len(all_recalls), 1),
        "overall_min_recall_pct": min(all_recalls),
        "per_probe": per_probe,
        "irrelevant_leakage_events": leakage,
    }


def analyze_qw9(results: dict) -> dict:
    """QW9: 独自計算hallucinationチェック(パターン検出→目視確認対象を出力)。"""
    flagged = []
    total = 0
    for pid, rec in results["qw9_probes"].items():
        for label, t in [("greedy", rec.get("greedy"))] + list(rec.get("sampled", {}).items()):
            if t is None:
                continue
            total += 1
            if Q9_UNGROUNDED_CALC_PATTERN.search(t):
                flagged.append({"probe": pid, "seed": label, "text": t})
    return {
        "total": total,
        "flagged_count": len(flagged),
        "flagged_rate_pct": round(100 * len(flagged) / total, 1) if total else 0.0,
        "review_required": flagged,
    }


def analyze_qw11(results: dict) -> dict:
    """QW11: 因果・戦略hallucinationチェック(パターン検出→目視確認対象を出力)。"""
    flagged = []
    total = 0
    for pid, rec in results["qw11_probes"].items():
        for label, t in [("greedy", rec.get("greedy"))] + list(rec.get("sampled", {}).items()):
            if t is None:
                continue
            total += 1
            if Q11_STRATEGY_PATTERN.search(t):
                flagged.append({"probe": pid, "seed": label, "text": t})
    return {
        "total": total,
        "flagged_count": len(flagged),
        "flagged_rate_pct": round(100 * len(flagged) / total, 1) if total else 0.0,
        "review_required": flagged,
    }


def analyze_broad_recheck(results: dict) -> dict:
    probe_by_id = {p["id"]: p for p in BROAD_PROBES}
    all_recalls = []
    broad_recalls = []
    narrow_recalls = []
    complete_flags = []
    for pid, rec in results["broad_recheck"].items():
        probe = probe_by_id[pid]
        required = probe["required_facts"]
        cat = probe["category"]
        texts = all_texts(rec)
        for t in texts:
            r = recall_pct(t, required)
            all_recalls.append(r)
            complete_flags.append(r == 100.0)
            if cat in BROAD_CATEGORIES:
                broad_recalls.append(r)
            else:
                narrow_recalls.append(r)
    return {
        "overall_mean_recall_pct": round(sum(all_recalls) / len(all_recalls), 1),
        "complete_answer_rate_pct": round(100 * sum(complete_flags) / len(complete_flags), 1),
        "broad_mean_recall_pct": (
            round(sum(broad_recalls) / len(broad_recalls), 1) if broad_recalls else None
        ),
        "narrow_mean_recall_pct": (
            round(sum(narrow_recalls) / len(narrow_recalls), 1) if narrow_recalls else None
        ),
    }


def main() -> int:
    results = json.loads(
        (EVAL_DIR / "phase4w_comprehensive_results.json").read_text(encoding="utf-8")
    )

    analysis = {
        "newseed_facts": analyze_newseed_facts(results),
        "e36_e02_newseed": analyze_e36_e02_newseed(results),
        "naming_stress": analyze_naming_stress(results),
        "adversarial": analyze_adversarial(results),
        "conflicting": analyze_conflicting(results),
        "longcontext": analyze_longcontext(results),
        "qw9_calc_halluc": analyze_qw9(results),
        "qw11_causal_halluc": analyze_qw11(results),
        "broad_recheck": analyze_broad_recheck(results),
    }

    out_path = REPORTS_DIR / "phase4w_gate_analysis.json"
    out_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    # 目視確認対象を一箇所にまとめてUTF-8ダンプ(cp932コンソール回避)
    review_lines = []
    review_lines.append("=== naming_stress: A(genuine wrong-name) review ===")
    for item in analysis["naming_stress"]["review_required_A"]:
        review_lines.append(f"[{item['probe']}/{item['seed']}] {item['text']}")
    review_lines.append("")
    review_lines.append("=== adversarial: suspected hallucination review ===")
    for item in analysis["adversarial"]["review_required"]:
        review_lines.append(
            f"[{item['probe']}/{item['seed']}] Q: {item['question']}\n  A: {item['text']}"
        )
    review_lines.append("")
    review_lines.append("=== conflicting: wrong-value review ===")
    for item in analysis["conflicting"]["review_required"]:
        review_lines.append(
            f"[{item['probe']}/{item['seed']}] expected={item['expected']}\n  A: {item['text']}"
        )
    review_lines.append("")
    review_lines.append("=== longcontext: irrelevant-fact leakage review ===")
    for item in analysis["longcontext"]["irrelevant_leakage_events"]:
        review_lines.append(f"[{item['probe']}/{item['seed']}] leaked={item['leaked']}")
    review_lines.append("")
    review_lines.append("=== QW9: suspected ungrounded-calc review ===")
    for item in analysis["qw9_calc_halluc"]["review_required"]:
        review_lines.append(f"[{item['probe']}/{item['seed']}] {item['text']}")
    review_lines.append("")
    review_lines.append("=== QW11: suspected strategy/causal-claim review ===")
    for item in analysis["qw11_causal_halluc"]["review_required"]:
        review_lines.append(f"[{item['probe']}/{item['seed']}] {item['text']}")

    review_path = REPORTS_DIR / "_phase4w_review_required_utf8.txt"
    review_path.write_text("\n".join(review_lines), encoding="utf-8")

    print(f"Saved -> {out_path}")
    print(f"Saved -> {review_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
