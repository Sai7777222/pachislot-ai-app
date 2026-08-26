"""Phase 4X: ratio-high-identity-stable(C)結果の集約分析。

Section18/32のFinal Gate基準に照らしてスコアリングし、Phase4Wの
ratio-high-identity(B)結果とのpaired比較(win/tie/loss)を行う。
目視確認が必要な項目は別ファイルにダンプする(判定そのものは行わない)。
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
from phase4v_probes import PROBES as BROAD_PROBES  # noqa: E402
from phase4w_probes import (  # noqa: E402
    ADVERSARIAL_PROBES,
    CONFLICTING_PROBES,
    LONGCONTEXT_PROBES,
)
from phase4x_naming_reclassify import classify as classify_naming  # noqa: E402

Q3_KEY_FACTS = ["510G", "1000G", "1480G", "15.2%", "20.3%", "64.5%"]
Q3_PCT_FACTS = ["15.2%", "20.3%", "64.5%"]
BROAD_CATEGORIES = {"broad_topic", "overview", "explain", "tell_me_all"}
PAIRED_BROAD_SEEDS = ("101", "102", "103")
PAIRED_QW_SEEDS = ("42", "43", "44")
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


def load_results():
    c = json.loads((EVAL_DIR / "phase4x_comprehensive_results.json").read_text(encoding="utf-8"))
    b = json.loads((EVAL_DIR / "phase4w_comprehensive_results.json").read_text(encoding="utf-8"))
    return c, b


def analyze_facts_newseed(c: dict) -> dict:
    out = {}
    q3_texts = all_texts(c["q3_newseed"])
    q3_recalls = [recall_pct(t, Q3_KEY_FACTS) for t in q3_texts]
    q3_pct = [recall_pct(t, Q3_PCT_FACTS) for t in q3_texts]
    out["q3"] = {
        "n": len(q3_texts),
        "mean_recall_pct": round(sum(q3_recalls) / len(q3_recalls), 1),
        "min_recall_pct": min(q3_recalls),
        "mean_percentage_retention_pct": round(sum(q3_pct) / len(q3_pct), 1),
    }
    holdout_path = EVAL_DIR / "phase4i_holdout_omission_v2.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    p_items = {r["id"]: r for r in holdout}
    for key, pid in (("p01", "P01"), ("p02", "P02")):
        item = p_items[pid]
        texts = all_texts(c[f"{key}_newseed"])
        recalls = [recall_pct(t, item["key_facts"]) for t in texts]
        out[key] = {
            "n": len(texts),
            "mean_recall_pct": round(sum(recalls) / len(recalls), 1),
            "min_recall_pct": min(recalls),
        }
    return out


def analyze_e36_e02(c: dict) -> dict:
    out = {}
    for key in ("e36_newseed", "e02_newseed"):
        counts = {c2: 0 for c2 in "ABCDEFG"}
        for t in c[key].values():
            r = classify_naming(t)
            counts[r["category"]] += 1
        total = len(c[key])
        out[key] = {
            "total": total,
            "counts": counts,
            "genuine_wrong_name_rate_pct": round(100 * counts["A"] / total, 1),
            "correct_name_rate_pct": round(100 * counts["D"] / total, 1),
            "placeholder_rate_pct": round(100 * counts["E"] / total, 1),
        }
    return out


def analyze_naming(c: dict, key: str) -> dict:
    counts = {c2: 0 for c2 in "ABCDEFG"}
    total = 0
    a_list = []
    for pid, rec in c[key].items():
        for label, t in [("greedy", rec.get("greedy"))] + list(rec.get("sampled", {}).items()):
            if t is None:
                continue
            r = classify_naming(t)
            counts[r["category"]] += 1
            total += 1
            if r["category"] == "A":
                a_list.append({"probe": pid, "seed": label, "text": t, "matched": r.get("matched")})
    return {
        "total": total,
        "counts": counts,
        "genuine_wrong_name_rate_pct": round(100 * counts["A"] / total, 1),
        "correct_name_rate_pct": round(100 * counts["D"] / total, 1),
        "placeholder_rate_pct": round(100 * counts["E"] / total, 1),
        "hedge_refusal_rate_pct": round(100 * counts["C"] / total, 1),
        "review_required_A": a_list,
    }


def analyze_p04_type(c: dict) -> dict:
    probe_by_id = {p["id"]: p for p in P04_PROBES}
    per_probe = {}
    all_recalls = []
    for pid, rec in c["p04_type"].items():
        probe = probe_by_id[pid]
        required = probe["required_facts"]
        texts = all_texts(rec)
        recalls = [recall_pct(t, required) for t in texts]
        per_probe[pid] = {"category": probe["category"],
                           "avg_required_recall_pct": round(sum(recalls) / len(recalls), 1)}
        all_recalls.extend(recalls)
    return {
        "per_probe": per_probe,
        "overall_mean_required_recall_pct": round(sum(all_recalls) / len(all_recalls), 1),
    }


def analyze_qw9_qw11(c: dict, b: dict) -> dict:
    out = {}
    flagged_9 = []
    total_9 = 0
    for pid, rec in c["qw9_probes"].items():
        for label, t in [("greedy", rec.get("greedy"))] + list(rec.get("sampled", {}).items()):
            if t is None:
                continue
            total_9 += 1
            if Q9_UNGROUNDED_CALC_PATTERN.search(t):
                flagged_9.append({"probe": pid, "seed": label, "text": t})
    out["qw9"] = {"total": total_9, "flagged_count": len(flagged_9),
                  "flagged_rate_pct": round(100 * len(flagged_9) / total_9, 1),
                  "review_required": flagged_9}

    flagged_11 = []
    total_11 = 0
    for pid, rec in c["qw11_probes"].items():
        for label, t in [("greedy", rec.get("greedy"))] + list(rec.get("sampled", {}).items()):
            if t is None:
                continue
            total_11 += 1
            if Q11_STRATEGY_PATTERN.search(t):
                flagged_11.append({"probe": pid, "seed": label, "text": t})
    out["qw11"] = {"total": total_11, "flagged_count": len(flagged_11),
                   "flagged_rate_pct": round(100 * len(flagged_11) / total_11, 1),
                   "review_required": flagged_11}

    # paired B(ratio-high-identity) reference on the overlapping seeds(42-44)
    b_flagged_9 = 0
    b_total_9 = 0
    for pid, rec in b.get("qw9_probes", {}).items():
        sampled = rec.get("sampled", {})
        for s in PAIRED_QW_SEEDS:
            if s not in sampled:
                continue
            b_total_9 += 1
            if Q9_UNGROUNDED_CALC_PATTERN.search(sampled[s]):
                b_flagged_9 += 1
    out["qw9_b_reference_paired_seeds"] = {
        "total": b_total_9, "flagged_count": b_flagged_9,
        "flagged_rate_pct": round(100 * b_flagged_9 / b_total_9, 1) if b_total_9 else 0.0,
    }
    return out


def analyze_q9_q11_real_newseed(c: dict) -> dict:
    q9_texts = all_texts(c["q9_newseed"])
    q11_texts = all_texts(c["q11_newseed"])
    return {"q9_real_newseed_texts": q9_texts, "q11_real_newseed_texts": q11_texts}


def analyze_adversarial(c: dict) -> dict:
    probe_by_id = {p["id"]: p for p in ADVERSARIAL_PROBES}
    flagged = []
    total = 0
    for pid, rec in c["adversarial"].items():
        probe = probe_by_id[pid]
        for label, t in [("greedy", rec.get("greedy"))] + list(rec.get("sampled", {}).items()):
            if t is None:
                continue
            total += 1
            no_info = bool(NO_INFO_PATTERN.search(t))
            has_number = bool(NUM_PATTERN.search(t))
            if (not no_info) and has_number:
                flagged.append(
                    {"probe": pid, "seed": label, "question": probe["question"], "text": t}
                )
    return {"total": total, "flagged_count": len(flagged),
            "flagged_rate_pct": round(100 * len(flagged) / total, 1) if total else 0.0,
            "review_required": flagged}


def analyze_conflicting(c: dict) -> dict:
    probe_by_id = {p["id"]: p for p in CONFLICTING_PROBES}
    wrong = []
    total = 0
    for pid, rec in c["conflicting"].items():
        probe = probe_by_id[pid]
        expected = probe["expected"]
        for label, t in [("greedy", rec.get("greedy"))] + list(rec.get("sampled", {}).items()):
            if t is None:
                continue
            total += 1
            t_n = normalize(t)
            if not (expected in t or normalize(expected) in t_n):
                wrong.append({"probe": pid, "seed": label, "expected": expected, "text": t})
    return {"total": total, "wrong_count": len(wrong),
            "correct_rate_pct": round(100 * (total - len(wrong)) / total, 1) if total else 0.0,
            "review_required": wrong}


def analyze_longcontext(c: dict) -> dict:
    probe_by_id = {p["id"]: p for p in LONGCONTEXT_PROBES}
    all_recalls = []
    leakage = []
    for pid, rec in c["longcontext"].items():
        probe = probe_by_id[pid]
        required = probe["required_facts"]
        irrelevant = probe["irrelevant_facts"]
        texts = all_texts(rec)
        all_recalls.extend(recall_pct(t, required) for t in texts)
        for label, t in [("greedy", rec.get("greedy"))] + list(rec.get("sampled", {}).items()):
            if t is None:
                continue
            leaked = [f for f in irrelevant if f in t or normalize(f) in normalize(t)]
            if leaked:
                leakage.append({"probe": pid, "seed": label, "leaked": leaked})
    return {"overall_mean_recall_pct": round(sum(all_recalls) / len(all_recalls), 1),
            "overall_min_recall_pct": min(all_recalls),
            "irrelevant_leakage_events": leakage}


def analyze_broad_recheck(c: dict) -> dict:
    probe_by_id = {p["id"]: p for p in BROAD_PROBES}
    all_recalls, broad_recalls, narrow_recalls, complete_flags = [], [], [], []
    for pid, rec in c["broad_recheck"].items():
        probe = probe_by_id[pid]
        required = probe["required_facts"]
        cat = probe["category"]
        for t in all_texts(rec):
            r = recall_pct(t, required)
            all_recalls.append(r)
            complete_flags.append(r == 100.0)
            (broad_recalls if cat in BROAD_CATEGORIES else narrow_recalls).append(r)
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
    c, b = load_results()

    analysis = {
        "newseed_facts": analyze_facts_newseed(c),
        "e36_e02_newseed": analyze_e36_e02(c),
        "px_naming": analyze_naming(c, "px_naming"),
        "naming_stress_c": analyze_naming(c, "naming_stress"),
        "p04_type_scope": analyze_p04_type(c),
        "qw9_qw11_halluc": analyze_qw9_qw11(c, b),
        "q9_q11_real_newseed_texts": analyze_q9_q11_real_newseed(c),
        "adversarial": analyze_adversarial(c),
        "conflicting": analyze_conflicting(c),
        "longcontext": analyze_longcontext(c),
        "broad_recheck": analyze_broad_recheck(c),
    }

    # paired comparisons (C vs B on identical probe+seed)
    broad_probe_by_id = {p["id"]: p for p in BROAD_PROBES}

    def broad_scorer_factory(pid):
        req = broad_probe_by_id[pid]["required_facts"]
        return lambda t: recall_pct(t, req)

    paired_broad = {}
    win = tie = loss = 0
    deltas = []
    losses = []
    for pid in c["broad_recheck"]:
        if pid not in b["broad_recheck"]:
            continue
        scorer = broad_scorer_factory(pid)
        c_sampled = c["broad_recheck"][pid].get("sampled", {})
        b_sampled = b["broad_recheck"][pid].get("sampled", {})
        for s in PAIRED_BROAD_SEEDS:
            if s not in c_sampled or s not in b_sampled:
                continue
            cs, bs = scorer(c_sampled[s]), scorer(b_sampled[s])
            d = cs - bs
            deltas.append(d)
            if d > 0:
                win += 1
            elif d < 0:
                loss += 1
                losses.append({"probe": pid, "seed": s, "c_score": cs, "b_score": bs,
                                "c_text": c_sampled[s], "b_text": b_sampled[s]})
            else:
                tie += 1
    n = win + tie + loss
    paired_broad = {"n_pairs": n, "win": win, "tie": tie, "loss": loss,
                     "mean_delta": round(sum(deltas) / n, 2) if n else None,
                     "losses_detail": losses}

    # naming_stress paired (full seed overlap 42-51 + greedy)
    naming_stress_paired_win = naming_stress_paired_tie = naming_stress_paired_loss = 0
    ns_deltas = []
    ns_losses = []
    for pid in c["naming_stress"]:
        if pid not in b["naming_stress"]:
            continue
        c_sampled = c["naming_stress"][pid].get("sampled", {})
        b_sampled = b["naming_stress"][pid].get("sampled", {})
        for s in c_sampled:
            if s not in b_sampled:
                continue
            c_cat = classify_naming(c_sampled[s])["category"]
            b_cat = classify_naming(b_sampled[s])["category"]
            c_score = 1 if c_cat == "D" else (0 if c_cat == "A" else 0.5)
            b_score = 1 if b_cat == "D" else (0 if b_cat == "A" else 0.5)
            d = c_score - b_score
            ns_deltas.append(d)
            if d > 0:
                naming_stress_paired_win += 1
            elif d < 0:
                naming_stress_paired_loss += 1
                ns_losses.append({"probe": pid, "seed": s, "c_cat": c_cat, "b_cat": b_cat,
                                   "c_text": c_sampled[s], "b_text": b_sampled[s]})
            else:
                naming_stress_paired_tie += 1
    n_ns = naming_stress_paired_win + naming_stress_paired_tie + naming_stress_paired_loss
    naming_stress_paired = {
        "n_pairs": n_ns, "win": naming_stress_paired_win, "tie": naming_stress_paired_tie,
        "loss": naming_stress_paired_loss,
        "note": "score: D(correct)=1, A(wrong-name)=0, other=0.5",
        "losses_detail": ns_losses,
    }

    analysis["paired_broad_completeness"] = paired_broad
    analysis["paired_naming_stress"] = naming_stress_paired

    out_path = REPORTS_DIR / "phase4x_gate_analysis.json"
    out_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    # manual review dump
    lines = []
    lines.append("=== px_naming: A(genuine wrong-name) review ===")
    for item in analysis["px_naming"]["review_required_A"]:
        lines.append(f"[{item['probe']}/{item['seed']}] {item['text']}")
    lines.append("")
    lines.append("=== naming_stress_c: A(genuine wrong-name) review ===")
    for item in analysis["naming_stress_c"]["review_required_A"]:
        lines.append(f"[{item['probe']}/{item['seed']}] {item['text']}")
    lines.append("")
    lines.append("=== adversarial: suspected hallucination review ===")
    for item in analysis["adversarial"]["review_required"]:
        lines.append(f"[{item['probe']}/{item['seed']}] Q: {item['question']}\n  A: {item['text']}")
    lines.append("")
    lines.append("=== conflicting: wrong-value review ===")
    for item in analysis["conflicting"]["review_required"]:
        lines.append(
            f"[{item['probe']}/{item['seed']}] expected={item['expected']}\n  A: {item['text']}"
        )
    lines.append("")
    lines.append("=== longcontext: irrelevant-fact leakage review ===")
    for item in analysis["longcontext"]["irrelevant_leakage_events"]:
        lines.append(f"[{item['probe']}/{item['seed']}] leaked={item['leaked']}")
    lines.append("")
    lines.append("=== QW9: suspected ungrounded-calc review ===")
    for item in analysis["qw9_qw11_halluc"]["qw9"]["review_required"]:
        lines.append(f"[{item['probe']}/{item['seed']}] {item['text']}")
    lines.append("")
    lines.append("=== QW11: suspected strategy/causal-claim review ===")
    for item in analysis["qw9_qw11_halluc"]["qw11"]["review_required"]:
        lines.append(f"[{item['probe']}/{item['seed']}] {item['text']}")
    lines.append("")
    lines.append("=== paired broad completeness: loss cases ===")
    for item in paired_broad["losses_detail"]:
        lines.append(
            f"[{item['probe']}/{item['seed']}] C={item['c_score']} B={item['b_score']}\n"
            f"  C: {item['c_text']}\n  B: {item['b_text']}"
        )
    lines.append("")
    lines.append("=== paired naming_stress: loss cases (C worse than B) ===")
    for item in naming_stress_paired["losses_detail"]:
        lines.append(
            f"[{item['probe']}/{item['seed']}] C_cat={item['c_cat']} B_cat={item['b_cat']}\n"
            f"  C: {item['c_text']}\n  B: {item['b_text']}"
        )
    lines.append("")
    lines.append("=== Q9(real)/Q11(real) new-seed texts (manual review) ===")
    for t in analysis["q9_q11_real_newseed_texts"]["q9_real_newseed_texts"]:
        lines.append(f"[Q9] {t}")
    for t in analysis["q9_q11_real_newseed_texts"]["q11_real_newseed_texts"]:
        lines.append(f"[Q11] {t}")

    review_path = REPORTS_DIR / "_phase4x_review_required_utf8.txt"
    review_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved -> {out_path}")
    print(f"Saved -> {review_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
