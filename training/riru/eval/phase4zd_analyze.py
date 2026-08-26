"""Phase 4ZD Section13: Stage1 naming220結果の分類・paired比較・判定。

B_HF_BF16_EAGER と D_LLAMA_BF16_CPU の naming220結果を、
Phase4Z/4Xで改善済みの最新naming classifier(phase4z_naming_classify.classify_naming)
で分類し、genuine wrong-name率・paired比較(B safe -> D unsafe)・critical loss率を算出する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
sys.path.insert(0, str(EVAL_DIR))
REPORTS_DIR = TRAINING_ROOT / "reports"

from phase4z_naming_classify import classify_naming  # noqa: E402

CONDITIONS = ["A_LEGACY_4BIT", "B_HF_BF16_EAGER", "C_HF_BF16_SDPA", "D_LLAMA_BF16_CPU"]


def flatten(data: dict) -> list[dict]:
    """{'probe_id': {'greedy': text, 'sampled': {seed: text}}} -> flat list of
    {'probe_id':, 'kind':, 'key':, 'text':}"""
    out = []
    for pid, entry in data["results"].items():
        out.append({"probe_id": pid, "kind": "greedy", "key": "greedy", "text": entry["greedy"]})
        for seed, text in entry.get("sampled", {}).items():
            out.append({"probe_id": pid, "kind": "sampled", "key": seed, "text": text})
    return out


def classify_all(flat: list[dict]) -> list[dict]:
    out = []
    for item in flat:
        cls = classify_naming(item["text"], is_naming_context=True)
        out.append({**item, "category": cls["category"], "reason": cls.get("reason")})
    return out


def summarize(classified: list[dict]) -> dict:
    n = len(classified)
    counts = {}
    for c in classified:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    genuine_wrong = counts.get("A", 0)
    return {
        "n_total": n,
        "category_counts": counts,
        "category_pct": {k: round(v / n * 100, 2) for k, v in counts.items()},
        "genuine_wrong_name_count": genuine_wrong,
        "genuine_wrong_name_pct": round(genuine_wrong / n * 100, 2),
    }


def main() -> int:
    stage1_results = {}
    for cond in ["B_HF_BF16_EAGER", "D_LLAMA_BF16_CPU"]:
        f = REPORTS_DIR / f"phase4zd_naming220_{cond}.json"
        if not f.exists():
            print(f"missing: {f}, skip")
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        flat = flatten(data)
        classified = classify_all(flat)
        summary = summarize(classified)
        stage1_results[cond] = {"summary": summary, "classified": classified}
        print(cond, summary)

    identity_analysis = {c: v["summary"] for c, v in stage1_results.items()}
    (REPORTS_DIR / "phase4zd_identity_analysis.json").write_text(
        json.dumps(identity_analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved -> phase4zd_identity_analysis.json")

    # paired comparison: B safe -> D unsafe (per identical probe_id+kind+key)
    if "B_HF_BF16_EAGER" in stage1_results and "D_LLAMA_BF16_CPU" in stage1_results:
        b_items = {(c["probe_id"], c["kind"], c["key"]): c for c in stage1_results["B_HF_BF16_EAGER"]["classified"]}
        d_items = {(c["probe_id"], c["kind"], c["key"]): c for c in stage1_results["D_LLAMA_BF16_CPU"]["classified"]}
        keys = sorted(set(b_items) & set(d_items))

        def is_safe(cat):
            return cat in ("E",)  # correct name only counts as safe for this paired check

        win = tie = loss = critical_loss = 0
        pairs = []
        for k in keys:
            bc, dc = b_items[k]["category"], d_items[k]["category"]
            b_safe, d_safe = is_safe(bc), is_safe(dc)
            if b_safe and d_safe:
                tie += 1
                status = "TIE"
            elif b_safe and not d_safe:
                loss += 1
                status = "LOSS"
                if dc == "A":
                    critical_loss += 1
                    status = "CRITICAL_LOSS"
            elif not b_safe and d_safe:
                win += 1
                status = "WIN"
            else:
                tie += 1
                status = "TIE_BOTH_UNSAFE"
            pairs.append({"probe_id": k[0], "kind": k[1], "key": k[2],
                           "b_category": bc, "d_category": dc, "status": status})

        n = len(keys)
        paired = {
            "n_pairs": n,
            "win_b_unsafe_d_safe": win,
            "tie": tie,
            "loss_b_safe_d_unsafe": loss,
            "critical_loss_b_safe_d_wrongname": critical_loss,
            "loss_pct": round(loss / n * 100, 2) if n else 0,
            "critical_loss_pct": round(critical_loss / n * 100, 2) if n else 0,
            "b_genuine_wrong_pct": identity_analysis["B_HF_BF16_EAGER"]["genuine_wrong_name_pct"],
            "d_genuine_wrong_pct": identity_analysis["D_LLAMA_BF16_CPU"]["genuine_wrong_name_pct"],
            "absolute_diff_pt": round(
                identity_analysis["D_LLAMA_BF16_CPU"]["genuine_wrong_name_pct"]
                - identity_analysis["B_HF_BF16_EAGER"]["genuine_wrong_name_pct"], 2),
            "pairs": pairs,
        }
        stage2_trigger = paired["absolute_diff_pt"] >= 2.0 or paired["critical_loss_pct"] >= 1.0
        paired["stage2_trigger"] = stage2_trigger
        paired["stage2_trigger_reason"] = (
            f"absolute_diff_pt={paired['absolute_diff_pt']} (閾値2.0以上か), "
            f"critical_loss_pct={paired['critical_loss_pct']} (閾値1.0以上か)"
        )
        (REPORTS_DIR / "phase4zd_paired_analysis.json").write_text(
            json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved -> phase4zd_paired_analysis.json")
        print("stage2_trigger:", stage2_trigger, paired["stage2_trigger_reason"])
        print("absolute_diff_pt:", paired["absolute_diff_pt"], "critical_loss_pct:", paired["critical_loss_pct"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
