# ruff: noqa: E501
"""Phase 4I: prompt_experiment結果の定量分析。

- 4I-2: 再現性確認 (seed 42/43/44でのQ3回答比較)
- 4I-3/4/5: prompt A/B/C/D x {base, v2} x {Q3 + P01-P10} の
  重要情報網羅率 (relevant recall) / 不要情報混入率 / 平均回答長を集計する。
"""

from __future__ import annotations

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "phase4i_prompt_experiment_results.json"
HOLDOUT_PATH = EVAL_DIR / "phase4i_holdout_omission_v2.json"

# Q3 (実際の本番ケース) のground truth。分析専用にここで定義する
# (build_phase4h_dataset.py側のOMISSION_GROUND_TRUTHと同一定義)。
Q3_GROUND_TRUTH = {
    "key_facts": ["510G", "1000G", "1480G", "15.2%", "20.3%", "64.5%"],
    "irrelevant_markers": ["裏天国", "天空の扉", "雷雨", "0テンパイ"],
}


def load_ground_truth() -> dict:
    gt = {"Q3": Q3_GROUND_TRUTH}
    holdout = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    for item in holdout:
        gt[item["id"]] = {
            "key_facts": item["key_facts"],
            "irrelevant_markers": item["irrelevant_markers"],
        }
    return gt


def main() -> int:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    ground_truth = load_ground_truth()

    # --- 4I-2: 再現性確認 ---
    repro = results["reproducibility_check"]
    repro_texts = {k: v["text"] for k, v in repro.items()}
    all_identical = len(set(repro_texts.values())) == 1
    repro_report = {
        "texts_by_seed": repro_texts,
        "all_seeds_identical_output": all_identical,
        "note": (
            "temperature=0.3・seed固定でも do_sample=True のため、"
            "異なるseed間で完全一致するとは限らない。同一seedなら再現するはずだが、"
            "ここではseed 42/43/44 (異なるseed) を比較しているため、"
            "多少の表現ゆれは正常。key_fact網羅の有無が一貫しているかに注目する。"
        ),
    }
    for k, v in repro.items():
        text = v["text"]
        found = [f for f in Q3_GROUND_TRUTH["key_facts"] if f in text]
        repro_report.setdefault("key_fact_coverage_by_seed", {})[k] = {
            "coverage_pct": round(len(found) / len(Q3_GROUND_TRUTH["key_facts"]) * 100, 1),
            "found": found,
        }

    # --- 4I-3/4/5: prompt sweep 分析 ---
    sweep = results["prompt_sweep"]
    condition_totals: dict[str, dict[str, float]] = {}
    per_item_summary: dict[str, dict] = {}

    for item_id, item_data in sweep.items():
        gt = ground_truth.get(item_id)
        if gt is None:
            continue
        per_item_summary[item_id] = {"question": item_data["question"], "conditions": {}}
        for cond_key, gen in item_data["conditions"].items():
            text = gen["text"]
            key_found = [k for k in gt["key_facts"] if k in text]
            irr_found = [k for k in gt["irrelevant_markers"] if k in text]
            coverage_pct = round(len(key_found) / len(gt["key_facts"]) * 100, 1)
            leak_pct = round(len(irr_found) / max(len(gt["irrelevant_markers"]), 1) * 100, 1)
            per_item_summary[item_id]["conditions"][cond_key] = {
                "text": text,
                "key_facts_found": key_found,
                "key_facts_missing": [k for k in gt["key_facts"] if k not in text],
                "key_fact_coverage_pct": coverage_pct,
                "irrelevant_markers_leaked": irr_found,
                "irrelevant_leak_pct": leak_pct,
                "length": len(text),
            }
            t = condition_totals.setdefault(
                cond_key,
                {"key_hits": 0, "key_total": 0, "irr_hits": 0, "irr_total": 0, "lengths": []},
            )
            t["key_hits"] += len(key_found)
            t["key_total"] += len(gt["key_facts"])
            t["irr_hits"] += len(irr_found)
            t["irr_total"] += len(gt["irrelevant_markers"])
            t["lengths"].append(len(text))

    condition_summary = {}
    for cond_key, t in condition_totals.items():
        n = len(t["lengths"])
        condition_summary[cond_key] = {
            "key_fact_coverage_rate_pct": round(t["key_hits"] / max(t["key_total"], 1) * 100, 1),
            "irrelevant_leak_rate_pct": round(t["irr_hits"] / max(t["irr_total"], 1) * 100, 1),
            "avg_length": round(sum(t["lengths"]) / n, 1) if n else 0,
        }

    # Q3のみの条件別サマリ (本命ケースなので単独でも見られるようにする)
    q3_only = {
        cond: per_item_summary["Q3"]["conditions"][cond]
        for cond in per_item_summary.get("Q3", {}).get("conditions", {})
    }

    out = {
        "reproducibility_check": repro_report,
        "condition_summary_all_11_items": condition_summary,
        "q3_only_by_condition": q3_only,
        "per_item_detail": per_item_summary,
    }
    (EVAL_DIR / "phase4i_prompt_experiment_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved analysis")
    print(json.dumps({"condition_summary_all_11_items": condition_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
