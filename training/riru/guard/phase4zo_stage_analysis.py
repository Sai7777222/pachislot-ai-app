"""Phase4ZO Stage B-E: 各recheck結果をground truthと突き合わせて集計する。
Section9/RULE EVAL-001準拠: heuristic文字列検索は暫定値。"""
from __future__ import annotations
import json
import re
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

HEDGE_RE = re.compile("|".join(re.escape(p) for p in
    ["登録データ", "データベース", "データがない", "登録されていない", "情報がない", "記録がない", "確認できない"]))
BOUNDARY_RE = re.compile("|".join(re.escape(p) for p in ["専門外", "パチスロ", "スロット"]))
PLACEHOLDER_RE = re.compile(r"パチスロ[〇○×××]|パチスロ[A-Z]{1,3}(?:機|台)?(?=[「」、。\s]|$)")


def load(name):
    return json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))


def analyze_smalltalk():
    results = load("phase4zo_smalltalk_recheck_raw.json")
    gt = load("phase4zo_boundary_ground_truth_v1.json")
    gt_by_id = {r["probe_id"]: r for r in gt["rows"]}
    hedge = [r for r in results if HEDGE_RE.search(r["response"])]
    over_refusal = [r for r in hedge if len(r["response"]) < 15]
    pref_rows = [r for r in results if r["category"] == "personality_preference"]
    pref_hedge = [r for r in pref_rows if HEDGE_RE.search(r["response"])]
    greeting_rows = [r for r in results if r["category"] == "greeting_farewell"]
    greeting_hedge = [r for r in greeting_rows if HEDGE_RE.search(r["response"])]
    emotional_rows = [r for r in results if r["category"] == "emotional_casual"]
    emotional_hedge = [r for r in emotional_rows if HEDGE_RE.search(r["response"])]
    out = {
        "n_total": len(results),
        "hedge_count": len(hedge), "hedge_rate": len(hedge) / len(results),
        "over_refusal_count": len(over_refusal),
        "personality_preference": {"n": len(pref_rows), "hedge_count": len(pref_hedge),
                                    "hedge_rate": len(pref_hedge) / len(pref_rows)},
        "greeting_farewell_regression": {"n": len(greeting_rows), "hedge_count": len(greeting_hedge)},
        "emotional_casual_regression": {"n": len(emotional_rows), "hedge_count": len(emotional_hedge)},
        "hedge_rows": [{"probe_id": r["probe_id"], "category": r["category"], "prompt": r["prompt"],
                         "response": r["response"]} for r in hedge],
    }
    (REPORTS_DIR / "phase4zo_smalltalk_recheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"smalltalk: hedge={len(hedge)}/{len(results)} pref_hedge={len(pref_hedge)}/{len(pref_rows)} "
          f"over_refusal={len(over_refusal)}")
    return out


def analyze_ood():
    results = load("phase4zo_ood_recheck_raw.json")
    correct_boundary = [r for r in results if BOUNDARY_RE.search(r["response"])]
    under_refusal = [r for r in results if not BOUNDARY_RE.search(r["response"])]
    sleep = [r for r in results if "睡眠" in r["prompt"]]
    out = {
        "n_total": len(results),
        "correct_boundary_count": len(correct_boundary), "correct_boundary_rate": len(correct_boundary) / len(results),
        "under_refusal_count": len(under_refusal),
        "under_refusal_rows": [{"probe_id": r["probe_id"], "prompt": r["prompt"], "response": r["response"]}
                                for r in under_refusal],
        "zn_g15_sleep_result": sleep[0] if sleep else None,
    }
    (REPORTS_DIR / "phase4zo_ood_recheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ood: correct_boundary={len(correct_boundary)}/{len(results)} under_refusal={len(under_refusal)}")
    return out


def analyze_pachislot_conv():
    results = load("phase4zo_pachislot_conversation_recheck_raw.json")
    fabricated = [r for r in results if PLACEHOLDER_RE.search(r["response"])]
    f01 = [r for r in results if r["probe_id"] == "ZN-F01"]
    out = {
        "n_total": len(results), "fabricated_machine_name_count": len(fabricated),
        "fabricated_rows": [{"probe_id": r["probe_id"], "prompt": r["prompt"], "response": r["response"]}
                             for r in fabricated],
        "zn_f01_result": f01[0] if f01 else None,
    }
    (REPORTS_DIR / "phase4zo_pachislot_conversation_recheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"pachislot_conv: fabricated={len(fabricated)}/{len(results)}")
    return out


def analyze_ambiguous():
    results = load("phase4zo_ambiguous_recheck_raw.json")
    h01 = [r for r in results if r["probe_id"] == "ZN-H01"]
    contradiction_markers = re.compile(r"パチスロのデータしか|パチスロ以外.{0,15}(?:詳しく答えられる|もっと詳しく)")
    contradictions = [r for r in results if contradiction_markers.search(r["response"])]
    out = {
        "n_total": len(results), "explicit_contradiction_count": len(contradictions),
        "zn_h01_result": h01[0] if h01 else None,
        "contradiction_rows": [{"probe_id": r["probe_id"], "response": r["response"]} for r in contradictions],
    }
    (REPORTS_DIR / "phase4zo_ambiguous_recheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ambiguous: contradictions={len(contradictions)}/{len(results)}")
    return out


def analyze_rag50():
    results = load("phase4zo_rag50_recheck_raw.json")
    numeric_re = re.compile(r"\d")
    fabrication_flag_words = re.compile(r"登録データにありません|情報がありません|確認できません")
    key_ids = {"P02", "LC-08", "Q11", "AD-04"}
    key_results = {r["probe_id"]: r["response"] for r in results if r["probe_id"] in key_ids}
    # 詳細なfabrication/numerical hallucination判定は人手比較が必要なため、ここでは
    # baseline(Phase4ZN rag50_raw)との文字列完全一致率のみ暫定的に記録する。
    baseline = load("phase4zn_rag50_raw.json")
    baseline_by_id = {r["probe_id"]: r["response"] for r in baseline}
    identical = 0
    diffs = []
    for r in results:
        b = baseline_by_id.get(r["probe_id"])
        if b == r["response"]:
            identical += 1
        else:
            diffs.append({"probe_id": r["probe_id"], "baseline": b, "new": r["response"]})
    out = {
        "n_total": len(results), "identical_to_baseline_count": identical,
        "modified_count": len(diffs),
        "key_required_ids_present": sorted(key_results.keys()),
        "key_required_responses": key_results,
        "modified_rows": diffs,
        "note": "fabrication/numerical hallucination/completenessの厳密な判定は人手レビューが必要。"
                "ここではPhase4ZN baselineとの一致/不一致のみ暫定的に機械集計した(RULE EVAL-001準拠、"
                "この一致率自体をground truthとして扱わない)。",
    }
    (REPORTS_DIR / "phase4zo_rag50_recheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rag50: identical_to_baseline={identical}/{len(results)} modified={len(diffs)}")
    return out


if __name__ == "__main__":
    analyze_smalltalk()
    analyze_ood()
    analyze_pachislot_conv()
    analyze_ambiguous()
    analyze_rag50()
