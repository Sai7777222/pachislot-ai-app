"""Phase 4ZE Section13,17: identity評価結果の分類・集計・regression比較。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
sys.path.insert(0, str(EVAL_DIR))
REPORTS_DIR = TRAINING_ROOT / "reports"

from phase4z_naming_classify import classify_naming  # noqa: E402


def flatten(data: dict) -> list[dict]:
    out = []
    for pid, entry in data["results"].items():
        out.append({"probe_id": pid, "set": entry.get("set"), "kind": "greedy", "key": "greedy",
                     "text": entry["greedy"]})
        for seed, text in entry.get("sampled", {}).items():
            out.append({"probe_id": pid, "set": entry.get("set"), "kind": "sampled", "key": seed,
                         "text": text})
    return out


def classify_all(flat: list[dict], is_naming_context: bool = True) -> list[dict]:
    out = []
    for item in flat:
        cls = classify_naming(item["text"], is_naming_context=is_naming_context)
        out.append({**item, "category": cls["category"], "reason": cls.get("reason")})
    return out


def summarize(classified: list[dict]) -> dict:
    n = len(classified)
    counts = {}
    for c in classified:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    genuine_wrong = counts.get("A", 0)
    correct = counts.get("E", 0)
    hedge = counts.get("B", 0)
    placeholder = counts.get("C", 0)
    generic = counts.get("D", 0)
    intrusion = counts.get("F", 0)
    other = counts.get("G", 0)
    return {
        "n_total": n, "category_counts": counts,
        "category_pct": {k: round(v / n * 100, 2) for k, v in counts.items()} if n else {},
        "genuine_wrong_name_pct": round(genuine_wrong / n * 100, 2) if n else 0,
        "correct_name_pct": round(correct / n * 100, 2) if n else 0,
        "hedge_pct": round(hedge / n * 100, 2) if n else 0,
        "placeholder_pct": round(placeholder / n * 100, 2) if n else 0,
        "generic_role_pct": round(generic / n * 100, 2) if n else 0,
        "identity_intrusion_pct": round(intrusion / n * 100, 2) if n else 0,
        "other_no_name_pct": round(other / n * 100, 2) if n else 0,
    }


def main() -> int:
    out_all = {}
    for attn_impl in ["eager", "sdpa"]:
        f = REPORTS_DIR / f"phase4ze_identity_{attn_impl}.json"
        if not f.exists():
            print(f"missing: {f}")
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        flat = flatten(data)
        classified = classify_all(flat, is_naming_context=True)
        summary = summarize(classified)
        # by-set breakdown
        by_set = {}
        sets = sorted({c["set"] for c in classified})
        for s in sets:
            sub = [c for c in classified if c["set"] == s]
            by_set[s] = summarize(sub)
        out_all[attn_impl] = {"summary": summary, "by_set": by_set}
        print(attn_impl, summary)
        (REPORTS_DIR / f"_phase4ze_identity_classified_{attn_impl}.json").write_text(
            json.dumps(classified, ensure_ascii=False, indent=2), encoding="utf-8")

    (REPORTS_DIR / "phase4ze_identity_analysis.json").write_text(
        json.dumps(out_all, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved -> phase4ze_identity_analysis.json")

    # identity intrusion via regression probes (non-naming context: does the model volunteer
    # its name unprompted?) -- checked separately via classify_naming(is_naming_context=False)
    intrusion_summary = {}
    for attn_impl in ["eager"]:
        f = REPORTS_DIR / f"phase4ze_regression_{attn_impl}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        flat = []
        for pid, entry in data["results"].items():
            flat.append({"probe_id": pid, "kind": "greedy", "text": entry["greedy"]})
            for seed, text in entry.get("sampled", {}).items():
                flat.append({"probe_id": pid, "kind": "sampled", "text": text})
        classified = classify_all(flat, is_naming_context=False)
        n_intrusion = sum(1 for c in classified if c["category"] == "F")
        intrusion_summary[attn_impl] = {"n_total": len(classified), "n_intrusion": n_intrusion,
                                          "intrusion_pct": round(n_intrusion / len(classified) * 100, 2) if classified else 0}
    (REPORTS_DIR / "phase4ze_regression_intrusion_check.json").write_text(
        json.dumps(intrusion_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("intrusion (regression probes, non-naming context):", intrusion_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
