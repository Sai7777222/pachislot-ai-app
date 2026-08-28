"""Phase4ZR Stage A-D: conservative dispatchを全260probeへ適用し、precision/recall/
UNKNOWN率、dangerous misroute、RAG50安全性、small-talk/OOD coverageを算出する。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import Counter

GUARD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GUARD_DIR))
REPORTS_DIR = GUARD_DIR.parent / "reports"

from phase4zr_conservative_dispatch import dispatch, UNKNOWN  # noqa: E402
from phase4zp_router import PACHISLOT_FACTUAL, PACHISLOT_CONVERSATIONAL, SMALL_TALK, OOD_FACTUAL  # noqa: E402

MODES = [SMALL_TALK, PACHISLOT_FACTUAL, PACHISLOT_CONVERSATIONAL, OOD_FACTUAL]


def main():
    gt = json.loads((REPORTS_DIR / "phase4zr_ground_truth.json").read_text(encoding="utf-8"))
    rows = gt["rows"]

    results = []
    for row in rows:
        r = dispatch(row["prompt"])
        results.append({"probe_id": row["probe_id"], "expected_mode": row["expected_mode"],
                         "prompt": row["prompt"], "dispatched_mode": r.mode, "confident": r.confident,
                         "matched_rule": r.matched_rule, "matched_keyword": r.matched_keyword,
                         "is_rag50": row["probe_id"].startswith("RAG50-"),
                         "original_id": row.get("original_id")})

    (REPORTS_DIR / "phase4zr_dispatch_results.json").write_text(
        json.dumps({"n_total": len(results), "rows": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- confusion matrix (UNKNOWN as its own column) ---
    confusion = {m: {**{k: 0 for k in MODES}, UNKNOWN: 0} for m in MODES}
    for r in results:
        confusion[r["expected_mode"]][r["dispatched_mode"]] += 1
    (REPORTS_DIR / "phase4zr_confusion_matrix.json").write_text(
        json.dumps(confusion, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- per-mode precision/recall/unknown rate ---
    per_mode = {}
    dangerous_misroutes = []  # PACHISLOT_FACTUAL dispatched to SMALL_TALK or OOD_FACTUAL
    for m in MODES:
        total = confusion[m]
        n = sum(total.values())
        correct = total[m]
        unk = total[UNKNOWN]
        # precision: among rows dispatched to m (regardless of expected), how many were truly m
        dispatched_to_m = sum(confusion[e][m] for e in MODES)
        tp = confusion[m][m]
        precision = tp / dispatched_to_m if dispatched_to_m else None
        recall = correct / n if n else None
        unknown_rate = unk / n if n else None
        per_mode[m] = {"n": n, "recall": recall, "precision": precision, "unknown_rate": unknown_rate,
                        "correct": correct, "unknown": unk}

    for r in results:
        if r["expected_mode"] == PACHISLOT_FACTUAL and r["dispatched_mode"] in (SMALL_TALK, OOD_FACTUAL):
            dangerous_misroutes.append(r)

    rag50 = [r for r in results if r["is_rag50"]]
    rag50_dangerous = [r for r in rag50 if r["dispatched_mode"] in (SMALL_TALK, OOD_FACTUAL)]
    rag50_safe = [r for r in rag50 if r["dispatched_mode"] in (PACHISLOT_FACTUAL, UNKNOWN)]
    mandatory = ["P02", "LC-08", "Q11", "Q17", "AD-04"]
    mandatory_routing = {pid: next((r["dispatched_mode"] for r in rag50 if r["original_id"] == pid), None)
                          for pid in mandatory}

    rag50_out = {
        "purpose": "Stage B: RAG50全件がPACHISLOT_FACTUALまたはUNKNOWNのいずれかであることを確認する"
                   "(SMALL_TALK/OODへのdangerous misrouteは0が目標)。",
        "n_total": len(rag50), "dangerous_misroute_count": len(rag50_dangerous),
        "safe_count": len(rag50_safe),
        "dangerous_misroute_rows": rag50_dangerous,
        "mandatory_probe_routing": mandatory_routing,
        "mandatory_all_safe": all(v not in (SMALL_TALK, OOD_FACTUAL) for v in mandatory_routing.values()),
        "mandatory_all_ideal_factual_or_unknown": all(v in (PACHISLOT_FACTUAL, UNKNOWN) for v in mandatory_routing.values()),
        "target": "dangerous misroute(SMALL_TALK/OOD_FACTUALへの誤route) = 0/50 (Section8の定義通り)。"
                   "PACHISLOT_CONVERSATIONALはdangerousではないが、RAG groundingが保証されないという"
                   "意味でidealではない(mandatory_all_ideal_factual_or_unknownで別途追跡)。",
    }
    (REPORTS_DIR / "phase4zr_rag50_safety.json").write_text(
        json.dumps(rag50_out, ensure_ascii=False, indent=2), encoding="utf-8")

    smalltalk_rows = [r for r in results if r["expected_mode"] == SMALL_TALK]
    st_out = {
        "purpose": "Stage C: SMALL_TALK65+ZP-GT30(計95件)の直接分類率・UNKNOWN率・危険な誤route率。",
        "n_total": len(smalltalk_rows),
        "direct_small_talk": sum(1 for r in smalltalk_rows if r["dispatched_mode"] == SMALL_TALK),
        "unknown": sum(1 for r in smalltalk_rows if r["dispatched_mode"] == UNKNOWN),
        "misrouted_to_pachislot_factual": sum(1 for r in smalltalk_rows if r["dispatched_mode"] == PACHISLOT_FACTUAL),
        "misrouted_to_other": [r for r in smalltalk_rows if r["dispatched_mode"] not in (SMALL_TALK, UNKNOWN)],
    }
    (REPORTS_DIR / "phase4zr_smalltalk_coverage.json").write_text(
        json.dumps(st_out, ensure_ascii=False, indent=2), encoding="utf-8")

    ood_rows = [r for r in results if r["expected_mode"] == OOD_FACTUAL]
    ood_out = {
        "purpose": "Stage D: OOD15+ZP-GT30(計45件)の直接分類率・UNKNOWN率・危険な誤route率(PACHISLOTへ送ることを最重要で防止)。",
        "n_total": len(ood_rows),
        "direct_ood": sum(1 for r in ood_rows if r["dispatched_mode"] == OOD_FACTUAL),
        "unknown": sum(1 for r in ood_rows if r["dispatched_mode"] == UNKNOWN),
        "misrouted_to_pachislot_factual": sum(1 for r in ood_rows if r["dispatched_mode"] == PACHISLOT_FACTUAL),
        "misrouted_to_pachislot_conversational": sum(1 for r in ood_rows if r["dispatched_mode"] == PACHISLOT_CONVERSATIONAL),
        "misrouted_rows": [r for r in ood_rows if r["dispatched_mode"] in (PACHISLOT_FACTUAL, PACHISLOT_CONVERSATIONAL)],
    }
    (REPORTS_DIR / "phase4zr_ood_coverage.json").write_text(
        json.dumps(ood_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("per_mode:", json.dumps(per_mode, ensure_ascii=False, indent=2))
    print("dangerous_misroutes (PACHISLOT_FACTUAL -> SMALL_TALK/OOD):", len(dangerous_misroutes))
    print("RAG50 dangerous:", len(rag50_dangerous), "/", len(rag50))
    print("mandatory_routing:", mandatory_routing)
    print("smalltalk direct/unknown:", st_out["direct_small_talk"], "/", st_out["unknown"])
    print("ood direct/unknown:", ood_out["direct_ood"], "/", ood_out["unknown"])

    return per_mode, dangerous_misroutes


if __name__ == "__main__":
    main()
