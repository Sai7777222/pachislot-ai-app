"""Phase4FM Section22: GT260ルーター回帰確認。
dispatch()自体はFC4/FM双方で無変更(routing vocabulary凍結)。モデレーションは
dispatchより前の別レイヤーであり、GT260の各行はいずれも合成テストマーカーを
含まないため、モデレーションはこの検証結果に一切影響しない(Section22の指示
『ブロック対象の合成入力に基づいてrouter指標を再定義しない』を字義通り守り、
GT260自体にはモデレーションを適用せず、純粋なdispatch() driftのみを見る)。"""
from __future__ import annotations
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
REPORTS_DIR = PROJECT_ROOT / "training" / "riru" / "reports"

from pachislot_ai.dispatch import dispatch  # noqa: E402

DANGEROUS_MISROUTE_TARGETS = {"SMALL_TALK", "OOD_FACTUAL"}


def main():
    gt = json.loads((REPORTS_DIR / "phase4zt_ground_truth.json").read_text(encoding="utf-8"))
    fc4 = json.loads((REPORTS_DIR / "phase4fc4_router_gt260.json").read_text(encoding="utf-8"))
    fc4_mode_by_id = {r["id"]: r["mode"] for r in fc4["rows"]}

    rows_out = []
    mode_counts = {}
    dangerous_misroute = 0
    drift_vs_fc4 = 0
    for row in gt["rows"]:
        query = row["prompt"]
        expected_mode = row.get("expected_mode")
        rid = row.get("probe_id")
        result = dispatch(query)
        mode = result.mode
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if expected_mode == "PACHISLOT_FACTUAL" and mode in DANGEROUS_MISROUTE_TARGETS:
            dangerous_misroute += 1
        fc4_mode = fc4_mode_by_id.get(rid)
        if fc4_mode is not None and fc4_mode != mode:
            drift_vs_fc4 += 1
        rows_out.append({
            "id": rid, "prompt": query, "expected_mode": expected_mode,
            "mode": mode, "matched_rule": result.matched_rule, "fc4_mode": fc4_mode,
        })

    out = {
        "phase": "Phase4FM",
        "section": "Section22 - routing regression (GT260)",
        "n_total": len(rows_out),
        "no_router_tuning": True,
        "moderation_applied_to_gt260": False,
        "moderation_note": "Section22の指示通り、GT260自体にはモデレーションを適用していない"
                            "(全260件とも合成テストマーカーを含まないため、適用しても結果は不変)。",
        "dangerous_misroute": dangerous_misroute,
        "mode_counts": mode_counts,
        "drift_vs_fc4_dispatch_result": drift_vs_fc4,
        "fc4_dangerous_misroute_reference": fc4.get("dangerous_misroute"),
        "gate_verdict": "PASS" if dangerous_misroute == 0 and drift_vs_fc4 == 0 else "FAIL",
        "rows": rows_out,
    }
    out_path = REPORTS_DIR / "phase4fm_router_regression.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"n_total={out['n_total']} dangerous_misroute={dangerous_misroute} drift_vs_fc4={drift_vs_fc4} mode_counts={mode_counts}")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
