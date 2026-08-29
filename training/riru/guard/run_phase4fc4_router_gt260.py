"""Phase4FC4 Section18 (Stage I): GT260ルーター回帰確認。
dispatch()自体はFC4で変更していない(routing vocabulary凍結)ため、新しい研究
ではなく「変更していないことの検証」として、現行コードでdispatch()を再実行し
FC3時点の結果とdrift 0であることを確認する。生成は一切行わない。"""
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
    fc3 = json.loads((REPORTS_DIR / "phase4fc3_router_gt260.json").read_text(encoding="utf-8"))
    fc3_mode_by_id = {r["probe_id"]: r["production_mode"] for r in fc3["rows"]}

    rows_out = []
    mode_counts = {}
    dangerous_misroute = 0
    drift_vs_fc3 = 0
    for row in gt["rows"]:
        query = row["prompt"]
        expected_mode = row.get("expected_mode")
        rid = row.get("probe_id")
        result = dispatch(query)
        mode = result.mode
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        # 「危険な誤配送」= 期待値がPACHISLOT_FACTUALなのに実際はSMALL_TALK/OOD_FACTUALへ
        # ルーティングされ、数値回答が必要な質問なのにconfident-boundary扱いされてしまうケース
        if expected_mode == "PACHISLOT_FACTUAL" and mode in DANGEROUS_MISROUTE_TARGETS:
            dangerous_misroute += 1
        fc3_mode = fc3_mode_by_id.get(rid)
        if fc3_mode is not None and fc3_mode != mode:
            drift_vs_fc3 += 1
        rows_out.append({
            "id": rid, "prompt": query, "expected_mode": expected_mode,
            "mode": mode, "matched_rule": result.matched_rule, "fc3_mode": fc3_mode,
        })

    out = {
        "phase": "Phase4FC4",
        "section": "Section18 - Stage I: GT260 routing regression",
        "n_total": len(rows_out),
        "no_router_tuning": True,
        "dangerous_misroute": dangerous_misroute,
        "mode_counts": mode_counts,
        "drift_vs_fc3_dispatch_result": drift_vs_fc3,
        "fc3_dangerous_misroute_reference": fc3.get("dangerous_misroute"),
        "fc3_unknown_count_reference": fc3.get("unknown_count"),
        "gate_verdict": "PASS" if dangerous_misroute == 0 else "FAIL",
        "rows": rows_out,
    }
    out_path = REPORTS_DIR / "phase4fc4_router_gt260.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"n_total={out['n_total']} dangerous_misroute={dangerous_misroute} mode_counts={mode_counts}")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
