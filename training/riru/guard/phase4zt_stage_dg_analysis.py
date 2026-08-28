"""Phase4ZT Stage D-G: 選定されたPolicy C(C3)を使い、direct-mode(既存ZP output、無変更)
とUNKNOWN-mode(C3で今回生成)を結合してsmall-talk65/OOD15/RAG50/conversational10の
完全なregressionを構成する。追加generationは不要(既存C3 unknown77.jsonとZPのdirect outputを
組み合わせるのみ)。"""
from __future__ import annotations
import json
import re
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
HEDGE_RE = re.compile("|".join(re.escape(p) for p in
    ["登録データ", "データベース", "データがない", "登録されていない", "情報がない", "記録がない", "確認できない"]))
BOUNDARY_RE = re.compile("|".join(re.escape(p) for p in ["専門外", "パチスロ", "スロット", "専門分野", "専門家"]))
PLACEHOLDER_RE = re.compile(r"パチスロ[〇○×××]|パチスロ[A-Z]{1,3}(?:機|台)?(?=[「」、。\s]|$)")


def load(name):
    return json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))


def main():
    dispatch = {r["probe_id"]: r for r in load("phase4zr_dispatch_results.json")["rows"]}
    c3_unknown = {r["probe_id"]: r for r in load("phase4zt_unknown77.json")["C3"]}
    smalltalk_direct = {r["probe_id"]: r for r in load("phase4zp_smalltalk_recheck_raw.json")}
    ood_direct = {r["probe_id"]: r for r in load("phase4zp_ood_recheck_raw.json")}
    conv_direct = {r["probe_id"]: r for r in load("phase4zp_pachislot_conversation_recheck_raw.json")}
    rag50_baseline = {r["probe_id"]: r for r in load("phase4zn_rag50_raw.json")}

    # --- Stage D: small-talk65 ---
    gt = load("phase4zt_ground_truth.json")
    zn_st_ids = [r["probe_id"] for r in gt["rows"] if r["probe_id"].startswith("ZQ-ZN-") and r["expected_mode"] == "SMALL_TALK"]
    st_rows = []
    for pid in zn_st_ids:
        orig_id = pid.replace("ZQ-", "")
        route = dispatch.get(pid, {}).get("dispatched_mode")
        if route == "SMALL_TALK":
            r = smalltalk_direct.get(orig_id)
            resp = r["response"] if r else None
            source = "direct_smalltalk_policy_ZP_unchanged"
        else:  # UNKNOWN
            r = c3_unknown.get(pid)
            resp = r["response"] if r else None
            source = "policy_C3"
        if resp is None:
            continue
        st_rows.append({"probe_id": orig_id, "route": route, "source": source, "response": resp,
                         "hedge": bool(HEDGE_RE.search(resp)), "placeholder": bool(PLACEHOLDER_RE.search(resp))})
    n_st = len(st_rows)
    st_hedge = sum(1 for r in st_rows if r["hedge"])
    st_out = {"purpose": "Stage D: ZN small-talk65 (direct-mode=既存ZP output無変更 + UNKNOWN=Policy C3)",
              "n_total": n_st, "hedge_count": st_hedge, "hedge_rate": st_hedge / n_st if n_st else None,
              "target": "<=5%", "placeholder_count": sum(1 for r in st_rows if r["placeholder"]), "rows": st_rows}
    (REPORTS_DIR / "phase4zt_smalltalk_recheck.json").write_text(json.dumps(st_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stage D: n={n_st} hedge={st_hedge} rate={st_hedge/n_st:.3f}")

    # --- Stage E: OOD15 ---
    zn_ood_ids = [r["probe_id"] for r in gt["rows"] if r["probe_id"].startswith("ZQ-ZN-") and r["expected_mode"] == "OOD_FACTUAL"]
    ood_rows = []
    for pid in zn_ood_ids:
        orig_id = pid.replace("ZQ-", "")
        route = dispatch.get(pid, {}).get("dispatched_mode")
        if route == "OOD_FACTUAL":
            r = ood_direct.get(orig_id)
            resp = r["response"] if r else None
            source = "direct_ood_policy_ZP_unchanged"
        else:  # UNKNOWN
            r = c3_unknown.get(pid)
            resp = r["response"] if r else None
            source = "policy_C3"
        if resp is None:
            continue
        ood_rows.append({"probe_id": orig_id, "route": route, "source": source, "response": resp,
                          "boundary_marker": bool(BOUNDARY_RE.search(resp))})
    n_ood = len(ood_rows)
    ood_correct = sum(1 for r in ood_rows if r["boundary_marker"])
    ood_out = {"purpose": "Stage E: ZN OOD15 (direct-mode=既存ZP output無変更 + UNKNOWN=Policy C3)",
               "n_total": n_ood, "correct_boundary_count": ood_correct, "target": ">=14/15", "rows": ood_rows}
    (REPORTS_DIR / "phase4zt_ood_recheck.json").write_text(json.dumps(ood_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stage E: n={n_ood} correct_boundary(auto)={ood_correct}")

    # --- Stage F: RAG50 (direct PACHISLOT_FACTUAL unchanged + LC-08 UNKNOWN via C3) ---
    gt_rag = [r for r in gt["rows"] if r["probe_id"].startswith("RAG50-")]
    rag_rows = []
    for r in gt_rag:
        orig_id = r["original_id"]
        route = dispatch.get(r["probe_id"], {}).get("dispatched_mode")
        if route == "PACHISLOT_FACTUAL":
            base = rag50_baseline.get(orig_id)
            resp = base["response"] if base else None
            source = "direct_pachislot_factual_unchanged"
        elif route == "UNKNOWN":
            c3r = c3_unknown.get(r["probe_id"])
            resp = c3r["response"] if c3r else None
            source = "policy_C3"
        else:
            resp = None
            source = f"routed_to_{route}_not_covered_in_rag50_gt"
        rag_rows.append({"probe_id": orig_id, "route": route, "source": source, "response": resp})
    mandatory = ["P02", "LC-08", "Q11", "Q17", "AD-04"]
    mandatory_status = {m: next((r for r in rag_rows if r["probe_id"] == m), None) for m in mandatory}
    rag_out = {"purpose": "Stage F: RAG50 golden regression (direct PACHISLOT_FACTUAL=既存Phase4ZN raw output"
                          "無変更 + UNKNOWN[LC-08のみ]=Policy C3)。",
               "n_total": len(rag_rows), "mandatory_probes": mandatory_status, "rows": rag_rows}
    (REPORTS_DIR / "phase4zt_rag50_recheck.json").write_text(json.dumps(rag_out, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "phase4zt_rag_equivalence.json").write_text(json.dumps({
        "purpose": "既存PACHISLOT_FACTUAL directパスは、prompt/context/generation configとも一切変更していない"
                   "(Policy Cはdispatch=UNKNOWNの場合のみ介入し、確信度の高いPACHISLOT_FACTUALには一切触れない設計)。",
        "unchanged_by_construction": True,
        "evidence": "phase4zt_path_trace.jsonのdirect-mode traceにおいて、is_unknown=falseの場合"
                    "selected_policy='existing_mode_specific_prompt (Phase4ZP, unchanged)'固定であり、"
                    "Policy Cのいずれのvariantも関与しない。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stage F: mandatory routes = {{k: v['route'] for k, v in mandatory_status.items()}}" if False else
          f"Stage F mandatory: {[(k, v['route'] if v else None) for k, v in mandatory_status.items()]}")

    # --- Stage G: pachislot conversational10 ---
    conv_ids = [r["probe_id"] for r in gt["rows"] if r["probe_id"].startswith("ZQ-ZN-F")]
    conv_rows = []
    for pid in conv_ids:
        orig_id = pid.replace("ZQ-", "")
        route = dispatch.get(pid, {}).get("dispatched_mode")
        if route == "PACHISLOT_CONVERSATIONAL":
            r = conv_direct.get(orig_id)
            resp = r["response"] if r else None
            source = "direct_conversational_unchanged"
        else:
            r = c3_unknown.get(pid)
            resp = r["response"] if r else None
            source = "policy_C3"
        if resp is None:
            continue
        conv_rows.append({"probe_id": orig_id, "route": route, "source": source, "response": resp,
                           "placeholder": bool(PLACEHOLDER_RE.search(resp))})
    conv_out = {"purpose": "Stage G: pachislot conversational10 (direct=既存ZP output無変更 + UNKNOWN=Policy C3)",
                "n_total": len(conv_rows), "fabricated_machine_name_count": sum(1 for r in conv_rows if r["placeholder"]),
                "rows": conv_rows}
    (REPORTS_DIR / "phase4zt_pachislot_conversation.json").write_text(json.dumps(conv_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stage G: n={len(conv_rows)} fabricated={conv_out['fabricated_machine_name_count']}")


if __name__ == "__main__":
    main()
