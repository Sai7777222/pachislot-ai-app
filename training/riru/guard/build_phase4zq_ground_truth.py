"""Phase4ZQ Section3-4: retrieval実行前にfreezeする独立ground truth。

既存の人間定義categoryを最大限再利用する(Phase4ZN/ZP由来)。retrieval結果や
router予測は一切参照しない。新規に作成した部分は無い(Section4: 既存資産の
再利用を優先)。"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
sys.path.insert(0, str(GUARD_DIR))
sys.path.insert(0, str(TRAINING_ROOT / "eval"))
REPORTS_DIR = TRAINING_ROOT / "reports"


def build_rag50():
    """既存RAG probe poolから、必須ID(P02/P04/LC-08/Q11/Q15/Q17/AD-04)とQ1-Q17を
    全件含み、50件になるまでID順で埋める。"""
    from phase4zf_rag_stress_eval import load_rag_probe_pool
    pool = load_rag_probe_pool()
    by_id = {p["id"]: p for p in pool}
    must_have = {"P02", "P04", "LC-08", "Q11", "Q15", "Q17", "AD-04"} | {f"Q{i}" for i in range(1, 18)}
    selected_ids = [pid for pid in must_have if pid in by_id]
    remaining = [p["id"] for p in pool if p["id"] not in selected_ids]
    for pid in remaining:
        if len(selected_ids) >= 50:
            break
        selected_ids.append(pid)
    rows = []
    for pid in selected_ids:
        p = by_id[pid]
        rows.append({"probe_id": f"RAG50-{pid}", "expected_mode": "PACHISLOT_FACTUAL",
                     "prompt": p["question"], "source": "phase4zf_rag_stress_eval (existing RAG50 set)",
                     "original_id": pid})
    return rows


def build_pachislot_factual_extra():
    from phase4zp_router import PACHISLOT_FACTUAL  # noqa
    import build_phase4zp_router_ground_truth as zp
    rows = []
    for i, prompt in enumerate(zp.PACHISLOT_FACTUAL_PROBES, start=1):
        rows.append({"probe_id": f"ZQ-PF{i:02d}", "expected_mode": "PACHISLOT_FACTUAL", "prompt": prompt,
                     "source": "phase4zp router ground truth (existing, human-labeled)"})
    return rows


def build_pachislot_conversational():
    import build_phase4zp_router_ground_truth as zp
    from phase4zn_unattended_probes import ALL_PROBES as ZN
    rows = []
    for i, prompt in enumerate(zp.PACHISLOT_CONVERSATIONAL_PROBES, start=1):
        rows.append({"probe_id": f"ZQ-PC{i:02d}", "expected_mode": "PACHISLOT_CONVERSATIONAL", "prompt": prompt,
                     "source": "phase4zp router ground truth (existing, human-labeled)"})
    zn_conv = [p for p in ZN if p["category"] == "pachislot_conversational"]
    for p in zn_conv:
        rows.append({"probe_id": f"ZQ-{p['id']}", "expected_mode": "PACHISLOT_CONVERSATIONAL", "prompt": p["prompt"],
                     "source": "phase4zn_unattended_probes (existing, human-labeled)"})
    return rows


def build_small_talk():
    import build_phase4zp_router_ground_truth as zp
    from phase4zn_unattended_probes import ALL_PROBES as ZN
    rows = []
    for i, prompt in enumerate(zp.SMALL_TALK_PROBES, start=1):
        rows.append({"probe_id": f"ZQ-ST{i:02d}", "expected_mode": "SMALL_TALK", "prompt": prompt,
                     "source": "phase4zp router ground truth (existing, human-labeled)"})
    smalltalk_cats = {"greeting_farewell", "emotional_casual", "personality_preference", "social_small_talk"}
    zn_st = [p for p in ZN if p["category"] in smalltalk_cats]
    for p in zn_st:
        rows.append({"probe_id": f"ZQ-{p['id']}", "expected_mode": "SMALL_TALK", "prompt": p["prompt"],
                     "source": "phase4zn_unattended_probes (existing, human-labeled)"})
    return rows


def build_ood_factual():
    import build_phase4zp_router_ground_truth as zp
    from phase4zn_unattended_probes import ALL_PROBES as ZN
    rows = []
    for i, prompt in enumerate(zp.OOD_FACTUAL_PROBES, start=1):
        rows.append({"probe_id": f"ZQ-OD{i:02d}", "expected_mode": "OOD_FACTUAL", "prompt": prompt,
                     "source": "phase4zp router ground truth (existing, human-labeled)"})
    zn_ood = [p for p in ZN if p["category"] == "ood_factual"]
    for p in zn_ood:
        rows.append({"probe_id": f"ZQ-{p['id']}", "expected_mode": "OOD_FACTUAL", "prompt": p["prompt"],
                     "source": "phase4zn_unattended_probes (existing, human-labeled)"})
    return rows


def main():
    rag50 = build_rag50()
    pf_extra = build_pachislot_factual_extra()
    pc = build_pachislot_conversational()
    st = build_small_talk()
    od = build_ood_factual()

    all_rows = rag50 + pf_extra + pc + st + od
    for row in all_rows:
        row["annotation_source"] = "reused_existing_human_labeled_category_no_retrieval_seen"
        row["frozen"] = True

    counts = {
        "PACHISLOT_FACTUAL": len(rag50) + len(pf_extra),
        "PACHISLOT_CONVERSATIONAL": len(pc), "SMALL_TALK": len(st), "OOD_FACTUAL": len(od),
    }
    out = {
        "purpose": "Phase4ZQ Section3-4: retrieval実行前にfreezeした独立ground truth。既存の人間定義"
                   "categoryを再利用(Phase4ZN/ZP由来)。retrieval結果を一切参照していない。",
        "total": len(all_rows), "counts_by_mode": counts,
        "rag50_included_ids": [r["original_id"] for r in rag50],
        "rag50_must_have_check": {
            pid: (pid in [r["original_id"] for r in rag50])
            for pid in ["P02", "P04", "LC-08", "Q11", "Q15", "Q17", "AD-04"]
        },
        "rows": all_rows,
    }
    out_path = REPORTS_DIR / "phase4zq_ground_truth.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # freeze: hash the GT content immediately after writing, before any retrieval call.
    content = out_path.read_bytes()
    h = hashlib.sha256(content).hexdigest()
    (REPORTS_DIR / "phase4zq_ground_truth_hash.txt").write_text(
        f"sha256: {h}\nfile: phase4zq_ground_truth.json\nfrozen_before_retrieval: true\n"
        f"total_rows: {len(all_rows)}\n", encoding="utf-8")

    print(f"total={len(all_rows)} counts={counts}")
    print(f"rag50_must_have_check={out['rag50_must_have_check']}")
    print(f"sha256={h}")


if __name__ == "__main__":
    main()
