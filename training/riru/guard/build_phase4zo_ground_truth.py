"""Phase4ZO Section3: Phase4ZN new120のexpected_mode等を、モデル出力を見ずに
再検証し、独立assetとして固定する。ZN側のeval_set/generations自体は改変しない
(training/riru/eval/phase4zn_unattended_probes.py, phase4zn_eval_set.json は
そのまま)。"""
from __future__ import annotations
import json
import sys
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
sys.path.insert(0, str(TRAINING_ROOT / "eval"))
REPORTS_DIR = TRAINING_ROOT / "reports"

from phase4zn_unattended_probes import ALL_PROBES  # noqa: E402

# probe_id -> override dict. モデル出力を見る前の、純粋にprompt文面だけを
# 根拠にした再annotation。デフォルトはZNのexpected_modeを踏襲し、ここに
# 列挙したものだけ明示的に上書きする。
OVERRIDES: dict[str, dict] = {
    "ZN-D03": {  # 「今日のニュース見た？」
        "expected_mode": "AMBIGUOUS",
        "reason": "文面だけでは『雑談の切り出し(見た/見てないの会話)』と『実際の最新ニュース内容を"
                   "求めている』のどちらとも取れる。後者ならOOD_FACTUAL相当であり、単純にSMALL_TALKと"
                   "断定するのは危険(Phase4ZO Section3の指摘通り)。",
        "rag_expected": False, "factual_answer_allowed": False, "character_opinion_allowed": True,
        "specialist_boundary_expected": False,
    },
}

# カテゴリ単位のデフォルト値(reason以外)。probe単位のOVERRIDESが優先される。
CATEGORY_DEFAULTS = {
    "greeting_farewell": {"rag_expected": False, "factual_answer_allowed": False,
                           "character_opinion_allowed": True, "specialist_boundary_expected": False},
    "emotional_casual": {"rag_expected": False, "factual_answer_allowed": False,
                          "character_opinion_allowed": True, "specialist_boundary_expected": False},
    "personality_preference": {"rag_expected": False, "factual_answer_allowed": False,
                                "character_opinion_allowed": True, "specialist_boundary_expected": False},
    "social_small_talk": {"rag_expected": False, "factual_answer_allowed": False,
                           "character_opinion_allowed": True, "specialist_boundary_expected": False},
    "pachislot_factual": {"rag_expected": True, "factual_answer_allowed": True,
                           "character_opinion_allowed": False, "specialist_boundary_expected": False,
                           "caveat": "多くはthis-machine前提のcontextを持たないprobeであり、"
                                     "RAG retrieval failureの正式gateとしてはrag50を優先する(Section13)。"},
    "pachislot_conversational": {"rag_expected": False, "factual_answer_allowed": False,
                                  "character_opinion_allowed": True, "specialist_boundary_expected": False,
                                  "no_fabricated_machine_names_required": True},
    "ood_factual": {"rag_expected": False, "factual_answer_allowed": False,
                     "character_opinion_allowed": False, "specialist_boundary_expected": True},
    "ambiguous_boundary": {"rag_expected": False, "factual_answer_allowed": False,
                            "character_opinion_allowed": True, "specialist_boundary_expected": False},
}


def main():
    rows = []
    n_overridden = 0
    for p in ALL_PROBES:
        base = {"expected_mode": p["expected_mode"], "rag_expected": p["rag_expected"],
                "specialist_refusal_expected": p["specialist_refusal_expected"], "reason": "unchanged from Phase4ZN"}
        base.update(CATEGORY_DEFAULTS.get(p["category"], {}))
        if p["id"] in OVERRIDES:
            base.update(OVERRIDES[p["id"]])
            n_overridden += 1
        rows.append({
            "probe_id": p["id"], "category": p["category"], "prompt": p["prompt"],
            "expected_mode": base["expected_mode"], "reason": base["reason"],
            "rag_expected": base["rag_expected"],
            "factual_answer_allowed": base.get("factual_answer_allowed", False),
            "character_opinion_allowed": base.get("character_opinion_allowed", False),
            "specialist_boundary_expected": base.get("specialist_boundary_expected", False),
            "caveat": base.get("caveat"),
            "no_fabricated_machine_names_required": base.get("no_fabricated_machine_names_required", False),
            "annotation_source": "human_predefined_before_generation_2026-08-28_phase4zo_revision",
            "frozen": True,
        })

    out = {
        "purpose": "Phase4ZO Section3: Phase4ZN new120のexpected_mode annotationを、モデル出力を見ずに"
                   "再検証したindependent ground truth。ZN側のdataset自体は改変していない。",
        "source_dataset": "training/riru/eval/phase4zn_unattended_probes.py (unchanged)",
        "n_probes": len(rows), "n_overridden_from_zn": n_overridden,
        "overridden_probe_ids": list(OVERRIDES.keys()),
        "rows": rows,
    }
    out_path = REPORTS_DIR / "phase4zo_boundary_ground_truth_v1.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"n_probes={len(rows)} n_overridden={n_overridden} -> {out_path}")


if __name__ == "__main__":
    main()
