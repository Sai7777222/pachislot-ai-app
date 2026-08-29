"""Phase4FV Stage C(phantom entity, 22probes) + Stage D(concept binding, 12probes)。
Stage Aの結論(P1/P2ともに既知失敗ゲートを達成できず)を踏まえ、Section23の必須報告項目を
埋めるため、最も改善幅が大きかったP2候補で規模を広げた回帰評価を行う。P0は比較のため
新規14/9probeのみ生成し、既存8/3probeはPhase4FUのデータを再利用する。"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
REPORTS_DIR = TRAINING_ROOT / "reports"
sys.path.insert(0, str(GUARD_DIR))
from run_phase4fv_stage_ab import load_model, generate, analyze, build_context_string  # noqa: E402

P2_PATH = GUARD_DIR / "phase4fv_prompts" / "p2_explicit_entity_binding.jinja2"
PROJECT_ROOT = GUARD_DIR.parents[2]
P0_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"


def main():
    p0 = P0_PATH.read_text(encoding="utf-8")
    p2 = P2_PATH.read_text(encoding="utf-8")

    phantom_new = json.loads((REPORTS_DIR / "phase4fv_phantom_new_contexts.json").read_text(encoding="utf-8"))
    concept_new = json.loads((REPORTS_DIR / "phase4fv_concept_new_contexts.json").read_text(encoding="utf-8"))
    fu_precomputed = json.loads((REPORTS_DIR / "phase4fu_precomputed_contexts.json").read_text(encoding="utf-8"))
    fu_by_id = {p["id"]: p for p in fu_precomputed}
    fu_stage_b = json.loads((REPORTS_DIR / "phase4fu_stage_b_base_generations.json").read_text(encoding="utf-8"))
    fu_stage_b_by_id = {r["id"]: r for r in fu_stage_b}

    reused_phantom_ids = ["FU-A03", "FU-B03", "FU-B04", "FU-B05", "FU-D05", "FU-F03", "FU-F04", "FU-F05"]
    reused_concept_ids = ["FU-D01", "FU-D03", "FU-D05"]

    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    # ---- Stage C: phantom entity ----
    stage_c = []
    for pid in reused_phantom_ids:
        p = fu_by_id[pid]
        base = fu_stage_b_by_id[pid]
        ctx = build_context_string(p["retrieved_chunks"])
        p2_text = generate(model, tokenizer, p2, p["prompt"], context=ctx, seed=42, do_sample=False)
        stage_c.append({"id": pid, "prompt": p["prompt"], "source": "reused_from_FU",
                         "P0": {"response": base["response"], "has_numeric_claim": base["has_numeric_claim"], "abstain_or_hedge": base["abstain_or_hedge"]},
                         "P2": analyze(p2_text)})
        print(f"[StageC-reused] {pid} done")
    for p in phantom_new:
        ctx = build_context_string(p["retrieved_chunks"])
        p0_text = generate(model, tokenizer, p0, p["prompt"], context=ctx, seed=42, do_sample=False)
        p2_text = generate(model, tokenizer, p2, p["prompt"], context=ctx, seed=42, do_sample=False)
        stage_c.append({"id": p["id"], "prompt": p["prompt"], "type": p["type"], "source": "new",
                         "P0": analyze(p0_text), "P2": analyze(p2_text)})
        print(f"[StageC-new] {p['id']} done")
    (REPORTS_DIR / "phase4fv_phantom_entity.json").write_text(
        json.dumps(stage_c, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Stage D: concept binding ----
    stage_d = []
    for pid in reused_concept_ids:
        p = fu_by_id[pid]
        base = fu_stage_b_by_id[pid]
        ctx = build_context_string(p["retrieved_chunks"])
        p2_text = generate(model, tokenizer, p2, p["prompt"], context=ctx, seed=42, do_sample=False)
        stage_d.append({"id": pid, "prompt": p["prompt"], "source": "reused_from_FU",
                         "P0": {"response": base["response"], "has_numeric_claim": base["has_numeric_claim"], "abstain_or_hedge": base["abstain_or_hedge"]},
                         "P2": analyze(p2_text)})
        print(f"[StageD-reused] {pid} done")
    for p in concept_new:
        ctx = build_context_string(p["retrieved_chunks"])
        p0_text = generate(model, tokenizer, p0, p["prompt"], context=ctx, seed=42, do_sample=False)
        p2_text = generate(model, tokenizer, p2, p["prompt"], context=ctx, seed=42, do_sample=False)
        stage_d.append({"id": p["id"], "prompt": p["prompt"], "pair": p["pair"], "source": "new",
                         "P0": analyze(p0_text), "P2": analyze(p2_text)})
        print(f"[StageD-new] {p['id']} done")
    (REPORTS_DIR / "phase4fv_concept_binding.json").write_text(
        json.dumps(stage_d, ensure_ascii=False, indent=2), encoding="utf-8")

    print("STAGE C/D DONE")


if __name__ == "__main__":
    main()
