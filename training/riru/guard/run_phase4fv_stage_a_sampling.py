"""Phase4FV Stage A(known failure regression)の本実行。
greedy decodingではP1/P2の追加指示によって退化的反復ループが生じることが分かったため
(phase4fv_q6_sampling_diagnostic.jsonで確認済み)、本番の実運用設定(temperature=0.7 sampling、
Settings.llm_temperature=0.7)によりgreedyは診断参考データとして残しつつ、5回のsamplingを
主要な合否判定に用いる。"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
PROJECT_ROOT = GUARD_DIR.parents[2]
REPORTS_DIR = TRAINING_ROOT / "reports"

sys.path.insert(0, str(GUARD_DIR))
from run_phase4fv_stage_ab import load_model, generate, analyze, build_context_string  # noqa: E402

P0_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
P1_PATH = GUARD_DIR / "phase4fv_prompts" / "p1_minimal_grounding.jinja2"
P2_PATH = GUARD_DIR / "phase4fv_prompts" / "p2_explicit_entity_binding.jinja2"

KNOWN_FAILURE_IDS = ["FU-D01", "FU-B05", "FU-A03", "FU-E02"]
N_SAMPLES = 5


def main():
    p0 = P0_PATH.read_text(encoding="utf-8")
    p1 = P1_PATH.read_text(encoding="utf-8")
    p2 = P2_PATH.read_text(encoding="utf-8")

    fu_precomputed = json.loads((REPORTS_DIR / "phase4fu_precomputed_contexts.json").read_text(encoding="utf-8"))
    fu_by_id = {p["id"]: p for p in fu_precomputed}

    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    out = []
    for pid in KNOWN_FAILURE_IDS:
        p = fu_by_id[pid]
        ctx = build_context_string(p["retrieved_chunks"])
        row = {"id": pid, "prompt": p["prompt"], "candidates": {}}
        for cname, cprompt in (("P0", p0), ("P1", p1), ("P2", p2)):
            greedy_text = generate(model, tokenizer, cprompt, p["prompt"], context=ctx, seed=42, do_sample=False)
            greedy = analyze(greedy_text)
            samples = []
            for i in range(N_SAMPLES):
                text = generate(model, tokenizer, cprompt, p["prompt"], context=ctx, seed=100 + i,
                                 do_sample=True, temperature=0.7)
                samples.append({"run": i, **analyze(text)})
            row["candidates"][cname] = {"greedy": greedy, "sampling_temp0.7": samples}
            print(f"[StageA-sampling] {pid}/{cname} done")
        out.append(row)

    (REPORTS_DIR / "phase4fv_known_failures.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STAGE A SAMPLING DONE")


if __name__ == "__main__":
    main()
