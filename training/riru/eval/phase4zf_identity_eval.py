"""Phase 4ZF Section7-9: overnight stress evaluation (HF backends, ZE candidate frozen).

Probe pool = Phase4ZE's own 104-probe suite (ZEH holdout27 + naming_stress20 +
heldout_naming24 + e36family17 + e02family16) + phase4zf_stress_probes(40:
wrong_name_induction15 + role_confusion15 + correction_stress10) = 144 probes.
Sampling: greedy + seed101-103 (4/probe) = 576 generations/backend.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
PROJECT_ROOT = EVAL_DIR.parents[2]
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(TRAINING_ROOT))
REPORTS_DIR = TRAINING_ROOT / "reports"

from phase4ze_identity_eval import load_model, generate_reply, SYSTEM_PROMPT_PATH  # noqa: E402

SEEDS = (101, 102, 103)


def load_probe_pool() -> list[dict]:
    from phase4ze_holdout_probes import ALL_PROBES as ZE_HOLDOUT
    from phase4z_probes import PROBE_SET_A, PROBE_SET_B, PROBE_SET_C, PROBE_SET_D
    from phase4zf_stress_probes import WRONG_NAME_INDUCTION, ROLE_NAME_CONFUSION, IDENTITY_CORRECTION_STRESS

    probes = []
    for p in ZE_HOLDOUT:
        probes.append({"set": "phase4ze_holdout", "id": p["id"], "prompt": p["prompt"]})
    for p in PROBE_SET_A:
        probes.append({"set": "phase4w_naming_stress", "id": p["id"], "prompt": p["prompt"]})
    for p in PROBE_SET_B:
        probes.append({"set": "phase4x_heldout_naming", "id": p["id"], "prompt": p["prompt"]})
    for p in PROBE_SET_C:
        probes.append({"set": "e36_family", "id": p["id"], "prompt": p["prompt"]})
    for p in PROBE_SET_D:
        probes.append({"set": "e02_family", "id": p["id"], "prompt": p["prompt"]})
    for p in WRONG_NAME_INDUCTION:
        probes.append({"set": "zf_wrong_name_induction", "id": p["id"], "prompt": p["prompt"]})
    for p in ROLE_NAME_CONFUSION:
        probes.append({"set": "zf_role_name_confusion", "id": p["id"], "prompt": p["prompt"]})
    for p in IDENTITY_CORRECTION_STRESS:
        probes.append({"set": "zf_identity_correction_stress", "id": p["id"], "prompt": p["prompt"]})
    return probes


def main(attn_impl: str) -> int:
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)
    probes = load_probe_pool()

    results = {}
    t0 = time.time()
    for p in probes:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": p["prompt"]}]
        greedy = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
        sampled = {str(s): generate_reply(model, tokenizer, messages, seed=s, do_sample=True) for s in SEEDS}
        results[p["id"]] = {"set": p["set"], "greedy": greedy, "sampled": sampled}
        if len(results) % 10 == 0:
            elapsed = time.time() - t0
            print(f"{len(results)}/{len(probes)} done ({elapsed:.1f}s, "
                  f"eta {(elapsed/len(results))*(len(probes)-len(results)):.0f}s)")

    out = {"attn_impl": attn_impl, "n_probes": len(probes), "seeds": list(SEEDS),
           "n_generations": len(probes) * (1 + len(SEEDS)), "results": results}
    out_path = REPORTS_DIR / f"phase4zf_identity_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(f"total_generations={out['n_generations']}")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-impl", required=True, choices=["eager", "sdpa"])
    args = parser.parse_args()
    sys.exit(main(args.attn_impl))
