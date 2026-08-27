"""Phase 4ZF Section17: 専門外質問/雑談の観測評価(現状観測、hard FAILへ直結させない)。
主にeager backendで実施(観測目的のためbackend網羅は優先度低)。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
PROJECT_ROOT = EVAL_DIR.parents[2]
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(TRAINING_ROOT))
REPORTS_DIR = TRAINING_ROOT / "reports"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"


def main(attn_impl: str) -> int:
    from phase4ze_identity_eval import load_model, generate_reply
    from phase4zf_ood_probes import ALL_PROBES

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)

    results = {}
    t0 = time.time()
    for p in ALL_PROBES:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": p["prompt"]}]
        greedy = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
        results[p["id"]] = {"prompt": p["prompt"], "greedy": greedy}
        print(f"{p['id']} done ({time.time()-t0:.1f}s)")

    out = {"attn_impl": attn_impl, "n_probes": len(ALL_PROBES), "results": results}
    out_path = REPORTS_DIR / f"phase4zf_ood_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args()
    sys.exit(main(args.attn_impl))
