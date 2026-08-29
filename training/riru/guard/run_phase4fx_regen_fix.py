# -*- coding: utf-8 -*-
"""Phase4FX: Q1 extraction修正後、影響を受けた19probeのA2のみ再生成する。"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
PROJECT_ROOT = TRAINING_ROOT.parents[1]
REPORTS_DIR = TRAINING_ROOT / "reports"
PRODUCTION_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

sys.path.insert(0, str(GUARD_DIR))
from run_phase4fx_generation import load_model, generate, analyze  # noqa: E402

AFFECTED_IDS = ["FU-D05", "FV-P02", "FV-P04", "FV-P05", "FV-P07", "FV-P08", "FV-P09", "FV-P10",
                "FV-P11", "FV-P12", "CB-FU-D05", "FV-C06", "FV-C07", "FV-C08", "FV-C09",
                "FX-CB02", "FX-CB04", "FX-CB06", "FX-CB08"]


def main():
    production_prompt = PRODUCTION_PROMPT_PATH.read_text(encoding="utf-8")
    assembled = {r["id"]: r for r in json.loads((REPORTS_DIR / "phase4fx_context_assembly.json").read_text(encoding="utf-8"))}
    results = {r["id"]: r for r in json.loads((REPORTS_DIR / "phase4fx_generation_results.json").read_text(encoding="utf-8"))}

    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    for pid in AFFECTED_IDS:
        p = assembled[pid]
        t0 = time.time()
        text = generate(model, tokenizer, production_prompt, p["prompt"], context=p["A2"], seed=42)
        results[pid]["A2"] = {**analyze(text), "latency_sec": round(time.time() - t0, 2)}
        results[pid]["query_entities"] = p["query_entities"]
        print(f"[regen] {pid} done: {text[:60]}")

    out = list(results.values())
    (REPORTS_DIR / "phase4fx_generation_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("REGEN DONE")


if __name__ == "__main__":
    main()
