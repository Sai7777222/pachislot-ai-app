"""Phase4FV Stage E: RAG50 golden regression。P2候補(Stage Aで最も改善幅が大きかった候補)で
全50件を再生成し、Phase4ZN由来のP0ベースライン(phase4zn_rag50_raw.json)と比較する。"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
PROJECT_ROOT = GUARD_DIR.parents[2]
REPORTS_DIR = TRAINING_ROOT / "reports"
sys.path.insert(0, str(GUARD_DIR))
sys.path.insert(0, str(TRAINING_ROOT / "eval"))
from run_phase4fv_stage_ab import load_model, generate, analyze  # noqa: E402
import phase4zf_rag_stress_eval as pool_mod  # noqa: E402

P2_PATH = GUARD_DIR / "phase4fv_prompts" / "p2_explicit_entity_binding.jinja2"

MANDATORY_IDS = ["P02", "P04", "LC-08", "Q6", "Q11", "Q15", "Q17", "AD-04"]


def main():
    p2 = P2_PATH.read_text(encoding="utf-8")
    pool = pool_mod.load_rag_probe_pool()
    by_id = {p["id"]: p for p in pool}
    rag50_baseline = json.loads((REPORTS_DIR / "phase4zn_rag50_raw.json").read_text(encoding="utf-8"))

    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    out = []
    for base in rag50_baseline:
        pid = base["probe_id"]
        p = by_id[pid]
        text = generate(model, tokenizer, p2, p["question"], context=p["context"], seed=42, do_sample=False)
        row = {"probe_id": pid, "prompt": p["question"], "P0_baseline_response": base["response"],
               "P2": analyze(text), "mandatory": pid in MANDATORY_IDS}
        out.append(row)
        print(f"[StageE] {pid} done")

    (REPORTS_DIR / "phase4fv_rag50.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STAGE E DONE")


if __name__ == "__main__":
    main()
