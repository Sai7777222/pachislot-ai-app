"""Phase 4ZG Section21: Phase4ZG candidate adapter(lora-riru-qwen-phase4zg-identity-hardened)を
Base Qwen2.5-14B-InstructへLoRA mergeする。HF Gate PASS後にのみ実行する。

CPU上でbf16のまま処理する(Phase4Y/4ZEと同一の安全な方式)。既存のBaseモデル・adapter・
既存merged HF(riru-qwen-final-hf, riru-phase4ze-identity-margin-hf)・他のcandidate/adapterは
一切変更・上書きしない。merge先は新規ディレクトリとする。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

TRAINING_ROOT = Path(__file__).resolve().parent
BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")
MERGED_OUTPUT_DIR = TRAINING_ROOT / "merged" / "riru-phase4zg-identity-hardened-hf"


def main() -> int:
    if MERGED_OUTPUT_DIR.exists() and any(MERGED_OUTPUT_DIR.iterdir()):
        print(f"STOP: output dir already exists and is non-empty: {MERGED_OUTPUT_DIR}")
        return 1

    t0 = time.perf_counter()
    print(f"Loading base model (CPU, bf16): {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    print(f"  base loaded ({time.perf_counter() - t0:.1f}s)")

    print(f"Loading adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print(f"  adapter loaded ({time.perf_counter() - t0:.1f}s)")

    print("Merging LoRA weights into base (merge_and_unload)...")
    merged_model = model.merge_and_unload()
    print(f"  merge done ({time.perf_counter() - t0:.1f}s)")

    MERGED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model -> {MERGED_OUTPUT_DIR}")
    merged_model.save_pretrained(str(MERGED_OUTPUT_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_OUTPUT_DIR))
    print(f"  save done ({time.perf_counter() - t0:.1f}s)")

    manifest = {
        "base_model_path": BASE_MODEL_PATH, "adapter_path": ADAPTER_PATH,
        "merged_output_dir": str(MERGED_OUTPUT_DIR), "merge_dtype": "bfloat16", "merge_device": "cpu",
        "total_wall_clock_sec": round(time.perf_counter() - t0, 1),
    }
    (TRAINING_ROOT / "reports" / "phase4zg_merge_run_info.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Merge complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
