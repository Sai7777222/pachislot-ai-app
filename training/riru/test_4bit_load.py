"""Phase 4D: Qwen2.5-14B-Instruct HF版の4bit NF4ロード試験。

学習はまだ行わない。ロード・簡単なforward/generate・アンロードのみを確認する。
"""

from __future__ import annotations

import gc
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"


def nvidia_smi_snapshot() -> dict:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        mem_used, mem_total, util = (x.strip() for x in out.stdout.strip().split(","))
        return {
            "vram_used_mib": int(mem_used),
            "vram_total_mib": int(mem_total),
            "gpu_util_pct": int(util),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def main() -> int:
    print("=== VRAM before load ===")
    print(nvidia_smi_snapshot())

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("pad_token:", tokenizer.pad_token, "eos_token:", tokenizer.eos_token)
    print("special tokens map:", tokenizer.special_tokens_map)

    t0 = time.perf_counter()
    print("Loading model in 4bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    load_time = time.perf_counter() - t0
    print(f"Loaded in {load_time:.2f}s")

    print("=== VRAM after load ===")
    print(nvidia_smi_snapshot())

    # dtype / 量子化状態の確認
    first_param = next(model.parameters())
    print("model dtype (first param):", first_param.dtype)
    print("model device:", next(model.parameters()).device)
    is_loaded_in_4bit = getattr(model, "is_loaded_in_4bit", None)
    print("is_loaded_in_4bit:", is_loaded_in_4bit)
    # 全パラメータがCUDA上にあるか (CPU fallbackが起きていないか)
    devices = {p.device.type for p in model.parameters()}
    print("parameter device types (CPU fallbackがあればここに'cpu'が混ざる):", devices)

    # chat template + 簡単なgenerateの確認
    messages = [
        {"role": "system", "content": "あなたは親切な日本語アシスタントです。"},
        {"role": "user", "content": "こんにちは。一言だけ挨拶して。"},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    print("apply_chat_template output (テンプレート適用後の文字列):")
    print(repr(prompt_text))
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_ids = encoded["input_ids"]

    print("Running short generate()...")
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen_time = time.perf_counter() - t0
    generated = tokenizer.decode(output_ids[0][prompt_ids.shape[1] :], skip_special_tokens=True)
    print(f"generate() OK in {gen_time:.2f}s")
    print("Generated text:", repr(generated))

    print("=== VRAM during/after generate ===")
    print(nvidia_smi_snapshot())

    print("Unloading model...")
    del model
    del prompt_ids
    del output_ids
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(1.0)

    print("=== VRAM after unload ===")
    print(nvidia_smi_snapshot())

    print("ALL CHECKS COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
