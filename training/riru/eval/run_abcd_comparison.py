"""Phase 4H-7: ベースQwen (A) vs リルLoRA v1 (B) vs v2 (C) vs v3 (D) の4者比較評価。

- ベースモデルは1回だけ4bit NF4でロードし、PEFTの複数named adapter機能
  (`load_adapter(..., adapter_name=...)` + `set_adapter(...)` +
  `disable_adapter()`) を使って、同一メモリ上の重みで4条件を公平に比較する。
- 評価対象:
  1. training/riru/eval/riru_eval_set_v1.jsonl (39問、キャラクター性評価用、Phase4E/4G同一)
  2. training/riru/eval/structured_rag_17q_context.json (既存17問、Q3を含む)
  3. training/riru/eval/riru_eval_set_kimi_v2.jsonl (Phase4H-4新規、キミ自然使用/非使用テスト)
  4. training/riru/eval/riru_eval_set_omission_v2.json (Phase4H-4新規、Q3型省略問題の
     複数RAG構造での汎化テスト。学習データとは異なる架空値・トピックを使用)
- generation条件 (system prompt / max_new_tokens / temperature / seed) は
  A/B/C/Dすべて完全に同一にする。
- adapterのmerge・GGUF変換は行わない。読み込みのみ。v1/v2/v3 adapterはいずれも
  読み取り専用でロードするのみで、一切変更しない。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parents[1]
EVAL_DIR = Path(__file__).resolve().parent

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_V1_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v1")
ADAPTER_V2_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v2")
ADAPTER_V3_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v3")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
SEED = 42

CONDITIONS = ("A_base", "B_v1", "C_v2", "D_v3")


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


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def build_model_and_tokenizer():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_V1_PATH, adapter_name="v1")
    model.load_adapter(ADAPTER_V2_PATH, adapter_name="v2")
    model.load_adapter(ADAPTER_V3_PATH, adapter_name="v3")
    model.eval()
    return model, tokenizer


def generate_reply(model, tokenizer, messages: list[dict], condition: str) -> dict:
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]

    torch.manual_seed(SEED)
    t0 = time.perf_counter()
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    with torch.no_grad():
        if condition == "A_base":
            with model.disable_adapter():
                output_ids = model.generate(**encoded, **gen_kwargs)
        elif condition in ("B_v1", "C_v2", "D_v3"):
            model.set_adapter(condition.split("_")[1])
            output_ids = model.generate(**encoded, **gen_kwargs)
        else:
            raise ValueError(f"unknown condition: {condition}")
    elapsed = time.perf_counter() - t0
    completion_ids = output_ids[0][prompt_len:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    n_tokens = len(completion_ids)
    return {
        "text": text.strip(),
        "completion_tokens": n_tokens,
        "elapsed_sec": round(elapsed, 3),
        "tokens_per_sec": round(n_tokens / elapsed, 2) if elapsed > 0 else None,
    }


def run_single_turn(model, tokenizer, system_prompt: str, rag_context: str | None, user_text: str):
    results = {}
    for cond in CONDITIONS:
        messages = [{"role": "system", "content": system_prompt}]
        if rag_context:
            messages.append({"role": "system", "content": rag_context})
        messages.append({"role": "user", "content": user_text})
        results[cond] = generate_reply(model, tokenizer, messages, cond)
    return results


def run_multiturn(model, tokenizer, system_prompt: str, turns: list[str]):
    results = {}
    for cond in CONDITIONS:
        messages = [{"role": "system", "content": system_prompt}]
        turn_log = []
        for i in range(0, len(turns), 2):
            user_text = turns[i]
            messages.append({"role": "user", "content": user_text})
            gen = generate_reply(model, tokenizer, messages, cond)
            messages.append({"role": "assistant", "content": gen["text"]})
            turn_log.append({"user": user_text, "assistant": gen["text"], "meta": gen})
        results[cond] = turn_log
    return results


def main() -> int:
    print("Loading base model + v1/v2/v3 LoRA adapters (named adapters on one base)...")
    vram_before = nvidia_smi_snapshot()
    t0 = time.perf_counter()
    model, tokenizer = build_model_and_tokenizer()
    load_time = time.perf_counter() - t0
    vram_after_load = nvidia_smi_snapshot()
    print(f"Loaded in {load_time:.2f}s. VRAM: {vram_after_load}")

    system_prompt = load_system_prompt()

    eval_items = []
    with open(EVAL_DIR / "riru_eval_set_v1.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_items.append(json.loads(line))

    kimi_items = []
    with open(EVAL_DIR / "riru_eval_set_kimi_v2.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                kimi_items.append(json.loads(line))

    rag_context_path = EVAL_DIR / "structured_rag_17q_context.json"
    rag_items = json.loads(rag_context_path.read_text(encoding="utf-8"))

    omission_path = EVAL_DIR / "riru_eval_set_omission_v2.json"
    omission_items = json.loads(omission_path.read_text(encoding="utf-8"))

    all_results = {
        "character_eval": [],
        "kimi_eval_v2": [],
        "structured_rag_eval": [],
        "omission_eval_v2": [],
    }

    print(f"Running {len(eval_items)} character eval items x4 conditions...")
    for item in eval_items:
        print(f"  {item['id']} ({item['category']})")
        if item["type"] == "single":
            res = run_single_turn(model, tokenizer, system_prompt, None, item["prompt"])
        else:
            res = run_multiturn(model, tokenizer, system_prompt, item["turns"])
        all_results["character_eval"].append(
            {"id": item["id"], "category": item["category"], "type": item["type"], "results": res}
        )

    print(f"Running {len(kimi_items)} kimi-v2 eval items x4 conditions...")
    for item in kimi_items:
        print(f"  {item['id']} ({item['category']})")
        res = run_single_turn(model, tokenizer, system_prompt, None, item["prompt"])
        all_results["kimi_eval_v2"].append(
            {"id": item["id"], "category": item["category"], "type": item["type"], "results": res}
        )

    print(f"Running {len(rag_items)} structured DB/RAG eval items x4 conditions...")
    for item in rag_items:
        print(f"  {item['id']}")
        res = run_single_turn(
            model, tokenizer, system_prompt, item["rag_context_text"], item["question"]
        )
        all_results["structured_rag_eval"].append(
            {
                "id": item["id"],
                "group": item["group"],
                "question": item["question"],
                "results": res,
            }
        )

    print(f"Running {len(omission_items)} omission-v2 held-out eval items x4 conditions...")
    for item in omission_items:
        print(f"  {item['id']}")
        res = run_single_turn(
            model, tokenizer, system_prompt, item["rag_context_text"], item["question"]
        )
        all_results["omission_eval_v2"].append(
            {
                "id": item["id"],
                "group": item["group"],
                "question": item["question"],
                "results": res,
            }
        )

    vram_peak = nvidia_smi_snapshot()

    out_path = EVAL_DIR / "abcd_comparison_results.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "load_time_sec": round(load_time, 2),
        "vram_before_mib": vram_before.get("vram_used_mib"),
        "vram_after_load_mib": vram_after_load.get("vram_used_mib"),
        "vram_during_eval_mib": vram_peak.get("vram_used_mib"),
        "num_character_items": len(eval_items),
        "num_kimi_v2_items": len(kimi_items),
        "num_structured_rag_items": len(rag_items),
        "num_omission_v2_items": len(omission_items),
        "conditions": list(CONDITIONS),
        "adapter_v1_path": ADAPTER_V1_PATH,
        "adapter_v2_path": ADAPTER_V2_PATH,
        "adapter_v3_path": ADAPTER_V3_PATH,
        "base_model_path": BASE_MODEL_PATH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "seed": SEED,
    }
    (EVAL_DIR / "abcd_comparison_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved results -> {out_path}")
    print(f"Meta: {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
