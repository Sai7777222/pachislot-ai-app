"""Phase 4N-8: module別 LoRA ablation (v4)。

q_proj/k_proj/v_proj/o_proj のうち特定moduleへのLoRA寄与のみを、
adapterファイルを書き換えずに一時的に無効化(scale=0)して評価する。
実装はphase4n_lora_scale_experiment.pyのscaled_lora()を再利用する。

条件: full(通常v4) / q_off / k_off / v_off / o_off / qk_off / vo_off

QLoRA/LoRA学習は行わない。v5 adapterは作成しない。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4n_lora_scale_experiment import (  # noqa: E402
    build_messages,
    generate,
    q3_fact_check,
    scaled_lora,
)

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parents[1]
EVAL_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_V4_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v4")

PERSONA_REP_IDS = ["E01", "E14", "E17", "E27", "E36"]

MODULE_CONDITIONS = {
    "full_v4": None,
    "q_off": ["q_proj"],
    "k_off": ["k_proj"],
    "v_off": ["v_proj"],
    "o_off": ["o_proj"],
    "qk_off": ["q_proj", "k_proj"],
    "vo_off": ["v_proj", "o_proj"],
}


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
        BASE_MODEL_PATH, quantization_config=quant_config, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_V4_PATH, adapter_name="v4")
    model.set_adapter("v4")
    model.eval()
    return model, tokenizer


def main() -> int:
    print("Loading base model + v4 adapter (module ablation)...")
    model, tokenizer = build_model_and_tokenizer()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    rag_17q = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    q3_item = next(r for r in rag_17q if r["id"] == "Q3")
    eval_39 = [
        json.loads(line)
        for line in (EVAL_DIR / "riru_eval_set_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_39_by_id = {x["id"]: x for x in eval_39}

    q3_messages = build_messages(system_prompt, q3_item["rag_context_text"], q3_item["question"])

    results = {}
    t0 = time.perf_counter()
    for cond_name, module_types in MODULE_CONDITIONS.items():
        print(f"  condition={cond_name} module_types={module_types}")
        with scaled_lora(model, "v4", factor=0.0, module_types=module_types) as n_off:
            greedy_text = generate(model, tokenizer, q3_messages, do_sample=False, seed=42)
            sampled_text = generate(model, tokenizer, q3_messages, do_sample=True, seed=42)
            persona = {}
            for eid in PERSONA_REP_IDS:
                item = eval_39_by_id[eid]
                p_messages = build_messages(system_prompt, None, item["prompt"])
                p_text = generate(model, tokenizer, p_messages, do_sample=True, seed=42)
                persona[eid] = {"category": item["category"], "text": p_text}
        results[cond_name] = {
            "n_layers_scaled_to_zero": n_off,
            "q3_greedy": q3_fact_check(greedy_text),
            "q3_sampled_seed42": q3_fact_check(sampled_text),
            "persona_representative": persona,
        }

    out_path = EVAL_DIR / "phase4n_module_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path} ({time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
