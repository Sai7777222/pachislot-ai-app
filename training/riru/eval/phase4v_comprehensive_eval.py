"""Phase 4V: Broad-Question Completeness 診断 本評価。

A=Base / B=v4 / C=ratio_high / D=ratio_high_identity の4条件で、
phase4v_probes.PROBES (36問: 6 context family x 6 question variant) を
greedy + 5seed(42-46) で評価する(36 x 6 x 4 = 864生成)。

QLoRA/LoRA学習は行わない。adapterは読み取り専用でロードするのみ。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parents[1]
EVAL_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(EVAL_DIR))
from phase4v_probes import PROBES  # noqa: E402

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_V4_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v4")
ADAPTER_HIGH_PATH = str(TRAINING_ROOT / "lora-riru-qwen-ratio-high")
ADAPTER_IDENTITY_PATH = str(TRAINING_ROOT / "lora-riru-qwen-ratio-high-identity")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
SEEDS_5 = (42, 43, 44, 45, 46)

CONDITIONS = ("A_base", "B_v4", "C_high", "D_identity")
ADAPTER_NAME_FOR_CONDITION = {"B_v4": "v4", "C_high": "high", "D_identity": "identity"}


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
    model.load_adapter(ADAPTER_HIGH_PATH, adapter_name="high")
    model.load_adapter(ADAPTER_IDENTITY_PATH, adapter_name="identity")
    model.eval()
    return model, tokenizer


def generate_reply(model, tokenizer, messages, condition, seed, do_sample=True):
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
    else:
        gen_kwargs.update(do_sample=False)
    with torch.no_grad():
        if condition == "A_base":
            with model.disable_adapter():
                output_ids = model.generate(**encoded, **gen_kwargs)
        else:
            model.set_adapter(ADAPTER_NAME_FOR_CONDITION[condition])
            output_ids = model.generate(**encoded, **gen_kwargs)
    completion_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def run_single(model, tokenizer, system_prompt, rag_context, question, cond, seed, do_sample=True):
    messages = [{"role": "system", "content": system_prompt}]
    if rag_context:
        messages.append({"role": "system", "content": rag_context})
    messages.append({"role": "user", "content": question})
    return generate_reply(model, tokenizer, messages, cond, seed, do_sample)


def main() -> int:
    print("Loading base model + v4/ratio-high/ratio-high-identity adapters...")
    model, tokenizer = build_model_and_tokenizer()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    results: dict = {}
    t0 = time.perf_counter()
    for probe in PROBES:
        results[probe["id"]] = {
            "category": probe["category"], "family": probe["family"], "question": probe["question"],
            "conditions": {},
        }
        for cond in CONDITIONS:
            greedy_text = run_single(
                model, tokenizer, system_prompt, probe["context"], probe["question"],
                cond, 42, do_sample=False,
            )
            sampled = {}
            for seed in SEEDS_5:
                text = run_single(
                    model, tokenizer, system_prompt, probe["context"], probe["question"], cond, seed
                )
                sampled[str(seed)] = text
            results[probe["id"]]["conditions"][cond] = {"greedy": greedy_text, "sampled": sampled}
        print(f"  {probe['id']} done ({time.perf_counter() - t0:.1f}s elapsed)")

    out_path = EVAL_DIR / "phase4v_comprehensive_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path} (total {time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
