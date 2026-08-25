"""Phase 4I-8/13補助: temperature変更が事実精度(数値回答・Q3省略)に与える影響を確認する。

v2固定・現行system prompt (override無し)。Q1/Q2/Q4 (単純数値) と Q3 (省略問題の本命ケース)
について、temperature 0.3/0.5/0.7/0.9 x seed 2本で生成する。
QLoRA/LoRA学習は行わない。DB/RAG/adapterは一切変更しない。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parents[1]
EVAL_DIR = Path(__file__).resolve().parent

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_V2_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v2")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
RAG_17Q_PATH = EVAL_DIR / "structured_rag_17q_context.json"

MAX_NEW_TOKENS = 300
TOP_P = 0.9
TEMPERATURES = (0.3, 0.5, 0.7, 0.9)
SEEDS = (42, 43)
TARGET_IDS = ("Q1", "Q2", "Q4", "Q3")

EXPECTED_VALUES = {
    "Q1": ["114.6%"],
    "Q2": ["1/295"],
    "Q4": ["1/37.6"],
    "Q3": ["510G", "1000G", "1480G", "15.2%", "20.3%", "64.5%"],
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
    model = PeftModel.from_pretrained(base_model, ADAPTER_V2_PATH, adapter_name="v2")
    model.set_adapter("v2")
    model.eval()
    return model, tokenizer


def generate_reply(model, tokenizer, system_prompt, rag_context, question, temperature, seed):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": rag_context},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion_ids = output_ids[0][prompt_len:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return text


def main() -> int:
    print("Loading base model + v2 adapter...")
    model, tokenizer = build_model_and_tokenizer()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    rag_17q = json.loads(RAG_17Q_PATH.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rag_17q}

    results: dict = {}
    for qid in TARGET_IDS:
        item = by_id[qid]
        results[qid] = {"question": item["question"], "by_temperature": {}}
        for temp in TEMPERATURES:
            gens = []
            for seed in SEEDS:
                print(f"  {qid} temp={temp} seed={seed}")
                text = generate_reply(
                    model, tokenizer, system_prompt, item["rag_context_text"], item["question"],
                    temp, seed,
                )
                expected = EXPECTED_VALUES[qid]
                found = [v for v in expected if v in text]
                gens.append(
                    {
                        "seed": seed,
                        "text": text,
                        "expected_values": expected,
                        "values_found": found,
                        "coverage_pct": round(len(found) / len(expected) * 100, 1),
                    }
                )
            results[qid]["by_temperature"][str(temp)] = gens

    out_path = EVAL_DIR / "phase4i_factual_temperature_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
