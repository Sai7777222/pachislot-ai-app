"""Phase 4L-7〜4L-19: v4本評価。A=Base / B=v2 / C=v4 / D=v3(参考) の4条件で、

  1. Q3 (実際の本番RAGコンテキスト) を5seed(42,43,44,45,46) x temperature=0.3で評価
  2. Phase4I held-out (Q3 + P01〜P10、計11問) をseed=42で評価
  3. 既存structured RAG 17問をseed=42で評価
  4. 既存キャラクター評価セット(39問)をseed=42で評価

generation条件は max_new_tokens=300 / temperature=0.3 で統一。
「キミ」用system prompt変更はこのフェーズでは行わない (現行system.jinja2のまま)。

QLoRA/LoRA学習は行わない。v1/v2/v3/v4 adapterはいずれも読み取り専用でロードする。
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

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_V2_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v2")
ADAPTER_V3_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v3")
ADAPTER_V4_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v4")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
SEED_DEFAULT = 42
Q3_SEEDS = (42, 43, 44, 45, 46)

CONDITIONS = ("A_base", "B_v2", "C_v4", "D_v3")


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
    model.load_adapter(ADAPTER_V4_PATH, adapter_name="v4")
    model.load_adapter(ADAPTER_V3_PATH, adapter_name="v3")
    model.eval()
    return model, tokenizer


def generate_reply(model, tokenizer, messages: list[dict], condition: str, seed: int) -> dict:
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]

    torch.manual_seed(seed)
    t0 = time.perf_counter()
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    with torch.no_grad():
        if condition == "A_base":
            with model.disable_adapter():
                output_ids = model.generate(**encoded, **gen_kwargs)
        else:
            model.set_adapter(condition.split("_")[1])
            output_ids = model.generate(**encoded, **gen_kwargs)
    elapsed = time.perf_counter() - t0
    completion_ids = output_ids[0][prompt_len:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "completion_tokens": len(completion_ids),
        "elapsed_sec": round(elapsed, 3),
    }


def run_single_turn(model, tokenizer, system_prompt, rag_context, question, condition, seed):
    messages = [{"role": "system", "content": system_prompt}]
    if rag_context:
        messages.append({"role": "system", "content": rag_context})
    messages.append({"role": "user", "content": question})
    return generate_reply(model, tokenizer, messages, condition, seed)


def run_multiturn(model, tokenizer, system_prompt, turns, condition, seed):
    messages = [{"role": "system", "content": system_prompt}]
    turn_log = []
    for i in range(0, len(turns), 2):
        user_text = turns[i]
        messages.append({"role": "user", "content": user_text})
        gen = generate_reply(model, tokenizer, messages, condition, seed)
        messages.append({"role": "assistant", "content": gen["text"]})
        turn_log.append({"user": user_text, "assistant": gen["text"], "meta": gen})
    return turn_log


def main() -> int:
    print("Loading base model + v2/v3/v4 LoRA adapters (named adapters on one base)...")
    model, tokenizer = build_model_and_tokenizer()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    rag17q_path = EVAL_DIR / "structured_rag_17q_context.json"
    rag_17q = json.loads(rag17q_path.read_text(encoding="utf-8"))
    q3 = next(r for r in rag_17q if r["id"] == "Q3")
    holdout_path = EVAL_DIR / "phase4i_holdout_omission_v2.json"
    holdout_p = json.loads(holdout_path.read_text(encoding="utf-8"))
    eval_39 = [
        json.loads(line)
        for line in (EVAL_DIR / "riru_eval_set_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    results: dict = {"q3_multiseed": {}, "holdout_11": {}, "structured_17q": {}, "character_39": {}}

    # --- Q3 5-seed evaluation ---
    print("=== Q3 multi-seed (5 seeds x 4 conditions) ===")
    for cond in CONDITIONS:
        results["q3_multiseed"][cond] = {}
        for seed in Q3_SEEDS:
            print(f"  Q3 {cond} seed={seed}")
            gen = run_single_turn(
                model, tokenizer, system_prompt, q3["rag_context_text"], q3["question"], cond, seed
            )
            results["q3_multiseed"][cond][str(seed)] = gen

    # --- Phase4I held-out (Q3 + P01-P10), seed=42 ---
    print("=== Phase4I held-out (Q3 + P01-P10) seed=42 ===")
    holdout_items = [
        {"id": "Q3", "question": q3["question"], "rag_context_text": q3["rag_context_text"]}
    ]
    holdout_items += holdout_p
    for item in holdout_items:
        print(f"  {item['id']}")
        results["holdout_11"][item["id"]] = {"question": item["question"], "conditions": {}}
        for cond in CONDITIONS:
            gen = run_single_turn(
                model, tokenizer, system_prompt,
                item["rag_context_text"], item["question"], cond, SEED_DEFAULT,
            )
            results["holdout_11"][item["id"]]["conditions"][cond] = gen

    # --- structured 17q, seed=42 ---
    print("=== structured 17q seed=42 ===")
    for item in rag_17q:
        print(f"  {item['id']}")
        results["structured_17q"][item["id"]] = {"question": item["question"], "conditions": {}}
        for cond in CONDITIONS:
            gen = run_single_turn(
                model, tokenizer, system_prompt,
                item["rag_context_text"], item["question"], cond, SEED_DEFAULT,
            )
            results["structured_17q"][item["id"]]["conditions"][cond] = gen

    # --- character 39, seed=42 ---
    print("=== character 39 seed=42 ===")
    for item in eval_39:
        print(f"  {item['id']} ({item['category']})")
        results["character_39"][item["id"]] = {
            "category": item["category"], "type": item["type"], "conditions": {}
        }
        for cond in CONDITIONS:
            if item["type"] == "single":
                gen = run_single_turn(
                    model, tokenizer, system_prompt, None, item["prompt"], cond, SEED_DEFAULT
                )
            else:
                gen = run_multiturn(
                    model, tokenizer, system_prompt, item["turns"], cond, SEED_DEFAULT
                )
            results["character_39"][item["id"]]["conditions"][cond] = gen

    out_path = EVAL_DIR / "phase4l_comprehensive_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
