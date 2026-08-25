"""Phase 4O-6〜4O-15: v5-qkv 対照実験 本評価。

A=Base / B=v4(q/k/v/o) / C=v5-qkv(q/k/vのみ) の3条件で:

  1. Q3 (実本番RAG) を5seed(42-46) x temperature=0.3 で sampled評価 + greedy評価
  2. Phase4I held-out (Q3 + P01-P10、計11問) をseed=42で評価 (P01/P02重点)
  3. Q11 を5seedで評価 (ヤメ時アドバイス等のhallucination重点検査)
  4. Q9 を5seedで評価 (派生計算hallucination重点検査)
  5. E36 を5seedで評価 (persona安定性重点検査)
  6. 既存structured RAG 17問をseed=42で評価
  7. 既存character39をseed=42で評価 (Base/v4/v5-qkv)

generation条件: max_new_tokens=300 / temperature=0.3 (sampled) / greedy=do_sample=False。
system prompt変更なし。QLoRA/LoRA学習は行わない。adapterは読み取り専用ロードのみ。
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
ADAPTER_V4_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v4")
ADAPTER_V5QKV_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v5-qkv")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
SEED_DEFAULT = 42
Q3_SEEDS = (42, 43, 44, 45, 46)

CONDITIONS = ("A_base", "B_v4", "C_v5qkv")
ADAPTER_NAME_FOR_CONDITION = {"B_v4": "v4", "C_v5qkv": "v5qkv"}

Q3_KEY_FACTS = ["510G", "1000G", "1480G", "15.2%", "20.3%", "64.5%"]
Q3_EXTRA_MARKERS = ["33.2%", "Z-ZONE"]


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
    model.load_adapter(ADAPTER_V5QKV_PATH, adapter_name="v5qkv")
    model.eval()
    return model, tokenizer


def generate_reply(
    model, tokenizer, messages: list[dict], condition: str, seed: int, do_sample: bool = True
) -> dict:
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]

    torch.manual_seed(seed)
    t0 = time.perf_counter()
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
    elapsed = time.perf_counter() - t0
    completion_ids = output_ids[0][prompt_len:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "completion_tokens": len(completion_ids),
        "elapsed_sec": round(elapsed, 3),
    }


def run_single_turn(
    model, tokenizer, system_prompt, rag_context, question, condition, seed, do_sample=True
):
    messages = [{"role": "system", "content": system_prompt}]
    if rag_context:
        messages.append({"role": "system", "content": rag_context})
    messages.append({"role": "user", "content": question})
    return generate_reply(model, tokenizer, messages, condition, seed, do_sample)


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


def q3_fact_check(text: str) -> dict:
    found = [k for k in Q3_KEY_FACTS if k in text]
    extra = [k for k in Q3_EXTRA_MARKERS if k in text]
    return {
        "text": text,
        "length": len(text),
        "key_facts_found": found,
        "recall_pct": round(len(found) / len(Q3_KEY_FACTS) * 100, 1),
        "extra_markers_found": extra,
    }


def main() -> int:
    print("Loading base model + v4/v5-qkv LoRA adapters (named adapters on one base)...")
    model, tokenizer = build_model_and_tokenizer()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    rag17q_path = EVAL_DIR / "structured_rag_17q_context.json"
    rag_17q = json.loads(rag17q_path.read_text(encoding="utf-8"))
    q3 = next(r for r in rag_17q if r["id"] == "Q3")
    q9 = next(r for r in rag_17q if r["id"] == "Q9")
    q11 = next(r for r in rag_17q if r["id"] == "Q11")
    holdout_path = EVAL_DIR / "phase4i_holdout_omission_v2.json"
    holdout_p = json.loads(holdout_path.read_text(encoding="utf-8"))
    eval_39 = [
        json.loads(line)
        for line in (EVAL_DIR / "riru_eval_set_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    e36 = next(x for x in eval_39 if x["id"] == "E36")

    results: dict = {
        "q3_multiseed_sampled": {},
        "q3_greedy": {},
        "holdout_11": {},
        "q11_multiseed": {},
        "q9_multiseed": {},
        "e36_multiseed": {},
        "structured_17q": {},
        "character_39": {},
    }

    print("=== Q3 multi-seed sampled (5 seeds x 3 conditions) ===")
    for cond in CONDITIONS:
        results["q3_multiseed_sampled"][cond] = {}
        for seed in Q3_SEEDS:
            print(f"  Q3 sampled {cond} seed={seed}")
            gen = run_single_turn(
                model, tokenizer, system_prompt, q3["rag_context_text"], q3["question"], cond, seed
            )
            results["q3_multiseed_sampled"][cond][str(seed)] = q3_fact_check(gen["text"])

    print("=== Q3 greedy (3 conditions) ===")
    for cond in CONDITIONS:
        print(f"  Q3 greedy {cond}")
        gen = run_single_turn(
            model, tokenizer, system_prompt, q3["rag_context_text"], q3["question"], cond,
            SEED_DEFAULT, do_sample=False,
        )
        results["q3_greedy"][cond] = q3_fact_check(gen["text"])

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
            key_facts = item.get("key_facts")
            entry = dict(gen)
            if key_facts:
                found = [f for f in key_facts if f in gen["text"]]
                entry["key_facts_found"] = found
                entry["recall_pct"] = round(len(found) / len(key_facts) * 100, 1)
                leaked = [m for m in item.get("irrelevant_markers", []) if m in gen["text"]]
                entry["irrelevant_leaked"] = leaked
            results["holdout_11"][item["id"]]["conditions"][cond] = entry

    print("=== Q11 multi-seed (5 seeds x 3 conditions, hallucination focus) ===")
    for cond in CONDITIONS:
        results["q11_multiseed"][cond] = {}
        for seed in Q3_SEEDS:
            print(f"  Q11 {cond} seed={seed}")
            gen = run_single_turn(
                model, tokenizer, system_prompt,
                q11["rag_context_text"], q11["question"], cond, seed,
            )
            results["q11_multiseed"][cond][str(seed)] = gen

    print("=== Q9 multi-seed (5 seeds x 3 conditions) ===")
    for cond in CONDITIONS:
        results["q9_multiseed"][cond] = {}
        for seed in Q3_SEEDS:
            print(f"  Q9 {cond} seed={seed}")
            gen = run_single_turn(
                model, tokenizer, system_prompt, q9["rag_context_text"], q9["question"], cond, seed
            )
            results["q9_multiseed"][cond][str(seed)] = gen

    print("=== E36 multi-seed (5 seeds x 3 conditions) ===")
    for cond in CONDITIONS:
        results["e36_multiseed"][cond] = {}
        for seed in Q3_SEEDS:
            print(f"  E36 {cond} seed={seed}")
            gen = run_single_turn(model, tokenizer, system_prompt, None, e36["prompt"], cond, seed)
            results["e36_multiseed"][cond][str(seed)] = gen

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

    out_path = EVAL_DIR / "phase4o_comprehensive_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
