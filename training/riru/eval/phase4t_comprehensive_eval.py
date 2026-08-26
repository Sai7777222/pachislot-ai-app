"""Phase 4T: P04型質問スコープ診断 + 誤名乗り/placeholder重点診断。

A=Base / B=v4 / C=ratio_high の3条件で、
  1. P04型held-out probe(22問) をgreedy+5seed(42-46)で評価
  2. 誤名乗り/自己紹介paraphrase probe(22問) をgreedy+10seed(42-51)で評価
     (v4/ratio_highが主対象、参考としてBaseも同条件で取得)
  3. 既存E36(character39内)をgreedy+10seedで追加評価 (placeholder再現性確認用)

生成条件は既存Phase4O/4P/4Q/4Sと同一 (max_new_tokens=300, temperature=0.3, top_p=0.9)。
QLoRA/LoRA学習は行わない。既存adapter/データは一切変更しない。
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
from phase4t_probes import NAMING_PROBES, P04_PROBES  # noqa: E402

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_V4_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v4")
ADAPTER_HIGH_PATH = str(TRAINING_ROOT / "lora-riru-qwen-ratio-high")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
P04_SEEDS = (42, 43, 44, 45, 46)
NAMING_SEEDS = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)

CONDITIONS = ("A_base", "B_v4", "C_high")
# 誤名乗り/placeholder診断はratio-highを主対象、v4を比較対象とする (指示5節)。
# Baseはpersona未学習で誤名乗りリスクの主眼ではないため、この2ブロックはv4/highに限定する。
NAMING_CONDITIONS = ("B_v4", "C_high")
ADAPTER_NAME_FOR_CONDITION = {"B_v4": "v4", "C_high": "high"}


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
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return text


def run_single(model, tokenizer, system_prompt, rag_context, question, cond, seed, do_sample=True):
    messages = [{"role": "system", "content": system_prompt}]
    if rag_context:
        messages.append({"role": "system", "content": rag_context})
    messages.append({"role": "user", "content": question})
    return generate_reply(model, tokenizer, messages, cond, seed, do_sample)


def main() -> int:
    print("Loading base model + v4/ratio-high adapters...")
    model, tokenizer = build_model_and_tokenizer()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    results: dict = {"p04_probes": {}, "naming_probes": {}, "e36_extended": {}}
    t0 = time.perf_counter()

    print("=== P04-type probes (22 x greedy+5seed x 3 conditions) ===")
    for probe in P04_PROBES:
        results["p04_probes"][probe["id"]] = {"question": probe["question"], "conditions": {}}
        for cond in CONDITIONS:
            greedy_text = run_single(
                model, tokenizer, system_prompt, probe["context"], probe["question"],
                cond, 42, do_sample=False,
            )
            sampled = {}
            for seed in P04_SEEDS:
                text = run_single(
                    model, tokenizer, system_prompt, probe["context"], probe["question"],
                    cond, seed,
                )
                sampled[str(seed)] = text
            results["p04_probes"][probe["id"]]["conditions"][cond] = {
                "greedy": greedy_text, "sampled": sampled,
            }
        print(f"  {probe['id']} done ({time.perf_counter() - t0:.1f}s elapsed)")

    print("=== naming probes (22 x greedy+10seed x 2 conditions: v4/high) ===")
    for probe in NAMING_PROBES:
        results["naming_probes"][probe["id"]] = {"prompt": probe["prompt"], "conditions": {}}
        for cond in NAMING_CONDITIONS:
            greedy_text = run_single(
                model, tokenizer, system_prompt, None, probe["prompt"], cond, 42, do_sample=False
            )
            sampled = {}
            for seed in NAMING_SEEDS:
                text = run_single(
                    model, tokenizer, system_prompt, None, probe["prompt"], cond, seed
                )
                sampled[str(seed)] = text
            results["naming_probes"][probe["id"]]["conditions"][cond] = {
                "greedy": greedy_text, "sampled": sampled,
            }
        print(f"  {probe['id']} done ({time.perf_counter() - t0:.1f}s elapsed)")

    print("=== E36 extended (10 seeds x v4/high) ===")
    eval_39 = [
        json.loads(line)
        for line in (EVAL_DIR / "riru_eval_set_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    e36_item = next(x for x in eval_39 if x["id"] == "E36")
    for cond in NAMING_CONDITIONS:
        sampled = {}
        for seed in NAMING_SEEDS:
            text = run_single(model, tokenizer, system_prompt, None, e36_item["prompt"], cond, seed)
            sampled[str(seed)] = text
        results["e36_extended"][cond] = sampled
    print(f"  done ({time.perf_counter() - t0:.1f}s elapsed)")

    out_path = EVAL_DIR / "phase4t_comprehensive_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path} (total {time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
