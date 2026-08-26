"""Phase 4U: ratio_high_identity 本評価。

Phase4S/4Tで既に取得済みのv4/ratio-high結果(同一probe・同一seed・同一generation設定)を
読み取り専用で再利用し、新規学習したratio_high_identityについてのみ新規生成する。
これにより不要な再生成を避けつつ、A_v4/B_ratio_high/C_identityの3-way比較を実現する。

新規生成対象 (ratio_high_identityのみ、必要に応じてv4/ratio_highも一部追加生成):
  1. naming probes (22問 x greedy+10seed) : identityのみ新規、v4/highはPhase4Tデータを再利用
  2. E36/E02 extended (各20seed) : v4/ratio-high/identity 3条件とも新規生成(既存は10seedのみのため)
  3. P04型22probe (Phase4T phase4t_probes.P04_PROBES) : identityのみ新規、v4/highは
     Phase4Tデータ再利用
  4. Q3 (greedy+5seed) / P01/P02/P04(元のholdout単体) / Q9(5seed) / Q11(5seed) /
     structured17(seed42) / character39(seed42) : identityのみ新規、v4/highはPhase4Sデータ再利用

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
from phase4t_probes import NAMING_PROBES, P04_PROBES  # noqa: E402

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_V4_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v4")
ADAPTER_HIGH_PATH = str(TRAINING_ROOT / "lora-riru-qwen-ratio-high")
ADAPTER_IDENTITY_PATH = str(TRAINING_ROOT / "lora-riru-qwen-ratio-high-identity")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
SEEDS_5 = (42, 43, 44, 45, 46)
SEEDS_10 = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
SEEDS_20 = tuple(range(42, 62))

ALL_CONDITIONS = ("B_v4", "C_high", "D_identity")
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


def run_multiturn(model, tokenizer, system_prompt, turns, cond, seed):
    messages = [{"role": "system", "content": system_prompt}]
    turn_log = []
    for i in range(0, len(turns), 2):
        user_text = turns[i]
        messages.append({"role": "user", "content": user_text})
        text = generate_reply(model, tokenizer, messages, cond, seed)
        messages.append({"role": "assistant", "content": text})
        turn_log.append({"user": user_text, "assistant": text})
    return turn_log


def main() -> int:
    print("Loading base model + v4/ratio-high/ratio-high-identity adapters...")
    model, tokenizer = build_model_and_tokenizer()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    rag_17q = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rag_17q}
    q3, q9, q11 = by_id["Q3"], by_id["Q9"], by_id["Q11"]

    holdout = json.loads(
        (EVAL_DIR / "phase4i_holdout_omission_v2.json").read_text(encoding="utf-8")
    )
    p01 = next(r for r in holdout if r["id"] == "P01")
    p02 = next(r for r in holdout if r["id"] == "P02")
    p04 = next(r for r in holdout if r["id"] == "P04")

    eval_39 = [
        json.loads(line)
        for line in (EVAL_DIR / "riru_eval_set_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_39_by_id = {x["id"]: x for x in eval_39}
    e36_item = eval_39_by_id["E36"]
    e02_item = eval_39_by_id["E02"]

    results: dict = {
        "naming_probes_identity": {},
        "e36_extended": {}, "e02_extended": {},
        "p04_type_probes_identity": {},
        "q3_greedy": {}, "q3_sampled": {}, "p01": {}, "p02": {}, "p04": {},
        "q9": {}, "q11": {}, "structured_17q": {}, "character_39": {},
    }
    t0 = time.perf_counter()

    print("=== naming probes (22 x greedy+10seed, identity only) ===")
    for probe in NAMING_PROBES:
        greedy_text = run_single(
            model, tokenizer, system_prompt, None, probe["prompt"], "D_identity", 42,
            do_sample=False,
        )
        sampled = {}
        for seed in SEEDS_10:
            text = run_single(
                model, tokenizer, system_prompt, None, probe["prompt"], "D_identity", seed
            )
            sampled[str(seed)] = text
        results["naming_probes_identity"][probe["id"]] = {
            "prompt": probe["prompt"], "greedy": greedy_text, "sampled": sampled,
        }
    print(f"  done ({time.perf_counter() - t0:.1f}s elapsed)")

    print("=== E36/E02 extended (20 seeds x 3 conditions) ===")
    for cond in ALL_CONDITIONS:
        results["e36_extended"][cond] = {}
        results["e02_extended"][cond] = {}
        for seed in SEEDS_20:
            results["e36_extended"][cond][str(seed)] = run_single(
                model, tokenizer, system_prompt, None, e36_item["prompt"], cond, seed
            )
            results["e02_extended"][cond][str(seed)] = run_single(
                model, tokenizer, system_prompt, None, e02_item["prompt"], cond, seed
            )
        print(f"  {cond} done ({time.perf_counter() - t0:.1f}s elapsed)")

    print("=== P04-type 22 probes (identity only) ===")
    for probe in P04_PROBES:
        greedy_text = run_single(
            model, tokenizer, system_prompt, probe["context"], probe["question"],
            "D_identity", 42, do_sample=False,
        )
        sampled = {}
        for seed in SEEDS_5:
            text = run_single(
                model, tokenizer, system_prompt, probe["context"], probe["question"],
                "D_identity", seed,
            )
            sampled[str(seed)] = text
        results["p04_type_probes_identity"][probe["id"]] = {
            "greedy": greedy_text, "sampled": sampled,
        }
    print(f"  done ({time.perf_counter() - t0:.1f}s elapsed)")

    print("=== Regression: Q3/P01/P02/P04/Q9/Q11/structured17/character39 (identity only) ===")
    cond = "D_identity"
    results["q3_greedy"][cond] = run_single(
        model, tokenizer, system_prompt, q3["rag_context_text"], q3["question"], cond, 42,
        do_sample=False,
    )
    results["q3_sampled"][cond] = {
        str(seed): run_single(
            model, tokenizer, system_prompt, q3["rag_context_text"], q3["question"], cond, seed
        )
        for seed in SEEDS_5
    }
    for pid, item in (("p01", p01), ("p02", p02), ("p04", p04)):
        results[pid][cond] = {
            str(seed): run_single(
                model, tokenizer, system_prompt,
                item["rag_context_text"], item["question"], cond, seed,
            )
            for seed in SEEDS_5
        }
    results["q9"][cond] = {
        str(seed): run_single(
            model, tokenizer, system_prompt, q9["rag_context_text"], q9["question"], cond, seed
        )
        for seed in SEEDS_5
    }
    results["q11"][cond] = {
        str(seed): run_single(
            model, tokenizer, system_prompt, q11["rag_context_text"], q11["question"], cond, seed
        )
        for seed in SEEDS_5
    }
    for item in rag_17q:
        text = run_single(
            model, tokenizer, system_prompt, item["rag_context_text"], item["question"], cond, 42
        )
        results["structured_17q"][item["id"]] = {"question": item["question"], "text": text}
    for item in eval_39:
        if item["type"] == "single":
            text = run_single(model, tokenizer, system_prompt, None, item["prompt"], cond, 42)
            results["character_39"][item["id"]] = {"category": item["category"], "text": text}
        else:
            turns = run_multiturn(model, tokenizer, system_prompt, item["turns"], cond, 42)
            results["character_39"][item["id"]] = {
                "category": item["category"],
                "text": " ".join(t["assistant"] for t in turns),
            }
    print(f"  done ({time.perf_counter() - t0:.1f}s elapsed)")

    out_path = EVAL_DIR / "phase4u_comprehensive_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path} (total {time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
