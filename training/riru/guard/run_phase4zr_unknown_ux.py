"""Phase4ZR Stage E: UNKNOWN 30件をPolicy A(既存strict RAG system prompt、変更なし)と
Policy B(clarification)で比較生成する。最大60 generations。Phase4ZG read-only。"""
from __future__ import annotations
import json
import random
import sys
import time
from pathlib import Path

import torch

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
PROJECT_ROOT = GUARD_DIR.parents[2]
sys.path.insert(0, str(GUARD_DIR))
REPORTS_DIR = TRAINING_ROOT / "reports"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ZG_ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")
BASE_SYSTEM_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
POLICY_B_PATH = GUARD_DIR / "phase4zr_unknown_ux_prompt_b.txt"


def sample_unknown(n=30, seed=4):
    d = json.loads((REPORTS_DIR / "phase4zr_dispatch_results.json").read_text(encoding="utf-8"))
    unk = [r for r in d["rows"] if r["dispatched_mode"] == "UNKNOWN"]
    by_mode = {}
    for r in unk:
        by_mode.setdefault(r["expected_mode"], []).append(r)
    rng = random.Random(seed)
    quota = {"SMALL_TALK": 15, "PACHISLOT_FACTUAL": 10, "OOD_FACTUAL": 3, "PACHISLOT_CONVERSATIONAL": 2}
    sample = []
    for mode, q in quota.items():
        pool = by_mode.get(mode, [])
        rng.shuffle(pool)
        sample.extend(pool[:q])
    return sample[:n]


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_zr")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system_prompt, user_text, seed=42):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**encoded, max_new_tokens=300, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    dt = time.perf_counter() - t0
    text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
    return text, dt


def main():
    sample = sample_unknown()
    print(f"sampled {len(sample)} UNKNOWN probes")
    policy_a_prompt = BASE_SYSTEM_PATH.read_text(encoding="utf-8")
    policy_b_prompt = POLICY_B_PATH.read_text(encoding="utf-8")

    model, tokenizer = load_model()
    results = []
    for r in sample:
        a_resp, a_dt = generate(model, tokenizer, policy_a_prompt, r["prompt"])
        b_resp, b_dt = generate(model, tokenizer, policy_b_prompt, r["prompt"])
        results.append({"probe_id": r["probe_id"], "expected_mode": r["expected_mode"], "prompt": r["prompt"],
                         "policy_a_strict_rag_response": a_resp, "policy_a_time_sec": a_dt,
                         "policy_b_clarification_response": b_resp, "policy_b_time_sec": b_dt})
        print(f"{r['probe_id']} done")

    out = {"n_sampled": len(results), "max_generations_budget": 60,
           "generations_used": len(results) * 2, "rows": results}
    (REPORTS_DIR / "phase4zr_unknown_ux_raw.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE, generations_used={len(results)*2}")


if __name__ == "__main__":
    main()
