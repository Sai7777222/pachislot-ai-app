# -*- coding: utf-8 -*-
"""Phase4FC2 Section11 (Gate F): 全50probeを、Phase4FWで凍結された静的contextのまま
(実DB retrievalを経由せず)実行する。実本番DB E2Eとは明確に分離して報告する。"""
from __future__ import annotations
import json
import time
from pathlib import Path

import torch

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
PROJECT_ROOT = GUARD_DIR.parents[2]
REPORTS_DIR = TRAINING_ROOT / "reports"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ZG_ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_fc2_static")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages, seed=42, max_new_tokens=512):
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    gen_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    elapsed = time.time() - gen_start
    text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
    return text, elapsed


def render_system_prompt() -> str:
    from jinja2 import Template
    return Template(SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")).render()


def main():
    system_prompt = render_system_prompt()
    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    target = json.loads((REPORTS_DIR / "phase4fw_target_responses.json").read_text(encoding="utf-8"))
    rag50 = [x for x in target if x.get("category") == "rag50"]
    print(f"n_rag50: {len(rag50)}")

    out = []
    for i, r in enumerate(rag50):
        messages = [{"role": "system", "content": system_prompt}]
        if r["context"]:
            messages.append({"role": "system", "content": r["context"]})
        messages.append({"role": "user", "content": r["prompt"]})
        text, elapsed = generate(model, tokenizer, messages)
        out.append({
            "id": r["id"], "prompt": r["prompt"], "frozen_context": r["context"],
            "historical_response": r.get("response"), "fc2_response": text, "latency_sec": elapsed,
        })
        print(f"[{i+1}/{len(rag50)}] {r['id']} latency={elapsed:.1f}s")

    out_path = REPORTS_DIR / "phase4fc2_static_rag50_generation_raw.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(out)} -> {out_path}")
    print("STATIC RAG50 GENERATION DONE")


if __name__ == "__main__":
    main()
