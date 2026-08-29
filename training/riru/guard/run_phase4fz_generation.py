# -*- coding: utf-8 -*-
"""Phase4FZ Section9-14: 実際の本番経路(system.jinja2 + entity_attribution + 修正済み
structured_lookup + Phase4ZG)による生成。"""
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
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_fz_production")
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
    print(f"system_prompt loaded ({len(system_prompt)} chars)")

    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    contexts = json.loads((REPORTS_DIR / "phase4fz_precomputed_contexts.json").read_text(encoding="utf-8"))
    out = []
    for i, c in enumerate(contexts):
        messages = [{"role": "system", "content": system_prompt}]
        if c["prompt_text"]:
            messages.append({"role": "system", "content": c["prompt_text"]})
        messages.append({"role": "user", "content": c["prompt"]})
        text, elapsed = generate(model, tokenizer, messages)
        out.append({
            "id": c["id"], "stage": c["stage"], "label": c["label"], "prompt": c["prompt"],
            "chunk_titles": c["chunk_titles"], "structured_source_count": c["structured_source_count"],
            "is_empty": c["is_empty"], "response": text, "latency_sec": elapsed,
        })
        print(f"[{i+1}/{len(contexts)}] {c['id']} ({c['stage']}) latency={elapsed:.1f}s")

    out_path = REPORTS_DIR / "phase4fz_generation_raw.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(out)} -> {out_path}")
    print("PHASE4FZ GENERATION DONE")


if __name__ == "__main__":
    main()
