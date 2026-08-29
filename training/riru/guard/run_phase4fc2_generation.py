# -*- coding: utf-8 -*-
"""Phase4FC2 Gates B/C/D/E/H/J/K: 実本番経路(system.jinja2 + entity_attribution +
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
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_fc2_production")
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


def run_single_turn_gate(model, tokenizer, system_prompt, gate_name, rows):
    out = []
    for i, c in enumerate(rows):
        messages = [{"role": "system", "content": system_prompt}]
        if c.get("prompt_text"):
            messages.append({"role": "system", "content": c["prompt_text"]})
        messages.append({"role": "user", "content": c["prompt"]})
        text, elapsed = generate(model, tokenizer, messages)
        row = {
            "id": c["id"], "category": c.get("category"), "prompt": c["prompt"],
            "chunk_titles": c.get("chunk_titles", []),
            "structured_source_count": c.get("structured_source_count", 0),
            "is_empty": c.get("is_empty"), "response": text, "latency_sec": elapsed,
        }
        out.append(row)
        print(f"[{gate_name} {i+1}/{len(rows)}] {c['id']} latency={elapsed:.1f}s")
    return out


def run_multiturn(model, tokenizer, system_prompt, scenarios):
    out = []
    for sc in scenarios:
        history = []
        turn_results = []
        for t_idx, turn in enumerate(sc["turns"]):
            messages = [{"role": "system", "content": system_prompt}]
            if turn.get("prompt_text"):
                messages.append({"role": "system", "content": turn["prompt_text"]})
            messages.extend(history)
            messages.append({"role": "user", "content": turn["user"]})
            text, elapsed = generate(model, tokenizer, messages)
            turn_results.append({
                "turn_index": t_idx, "user": turn["user"],
                "chunk_titles": turn.get("chunk_titles", []),
                "structured_source_count": turn.get("structured_source_count", 0),
                "response": text, "latency_sec": elapsed,
            })
            history.append({"role": "user", "content": turn["user"]})
            history.append({"role": "assistant", "content": text})
            print(f"[multiturn {sc['id']} turn{t_idx}] latency={elapsed:.1f}s")
        out.append({"id": sc["id"], "description": sc.get("description", ""), "turns": turn_results})
    return out


def main():
    system_prompt = render_system_prompt()
    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    contexts = json.loads((REPORTS_DIR / "phase4fc2_precomputed_contexts.json").read_text(encoding="utf-8"))

    all_results = {}
    for gate_name in ["gate_b_phantom", "gate_c_concept_binding", "gate_d_query_style",
                       "gate_e_production", "gate_h_boundary", "gate_k_identity"]:
        rows = contexts[gate_name]
        results = run_single_turn_gate(model, tokenizer, system_prompt, gate_name, rows)
        all_results[gate_name] = results
        # incremental save
        (REPORTS_DIR / "phase4fc2_generation_raw.json").write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    mt_results = run_multiturn(model, tokenizer, system_prompt, contexts["gate_j_multiturn"])
    all_results["gate_j_multiturn"] = mt_results

    out_path = REPORTS_DIR / "phase4fc2_generation_raw.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(v) for k, v in all_results.items() if k != "gate_j_multiturn")
    print(f"wrote {total} single-turn + {len(mt_results)} multiturn scenarios -> {out_path}")
    print("PHASE4FC2 GENERATION DONE")


if __name__ == "__main__":
    main()
