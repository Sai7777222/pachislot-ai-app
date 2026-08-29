# -*- coding: utf-8 -*-
"""Phase4FC3 Stage B-J: 実本番経路(dispatch統合後のChatService + Phase4ZG)による生成。"""
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
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_fc3_production")
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


def run_stage(model, tokenizer, system_prompt, rows):
    out = []
    for i, c in enumerate(rows):
        messages = [{"role": "system", "content": system_prompt}]
        if c.get("prompt_text"):
            messages.append({"role": "system", "content": c["prompt_text"]})
        messages.append({"role": "user", "content": c["prompt"]})
        text, elapsed = generate(model, tokenizer, messages)
        row = dict(c)
        row["response"] = text
        row["latency_sec"] = elapsed
        out.append(row)
        print(f"[{i+1}/{len(rows)}] {c['id']} mode={c.get('mode')} latency={elapsed:.1f}s")
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
            row = dict(turn)
            row["response"] = text
            row["latency_sec"] = elapsed
            turn_results.append(row)
            history.append({"role": "user", "content": turn["user"]})
            history.append({"role": "assistant", "content": text})
            print(f"[multiturn {sc['id']} turn{t_idx}] mode={turn.get('mode')} latency={elapsed:.1f}s")
        out.append({"id": sc["id"], "description": sc["description"], "turns": turn_results})
    return out


def main():
    system_prompt = render_system_prompt()
    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    contexts = json.loads((REPORTS_DIR / "phase4fc3_precomputed_contexts.json").read_text(encoding="utf-8"))

    all_results = {}
    for stage_name in ["stage_b_smalltalk65", "stage_c_identity", "stage_d_ood15",
                        "stage_e_conversational10", "stage_g_evidence_arbitration",
                        "stage_h_known_failure12", "stage_i_rag8"]:
        results = run_stage(model, tokenizer, system_prompt, contexts[stage_name])
        all_results[stage_name] = results
        (REPORTS_DIR / "phase4fc3_generation_raw.json").write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    mt_results = run_multiturn(model, tokenizer, system_prompt, contexts["stage_j_multiturn"])
    all_results["stage_j_multiturn"] = mt_results

    out_path = REPORTS_DIR / "phase4fc3_generation_raw.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(v) for k, v in all_results.items() if k != "stage_j_multiturn")
    print(f"wrote {total} single-turn + {len(mt_results)} multiturn scenarios -> {out_path}")
    print("PHASE4FC3 GENERATION DONE")


if __name__ == "__main__":
    main()
