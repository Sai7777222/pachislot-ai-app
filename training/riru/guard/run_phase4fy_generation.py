# -*- coding: utf-8 -*-
"""Phase4FY Stage B-J: 実際の本番経路(system.jinja2 + entity-attribution統合後の
RagPipeline context + Phase4ZG)による生成。CPU側で事前計算した
phase4fy_precomputed_contexts.json / phase4fy_boundary_contexts.json /
phase4fy_multiturn_contexts.json を読み、GPU側(.venv-qlora)ではモデルのロードと
生成のみを行う(既存 phase4zt/phase4fw と同じ分離パターン)。

システムプロンプトは config/prompts/system.jinja2 をそのまま使用(変更なし)。
ChatService._build_messages() と同じ構成: [system, rag_context.prompt_text(空でなければ), user...]
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import torch

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent  # .../training/riru
PROJECT_ROOT = GUARD_DIR.parents[2]  # guard -> riru -> training -> project root
REPORTS_DIR = TRAINING_ROOT / "reports"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ZG_ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

AMBIGUOUS_ZERO_SELECTION_IDS_FOR_A0_COMPARISON = [
    "P02", "P10", "AD-04", "PT-16", "LC-08", "FX-K07", "FX-K08",
    "RJ-10", "RJ-11", "RJ-14",
]


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_fy_production")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages, seed=42, max_new_tokens=512):
    """production同様 do_sample可能だが、regression判定の再現性のためgreedy(seed固定)を使う
    (本番 default は temperature=0.7 だが、Phase4FC以降の全phaseで一貫してこの評価手法を
    採用しており、新しい手法ではない)。"""
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


def run_single_turn(model, tokenizer, system_prompt):
    contexts = json.loads((REPORTS_DIR / "phase4fy_precomputed_contexts.json").read_text(encoding="utf-8"))
    out = []
    for i, c in enumerate(contexts):
        messages = [{"role": "system", "content": system_prompt}]
        if c["fy_prompt_text"]:
            messages.append({"role": "system", "content": c["fy_prompt_text"]})
        messages.append({"role": "user", "content": c["prompt"]})
        text, elapsed = generate(model, tokenizer, messages)
        row = {
            "id": c["id"], "stage": c["stage"], "category": c["category"], "prompt": c["prompt"],
            "query_entities": c["query_entities"],
            "raw_embedding_titles": c["raw_embedding_titles"],
            "selected_titles": c["selected_titles_after_entity_attribution"],
            "fy_is_empty": c["fy_is_empty"], "a0_is_empty": c["a0_is_empty"],
            "fy_response": text, "fy_latency_sec": elapsed,
        }
        if c["id"] in AMBIGUOUS_ZERO_SELECTION_IDS_FOR_A0_COMPARISON:
            a0_messages = [{"role": "system", "content": system_prompt}]
            if c["a0_prompt_text"]:
                a0_messages.append({"role": "system", "content": c["a0_prompt_text"]})
            a0_messages.append({"role": "user", "content": c["prompt"]})
            a0_text, a0_elapsed = generate(model, tokenizer, a0_messages)
            row["a0_response"] = a0_text
            row["a0_latency_sec"] = a0_elapsed
        out.append(row)
        print(f"[{i+1}/{len(contexts)}] {c['id']} ({c['stage']}) fy_latency={elapsed:.1f}s"
              + (f" +A0={row.get('a0_latency_sec', 0):.1f}s" if "a0_response" in row else ""))
    return out


def run_boundary(model, tokenizer, system_prompt):
    contexts = json.loads((REPORTS_DIR / "phase4fy_boundary_contexts.json").read_text(encoding="utf-8"))
    out = []
    for i, c in enumerate(contexts):
        messages = [{"role": "system", "content": system_prompt}]
        if c["fy_prompt_text"]:
            messages.append({"role": "system", "content": c["fy_prompt_text"]})
        messages.append({"role": "user", "content": c["prompt"]})
        text, elapsed = generate(model, tokenizer, messages)
        out.append({
            "id": c["id"], "category": c["category"], "prompt": c["prompt"],
            "query_entities": c["query_entities"], "fy_is_empty": c["fy_is_empty"],
            "selected_titles": c["selected_titles_after_entity_attribution"],
            "response": text, "latency_sec": elapsed,
        })
        print(f"[boundary {i+1}/{len(contexts)}] {c['id']} ({c['category']}) latency={elapsed:.1f}s")
    return out


def run_multiturn(model, tokenizer, system_prompt):
    scenarios = json.loads((REPORTS_DIR / "phase4fy_multiturn_contexts.json").read_text(encoding="utf-8"))
    out = []
    for sc in scenarios:
        history = []  # list of {"role": "user"/"assistant", "content": ...}
        turn_results = []
        for t_idx, turn in enumerate(sc["turns"]):
            messages = [{"role": "system", "content": system_prompt}]
            if turn["fy_prompt_text"]:
                messages.append({"role": "system", "content": turn["fy_prompt_text"]})
            messages.extend(history)
            messages.append({"role": "user", "content": turn["user"]})
            text, elapsed = generate(model, tokenizer, messages)
            turn_results.append({
                "turn_index": t_idx, "user": turn["user"], "expected_mode": turn["expected_mode"],
                "query_entities": turn["query_entities"], "fy_is_empty": turn["fy_is_empty"],
                "selected_titles": turn["selected_titles_after_entity_attribution"],
                "response": text, "latency_sec": elapsed,
            })
            history.append({"role": "user", "content": turn["user"]})
            history.append({"role": "assistant", "content": text})
            print(f"[multiturn {sc['id']} turn{t_idx}] latency={elapsed:.1f}s")
        out.append({"id": sc["id"], "description": sc["description"], "turns": turn_results})
    return out


def main():
    system_prompt = render_system_prompt()
    print(f"system_prompt loaded ({len(system_prompt)} chars)")

    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    single_turn_results = run_single_turn(model, tokenizer, system_prompt)
    (REPORTS_DIR / "phase4fy_generation_raw.json").write_text(
        json.dumps(single_turn_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(single_turn_results)} single-turn generations")

    boundary_results = run_boundary(model, tokenizer, system_prompt)
    (REPORTS_DIR / "phase4fy_boundary_generation_raw.json").write_text(
        json.dumps(boundary_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(boundary_results)} boundary generations")

    multiturn_results = run_multiturn(model, tokenizer, system_prompt)
    (REPORTS_DIR / "phase4fy_multiturn_generation_raw.json").write_text(
        json.dumps(multiturn_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(multiturn_results)} multiturn generations")

    print("PHASE4FY GENERATION DONE")


if __name__ == "__main__":
    main()
