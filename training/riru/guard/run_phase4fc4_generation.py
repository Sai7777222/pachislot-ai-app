# -*- coding: utf-8 -*-
"""Phase4FC4 Stage B/D/E/F/G/H/J: 実本番経路(mode-specific prompt統合後の
ChatService相当ロジック + Phase4ZG)による生成。

precomputed_contexts.json(dispatch()+build_rag_context()を現行コードで再計算済み)
を入力に、各行の`mode`からsystem promptを選択する(_select_system_promptと同じ
ロジックをここに複製する。ChatService本体を直接importしないのは、GPU/HF側の
.venv-qlora環境にpachislot_aiパッケージの全依存(FastAPI等)を入れていないため、
FC3から一貫して採用している「ロジックの複製」パターンを踏襲する)。
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import torch

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
PROJECT_ROOT = GUARD_DIR.parents[2]
REPORTS_DIR = TRAINING_ROOT / "reports"
PROMPTS_DIR = PROJECT_ROOT / "config" / "prompts"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ZG_ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")

_NO_RAG_CONTEXT_MODES = {"SMALL_TALK", "IDENTITY_PERSONA", "OOD_FACTUAL"}


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_fc4_production")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages, seed=42, max_new_tokens=220):
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


def render(path: Path) -> str:
    from jinja2 import Template
    return Template(path.read_text(encoding="utf-8")).render()


def load_prompts() -> dict:
    return {
        "FACTUAL_DEFAULT": render(PROMPTS_DIR / "system.jinja2"),
        "SMALL_TALK": render(PROMPTS_DIR / "small_talk.jinja2"),
        "IDENTITY_PERSONA": render(PROMPTS_DIR / "identity_persona.jinja2"),
        "OOD_FACTUAL": render(PROMPTS_DIR / "ood_boundary.jinja2"),
    }


def select_system_prompt(prompts: dict, mode: str) -> str:
    """ChatService._select_system_prompt()と同一ロジック(置き換え、積み増しなし)。"""
    return prompts.get(mode, prompts["FACTUAL_DEFAULT"])


def build_messages(prompts: dict, mode: str, prompt_text: str, user_content: str, history=None) -> list[dict]:
    """ChatService._build_messages()と同一ロジック。"""
    system_prompt = select_system_prompt(prompts, mode)
    messages = [{"role": "system", "content": system_prompt}]
    if prompt_text:
        messages.append({"role": "system", "content": prompt_text})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages


def run_stage(model, tokenizer, prompts, rows):
    out = []
    for i, c in enumerate(rows):
        messages = build_messages(prompts, c["mode"], c.get("prompt_text", ""), c["prompt"])
        n_system = sum(1 for m in messages if m["role"] == "system")
        text, elapsed = generate(model, tokenizer, messages)
        row = dict(c)
        row["response"] = text
        row["latency_sec"] = elapsed
        row["n_system_messages"] = n_system
        out.append(row)
        print(f"[{i+1}/{len(rows)}] {c['id']} mode={c['mode']} latency={elapsed:.1f}s")
    return out


def run_multiturn(model, tokenizer, prompts, scenarios):
    out = []
    for sc in scenarios:
        history: list[dict] = []
        turn_results = []
        for t_idx, turn in enumerate(sc["turns"]):
            messages = build_messages(prompts, turn["mode"], turn.get("prompt_text", ""), turn["user"], history)
            n_system = sum(1 for m in messages if m["role"] == "system")
            text, elapsed = generate(model, tokenizer, messages)
            row = dict(turn)
            row["response"] = text
            row["latency_sec"] = elapsed
            row["n_system_messages"] = n_system
            turn_results.append(row)
            history.append({"role": "user", "content": turn["user"]})
            history.append({"role": "assistant", "content": text})
            print(f"[multiturn {sc['id']} turn{t_idx}] mode={turn['mode']} "
                  f"n_sys={n_system} latency={elapsed:.1f}s")
        out.append({"id": sc["id"], "description": sc["description"], "turns": turn_results})
    return out


def main():
    prompts = load_prompts()
    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    contexts = json.loads((REPORTS_DIR / "phase4fc4_precomputed_contexts.json").read_text(encoding="utf-8"))

    all_results = {}
    for stage_name in ["stage_b_smalltalk65", "stage_d_identity23", "stage_e_ood15",
                        "stage_f_known_failure12", "stage_g_rag8", "stage_h_conversational10"]:
        results = run_stage(model, tokenizer, prompts, contexts[stage_name])
        all_results[stage_name] = results
        (REPORTS_DIR / "phase4fc4_generation_raw.json").write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    mt_results = run_multiturn(model, tokenizer, prompts, contexts["stage_j_multiturn"])
    all_results["stage_j_multiturn"] = mt_results

    out_path = REPORTS_DIR / "phase4fc4_generation_raw.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(v) for k, v in all_results.items() if k != "stage_j_multiturn")
    total_turns = sum(len(s["turns"]) for s in mt_results)
    print(f"wrote {total} single-turn + {total_turns} multiturn turns -> {out_path}")
    print("PHASE4FC4 GENERATION DONE")


if __name__ == "__main__":
    main()
