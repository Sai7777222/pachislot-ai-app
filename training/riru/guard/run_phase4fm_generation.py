# -*- coding: utf-8 -*-
"""Phase4FM Section20-23: 実本番経路(モデレーション統合後のChatService相当ロジック
+ Phase4ZG)による生成。ChatService.check_input()/_select_system_prompt()/
check_output()と同一ロジックをここに複製する(FC4から一貫したパターン: GPU/HF側の
.venv-qlora環境にpachislot_aiパッケージの全依存を入れていないため、ChatService
本体を直接importしない。moderationモジュール自体は軽量なのでここでは直接importする)。
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
PROJECT_ROOT = GUARD_DIR.parents[2]
REPORTS_DIR = TRAINING_ROOT / "reports"
PROMPTS_DIR = PROJECT_ROOT / "config" / "prompts"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
# 注意: pachislot_ai.core.config は pydantic-settings に依存しており、この
# .venv-qlora(GPU推論専用、FastAPI/pydantic-settings等はインストールしていない)
# には入っていないため、config.py を経由せずパスを直接構築する
# (FC4のGPUスクリプトが chat_service.py 全体をimportしなかったのと同じ理由)。
MODERATION_POLICY_PATH = PROJECT_ROOT / "config" / "moderation.yaml"
from pachislot_ai.moderation import ModerationEngine  # noqa: E402

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
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_fm_production")
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
    return prompts.get(mode, prompts["FACTUAL_DEFAULT"])


def build_messages(prompts: dict, mode: str, prompt_text: str, user_content: str, history=None) -> list[dict]:
    system_prompt = select_system_prompt(prompts, mode)
    messages = [{"role": "system", "content": system_prompt}]
    if prompt_text:
        messages.append({"role": "system", "content": prompt_text})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages


def run_stage(model, tokenizer, prompts, engine: ModerationEngine, rows):
    out = []
    for i, c in enumerate(rows):
        # Section9/11と同一の順序: 入力チェック -> (ブロックならLLM/RAGを呼ばない) ->
        # 生成 -> 出力チェック -> 最終content決定。
        input_mod = engine.check_input(c["prompt"])
        if not input_mod.allowed:
            row = dict(c)
            row["response"] = input_mod.safe_response
            row["moderation_action"] = "HARD_BLOCK_INPUT"
            row["llm_called"] = False
            row["latency_sec"] = 0.0
            out.append(row)
            print(f"[{i+1}/{len(rows)}] {c['id']} BLOCKED(input) no LLM call")
            continue

        messages = build_messages(prompts, c["mode"], c.get("prompt_text", ""), c["prompt"])
        text, elapsed = generate(model, tokenizer, messages)
        output_mod = engine.check_output(text)
        final_text = output_mod.safe_response if not output_mod.allowed else text

        row = dict(c)
        row["raw_response"] = text
        row["response"] = final_text
        row["moderation_action"] = "ALLOWED" if output_mod.allowed else "HARD_BLOCK_OUTPUT"
        row["llm_called"] = True
        row["latency_sec"] = elapsed
        out.append(row)
        print(f"[{i+1}/{len(rows)}] {c['id']} mode={c['mode']} moderation={row['moderation_action']} latency={elapsed:.1f}s")
    return out


def run_multiturn(model, tokenizer, prompts, engine: ModerationEngine, scenarios):
    out = []
    for sc in scenarios:
        history: list[dict] = []
        turn_results = []
        for t_idx, turn in enumerate(sc["turns"]):
            synthetic_blocked = turn.get("synthetic_blocked", False)

            if synthetic_blocked == "input":
                input_mod = engine.check_input(turn["user"])
                assert not input_mod.allowed, "precomputeでinputブロック対象と分類された発話がengine側で許可された"
                text = input_mod.safe_response
                llm_called = False
                elapsed = 0.0
                action = "HARD_BLOCK_INPUT"
            else:
                input_mod = engine.check_input(turn["user"])
                if not input_mod.allowed:
                    text = input_mod.safe_response
                    llm_called = False
                    elapsed = 0.0
                    action = "HARD_BLOCK_INPUT"
                else:
                    messages = build_messages(prompts, turn["mode"], turn.get("prompt_text", ""), turn["user"], history)
                    raw_text, elapsed = generate(model, tokenizer, messages)
                    output_mod = engine.check_output(raw_text)
                    text = output_mod.safe_response if not output_mod.allowed else raw_text
                    llm_called = True
                    action = "ALLOWED" if output_mod.allowed else "HARD_BLOCK_OUTPUT"

            row = {
                "user": turn["user"], "mode": turn.get("mode"), "response": text,
                "moderation_action": action, "llm_called": llm_called, "latency_sec": elapsed,
            }
            turn_results.append(row)
            # ブロックされたターンの生成text(=ユーザーに見せない安全応答)を履歴に
            # 積むのは問題ないが、そもそも入力自体を安全側で扱うため、ここでは
            # ブロックされたuserターンをそのまま履歴に残す設計とする(Section23の
            # 『ブロックされたターンが会話履歴を汚染しないことが望ましい』という
            # 指示を踏まえ、次項のhistory-audit分析で個別に評価する)。
            history.append({"role": "user", "content": turn["user"]})
            history.append({"role": "assistant", "content": text})
            print(f"[multiturn {sc['id']} turn{t_idx}] mode={turn.get('mode')} action={action} "
                  f"llm_called={llm_called} latency={elapsed:.1f}s")
        out.append({"id": sc["id"], "description": sc["description"], "turns": turn_results})
    return out


def main():
    prompts = load_prompts()
    engine = ModerationEngine.from_yaml(MODERATION_POLICY_PATH)
    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    contexts = json.loads((REPORTS_DIR / "phase4fm_precomputed_contexts.json").read_text(encoding="utf-8"))

    all_results = {}
    for stage_name in ["smalltalk20", "identity_representative", "ood_representative",
                        "known_failure12", "rag8"]:
        results = run_stage(model, tokenizer, prompts, engine, contexts[stage_name])
        all_results[stage_name] = results
        (REPORTS_DIR / "phase4fm_generation_raw.json").write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    mt_results = run_multiturn(model, tokenizer, prompts, engine, contexts["multiturn"])
    all_results["multiturn"] = mt_results

    out_path = REPORTS_DIR / "phase4fm_generation_raw.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(v) for k, v in all_results.items() if k != "multiturn")
    total_turns = sum(len(s["turns"]) for s in mt_results)
    llm_calls = sum(
        sum(1 for r in v if r.get("llm_called")) for k, v in all_results.items() if k != "multiturn"
    ) + sum(sum(1 for t in s["turns"] if t["llm_called"]) for s in mt_results)
    print(f"wrote {total} single-turn + {total_turns} multiturn turns "
          f"(actual LLM calls={llm_calls}) -> {out_path}")
    print("PHASE4FM GENERATION DONE")


if __name__ == "__main__":
    main()
