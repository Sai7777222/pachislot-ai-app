"""Phase4FC: true multi-turn setをaccepted architecture(conservative dispatch + Policy C3)
経由で実行する。会話履歴(user/assistant turns)を維持しつつ、各turnのsystem promptは
そのturnのdispatch結果に応じて動的に選択する(実運用のrouter+multi-turn LLMの挙動を模す)。"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
PROJECT_ROOT = GUARD_DIR.parents[2]
sys.path.insert(0, str(GUARD_DIR))
sys.path.insert(0, str(TRAINING_ROOT / "eval"))
REPORTS_DIR = TRAINING_ROOT / "reports"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ZG_ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")
STRICT_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
CLARIFY_PROMPT_PATH = GUARD_DIR / "phase4zr_unknown_ux_prompt_b.txt"
SMALLTALK_PROMPT_PATH = GUARD_DIR / "phase4zp_smalltalk_prompt.txt"
OOD_PROMPT_PATH = GUARD_DIR / "phase4zp_ood_prompt.txt"
CONV_PROMPT_PATH = GUARD_DIR / "phase4zp_pachislot_conversational_prompt.txt"

from phase4zr_conservative_dispatch import dispatch, UNKNOWN  # noqa: E402
from phase4zp_router import PACHISLOT_FACTUAL, PACHISLOT_CONVERSATIONAL, SMALL_TALK, OOD_FACTUAL  # noqa: E402
from phase4zt_policy_c import decide_c3  # noqa: E402
from phase4fc_multiturn_scenarios import SCENARIOS  # noqa: E402

PROMPTS = {
    SMALL_TALK: SMALLTALK_PROMPT_PATH.read_text(encoding="utf-8"),
    OOD_FACTUAL: OOD_PROMPT_PATH.read_text(encoding="utf-8"),
    PACHISLOT_CONVERSATIONAL: CONV_PROMPT_PATH.read_text(encoding="utf-8"),
}
STRICT_PROMPT = STRICT_PROMPT_PATH.read_text(encoding="utf-8")
CLARIFY_PROMPT = CLARIFY_PROMPT_PATH.read_text(encoding="utf-8")


def load_precomputed_contexts():
    # venv分離のため、retrieval(.venv側)は事前に別途実行しJSONへ保存済み。
    # このscript(.venv-qlora側、GPU generation用)はretrieverを直接importしない。
    return json.loads((REPORTS_DIR / "phase4fc_multiturn_contexts.json").read_text(encoding="utf-8"))


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_fc")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system_prompt, history, user_text, context=None, seed=42):
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    if context:
        messages.append({"role": "system", "content": f"[検索結果]\n{context}"})
    messages.append({"role": "user", "content": user_text})
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    with torch.no_grad():
        output_ids = model.generate(**encoded, max_new_tokens=300, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


def main():
    precomputed = load_precomputed_contexts()
    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    results = []
    for scenario in SCENARIOS:
        history = []
        turn_logs = []
        for i, turn in enumerate(scenario["turns"]):
            user_text = turn["user"]
            d = dispatch(user_text)
            mode = d.mode
            context = None
            policy_used = None
            pre = precomputed.get(user_text)
            if mode == UNKNOWN:
                if pre is not None:
                    c3 = decide_c3(user_text, pre["context"], pre["titles"], pre["texts"])
                    if c3.selected_path == "rag_with_context":
                        system_prompt = STRICT_PROMPT
                        context = pre["context"]
                        policy_used = "C3_rag_with_context"
                    else:
                        system_prompt = CLARIFY_PROMPT
                        policy_used = "C3_clarification"
                else:
                    system_prompt = CLARIFY_PROMPT
                    policy_used = "C3_clarification_context_unavailable_safe_fallback"
            elif mode == PACHISLOT_FACTUAL:
                if pre is not None:
                    context = pre["context"]
                system_prompt = STRICT_PROMPT
                policy_used = "direct_pachislot_factual"
            else:
                system_prompt = PROMPTS[mode]
                policy_used = f"direct_{mode.lower()}"

            response = generate(model, tokenizer, system_prompt, history, user_text, context=context)
            turn_logs.append({"turn": i + 1, "user": user_text, "expected_mode": turn["expected_mode"],
                               "dispatched_mode": mode, "policy_used": policy_used, "response": response})
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": response})
        results.append({"scenario_id": scenario["id"], "description": scenario["description"], "turns": turn_logs})
        print(f"{scenario['id']} done")

    (REPORTS_DIR / "phase4fc_multiturn_results.json").write_text(
        json.dumps({"n_scenarios": len(results), "n_turns": sum(len(r["turns"]) for r in results), "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("ALL SCENARIOS DONE")


if __name__ == "__main__":
    main()
