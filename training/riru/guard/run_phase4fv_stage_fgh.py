"""Phase4FV Stage F(insufficient context, 10probe x 2rep) + Stage G(Policy C3 rag-path regression、
Stage Fと同一probeで代用) + Stage H(multi-turn, 5 scenarios/16 turns)。P2候補のみで実行、
P0ベースラインはPhase4FU/Phase4FCの既存データを再利用する。dispatch/Policy C3/他モードprompt(小
talk/OOD/conversational/clarify)は一切変更せず、production RAG grounding promptだけをP2に
差し替えて実行する。"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

import torch

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
PROJECT_ROOT = GUARD_DIR.parents[2]
REPORTS_DIR = TRAINING_ROOT / "reports"
sys.path.insert(0, str(GUARD_DIR))
sys.path.insert(0, str(TRAINING_ROOT / "eval"))

from phase4zr_conservative_dispatch import dispatch, UNKNOWN  # noqa: E402
from phase4zp_router import PACHISLOT_FACTUAL, PACHISLOT_CONVERSATIONAL, SMALL_TALK, OOD_FACTUAL  # noqa: E402
from phase4zt_policy_c import decide_c3  # noqa: E402
from phase4fc_multiturn_scenarios import SCENARIOS  # noqa: E402

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ZG_ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")
P2_PATH = GUARD_DIR / "phase4fv_prompts" / "p2_explicit_entity_binding.jinja2"
CLARIFY_PROMPT_PATH = GUARD_DIR / "phase4zr_unknown_ux_prompt_b.txt"
SMALLTALK_PROMPT_PATH = GUARD_DIR / "phase4zp_smalltalk_prompt.txt"
OOD_PROMPT_PATH = GUARD_DIR / "phase4zp_ood_prompt.txt"
CONV_PROMPT_PATH = GUARD_DIR / "phase4zp_pachislot_conversational_prompt.txt"

NUMERIC_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?パーセント|1/\d+")
HEDGE_RE = re.compile("|".join(re.escape(p) for p in
    ["登録データ", "データベース", "データがない", "登録されていない", "情報がない", "記録がない", "確認できない", "見つかりません", "見つからない"]))


def analyze(text):
    return {"response": text, "has_numeric_claim": bool(NUMERIC_PATTERN.search(text)),
            "abstain_or_hedge": bool(HEDGE_RE.search(text))}


def build_context_string(chunks):
    return "\n".join(f"[{c['title']}] {c['text']}" for c in chunks)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_fv2")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system_prompt, user_text, history=None, context=None, seed=42,
             do_sample=False, temperature=None, max_new_tokens=300):
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    if context:
        messages.append({"role": "system", "content": f"[検索結果]\n{context}"})
    messages.append({"role": "user", "content": user_text})
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample,
                  "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id}
    if do_sample and temperature is not None:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        output_ids = model.generate(**encoded, **gen_kwargs)
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


def main():
    p2 = P2_PATH.read_text(encoding="utf-8")
    fu_precomputed = json.loads((REPORTS_DIR / "phase4fu_precomputed_contexts.json").read_text(encoding="utf-8"))
    fu_by_id = {p["id"]: p for p in fu_precomputed}
    fu_stress_baseline = json.loads((REPORTS_DIR / "phase4fu_insufficient_context_stress.json").read_text(encoding="utf-8"))
    stress_ids = sorted(set(r["id"] for r in fu_stress_baseline))

    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    # ---- Stage F/G: insufficient-context + Policy-C3 rag-path regression (shared probe set) ----
    stage_f = []
    for pid in stress_ids:
        p = fu_by_id[pid]
        ctx = build_context_string(p["retrieved_chunks"])
        for i, (sample, temp) in enumerate([(False, None), (True, 0.7)]):
            text = generate(model, tokenizer, p2, p["prompt"], context=ctx, seed=100 + i,
                             do_sample=sample, temperature=temp)
            stage_f.append({"id": pid, "prompt": p["prompt"], "run": i, **analyze(text)})
        print(f"[StageF] {pid} done")
    (REPORTS_DIR / "phase4fv_insufficient_context.json").write_text(
        json.dumps(stage_f, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Stage G: routing/dispatch integrity is hash-verified separately (no gen needed);
    #      here we also record dispatch outcomes for the same probes for traceability ----
    stage_g = []
    for pid in stress_ids:
        p = fu_by_id[pid]
        d = dispatch(p["prompt"])
        stage_g.append({"id": pid, "prompt": p["prompt"], "dispatched_mode": d.mode, "confident": d.confident})
    (REPORTS_DIR / "phase4fv_routing_regression.json").write_text(
        json.dumps({"note": "dispatch()自体は本フェーズ非対象(変更なし)。productionRAGprompt変更がPolicyC3のrag_with_context経路にのみ影響することをStage Fの生成結果と合わせて確認する。small-talk/OOD/conversational/clarify用promptファイルは一切変更していない(ファイルhashで確認済み)。",
                   "dispatch_trace": stage_g}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Stage H: multi-turn (5 scenarios / 16 turns), STRICT_PROMPT replaced with P2 ----
    precomputed_mt = json.loads((REPORTS_DIR / "phase4fc_multiturn_contexts.json").read_text(encoding="utf-8"))
    prompts_by_mode = {
        SMALL_TALK: SMALLTALK_PROMPT_PATH.read_text(encoding="utf-8"),
        OOD_FACTUAL: OOD_PROMPT_PATH.read_text(encoding="utf-8"),
        PACHISLOT_CONVERSATIONAL: CONV_PROMPT_PATH.read_text(encoding="utf-8"),
    }
    clarify_prompt = CLARIFY_PROMPT_PATH.read_text(encoding="utf-8")

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
            pre = precomputed_mt.get(user_text)
            if mode == UNKNOWN:
                if pre is not None:
                    c3 = decide_c3(user_text, pre["context"], pre["titles"], pre["texts"])
                    if c3.selected_path == "rag_with_context":
                        system_prompt = p2
                        context = pre["context"]
                        policy_used = "C3_rag_with_context_P2"
                    else:
                        system_prompt = clarify_prompt
                        policy_used = "C3_clarification"
                else:
                    system_prompt = clarify_prompt
                    policy_used = "C3_clarification_context_unavailable_safe_fallback"
            elif mode == PACHISLOT_FACTUAL:
                if pre is not None:
                    context = pre["context"]
                system_prompt = p2
                policy_used = "direct_pachislot_factual_P2"
            else:
                system_prompt = prompts_by_mode[mode]
                policy_used = f"direct_{mode.lower()}"

            response = generate(model, tokenizer, system_prompt, user_text, history=history, context=context)
            turn_logs.append({"turn": i + 1, "user": user_text, "expected_mode": turn["expected_mode"],
                               "dispatched_mode": mode, "policy_used": policy_used, "response": response})
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": response})
        results.append({"scenario_id": scenario["id"], "description": scenario["description"], "turns": turn_logs})
        print(f"[StageH] {scenario['id']} done")

    (REPORTS_DIR / "phase4fv_multiturn.json").write_text(
        json.dumps({"n_scenarios": len(results), "n_turns": sum(len(r["turns"]) for r in results), "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print("STAGE F/G/H DONE")


if __name__ == "__main__":
    main()
