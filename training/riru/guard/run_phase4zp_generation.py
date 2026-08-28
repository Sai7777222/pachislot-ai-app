"""Phase4ZP Stage B-D: router経由の生成。SMALL_TALK/OOD_FACTUAL/PACHISLOT_CONVERSATIONAL
の3モードはそれぞれ専用の軽量policy promptを使う。PACHISLOT_FACTUAL pathは既存
system.jinja2を一切変更せずそのまま使う(Stage Eの静的routing解析で router自体の
信頼性不足が既に確定したため、本スクリプトではPACHISLOT_FACTUAL側の実生成は
行わない -- 意味のない生成に GPU時間を使わないため)。"""
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
sys.path.insert(0, str(TRAINING_ROOT))
sys.path.insert(0, str(TRAINING_ROOT / "eval"))
REPORTS_DIR = TRAINING_ROOT / "reports"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ZG_ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")

from phase4zp_router import route, SMALL_TALK, PACHISLOT_FACTUAL, PACHISLOT_CONVERSATIONAL, OOD_FACTUAL  # noqa: E402

PROMPT_PATHS = {
    SMALL_TALK: GUARD_DIR / "phase4zp_smalltalk_prompt.txt",
    OOD_FACTUAL: GUARD_DIR / "phase4zp_ood_prompt.txt",
    PACHISLOT_CONVERSATIONAL: GUARD_DIR / "phase4zp_pachislot_conversational_prompt.txt",
}

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_zp")
    model.eval()
    _model, _tokenizer = model, tokenizer
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


def run_stage(probes, expected_mode_filter, out_name, retrieval_trace_name=None):
    model, tokenizer = load_model()
    results = []
    trace = []
    for p in probes:
        r = route(p["prompt"])
        retrieval_called = r.mode == PACHISLOT_FACTUAL  # 本stageではPACHISLOT_FACTUAL経路は生成しない
        trace.append({"probe_id": p["id"], "prompt": p["prompt"], "routed_mode": r.mode,
                       "matched_rule": r.matched_rule, "retrieval_call_simulated": retrieval_called})
        if r.mode == PACHISLOT_FACTUAL:
            # このstageではfactual path自体は生成しない(router信頼性が既に不十分と判明したため)。
            results.append({"probe_id": p["id"], "prompt": p["prompt"], "routed_mode": r.mode,
                             "response": None, "skipped_reason": "routed_to_pachislot_factual_not_generated_this_stage"})
            continue
        system_prompt = PROMPT_PATHS[r.mode].read_text(encoding="utf-8")
        raw, dt = generate(model, tokenizer, system_prompt, p["prompt"])
        results.append({"probe_id": p["id"], "category": p.get("category"), "prompt": p["prompt"],
                         "routed_mode": r.mode, "response": raw, "generation_time_sec": dt})
    (REPORTS_DIR / out_name).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if retrieval_trace_name:
        (REPORTS_DIR / retrieval_trace_name).write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{out_name}] {len(results)} done")
    return results


def main():
    from phase4zn_unattended_probes import ALL_PROBES

    smalltalk_cats = {"greeting_farewell", "emotional_casual", "personality_preference", "social_small_talk"}
    smalltalk_probes = [{"id": p["id"], "category": p["category"], "prompt": p["prompt"]}
                         for p in ALL_PROBES if p["category"] in smalltalk_cats]
    ood_probes = [{"id": p["id"], "category": p["category"], "prompt": p["prompt"]}
                  for p in ALL_PROBES if p["category"] == "ood_factual"]
    conv_probes = [{"id": p["id"], "category": p["category"], "prompt": p["prompt"]}
                   for p in ALL_PROBES if p["category"] == "pachislot_conversational"]

    assert len(smalltalk_probes) == 65 and len(ood_probes) == 15 and len(conv_probes) == 10

    run_stage(smalltalk_probes, SMALL_TALK, "phase4zp_smalltalk_recheck_raw.json", "phase4zp_retrieval_trace_smalltalk.json")
    run_stage(ood_probes, OOD_FACTUAL, "phase4zp_ood_recheck_raw.json", "phase4zp_retrieval_trace_ood.json")
    run_stage(conv_probes, PACHISLOT_CONVERSATIONAL, "phase4zp_pachislot_conversation_recheck_raw.json", "phase4zp_retrieval_trace_pachislot_factual.json")
    print("ALL STAGES DONE")


if __name__ == "__main__":
    main()
