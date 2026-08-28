"""Phase4ZO: system prompt-only causal test + regression driver。Phase4ZG read-only。
Stage A(causal20) / Stage B(smalltalk65) / Stage C(ood15) / Stage D(pachislot_conv10) /
Stage E(rag50) / ambiguous10 を、指定したsystem prompt variantで生成する。"""
from __future__ import annotations

import argparse
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
BASE_SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
THREE_MODE_PATH = GUARD_DIR / "phase4zo_three_mode_prompt.txt"
MINIMAL_PATH = GUARD_DIR / "phase4zo_minimal_prompt.txt"

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
        trust_remote_code=True, attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_zo")
    model.eval()
    _model, _tokenizer = model, tokenizer
    return model, tokenizer


def system_prompt_for(variant: str) -> str:
    base = BASE_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    if variant == "baseline":
        return base
    if variant == "minimal":
        return MINIMAL_PATH.read_text(encoding="utf-8")
    if variant == "three_mode":
        return base + "\n" + THREE_MODE_PATH.read_text(encoding="utf-8")
    raise ValueError(variant)


def generate(model, tokenizer, system_prompt, user_text, context=None, seed=42):
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": user_text})
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


def run_probes(probes, variant, out_name):
    model, tokenizer = load_model()
    system_prompt = system_prompt_for(variant)
    results = []
    for p in probes:
        raw, dt = generate(model, tokenizer, system_prompt, p["prompt"], context=p.get("context"))
        results.append({"probe_id": p["id"], "category": p.get("category"), "prompt": p["prompt"],
                         "system_prompt_variant": variant, "response": raw, "generation_time_sec": dt})
    out_path = REPORTS_DIR / out_name
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{variant}] {len(results)} generated -> {out_path}")
    return results


def _new120_by_ids(ids):
    from phase4zn_unattended_probes import ALL_PROBES
    by_id = {p["id"]: p for p in ALL_PROBES}
    return [by_id[i] for i in ids]


def stage_causal20(variant):
    from phase4zn_unattended_probes import ALL_PROBES
    probes = [p for p in ALL_PROBES if p["category"] == "personality_preference"]
    assert len(probes) == 20
    out_name = f"phase4zo_causal20_{variant}.json"
    run_probes(probes, variant, out_name)


def stage_smalltalk_recheck(variant="three_mode"):
    from phase4zn_unattended_probes import ALL_PROBES
    cats = {"greeting_farewell", "emotional_casual", "personality_preference", "social_small_talk"}
    probes = [p for p in ALL_PROBES if p["category"] in cats]
    assert len(probes) == 65
    run_probes(probes, variant, "phase4zo_smalltalk_recheck_raw.json")


def stage_ood_recheck(variant="three_mode"):
    from phase4zn_unattended_probes import ALL_PROBES
    probes = [p for p in ALL_PROBES if p["category"] == "ood_factual"]
    assert len(probes) == 15
    run_probes(probes, variant, "phase4zo_ood_recheck_raw.json")


def stage_pachislot_conv_recheck(variant="three_mode"):
    from phase4zn_unattended_probes import ALL_PROBES
    probes = [p for p in ALL_PROBES if p["category"] == "pachislot_conversational"]
    assert len(probes) == 10
    run_probes(probes, variant, "phase4zo_pachislot_conversation_recheck_raw.json")


def stage_ambiguous_recheck(variant="three_mode"):
    from phase4zn_unattended_probes import ALL_PROBES
    probes = [p for p in ALL_PROBES if p["category"] == "ambiguous_boundary"]
    assert len(probes) == 10
    run_probes(probes, variant, "phase4zo_ambiguous_recheck_raw.json")


def stage_rag50_recheck(variant="three_mode"):
    from phase4zf_rag_stress_eval import load_rag_probe_pool
    rag_pool = load_rag_probe_pool()
    required_ids = {"P02", "LC-08", "Q11", "AD-04"}
    required = [p for p in rag_pool if p["id"] in required_ids]
    extra = [p for p in rag_pool if p["id"] not in required_ids][:46]
    probes = required + extra
    normalized = [{"id": p["id"], "category": p.get("set"), "prompt": p["question"], "context": p.get("context")}
                  for p in probes]
    assert len(normalized) == 50
    run_probes(normalized, variant, "phase4zo_rag50_recheck_raw.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                         choices=["causal20", "smalltalk65", "ood15", "pachislot_conv10",
                                  "ambiguous10", "rag50"])
    parser.add_argument("--variant", default="three_mode", choices=["baseline", "minimal", "three_mode"])
    args = parser.parse_args()
    {
        "causal20": lambda: stage_causal20(args.variant),
        "smalltalk65": lambda: stage_smalltalk_recheck(args.variant),
        "ood15": lambda: stage_ood_recheck(args.variant),
        "pachislot_conv10": lambda: stage_pachislot_conv_recheck(args.variant),
        "ambiguous10": lambda: stage_ambiguous_recheck(args.variant),
        "rag50": lambda: stage_rag50_recheck(args.variant),
    }[args.stage]()
