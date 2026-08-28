"""Phase 4ZJ: Phase4ZJ candidate(instruction_override_resistance追加)の評価。

Stage1: Regression Guard(Phase4ZIで安全確認済みのprobe集合を再評価、悪化検知が目的)
Stage2: instruction_override core評価(baseline9probe再現 + 新規held-out16probe)
Stage3: 限定multi-turn確認(original6 + ZI追加32から代表12 = 18scenario)
Stage4: RAG/OOD causal safety確認
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
PROJECT_ROOT = EVAL_DIR.parents[2]
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(TRAINING_ROOT))
REPORTS_DIR = TRAINING_ROOT / "reports"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zj-instruction-override-hardened")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"


def load_model(attn_impl: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation=attn_impl,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, adapter_name="phase4zj")
    model.eval()
    return model, tokenizer


def generate_reply(model, tokenizer, messages, seed, do_sample) -> str:
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    gen_kwargs = dict(max_new_tokens=300, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=0.3, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)
    with torch.no_grad():
        output_ids = model.generate(**encoded, **gen_kwargs)
    completion_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


# ============================================================
# Stage1: Regression Guard
# ============================================================

REGRESSION_GUARD_IDS = {
    "identity_safe": ["ZHH-B02", "ZHH-B04", "ZHH-C02", "ZHH-C03", "ZHH-C04", "ZHH-C05", "ZHH-E02",
                       "ZHH-F01", "ZHH-F02", "ZHH-F03", "ZHH-F05", "ZHH-G01", "ZHH-G02", "ZHH-G03", "ZHH-G04",
                       "ZFB-02", "ZFB-08", "ZFB-11", "ZFB-13", "ZFB-14", "ZFC-03", "ZFC-04", "ZFC-13",
                       "ZFD-01", "ZFD-03", "ZFD-05", "ZFD-06", "ZFD-07", "ZFD-09", "ZFD-10"],
    "nickname_safe": ["ZHH-H02"],
    "role_name_safe": ["ZFC-04"],
    "no_name_control": ["ZHH-X01", "ZHH-X04"],
}


def load_regression_guard_pool() -> list[dict]:
    from phase4zg_holdout_probes import ALL_PROBES as ZGH
    from phase4zh_holdout_probes import ALL_PROBES as ZHH
    from phase4zf_stress_probes import WRONG_NAME_INDUCTION, ROLE_NAME_CONFUSION, IDENTITY_CORRECTION_STRESS

    by_id = {}
    for p in ZHH:
        by_id[p["id"]] = p["prompt"]
    for p in WRONG_NAME_INDUCTION + ROLE_NAME_CONFUSION + IDENTITY_CORRECTION_STRESS:
        by_id[p["id"]] = p["prompt"]

    probes = []
    all_ids = (REGRESSION_GUARD_IDS["identity_safe"] + REGRESSION_GUARD_IDS["nickname_safe"]
               + REGRESSION_GUARD_IDS["role_name_safe"] + REGRESSION_GUARD_IDS["no_name_control"])
    for pid in dict.fromkeys(all_ids):  # dedupe preserving order (ZFC-04 appears twice)
        if pid in by_id:
            probes.append({"set": "identity_guard", "id": pid, "prompt": by_id[pid]})

    from phase4zi_ood_sanity_probes import ALL_PROBES as OOD
    ood_guard = ["ZI-OD-01", "ZI-OD-02", "ZI-OD-03", "ZI-OD-04", "ZI-OD-05", "ZI-OD-06",
                 "ZI-OD-09", "ZI-OD-10", "ZI-OD-15", "ZI-OD-19", "ZI-OD-20", "ZI-OD-21"]
    for p in OOD:
        if p["id"] in ood_guard:
            probes.append({"set": "ood_guard", "id": p["id"], "prompt": p["prompt"]})
    return probes


def mode_stage1_guard(attn_impl: str) -> int:
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)
    probes = load_regression_guard_pool()

    results = {}
    for p in probes:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": p["prompt"]}]
        greedy = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
        results[p["id"]] = {"set": p["set"], "prompt": p["prompt"], "greedy": greedy}

    out = {"attn_impl": attn_impl, "n_probes": len(probes), "results": results}
    out_path = REPORTS_DIR / f"phase4zj_stage1_guard_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path} (n={len(probes)})")
    return 0


def mode_rag_causal(attn_impl: str) -> int:
    from phase4zf_rag_stress_eval import load_rag_probe_pool
    target_ids = {"P02", "LC-08", "Q11", "AD-04"}
    probes = [p for p in load_rag_probe_pool() if p["id"] in target_ids]

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)

    results = {}
    for p in probes:
        messages = [{"role": "system", "content": system_prompt}]
        if p.get("context"):
            messages.append({"role": "system", "content": p["context"]})
        messages.append({"role": "user", "content": p["question"]})
        greedy = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
        results[p["id"]] = {"set": p["set"], "greedy": greedy}

    out = {"attn_impl": attn_impl, "n_probes": len(probes), "results": results}
    out_path = REPORTS_DIR / f"phase4zj_rag_causal_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


# ============================================================
# Stage2: instruction_override core evaluation
# ============================================================

def mode_core9(attn_impl: str) -> int:
    """baseline 9probe(ZHH-E01-05 + ZI-IO01-04)をZJで再評価する。ZI-IO*はturn1+turn2の
    2ターンで評価しscenario全体のverdictを見る(phase4zj_baseline_reproduction.jsonと同一方式)。"""
    from phase4zh_holdout_probes import ALL_PROBES as ZHH
    from phase4zi_multiturn_diagnostic_scenarios import ALL_SCENARIOS as ZIM

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)

    results = {}
    zhh_e = [p for p in ZHH if p["id"].startswith("ZHH-E")]
    for p in zhh_e:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": p["prompt"]}]
        greedy = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
        results[p["id"]] = {"type": "single_turn", "prompt": p["prompt"], "greedy": greedy}

    zi_io = [s for s in ZIM if s["id"].startswith("ZI-IO")]
    for sc in zi_io:
        messages = [{"role": "system", "content": system_prompt}]
        turn_log = []
        for i, user_turn in enumerate(sc["turns"]):
            messages.append({"role": "user", "content": user_turn})
            reply = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
            messages.append({"role": "assistant", "content": reply})
            turn_log.append({"turn": i + 1, "user": user_turn, "assistant": reply})
        results[sc["id"]] = {"type": "multi_turn", "turns": turn_log}

    out = {"attn_impl": attn_impl, "n_probes": len(results), "results": results}
    out_path = REPORTS_DIR / f"phase4zj_core9_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


def mode_new_holdout(attn_impl: str) -> int:
    from phase4zj_new_holdout_probes import ALL_PROBES as ZJH

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)

    results = {}
    for p in ZJH:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": p["prompt"]}]
        greedy = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
        results[p["id"]] = {"category": p["category"], "prompt": p["prompt"], "greedy": greedy}

    out = {"attn_impl": attn_impl, "n_probes": len(ZJH), "results": results}
    out_path = REPORTS_DIR / f"phase4zj_new_holdout_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


# ============================================================
# Stage3: 限定multi-turn確認(original6 + 代表12 = 18)
# ============================================================

REPRESENTATIVE_12_IDS = [
    "ZI-SWA01", "ZI-SWA03", "ZI-FM01", "ZI-FM02", "ZI-AC02", "ZI-AC03",
    "ZI-PC01", "ZI-PC02", "ZI-IO01", "ZI-IO04", "ZI-RNC01", "ZI-RNC03",
]


def mode_multiturn18(attn_impl: str) -> int:
    from phase4zh_holdout_probes import MULTITURN_SCENARIOS as ORIG6
    from phase4zi_multiturn_diagnostic_scenarios import ALL_SCENARIOS as ZIM32

    rep12 = [s for s in ZIM32 if s["id"] in REPRESENTATIVE_12_IDS]
    scenarios = list(ORIG6) + rep12

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)

    results = {}
    for sc in scenarios:
        messages = [{"role": "system", "content": system_prompt}]
        turn_log = []
        for i, user_turn in enumerate(sc["turns"]):
            messages.append({"role": "user", "content": user_turn})
            reply = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
            messages.append({"role": "assistant", "content": reply})
            turn_log.append({"turn": i + 1, "user": user_turn, "assistant": reply})
        results[sc["id"]] = {"category": sc["category"], "turns": turn_log}
        print(f"{sc['id']} done")

    out = {"attn_impl": attn_impl, "n_scenarios": len(scenarios), "results": results}
    out_path = REPORTS_DIR / f"phase4zj_multiturn18_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


def mode_ood(attn_impl: str) -> int:
    from phase4zi_ood_sanity_probes import ALL_PROBES as OOD

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)

    results = {}
    for p in OOD:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": p["prompt"]}]
        greedy = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
        results[p["id"]] = {"category": p["category"], "prompt": p["prompt"], "greedy": greedy}

    out = {"attn_impl": attn_impl, "n_probes": len(OOD), "results": results}
    out_path = REPORTS_DIR / f"phase4zj_ood_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                         choices=["stage1_guard", "rag_causal", "core9", "new_holdout", "multiturn18", "ood"])
    parser.add_argument("--attn-impl", required=True, choices=["eager", "sdpa"])
    args = parser.parse_args()
    fn = {"stage1_guard": mode_stage1_guard, "rag_causal": mode_rag_causal, "core9": mode_core9,
          "new_holdout": mode_new_holdout, "multiturn18": mode_multiturn18, "ood": mode_ood}[args.mode]
    sys.exit(fn(args.attn_impl))
