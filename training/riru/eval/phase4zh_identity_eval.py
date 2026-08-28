"""Phase 4ZH Section24-26,30-33: Phase4ZH candidate(LoRA on top of base, true BF16)の評価。

Stage1 (quick gate): ZH新規held-out43 + ZG-stress40(wrong_name_induction/
role_name_confusion/identity_correction_stress) = 83probes、greedyのみ。
主要regressionの早期検知が目的。

Stage2 (large-scale gate): 既存Phase4ZG probe pool171 + Phase4ZH新規held-out43
= 214probes × greedy+seed101-103(4/probe) = 856 generations/backend。

multiturn: Phase4ZH新規のtrue multi-turn scenario(6件、2-3ターン)を、
実際に会話を1ターンずつ進めてモデル自身の生成応答をcontextに含めて評価する。
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
ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zh-structural-hardened")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

FORCED_PREFIX = "こんにちは〜！私はパチスロの専門アシスタントの"
RI_ID, RU_ID = 36723, 32610
SEEDS = (101, 102, 103)


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
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, adapter_name="phase4zh")
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


def build_full_text(tokenizer) -> str:
    from phase4z_probes import PROBE_SET_C
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    e36_original = PROBE_SET_C[0]["prompt"]
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": e36_original}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt_text + FORCED_PREFIX


def mode_margin(attn_impl: str) -> int:
    model, tokenizer = load_model(attn_impl)
    full_text = build_full_text(tokenizer)
    encoded = tokenizer(full_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**encoded, use_cache=False)
    last_logits = out.logits[0, -1, :].float().cpu()
    probs = torch.softmax(last_logits, dim=-1)
    topk = torch.topk(probs, 20)
    top_list = []
    for prob, idx in zip(topk.values.tolist(), topk.indices.tolist(), strict=True):
        top_list.append({"rank": len(top_list) + 1, "token_id": idx, "token_str": tokenizer.decode([idx]),
                          "prob": round(prob, 6), "logit": round(float(last_logits[idx]), 6)})
    ri_logit, ru_logit = float(last_logits[RI_ID]), float(last_logits[RU_ID])
    ri_rank = int((probs > probs[RI_ID]).sum().item()) + 1
    ru_rank = int((probs > probs[RU_ID]).sum().item()) + 1
    result = {
        "attn_impl": attn_impl, "ri_logit": round(ri_logit, 6), "ru_logit": round(ru_logit, 6),
        "ri_prob": round(float(probs[RI_ID]), 6), "ru_prob": round(float(probs[RU_ID]), 6),
        "ri_rank": ri_rank, "ru_rank": ru_rank,
        "margin_ri_minus_ru_logit": round(ri_logit - ru_logit, 6),
        "winner": "リ" if ri_logit > ru_logit else ("ル" if ru_logit > ri_logit else "TIE"),
        "top20": top_list,
    }
    out_path = REPORTS_DIR / f"phase4zh_margin_{attn_impl}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(result["winner"], result["margin_ri_minus_ru_logit"])
    return 0


def load_probe_pool_stage2() -> list[dict]:
    """Phase4ZG既存171probe + Phase4ZH新規held-out43probe = 214probes。"""
    from phase4zg_identity_eval import load_probe_pool as load_zg_pool
    from phase4zh_holdout_probes import ALL_PROBES as ZHH

    probes = load_zg_pool()
    for p in ZHH:
        probes.append({"set": "zh_holdout", "id": p["id"], "prompt": p["prompt"]})
    return probes


def load_probe_pool_stage1() -> list[dict]:
    """ZH新規held-out43 + ZG-stress40 = 83probes(greedyのみ、早期sanity check用)。"""
    from phase4zf_stress_probes import WRONG_NAME_INDUCTION, ROLE_NAME_CONFUSION, IDENTITY_CORRECTION_STRESS
    from phase4zh_holdout_probes import ALL_PROBES as ZHH

    probes = []
    for p in ZHH:
        probes.append({"set": "zh_holdout", "id": p["id"], "prompt": p["prompt"]})
    for p in WRONG_NAME_INDUCTION:
        probes.append({"set": "zf_wrong_name_induction", "id": p["id"], "prompt": p["prompt"]})
    for p in ROLE_NAME_CONFUSION:
        probes.append({"set": "zf_role_name_confusion", "id": p["id"], "prompt": p["prompt"]})
    for p in IDENTITY_CORRECTION_STRESS:
        probes.append({"set": "zf_identity_correction_stress", "id": p["id"], "prompt": p["prompt"]})
    return probes


def mode_identity(attn_impl: str, stage: str) -> int:
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)
    probes = load_probe_pool_stage1() if stage == "stage1" else load_probe_pool_stage2()
    seeds = () if stage == "stage1" else SEEDS

    results = {}
    t0 = time.time()
    for p in probes:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": p["prompt"]}]
        greedy = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
        sampled = {str(s): generate_reply(model, tokenizer, messages, seed=s, do_sample=True) for s in seeds}
        results[p["id"]] = {"set": p["set"], "greedy": greedy, "sampled": sampled}
        if len(results) % 10 == 0:
            elapsed = time.time() - t0
            print(f"{len(results)}/{len(probes)} done ({elapsed:.1f}s, "
                  f"eta {(elapsed/len(results))*(len(probes)-len(results)):.0f}s)")

    out = {"attn_impl": attn_impl, "stage": stage, "n_probes": len(probes), "seeds": list(seeds),
           "n_generations": len(probes) * (1 + len(seeds)), "results": results}
    out_path = REPORTS_DIR / f"phase4zh_identity_{stage}_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(f"total_generations={out['n_generations']}")
    return 0


def mode_multiturn(attn_impl: str) -> int:
    """Phase4ZH Section19: 真のmulti-turnシナリオ(6件)をターンごとに実際に会話を進めて評価する。"""
    from phase4zh_holdout_probes import MULTITURN_SCENARIOS

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)

    results = {}
    for sc in MULTITURN_SCENARIOS:
        messages = [{"role": "system", "content": system_prompt}]
        turn_log = []
        for i, user_turn in enumerate(sc["turns"]):
            messages.append({"role": "user", "content": user_turn})
            reply = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
            messages.append({"role": "assistant", "content": reply})
            turn_log.append({"turn": i + 1, "user": user_turn, "assistant": reply})
        results[sc["id"]] = {"category": sc["category"], "turns": turn_log}
        print(f"{sc['id']} ({sc['category']}) done, {len(sc['turns'])} turns")

    out = {"attn_impl": attn_impl, "n_scenarios": len(MULTITURN_SCENARIOS), "results": results}
    out_path = REPORTS_DIR / f"phase4zh_multiturn_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


def mode_regression(attn_impl: str) -> int:
    from phase4zf_rag_stress_eval import load_rag_probe_pool
    probes = load_rag_probe_pool()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)

    results = {}
    t0 = time.time()
    for p in probes:
        messages = [{"role": "system", "content": system_prompt}]
        if p.get("context"):
            messages.append({"role": "system", "content": p["context"]})
        messages.append({"role": "user", "content": p["question"]})
        greedy = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
        results[p["id"]] = {"set": p["set"], "greedy": greedy}
        if len(results) % 20 == 0:
            print(f"{len(results)}/{len(probes)} done ({time.time()-t0:.1f}s)")

    out = {"attn_impl": attn_impl, "n_probes": len(probes), "results": results}
    out_path = REPORTS_DIR / f"phase4zh_rag_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["margin", "identity", "multiturn", "regression"])
    parser.add_argument("--attn-impl", required=True, choices=["eager", "sdpa"])
    parser.add_argument("--stage", default="stage2", choices=["stage1", "stage2"])
    args = parser.parse_args()
    if args.mode == "margin":
        sys.exit(mode_margin(args.attn_impl))
    elif args.mode == "identity":
        sys.exit(mode_identity(args.attn_impl, args.stage))
    elif args.mode == "multiturn":
        sys.exit(mode_multiturn(args.attn_impl))
    else:
        sys.exit(mode_regression(args.attn_impl))
