"""Phase 4ZE Section12-17: Phase4ZE candidate(LoRA on top of base, true BF16, no
bitsandbytes量子化)のidentity/regression評価。

--mode margin      : E36 forced-prefixでの「リ」「ル」top20 logits/margin (Section14-15)
--mode identity     : Phase4ZE holdout(27) + Phase4W naming_stress(20) + Phase4X held-out
                      naming(24) + E36 family(17) + E02 family(16) = 104probes (Section12-13)
--mode regression   : Q3/P01/Q9/Q11/PT-01/AD-01/CF-01/LC-01の8問 (Section16)
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
ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4ze-identity-margin")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

FORCED_PREFIX = "こんにちは〜！私はパチスロの専門アシスタントの"
RI_ID, RU_ID = 36723, 32610
SEEDS_3 = (101, 102, 103)
TEMPERATURE = 0.3
TOP_P = 0.9
MAX_NEW_TOKENS = 300


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
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, adapter_name="phase4ze")
    model.eval()
    return model, tokenizer


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
        top_list.append({"rank": len(top_list) + 1, "token_id": idx,
                          "token_str": tokenizer.decode([idx]), "prob": round(prob, 6),
                          "logit": round(float(last_logits[idx]), 6)})
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
    out_path = REPORTS_DIR / f"phase4ze_margin_{attn_impl}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(result["winner"], result["margin_ri_minus_ru_logit"])
    return 0


def generate_reply(model, tokenizer, messages, seed, do_sample) -> str:
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS,
                       pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
    else:
        gen_kwargs.update(do_sample=False)
    with torch.no_grad():
        output_ids = model.generate(**encoded, **gen_kwargs)
    completion_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def load_identity_probes() -> list[dict]:
    from phase4ze_holdout_probes import ALL_PROBES as ZE_HOLDOUT
    from phase4z_probes import PROBE_SET_A, PROBE_SET_B, PROBE_SET_C, PROBE_SET_D

    probes = []
    for p in ZE_HOLDOUT:
        probes.append({"set": "phase4ze_holdout", "id": p["id"], "prompt": p["prompt"]})
    for p in PROBE_SET_A:
        probes.append({"set": "phase4w_naming_stress", "id": p["id"], "prompt": p["prompt"]})
    for p in PROBE_SET_B:
        probes.append({"set": "phase4x_heldout_naming", "id": p["id"], "prompt": p["prompt"]})
    for p in PROBE_SET_C:
        probes.append({"set": "e36_family", "id": p["id"], "prompt": p["prompt"]})
    for p in PROBE_SET_D:
        probes.append({"set": "e02_family", "id": p["id"], "prompt": p["prompt"]})
    return probes


def mode_identity(attn_impl: str, seeded: bool) -> int:
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)
    probes = load_identity_probes()

    results = {}
    t0 = time.time()
    for p in probes:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": p["prompt"]}]
        greedy = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
        entry = {"set": p["set"], "greedy": greedy}
        if seeded:
            entry["sampled"] = {
                str(s): generate_reply(model, tokenizer, messages, seed=s, do_sample=True) for s in SEEDS_3
            }
        results[p["id"]] = entry
        if len(results) % 10 == 0:
            print(f"{len(results)}/{len(probes)} done ({time.time()-t0:.1f}s)")

    out = {"attn_impl": attn_impl, "seeded": seeded, "n_probes": len(probes), "results": results}
    out_path = REPORTS_DIR / f"phase4ze_identity_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


def load_regression_probes() -> dict:
    rag17 = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rag17}
    q3, q9, q11 = by_id["Q3"], by_id["Q9"], by_id["Q11"]
    holdout = json.loads((EVAL_DIR / "phase4i_holdout_omission_v2.json").read_text(encoding="utf-8"))
    p01 = next(r for r in holdout if r["id"] == "P01")
    from phase4t_probes import P04_PROBES
    pt01 = next(p for p in P04_PROBES if p["id"] == "PT-01")
    from phase4w_probes import ADVERSARIAL_PROBES, CONFLICTING_PROBES, LONGCONTEXT_PROBES
    ad01 = next(p for p in ADVERSARIAL_PROBES if p["id"] == "AD-01")
    cf01 = next(p for p in CONFLICTING_PROBES if p["id"] == "CF-01")
    lc01 = next(p for p in LONGCONTEXT_PROBES if p["id"] == "LC-01")
    return {
        "Q3": {"context": q3["rag_context_text"], "question": q3["question"]},
        "P01": {"context": p01["rag_context_text"], "question": p01["question"]},
        "Q9": {"context": q9["rag_context_text"], "question": q9["question"]},
        "Q11": {"context": q11["rag_context_text"], "question": q11["question"]},
        "PT-01": {"context": pt01["context"], "question": pt01["question"]},
        "AD-01": {"context": ad01["context"], "question": ad01["question"]},
        "CF-01": {"context": cf01["context"], "question": cf01["question"]},
        "LC-01": {"context": lc01["context"], "question": lc01["question"]},
    }


def mode_regression(attn_impl: str, seeded: bool) -> int:
    probes = load_regression_probes()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)

    results = {}
    for pid, p in probes.items():
        messages = [{"role": "system", "content": system_prompt}]
        if p["context"]:
            messages.append({"role": "system", "content": p["context"]})
        messages.append({"role": "user", "content": p["question"]})
        greedy = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
        entry = {"greedy": greedy}
        if seeded:
            entry["sampled"] = {
                str(s): generate_reply(model, tokenizer, messages, seed=s, do_sample=True) for s in SEEDS_3
            }
        results[pid] = entry
        print(f"{pid} done")

    out = {"attn_impl": attn_impl, "seeded": seeded, "results": results}
    out_path = REPORTS_DIR / f"phase4ze_regression_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["margin", "identity", "regression"])
    parser.add_argument("--attn-impl", required=True, choices=["eager", "sdpa"])
    parser.add_argument("--seeded", action="store_true")
    args = parser.parse_args()

    if args.mode == "margin":
        sys.exit(mode_margin(args.attn_impl))
    elif args.mode == "identity":
        sys.exit(mode_identity(args.attn_impl, args.seeded))
    else:
        sys.exit(mode_regression(args.attn_impl, args.seeded))
