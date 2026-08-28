"""Phase 4ZI: Phase4ZG candidateを基準モデルとした read-only diagnostic評価。

学習は一切行わない。Phase4ZHで作成済みのheld-out/true multi-turnをPhase4ZG
adapter上でそのまま再評価し、Phase4ZHとのpaired比較を可能にする。

modes:
  heldout43       - Phase4ZH新規held-out43probeをZGで評価(greedy, Stage1と同条件)
  multiturn_orig  - Phase4ZH真のmulti-turn6シナリオをZGで再実行
  multiturn_new   - Phase4ZI新規diagnostic multi-turnシナリオをZGで実行
  turn1_isolation - 全multi-turnシナリオのturn1を独立single-turn probeとしてZGで評価
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
ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")
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
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, adapter_name="phase4zg_diag")
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


def mode_heldout43(attn_impl: str) -> int:
    from phase4zh_holdout_probes import ALL_PROBES as ZHH

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)

    results = {}
    t0 = time.time()
    for p in ZHH:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": p["prompt"]}]
        greedy = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
        results[p["id"]] = {"category": p["category"], "prompt": p["prompt"], "greedy": greedy}
        if len(results) % 10 == 0:
            elapsed = time.time() - t0
            print(f"{len(results)}/{len(ZHH)} done ({elapsed:.1f}s)")

    out = {"model": "phase4zg", "attn_impl": attn_impl, "n_probes": len(ZHH), "results": results}
    out_path = REPORTS_DIR / f"phase4zi_zg_heldout43_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


def _run_multiturn(scenarios: list[dict], attn_impl: str, out_name: str) -> int:
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
        print(f"{sc['id']} ({sc['category']}) done, {len(sc['turns'])} turns")

    out = {"model": "phase4zg", "attn_impl": attn_impl, "n_scenarios": len(scenarios), "results": results}
    out_path = REPORTS_DIR / f"{out_name}_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


def mode_multiturn_orig(attn_impl: str) -> int:
    from phase4zh_holdout_probes import MULTITURN_SCENARIOS
    return _run_multiturn(MULTITURN_SCENARIOS, attn_impl, "phase4zi_zg_multiturn_orig")


def mode_multiturn_new(attn_impl: str) -> int:
    from phase4zi_multiturn_diagnostic_scenarios import ALL_SCENARIOS
    return _run_multiturn(ALL_SCENARIOS, attn_impl, "phase4zi_zg_multiturn_new")


def mode_turn1_isolation(attn_impl: str) -> int:
    from phase4zh_holdout_probes import MULTITURN_SCENARIOS as ORIG
    from phase4zi_multiturn_diagnostic_scenarios import ALL_SCENARIOS as NEW

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)

    all_scenarios = [{"id": s["id"], "category": s["category"], "turn1": s["turns"][0]} for s in ORIG] + \
                    [{"id": s["id"], "category": s["category"], "turn1": s["turns"][0]} for s in NEW]

    results = {}
    for sc in all_scenarios:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": sc["turn1"]}]
        greedy = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
        results[sc["id"]] = {"category": sc["category"], "turn1_prompt": sc["turn1"], "greedy": greedy}

    out = {"model": "phase4zg", "attn_impl": attn_impl, "n_probes": len(all_scenarios), "results": results}
    out_path = REPORTS_DIR / f"phase4zi_turn1_isolation_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


def mode_ood_sanity(attn_impl: str) -> int:
    from phase4zi_ood_sanity_probes import ALL_PROBES as OOD

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)

    results = {}
    for p in OOD:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": p["prompt"]}]
        greedy = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
        results[p["id"]] = {"category": p["category"], "prompt": p["prompt"], "greedy": greedy}

    out = {"model": "phase4zg", "attn_impl": attn_impl, "n_probes": len(OOD), "results": results}
    out_path = REPORTS_DIR / f"phase4zi_ood_sanity_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
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

    out = {"model": "phase4zg", "attn_impl": attn_impl, "n_probes": len(probes), "results": results}
    out_path = REPORTS_DIR / f"phase4zi_rag_causal_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                         choices=["heldout43", "multiturn_orig", "multiturn_new", "turn1_isolation",
                                  "ood_sanity", "rag_causal"])
    parser.add_argument("--attn-impl", required=True, choices=["eager", "sdpa"])
    args = parser.parse_args()
    fn = {"heldout43": mode_heldout43, "multiturn_orig": mode_multiturn_orig,
          "multiturn_new": mode_multiturn_new, "turn1_isolation": mode_turn1_isolation,
          "ood_sanity": mode_ood_sanity, "rag_causal": mode_rag_causal}[args.mode]
    sys.exit(fn(args.attn_impl))
