"""Phase 4X: ratio-high-identity-stable(C)の本評価。

主対象はC(stable)。B(ratio-high-identity)はPhase4Wの既存結果を極力再利用し、
paired比較(同一probe・同一seed)が可能な範囲でのみ新規生成する。学習は行わない。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parents[1]
EVAL_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(EVAL_DIR))
from phase4t_probes import P04_PROBES  # noqa: E402
from phase4v_probes import PROBES as BROAD_PROBES  # noqa: E402
from phase4w_probes import (  # noqa: E402
    ADVERSARIAL_PROBES,
    CONFLICTING_PROBES,
    LONGCONTEXT_PROBES,
    NAMING_STRESS_PROBES,
    Q9_PROBES,
    Q11_PROBES,
)
from phase4x_probes import NAMING_PROBES as PX_NAMING  # noqa: E402

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_STABLE_PATH = str(TRAINING_ROOT / "lora-riru-qwen-ratio-high-identity-stable")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9

NEW_SEEDS_10 = tuple(range(101, 111))
SEEDS_10_NAMING = tuple(range(42, 52))
SEEDS_5 = (42, 43, 44, 45, 46)
SEEDS_4 = (42, 43, 44)
SEEDS_3_BROAD = (101, 102, 103)
PERSONA_SEEDS = (42, 43)


def build_model_and_tokenizer():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, quantization_config=quant_config, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_STABLE_PATH, adapter_name="stable")
    model.set_adapter("stable")
    model.eval()
    return model, tokenizer


def generate_reply(model, tokenizer, messages, seed, do_sample=True):
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
    else:
        gen_kwargs.update(do_sample=False)
    with torch.no_grad():
        output_ids = model.generate(**encoded, **gen_kwargs)
    completion_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def run_single(model, tokenizer, system_prompt, rag_context, question, seed, do_sample=True):
    messages = [{"role": "system", "content": system_prompt}]
    if rag_context:
        messages.append({"role": "system", "content": rag_context})
    messages.append({"role": "user", "content": question})
    return generate_reply(model, tokenizer, messages, seed, do_sample)


def sweep(model, tokenizer, system_prompt, context, question, seeds, greedy=True):
    out = {}
    if greedy:
        out["greedy"] = run_single(
            model, tokenizer, system_prompt, context, question, 42, do_sample=False
        )
    out["sampled"] = {
        str(s): run_single(model, tokenizer, system_prompt, context, question, s) for s in seeds
    }
    return out


def main() -> int:
    print("Loading base model + ratio-high-identity-stable adapter...")
    model, tokenizer = build_model_and_tokenizer()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    rag_17q = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rag_17q}
    q3, q9, q11 = by_id["Q3"], by_id["Q9"], by_id["Q11"]

    holdout = json.loads(
        (EVAL_DIR / "phase4i_holdout_omission_v2.json").read_text(encoding="utf-8")
    )
    p01 = next(r for r in holdout if r["id"] == "P01")
    p02 = next(r for r in holdout if r["id"] == "P02")

    eval_39 = [
        json.loads(line)
        for line in (EVAL_DIR / "riru_eval_set_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_39_by_id = {x["id"]: x for x in eval_39}
    e36_item = eval_39_by_id["E36"]
    e02_item = eval_39_by_id["E02"]
    persona_ids = ["E01", "E04", "E07", "E14", "E17", "E30", "E33", "E37"]

    results: dict = {}
    t0 = time.perf_counter()

    print("=== Q3/P01/P02 new-seed (101-110) ===")
    results["q3_newseed"] = sweep(
        model, tokenizer, system_prompt, q3["rag_context_text"], q3["question"], NEW_SEEDS_10
    )
    results["p01_newseed"] = sweep(
        model, tokenizer, system_prompt, p01["rag_context_text"], p01["question"], NEW_SEEDS_10,
        greedy=False,
    )
    results["p02_newseed"] = sweep(
        model, tokenizer, system_prompt, p02["rag_context_text"], p02["question"], NEW_SEEDS_10,
        greedy=False,
    )
    print(f"  done ({time.perf_counter() - t0:.1f}s)")

    print("=== Q9/Q11(real) new-seed (101-110) ===")
    results["q9_newseed"] = sweep(
        model, tokenizer, system_prompt, q9["rag_context_text"], q9["question"], NEW_SEEDS_10,
        greedy=False,
    )
    results["q11_newseed"] = sweep(
        model, tokenizer, system_prompt, q11["rag_context_text"], q11["question"], NEW_SEEDS_10,
        greedy=False,
    )
    print(f"  done ({time.perf_counter() - t0:.1f}s)")

    print("=== E36/E02 new-seed (101-110) ===")
    results["e36_newseed"] = {
        str(s): run_single(model, tokenizer, system_prompt, None, e36_item["prompt"], s)
        for s in NEW_SEEDS_10
    }
    results["e02_newseed"] = {
        str(s): run_single(model, tokenizer, system_prompt, None, e02_item["prompt"], s)
        for s in NEW_SEEDS_10
    }
    print(f"  done ({time.perf_counter() - t0:.1f}s)")

    print("=== Phase4X new naming probes (24, greedy+10seed) ===")
    results["px_naming"] = {}
    for probe in PX_NAMING:
        results["px_naming"][probe["id"]] = sweep(
            model, tokenizer, system_prompt, None, probe["prompt"], SEEDS_10_NAMING
        )
    print(f"  done ({time.perf_counter() - t0:.1f}s)")

    print("=== Phase4W naming stress (20, greedy+10seed, paired w/ Phase4W) ===")
    results["naming_stress"] = {}
    for probe in NAMING_STRESS_PROBES:
        results["naming_stress"][probe["id"]] = sweep(
            model, tokenizer, system_prompt, None, probe["prompt"], SEEDS_10_NAMING
        )
    print(f"  done ({time.perf_counter() - t0:.1f}s)")

    print("=== Scope correctness (Phase4T PT-01..22, greedy+3seed) ===")
    results["p04_type"] = {}
    for probe in P04_PROBES:
        results["p04_type"][probe["id"]] = sweep(
            model, tokenizer, system_prompt, probe["context"], probe["question"], SEEDS_4
        )
    print(f"  done ({time.perf_counter() - t0:.1f}s)")

    print("=== QW9/QW11 (Phase4W new probes, greedy+3seed, paired) ===")
    results["qw9_probes"] = {}
    for probe in Q9_PROBES:
        results["qw9_probes"][probe["id"]] = sweep(
            model, tokenizer, system_prompt, probe["context"], probe["question"], SEEDS_4
        )
    results["qw11_probes"] = {}
    for probe in Q11_PROBES:
        results["qw11_probes"][probe["id"]] = sweep(
            model, tokenizer, system_prompt, probe["context"], probe["question"], SEEDS_4
        )
    print(f"  done ({time.perf_counter() - t0:.1f}s)")

    print("=== Adversarial/Conflicting/Long-context (greedy+3seed, paired) ===")
    results["adversarial"] = {}
    for probe in ADVERSARIAL_PROBES:
        results["adversarial"][probe["id"]] = sweep(
            model, tokenizer, system_prompt, probe["context"], probe["question"], SEEDS_4
        )
    results["conflicting"] = {}
    for probe in CONFLICTING_PROBES:
        results["conflicting"][probe["id"]] = sweep(
            model, tokenizer, system_prompt, probe["context"], probe["question"], SEEDS_4
        )
    results["longcontext"] = {}
    for probe in LONGCONTEXT_PROBES:
        results["longcontext"][probe["id"]] = sweep(
            model, tokenizer, system_prompt, probe["context"], probe["question"], SEEDS_4
        )
    print(f"  done ({time.perf_counter() - t0:.1f}s)")

    print("=== Broad completeness recheck (Phase4V 36 probes, seeds 101-103, paired) ===")
    results["broad_recheck"] = {}
    for probe in BROAD_PROBES:
        results["broad_recheck"][probe["id"]] = sweep(
            model, tokenizer, system_prompt, probe["context"], probe["question"],
            SEEDS_3_BROAD, greedy=False,
        )
    print(f"  done ({time.perf_counter() - t0:.1f}s)")

    print("=== Persona sample (8 character39 items, greedy+2seed) ===")
    results["persona_sample"] = {}
    for pid in persona_ids:
        item = eval_39_by_id[pid]
        results["persona_sample"][pid] = sweep(
            model, tokenizer, system_prompt, None, item["prompt"], PERSONA_SEEDS
        )
    print(f"  done ({time.perf_counter() - t0:.1f}s)")

    out_path = EVAL_DIR / "phase4x_comprehensive_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path} (total {time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
