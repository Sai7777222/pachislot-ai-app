"""Phase 4Z: B_merged_hf 大規模identity診断評価。

phase4z_identity_eval_gguf.pyと同一のprobe set・seed・generation設定で、
merged HFモデル(4bit NF4量子化、Phase4Yの評価条件と統一)を評価する。
学習・再merge・merged HF変更は一切行わない。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parents[2]
TRAINING_ROOT = EVAL_DIR.parents[0]

sys.path.insert(0, str(EVAL_DIR))
from phase4t_probes import P04_PROBES  # noqa: E402
from phase4z_probes import PROBE_SET_A, PROBE_SET_B, PROBE_SET_C, PROBE_SET_D  # noqa: E402

MERGED_MODEL_PATH = str(TRAINING_ROOT / "merged" / "riru-qwen-final-hf")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9

SEEDS_20 = tuple(range(101, 121))
SEEDS_30 = tuple(range(101, 131))
SEEDS_10 = tuple(range(101, 111))
SEEDS_3 = (101, 102, 103)


def build_model_and_tokenizer():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH, quantization_config=quant_config, device_map="auto",
        trust_remote_code=True,
    )
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
    print(f"Loading merged HF (4bit NF4): {MERGED_MODEL_PATH}")
    t0 = time.perf_counter()
    model, tokenizer = build_model_and_tokenizer()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    results: dict = {"_meta": {"label": "B_merged_hf", "model_path": MERGED_MODEL_PATH,
                                "temperature": TEMPERATURE, "top_p": TOP_P,
                                "max_tokens": MAX_NEW_TOKENS}}
    t1 = time.perf_counter()

    print("=== Probe Set A: Phase4W naming stress (20 x greedy+20seed) ===")
    results["set_a_naming_stress"] = {}
    for p in PROBE_SET_A:
        results["set_a_naming_stress"][p["id"]] = sweep(
            model, tokenizer, system_prompt, None, p["prompt"], SEEDS_20
        )
    print(f"  done ({time.perf_counter() - t1:.1f}s)")

    print("=== Probe Set B: Phase4X held-out naming (24 x greedy+20seed) ===")
    results["set_b_heldout_naming"] = {}
    for p in PROBE_SET_B:
        results["set_b_heldout_naming"][p["id"]] = sweep(
            model, tokenizer, system_prompt, None, p["prompt"], SEEDS_20
        )
    print(f"  done ({time.perf_counter() - t1:.1f}s)")

    print("=== Probe Set C: E36 original+paraphrase (17 x greedy+30seed) ===")
    results["set_c_e36"] = {}
    for p in PROBE_SET_C:
        results["set_c_e36"][p["id"]] = sweep(
            model, tokenizer, system_prompt, None, p["prompt"], SEEDS_30
        )
    print(f"  done ({time.perf_counter() - t1:.1f}s)")

    print("=== Probe Set D: E02 original+paraphrase (16 x greedy+20seed) ===")
    results["set_d_e02"] = {}
    for p in PROBE_SET_D:
        results["set_d_e02"][p["id"]] = sweep(
            model, tokenizer, system_prompt, None, p["prompt"], SEEDS_20
        )
    print(f"  done ({time.perf_counter() - t1:.1f}s)")

    print("=== Scope: PT-01..22 (22 x greedy+10seed) ===")
    results["scope"] = {}
    for p in P04_PROBES:
        results["scope"][p["id"]] = sweep(
            model, tokenizer, system_prompt, p["context"], p["question"], SEEDS_10
        )
    print(f"  done ({time.perf_counter() - t1:.1f}s)")

    print("=== RAG safety sanity (6 probes x greedy+3seed) ===")
    rag17_path = EVAL_DIR / "structured_rag_17q_context.json"
    rag17 = json.loads(rag17_path.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rag17}
    from phase4w_probes import ADVERSARIAL_PROBES, CONFLICTING_PROBES, LONGCONTEXT_PROBES

    ad01 = next(p for p in ADVERSARIAL_PROBES if p["id"] == "AD-01")
    cf01 = next(p for p in CONFLICTING_PROBES if p["id"] == "CF-01")
    lc01 = next(p for p in LONGCONTEXT_PROBES if p["id"] == "LC-01")
    results["rag_safety"] = {
        "Q3": sweep(model, tokenizer, system_prompt, by_id["Q3"]["rag_context_text"],
                    by_id["Q3"]["question"], SEEDS_3),
        "Q9": sweep(model, tokenizer, system_prompt, by_id["Q9"]["rag_context_text"],
                    by_id["Q9"]["question"], SEEDS_3),
        "Q11": sweep(model, tokenizer, system_prompt, by_id["Q11"]["rag_context_text"],
                     by_id["Q11"]["question"], SEEDS_3),
        "AD-01": sweep(model, tokenizer, system_prompt, ad01["context"], ad01["question"],
                       SEEDS_3),
        "CF-01": sweep(model, tokenizer, system_prompt, cf01["context"], cf01["question"],
                       SEEDS_3),
        "LC-01": sweep(model, tokenizer, system_prompt, lc01["context"], lc01["question"],
                       SEEDS_3),
    }
    print(f"  done ({time.perf_counter() - t1:.1f}s)")

    out_path = EVAL_DIR / "phase4z_identity_results_hf.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path} (total {time.perf_counter() - t1:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
