"""Phase 4ZD Section10-12: HF条件(A/B/C)でのidentity生成評価。

--mode repro5      : E36 original greedyを5回(独立reload)実行し決定論性を確認 (Section10)
--mode paraphrase8 : Phase4ZAの8問(無改変)をgreedyで評価 (Section11)
--mode naming220   : Phase4W naming stress 20問 x (greedy + seed101-110) = 220生成 (Section12 Stage1)
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
REPORTS_DIR = TRAINING_ROOT / "reports"

from phase4zd_true_bf16_eval import CONDITIONS, MODEL_PATH  # noqa: E402
from phase4z_probes import PROBE_SET_A, PROBE_SET_B  # noqa: E402 (Phase4W/4X既存probe、無改変)

SEEDS_3 = (101, 102, 103)

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
SEEDS_10 = tuple(range(101, 111))

PARAPHRASE8_IDS = ["E36_ORIGINAL", "PZ36-12", "PZ36-06", "PZ36-14", "PZ36-15",
                   "PZ36-01", "PZ36-02", "PZ36-03"]


def build_model_and_tokenizer(condition: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = CONDITIONS[condition]
    kwargs = {"trust_remote_code": True, "device_map": "cuda:0"}
    if cfg["quant"] == "nf4_4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg["dtype"]]
        if cfg["attn_impl"]:
            kwargs["attn_implementation"] = cfg["attn_impl"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
    model.eval()
    return model, tokenizer


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


def run_single(model, tokenizer, system_prompt, question, seed, do_sample):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    return generate_reply(model, tokenizer, messages, seed, do_sample)


def mode_repro5(condition: str) -> int:
    from phase4z_probes import PROBE_SET_C
    e36_original = PROBE_SET_C[0]["prompt"]
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    results = []
    for i in range(5):
        t0 = time.time()
        model, tokenizer = build_model_and_tokenizer(condition)
        text = run_single(model, tokenizer, system_prompt, e36_original, seed=42, do_sample=False)
        results.append({"run": i + 1, "text": text, "elapsed_sec": round(time.time() - t0, 2)})
        del model
        torch.cuda.empty_cache()
        print(f"run {i+1}/5 done")

    all_identical = len({r["text"] for r in results}) == 1
    out = {"condition": condition, "mode": "repro5", "runs": results, "all_identical": all_identical}
    out_path = REPORTS_DIR / f"phase4zd_repro5_{condition}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}, all_identical={all_identical}")
    return 0


def mode_paraphrase8(condition: str) -> int:
    from phase4z_probes import PROBE_SET_C
    probe_by_id = {p["id"]: p["prompt"] for p in PROBE_SET_C}
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    model, tokenizer = build_model_and_tokenizer(condition)
    results = {}
    for pid in PARAPHRASE8_IDS:
        text = run_single(model, tokenizer, system_prompt, probe_by_id[pid], seed=42, do_sample=False)
        results[pid] = {"greedy": text}
        print(f"{pid} done")

    out = {"condition": condition, "mode": "paraphrase8", "results": results}
    out_path = REPORTS_DIR / f"phase4zd_paraphrase8_{condition}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


def mode_naming220(condition: str) -> int:
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = build_model_and_tokenizer(condition)

    results = {}
    t0 = time.time()
    for p in PROBE_SET_A:
        greedy = run_single(model, tokenizer, system_prompt, p["prompt"], seed=42, do_sample=False)
        sampled = {}
        for s in SEEDS_10:
            sampled[str(s)] = run_single(model, tokenizer, system_prompt, p["prompt"], seed=s, do_sample=True)
        results[p["id"]] = {"greedy": greedy, "sampled": sampled}
        print(f"{p['id']} done ({time.time()-t0:.1f}s elapsed)")

    out = {"condition": condition, "mode": "naming220", "n_probes": len(PROBE_SET_A),
           "seeds": list(SEEDS_10), "temperature": TEMPERATURE, "top_p": TOP_P,
           "results": results}
    out_path = REPORTS_DIR / f"phase4zd_naming220_{condition}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


def mode_stage2(condition: str) -> int:
    """Section14 Stage2: PROBE_SET_B(Phase4X held-out naming 24問) x (greedy+seed101-103)。"""
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = build_model_and_tokenizer(condition)

    results = {}
    t0 = time.time()
    for p in PROBE_SET_B:
        greedy = run_single(model, tokenizer, system_prompt, p["prompt"], seed=42, do_sample=False)
        sampled = {}
        for s in SEEDS_3:
            sampled[str(s)] = run_single(model, tokenizer, system_prompt, p["prompt"], seed=s, do_sample=True)
        results[p["id"]] = {"greedy": greedy, "sampled": sampled}
        print(f"{p['id']} done ({time.time()-t0:.1f}s elapsed)")

    out = {"condition": condition, "mode": "stage2", "n_probes": len(PROBE_SET_B),
           "seeds": list(SEEDS_3), "temperature": TEMPERATURE, "top_p": TOP_P,
           "results": results}
    out_path = REPORTS_DIR / f"phase4zd_stage2_{condition}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                         choices=["A_LEGACY_4BIT", "B_HF_BF16_EAGER", "C_HF_BF16_SDPA"])
    parser.add_argument("--mode", required=True,
                         choices=["repro5", "paraphrase8", "naming220", "stage2"])
    args = parser.parse_args()

    fn = {"repro5": mode_repro5, "paraphrase8": mode_paraphrase8, "naming220": mode_naming220,
          "stage2": mode_stage2}[args.mode]
    sys.exit(fn(args.condition))
