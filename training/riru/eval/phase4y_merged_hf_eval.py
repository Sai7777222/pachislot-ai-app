"""Phase 4Y Section12: merge後HFモデル(B_merged_hf)の基準出力を保存する。

新しいprocessで、adapterを一切ロードせず、merged HFモデル単独をロードして
推論する。A_lora_final(4bit NF4 QLoRA)と同じ量子化条件(4bit NF4)・同じ
system prompt・同じprobeで比較できるようにする。学習・データ変更は行わない。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parents[1]
EVAL_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(EVAL_DIR))
from phase4y_representative_probes import load_probes  # noqa: E402

MERGED_MODEL_PATH = str(TRAINING_ROOT / "merged" / "riru-qwen-final-hf")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
SEEDS = (42, 43, 44)


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


def main() -> int:
    print(f"Loading merged HF model (no adapter): {MERGED_MODEL_PATH}")
    model, tokenizer = build_model_and_tokenizer()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    probes = load_probes()

    results = {}
    t0 = time.perf_counter()
    for pid, probe in probes.items():
        out = {"greedy": run_single(
            model, tokenizer, system_prompt, probe["context"], probe["question"], 42,
            do_sample=False,
        )}
        out["sampled"] = {
            str(s): run_single(model, tokenizer, system_prompt, probe["context"],
                                probe["question"], s)
            for s in SEEDS
        }
        results[pid] = out
        print(f"  {pid} done ({time.perf_counter() - t0:.1f}s)")

    out_path = EVAL_DIR / "phase4y_b_merged_hf_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path} (total {time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
