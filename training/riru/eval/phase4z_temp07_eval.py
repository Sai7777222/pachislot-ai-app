"""Phase 4Z Section19: production実温度(0.7)でのE36 original+paraphrase独立評価。

0.3評価とは別集計とする。productionコード自体は変更しない。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parents[2]
TRAINING_ROOT = EVAL_DIR.parents[0]
sys.path.insert(0, str(EVAL_DIR))

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7
TOP_P = 0.9
SEEDS = (101, 102, 103)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["gguf", "hf"], required=True)
    args = parser.parse_args()

    from phase4z_probes import PROBE_SET_C

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    results = {}

    if args.engine == "gguf":
        from llama_cpp import Llama

        gguf_path = str(TRAINING_ROOT / "gguf" / "riru-qwen-final-bf16.gguf")
        llm = Llama(model_path=gguf_path, n_gpu_layers=99, n_ctx=2048, verbose=False)

        def gen(question, seed, do_sample):
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}]
            llm.set_seed(seed)
            kwargs = dict(messages=messages, max_tokens=MAX_NEW_TOKENS)
            if do_sample:
                kwargs.update(temperature=TEMPERATURE, top_p=TOP_P)
            else:
                kwargs.update(temperature=0.0)
            out = llm.create_chat_completion(**kwargs)
            return out["choices"][0]["message"]["content"].strip()
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_path = str(TRAINING_ROOT / "merged" / "riru-qwen-final-hf")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=quant_config, device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        def gen(question, seed, do_sample):
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}]
            prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                          tokenize=False)
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
            return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()

    for p in PROBE_SET_C:
        out = {"greedy": gen(p["prompt"], 42, do_sample=False)}
        out["sampled"] = {str(s): gen(p["prompt"], s, do_sample=True) for s in SEEDS}
        results[p["id"]] = out
        print(f"  {p['id']} done")

    out_path = EVAL_DIR / f"phase4z_temp07_results_{args.engine}.json"
    out_path.write_text(
        json.dumps({"_meta": {"temperature": TEMPERATURE, "top_p": TOP_P}, **results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
