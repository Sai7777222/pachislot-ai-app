"""Phase 4Z Section18: E36 originalのgreedy再現性確認(1回の独立プロセス呼び出し分)。

--engine gguf | hf を指定して、それぞれ独立プロセスとして5回呼び出す想定。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parents[2]
TRAINING_ROOT = EVAL_DIR.parents[0]
sys.path.insert(0, str(EVAL_DIR))

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
MAX_NEW_TOKENS = 300


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["gguf", "hf"], required=True)
    args = parser.parse_args()

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    from phase4z_probes import PROBE_SET_C

    e36_original = PROBE_SET_C[0]["prompt"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": e36_original},
    ]

    if args.engine == "gguf":
        from llama_cpp import Llama

        gguf_path = str(TRAINING_ROOT / "gguf" / "riru-qwen-final-bf16.gguf")
        llm = Llama(model_path=gguf_path, n_gpu_layers=99, n_ctx=2048, verbose=False)
        llm.set_seed(42)
        out = llm.create_chat_completion(messages=messages, max_tokens=MAX_NEW_TOKENS,
                                          temperature=0.0)
        text = out["choices"][0]["message"]["content"].strip()
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
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                      tokenize=False)
        encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = encoded["input_ids"].shape[1]
        torch.manual_seed(42)
        with torch.no_grad():
            output_ids = model.generate(
                **encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()

    out_path = EVAL_DIR / f"_phase4z_greedy_repro_{args.engine}_output.txt"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(text.replace("\n", "\\n") + "\n")
    print("RESULT:", text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
