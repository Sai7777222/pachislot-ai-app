"""Phase 4Z Section23: E36 originalの生成開始位置におけるtop-k logits比較。

既に同一であることを確認済みの416トークンのprompt(HF/GGUFでtoken列完全一致)を
用いて、生成開始位置(assistant応答の最初のトークン)でのtop-10候補と
確率を比較する。既存環境の変更は行わない。
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["gguf", "hf"], required=True)
    args = parser.parse_args()

    from phase4z_probes import PROBE_SET_C

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    e36_original = PROBE_SET_C[0]["prompt"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": e36_original},
    ]

    if args.engine == "hf":
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
        with torch.no_grad():
            out = model(**encoded)
        last_logits = out.logits[0, -1, :].float()
        probs = torch.softmax(last_logits, dim=-1)
        topk = torch.topk(probs, 10)
        results = []
        for prob, idx in zip(topk.values.tolist(), topk.indices.tolist(), strict=True):
            tok_str = tokenizer.decode([idx])
            results.append({"token_id": idx, "token_str": tok_str, "prob": round(prob, 6)})
    else:
        from llama_cpp import Llama

        gguf_path = str(TRAINING_ROOT / "gguf" / "riru-qwen-final-bf16.gguf")
        llm = Llama(model_path=gguf_path, n_gpu_layers=99, n_ctx=2048, verbose=False,
                    logits_all=False)
        # chat template監査(Section7)でHF/GGUF間のtoken列完全一致を確認済みの
        # Jinja2ChatFormatterを同じ要領で使い、E36_original messageをrenderする。
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        metadata = llm.metadata
        tmpl = metadata["tokenizer.chat_template"]
        eos = llm.detokenize([llm.token_eos()]).decode("utf-8", errors="ignore")
        bos = llm.detokenize([llm.token_bos()]).decode("utf-8", errors="ignore") \
            if llm.token_bos() != -1 else ""
        fmt = Jinja2ChatFormatter(template=tmpl, eos_token=eos, bos_token=bos,
                                   add_generation_prompt=True)
        rendered_text = fmt(messages=messages).prompt
        prompt_ids = llm.tokenize(rendered_text.encode("utf-8"), add_bos=False, special=True)
        llm.eval(prompt_ids)
        logits = llm.scores[len(prompt_ids) - 1] if llm.scores is not None else None
        import numpy as np

        logits_arr = np.array(logits, dtype=np.float64)
        exp = np.exp(logits_arr - logits_arr.max())
        probs = exp / exp.sum()
        top_idx = np.argsort(probs)[::-1][:10]
        results = []
        for idx in top_idx:
            tok_bytes = llm.detokenize([int(idx)])
            tok_str = tok_bytes.decode("utf-8", errors="replace")
            results.append({"token_id": int(idx), "token_str": tok_str,
                             "prob": round(float(probs[idx]), 6)})

    out_path = EVAL_DIR / f"_phase4z_logits_top10_{args.engine}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    for r in results:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
