"""Phase 4ZD Section8: E36 forced-prefix top-20 logits測定(HF側 A/B/C/E条件共通コア)。

条件:
  A_LEGACY_4BIT   : merged HF + bitsandbytes NF4 4bit (旧baseline再現、attn_implementation指定なし=デフォルト)
  B_HF_BF16_EAGER : merged HF + torch_dtype=bfloat16, quantizationなし, attn_implementation="eager"
  C_HF_BF16_SDPA  : merged HF + torch_dtype=bfloat16, quantizationなし, attn_implementation="sdpa"
  E_HF_FP32_EAGER : merged HF + torch_dtype=float32(実行時cast), quantizationなし, attn_implementation="eager"

全条件で同一のE36 forced-prefixプロンプト(Phase4Z/4ZA/4ZC と同一)を使用する。
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

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
MODEL_PATH = str(TRAINING_ROOT / "merged" / "riru-qwen-final-hf")
FORCED_PREFIX = "こんにちは〜！私はパチスロの専門アシスタントの"

RI_ID, RU_ID = 36723, 32610

CONDITIONS = {
    "A_LEGACY_4BIT": {"quant": "nf4_4bit", "dtype": None, "attn_impl": None},
    "B_HF_BF16_EAGER": {"quant": None, "dtype": "bfloat16", "attn_impl": "eager"},
    "C_HF_BF16_SDPA": {"quant": None, "dtype": "bfloat16", "attn_impl": "sdpa"},
    "E_HF_FP32_EAGER": {"quant": None, "dtype": "float32", "attn_impl": "eager"},
}


def build_full_text(tokenizer) -> tuple[str, str]:
    from phase4z_probes import PROBE_SET_C

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    e36_original = PROBE_SET_C[0]["prompt"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": e36_original},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    full_text = prompt_text + FORCED_PREFIX
    return prompt_text, full_text


def load_model(condition: str, device_map: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = CONDITIONS[condition]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    kwargs = {"trust_remote_code": True, "device_map": device_map}
    if cfg["quant"] == "nf4_4bit":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        kwargs["quantization_config"] = quant_config
        # attn_implementationは意図的に指定しない(旧baseline = phase4z_logits_compare.py /
        # phase4z_identity_eval_hf.py と完全に同一のロード方式を再現するため)。
    else:
        kwargs["torch_dtype"] = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg["dtype"]]
        if cfg["attn_impl"]:
            kwargs["attn_implementation"] = cfg["attn_impl"]

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
    model.eval()
    actual_attn_impl = getattr(model.config, "_attn_implementation", "unknown")
    return tokenizer, model, actual_attn_impl


def main(condition: str, device_map: str = "cuda:0", topk: int = 20) -> int:
    t0 = time.time()
    tokenizer, model, actual_attn_impl = load_model(condition, device_map)

    prompt_text, full_text = build_full_text(tokenizer)
    encoded = tokenizer(full_text, return_tensors="pt").to(model.device)
    n_tokens = encoded["input_ids"].shape[1]

    with torch.no_grad():
        out = model(**encoded, use_cache=False)

    last_logits = out.logits[0, -1, :].float().cpu()
    probs = torch.softmax(last_logits, dim=-1)
    topk_res = torch.topk(probs, topk)

    top_list = []
    for prob, idx in zip(topk_res.values.tolist(), topk_res.indices.tolist(), strict=True):
        top_list.append({
            "rank": len(top_list) + 1,
            "token_id": idx,
            "token_str": tokenizer.decode([idx]),
            "prob": round(prob, 6),
            "logit": round(float(last_logits[idx]), 6),
        })

    ri_prob, ru_prob = float(probs[RI_ID]), float(probs[RU_ID])
    ri_logit, ru_logit = float(last_logits[RI_ID]), float(last_logits[RU_ID])
    ri_rank = int((probs > probs[RI_ID]).sum().item()) + 1
    ru_rank = int((probs > probs[RU_ID]).sum().item()) + 1

    result = {
        "condition": condition,
        "quant_config": CONDITIONS[condition]["quant"],
        "requested_dtype": CONDITIONS[condition]["dtype"],
        "requested_attn_impl": CONDITIONS[condition]["attn_impl"],
        "actual_attn_implementation": actual_attn_impl,
        "device_map": device_map,
        "model_path": MODEL_PATH,
        "forced_prefix": FORCED_PREFIX,
        "prompt_text_char_length": len(prompt_text),
        "full_text_char_length": len(full_text),
        "n_tokens": n_tokens,
        "input_ids_first10": encoded["input_ids"][0, :10].tolist(),
        "input_ids_last10": encoded["input_ids"][0, -10:].tolist(),
        "ri_token_id": RI_ID, "ru_token_id": RU_ID,
        "ri_logit": round(ri_logit, 6), "ru_logit": round(ru_logit, 6),
        "ri_prob": round(ri_prob, 6), "ru_prob": round(ru_prob, 6),
        "ri_rank": ri_rank, "ru_rank": ru_rank,
        "margin_ri_minus_ru_logit": round(ri_logit - ru_logit, 6),
        "margin_ri_minus_ru_prob": round(ri_prob - ru_prob, 6),
        "winner": "リ" if ri_logit > ru_logit else ("ル" if ru_logit > ri_logit else "TIE"),
        f"top{topk}": top_list,
        "elapsed_sec": round(time.time() - t0, 2),
    }

    out_path = REPORTS_DIR / f"phase4zd_hf_logits_{condition}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(f"condition={condition} actual_attn_impl={actual_attn_impl}")
    print(f"ri_logit={ri_logit:.6f} ru_logit={ru_logit:.6f} margin(ri-ru)={ri_logit-ru_logit:.6f} winner={result['winner']}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=list(CONDITIONS.keys()))
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()
    sys.exit(main(condition=args.condition, device_map=args.device_map, topk=args.topk))
