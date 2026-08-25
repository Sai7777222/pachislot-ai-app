"""Phase 4O-18: Q3生成開始位置でのv4 vs v5-qkv logits比較。

Phase 4M/4Nと同方式: 最初のassistant生成位置における次トークン分布(top20)を
記録し、v4/v5-qkv/base間のfull-vocab logits差 (max abs / mean abs) を求める。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parents[1]
EVAL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = TRAINING_ROOT / "reports"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_V4_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v4")
ADAPTER_V5QKV_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v5-qkv")
TOP_K_LOGITS = 20


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
    model = PeftModel.from_pretrained(base_model, ADAPTER_V4_PATH, adapter_name="v4")
    model.load_adapter(ADAPTER_V5QKV_PATH, adapter_name="v5qkv")
    model.eval()
    return model, tokenizer


def get_top_logits(model, tokenizer, messages) -> tuple[dict, list[float]]:
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**encoded)
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, TOP_K_LOGITS)
    tokens = [tokenizer.decode([tid]) for tid in top_ids.tolist()]
    info = {"top_tokens": tokens, "top_probs": [round(x, 6) for x in top_probs.tolist()]}
    return info, logits.tolist()


def logits_diff(a: list[float], b: list[float]) -> dict:
    max_abs = 0.0
    sum_abs = 0.0
    for x, y in zip(a, b, strict=True):
        d = abs(x - y)
        sum_abs += d
        if d > max_abs:
            max_abs = d
    return {"max_abs_diff": round(max_abs, 6), "mean_abs_diff": round(sum_abs / len(a), 8)}


def main() -> int:
    print("Loading base model + v4/v5-qkv adapters for logits comparison...")
    model, tokenizer = build_model_and_tokenizer()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    rag_17q = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    q3 = next(r for r in rag_17q if r["id"] == "Q3")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": q3["rag_context_text"]},
        {"role": "user", "content": q3["question"]},
    ]

    full_logits = {}
    top_info = {}

    with model.disable_adapter():
        info, full = get_top_logits(model, tokenizer, messages)
        top_info["base"] = info
        full_logits["base"] = full

    model.set_adapter("v4")
    info, full = get_top_logits(model, tokenizer, messages)
    top_info["v4"] = info
    full_logits["v4"] = full

    model.set_adapter("v5qkv")
    info, full = get_top_logits(model, tokenizer, messages)
    top_info["v5qkv"] = info
    full_logits["v5qkv"] = full

    pairs = [("base", "v4"), ("base", "v5qkv"), ("v4", "v5qkv")]
    diffs = {f"{a}_vs_{b}": logits_diff(full_logits[a], full_logits[b]) for a, b in pairs}

    report = {"top20_logits": top_info, "logits_diffs": diffs}
    out_path = REPORTS_DIR / "phase4o_logits_compare.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diffs, ensure_ascii=False, indent=2))
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
