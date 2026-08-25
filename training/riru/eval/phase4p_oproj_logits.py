"""Phase 4P-15: Q3生成開始位置における o_proj scale別 first-token logits解析。

o_scale = 0.0 / 0.25 / 0.5 / 0.75 / 1.0 について、base / full v4 との差、
隣接scaleとの差、top1 token/probabilityを記録する。Q3 recallが急落するscale
付近で非線形な変化があるかを確認する。

adapterファイルは一切変更しない。QLoRA/LoRA学習は行わない。
"""

from __future__ import annotations

import json
import re
import sys
from contextlib import contextmanager
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
TOP_K_LOGITS = 20
LOGITS_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)


def find_lora_layers(model, adapter_name: str):
    results = []
    layer_pat = re.compile(r"\.layers\.(\d+)\.")
    module_types = ("q_proj", "k_proj", "v_proj", "o_proj")
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and adapter_name in scaling:
            m = layer_pat.search(name)
            layer_idx = int(m.group(1)) if m else None
            module_type = next((mt for mt in module_types if name.endswith(mt)), None)
            results.append((name, module, layer_idx, module_type))
    return results


@contextmanager
def oproj_only_scale(model, adapter_name: str, o_scale: float):
    layers = find_lora_layers(model, adapter_name)
    originals: dict[str, float] = {}
    try:
        for name, module, layer_idx, module_type in layers:
            originals[name] = module.scaling[adapter_name]
            if module_type == "o_proj":
                module.scaling[adapter_name] = originals[name] * o_scale
        for name, module, layer_idx, module_type in layers:
            if module_type != "o_proj":
                assert module.scaling[adapter_name] == originals[name]
        yield
    finally:
        for name, module, layer_idx, module_type in layers:
            module.scaling[adapter_name] = originals[name]


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
    model.set_adapter("v4")
    model.eval()
    return model, tokenizer


def get_top_logits(model, tokenizer, messages):
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


def logits_diff(a, b):
    max_abs = 0.0
    sum_abs = 0.0
    for x, y in zip(a, b, strict=True):
        d = abs(x - y)
        sum_abs += d
        if d > max_abs:
            max_abs = d
    return {"max_abs_diff": round(max_abs, 6), "mean_abs_diff": round(sum_abs / len(a), 8)}


def main() -> int:
    print("Loading base model + v4 adapter for o_proj-scale logits analysis...")
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

    for o_scale in LOGITS_SCALES:
        with oproj_only_scale(model, "v4", o_scale):
            info, full = get_top_logits(model, tokenizer, messages)
        key = f"o_scale_{o_scale}"
        top_info[key] = info
        full_logits[key] = full
        print(f"  o_scale={o_scale}: top1={info['top_tokens'][0]!r} p={info['top_probs'][0]}")

    diffs = {}
    scale_keys = [f"o_scale_{s}" for s in LOGITS_SCALES]
    for key in scale_keys:
        diffs[f"base_vs_{key}"] = logits_diff(full_logits["base"], full_logits[key])
        diffs[f"fullv4_vs_{key}"] = logits_diff(full_logits["o_scale_1.0"], full_logits[key])
    for a, b in zip(scale_keys, scale_keys[1:], strict=False):
        diffs[f"{a}_vs_{b}"] = logits_diff(full_logits[a], full_logits[b])

    report = {"top20_logits": top_info, "logits_diffs": diffs}
    out_path = REPORTS_DIR / "phase4p_oproj_logits_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
