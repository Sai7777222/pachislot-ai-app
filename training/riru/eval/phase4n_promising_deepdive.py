"""Phase 4N-8/9: 有望候補の深掘り評価 (v4)。

scale sweep / module ablation の結果から浮上した2つの有望候補:
  - scale=0.25 (全moduleを均等に0.25倍)
  - o_off (o_projのみLoRA寄与を0に、q/k/vはフル強度)

について、Q3単体の改善がQ3固有かどうかを見るためPhase4Iのheld-out P01/P02、
および既知のhallucinationパターンをQ9/Q11で確認する。

adapterファイルは一切書き換えない。QLoRA/LoRA学習は行わない。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4n_lora_scale_experiment import build_messages, generate, scaled_lora  # noqa: E402

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parents[1]
EVAL_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_V4_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v4")

CONDITIONS = {
    "full_v4": {"factor": 1.0, "module_types": None},
    "scale_0.25": {"factor": 0.25, "module_types": None},
    "o_off": {"factor": 0.0, "module_types": ["o_proj"]},
}


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


def recall_check(text: str, key_facts: list[str], irrelevant_markers: list[str]) -> dict:
    found = [f for f in key_facts if f in text]
    leaked = [m for m in irrelevant_markers if m in text]
    return {
        "text": text,
        "key_facts_found": found,
        "recall_pct": round(len(found) / len(key_facts) * 100, 1) if key_facts else None,
        "irrelevant_leaked": leaked,
    }


def main() -> int:
    print("Loading base model + v4 adapter (promising-scale deep dive)...")
    model, tokenizer = build_model_and_tokenizer()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    rag_17q = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    q9 = next(r for r in rag_17q if r["id"] == "Q9")
    q11 = next(r for r in rag_17q if r["id"] == "Q11")

    holdout_path = EVAL_DIR / "phase4i_holdout_omission_v2.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    p01 = next(r for r in holdout if r["id"] == "P01")
    p02 = next(r for r in holdout if r["id"] == "P02")

    results = {}
    for cond_name, cfg in CONDITIONS.items():
        print(f"  condition={cond_name}")
        with scaled_lora(model, "v4", factor=cfg["factor"], module_types=cfg["module_types"]):
            cond_out = {}
            for item in (p01, p02):
                msgs = build_messages(system_prompt, item["rag_context_text"], item["question"])
                text = generate(model, tokenizer, msgs, do_sample=True, seed=42)
                cond_out[item["id"]] = recall_check(
                    text, item["key_facts"], item.get("irrelevant_markers", [])
                )
            for item in (q9, q11):
                msgs = build_messages(system_prompt, item["rag_context_text"], item["question"])
                text = generate(model, tokenizer, msgs, do_sample=True, seed=42)
                cond_out[item["id"]] = {"text": text}
        results[cond_name] = cond_out

    out_path = EVAL_DIR / "phase4n_promising_deepdive_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
