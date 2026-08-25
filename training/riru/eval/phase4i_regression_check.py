"""Phase 4I-9: 最有力候補 (現行system prompt + 「キミ」軽量指示、temperature=0.7)
での既存構造化DB/RAG 17問の回帰確認。

v2固定。system promptファイルは変更しない (メモリ上でoverrideするのみ)。
QLoRA/LoRA学習は行わない。

比較対象:
  - baseline: 現行system prompt、temperature=0.3 (Phase4E/4G/4Hで使用してきた基準)
  - candidate: 現行system prompt + 「キミ」軽量指示、temperature=0.7 (本フェーズの推奨候補)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parents[1]
EVAL_DIR = Path(__file__).resolve().parent

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_V2_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v2")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
RAG_17Q_PATH = EVAL_DIR / "structured_rag_17q_context.json"

MAX_NEW_TOKENS = 300
TOP_P = 0.9

KIMI_LIGHT_INSTRUCTION = (
    "\n- ユーザーへの二人称として「キミ」を、自然な場面では使ってください。"
    "毎回答で無理に使用する必要はありません。"
)


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
    model = PeftModel.from_pretrained(base_model, ADAPTER_V2_PATH, adapter_name="v2")
    model.set_adapter("v2")
    model.eval()
    return model, tokenizer


def generate_reply(model, tokenizer, system_prompt, rag_context, question, temperature, seed=42):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": rag_context},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def main() -> int:
    print("Loading base model + v2 adapter...")
    model, tokenizer = build_model_and_tokenizer()
    base_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    candidate_prompt = base_prompt + KIMI_LIGHT_INSTRUCTION

    rag_17q = json.loads(RAG_17Q_PATH.read_text(encoding="utf-8"))

    results: dict = {}
    for item in rag_17q:
        qid = item["id"]
        print(f"  {qid}")
        baseline_text = generate_reply(
            model, tokenizer, base_prompt, item["rag_context_text"], item["question"],
            temperature=0.3,
        )
        candidate_text = generate_reply(
            model, tokenizer, candidate_prompt, item["rag_context_text"], item["question"],
            temperature=0.7,
        )
        results[qid] = {
            "question": item["question"],
            "baseline_prompt_temp0.3": baseline_text,
            "candidate_prompt_kimi_instruction_temp0.7": candidate_text,
        }

    out_path = EVAL_DIR / "phase4i_regression_check_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
