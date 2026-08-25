"""Phase 4I-6〜4I-7: 「キミ」自然使用のtemperature/system prompt軽量実験。

QLoRA/LoRA追加学習は一切行わない。リルLoRA v2 (読み取り専用ロード) を固定し、
Phase 4Hで作成した「キミ」肯定文脈評価問題 (riru_eval_set_kimi_v2.jsonl の
kimi_positive_v2_*、7問) と対照問題 (kimi_control_v2、5問) を使用する。

比較軸:
  - temperature: 0.3 / 0.5 / 0.7 / 0.9 (各3 seed)
  - system prompt: 現行のまま / 「キミ」軽量指示を追加 (ファイル自体は変更しない)

system promptファイル (config/prompts/system.jinja2) は一切変更しない。
このスクリプト内でメモリ上にoverride文字列を組み立てるのみ。
"""

from __future__ import annotations

import json
import re
import sys
import time
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
KIMI_EVAL_PATH = EVAL_DIR / "riru_eval_set_kimi_v2.jsonl"

MAX_NEW_TOKENS = 300
TOP_P = 0.9
TEMPERATURES = (0.3, 0.5, 0.7, 0.9)
SEEDS = (42, 43, 44)

KIMI_LIGHT_INSTRUCTION = (
    "\n- ユーザーへの二人称として「キミ」を、自然な場面では使ってください。"
    "毎回答で無理に使用する必要はありません。"
)

# 「不自然なキミ使用」の粗い検出パターン (文頭の機械的な「キミ、」、
# 1回答内でのキミ複数回連呼など)。最終判定は人間レビューを推奨するが、
# 自動での一次フラグ付けに使う。
MECHANICAL_KIMI_START_PATTERN = re.compile(r"^キミ[、,]")


def load_base_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


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
        BASE_MODEL_PATH,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_V2_PATH, adapter_name="v2")
    model.set_adapter("v2")
    model.eval()
    return model, tokenizer


def generate_reply(
    model, tokenizer, system_prompt: str, question: str, temperature: float, seed: int
) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]

    torch.manual_seed(seed)
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    completion_ids = output_ids[0][prompt_len:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "completion_tokens": len(completion_ids),
        "elapsed_sec": round(elapsed, 3),
        "temperature": temperature,
        "seed": seed,
        "kimi_count": text.count("キミ"),
        "mechanical_start_flag": bool(MECHANICAL_KIMI_START_PATTERN.match(text)),
    }


def main() -> int:
    print("Loading base model + v2 LoRA adapter (fixed, read-only)...")
    model, tokenizer = build_model_and_tokenizer()

    base_prompt = load_base_system_prompt()
    prompt_variants = {
        "no_instruction": base_prompt,
        "with_light_kimi_instruction": base_prompt + KIMI_LIGHT_INSTRUCTION,
    }

    kimi_items = []
    with open(KIMI_EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                kimi_items.append(json.loads(line))
    positive_items = [i for i in kimi_items if i["category"].startswith("kimi_positive")]
    control_items = [i for i in kimi_items if i["category"].startswith("kimi_control")]

    results: dict = {"positive": {}, "control": {}}

    print(f"=== positive items ({len(positive_items)}) x temps {TEMPERATURES} x seeds {SEEDS} ===")
    for prompt_name, prompt_text in prompt_variants.items():
        results["positive"][prompt_name] = {}
        for item in positive_items:
            results["positive"][prompt_name][item["id"]] = {
                "category": item["category"],
                "prompt": item["prompt"],
                "by_temperature": {},
            }
            for temp in TEMPERATURES:
                gens = []
                for seed in SEEDS:
                    print(f"  [{prompt_name}] {item['id']} temp={temp} seed={seed}")
                    gens.append(
                        generate_reply(model, tokenizer, prompt_text, item["prompt"], temp, seed)
                    )
                results["positive"][prompt_name][item["id"]]["by_temperature"][str(temp)] = gens

    print(f"=== control items ({len(control_items)}) x temp=0.7 x seeds {SEEDS} (leak check) ===")
    for prompt_name, prompt_text in prompt_variants.items():
        results["control"][prompt_name] = {}
        for item in control_items:
            gens = []
            for seed in SEEDS:
                print(f"  [{prompt_name}] {item['id']} temp=0.7 seed={seed}")
                gens.append(
                    generate_reply(model, tokenizer, prompt_text, item["prompt"], 0.7, seed)
                )
            results["control"][prompt_name][item["id"]] = {
                "category": item["category"],
                "prompt": item["prompt"],
                "generations": gens,
            }

    out_path = EVAL_DIR / "phase4i_kimi_temperature_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
