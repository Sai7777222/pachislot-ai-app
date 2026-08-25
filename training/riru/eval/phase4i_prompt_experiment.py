"""Phase 4I-2〜4I-5: Q3型「重要情報省略」問題の原因切り分け実験。

QLoRA/LoRA追加学習は一切行わない。ベースQwen2.5-14B-Instruct と
リルLoRA v2 (読み取り専用ロード) の2条件、system prompt A/B/C/D の
4条件で、Q3 (実際の本番RAGコンテキスト) + P01〜P10 (新規held-outテスト、
学習には未使用) を評価する。

system promptファイル (config/prompts/system.jinja2) は一切変更しない。
このスクリプト内でファイル内容を読み込み、メモリ上でA/B/C/D用に
override文字列を組み立てるのみ。

出力: phase4i_prompt_experiment_results.json (生の生成結果)
"""

from __future__ import annotations

import json
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
RAG_17Q_PATH = EVAL_DIR / "structured_rag_17q_context.json"
HOLDOUT_PATH = EVAL_DIR / "phase4i_holdout_omission_v2.json"

MAX_NEW_TOKENS = 300
TOP_P = 0.9

PROMPT_B_ADDITION = (
    "\n- 質問に直接関係する登録情報が複数ある場合、重要な数値・条件を省略せず回答してください。"
    "質問と関係のない情報は追加しないでください。"
)
PROMPT_C_ADDITION = PROMPT_B_ADDITION + (
    "\n- 回答を短くすることより、質問に直接関係する登録情報を正確かつ漏れなく伝えることを"
    "優先してください。"
)
PROMPT_D_ADDITION = (
    "\n- 回答を作成する際は、次の手順で考えてください（ただし手順や思考過程そのものは回答に書かず、"
    "最終的な回答文のみを出力してください）:"
    "\n  1. 質問対象を特定する"
    "\n  2. 提供されたデータから質問対象に直接関係する事実を抽出する"
    "\n  3. 関連する重要な数値・条件を回答に含める"
    "\n  4. 無関係な情報は含めない"
    "\n  5. データにない情報は推測しない"
)


def load_base_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def build_prompt_variants(base_prompt: str) -> dict[str, str]:
    return {
        "A_current": base_prompt,
        "B_omission_instruction": base_prompt + PROMPT_B_ADDITION,
        "C_prioritize_completeness": base_prompt + PROMPT_C_ADDITION,
        "D_structured_steps": base_prompt + PROMPT_D_ADDITION,
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
        BASE_MODEL_PATH,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_V2_PATH, adapter_name="v2")
    model.eval()
    return model, tokenizer


def generate_reply(
    model, tokenizer, system_prompt: str, rag_context: str, question: str,
    use_adapter: bool, temperature: float, seed: int,
) -> dict:
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
    t0 = time.perf_counter()
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=temperature,
        top_p=TOP_P,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    with torch.no_grad():
        if use_adapter:
            model.set_adapter("v2")
            output_ids = model.generate(**encoded, **gen_kwargs)
        else:
            with model.disable_adapter():
                output_ids = model.generate(**encoded, **gen_kwargs)
    elapsed = time.perf_counter() - t0
    completion_ids = output_ids[0][prompt_len:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return {
        "text": text.strip(),
        "completion_tokens": len(completion_ids),
        "elapsed_sec": round(elapsed, 3),
        "seed": seed,
        "temperature": temperature,
    }


def main() -> int:
    print("Loading base model + v2 LoRA adapter (read-only)...")
    model, tokenizer = build_model_and_tokenizer()

    base_prompt = load_base_system_prompt()
    prompt_variants = build_prompt_variants(base_prompt)

    rag_17q = json.loads(RAG_17Q_PATH.read_text(encoding="utf-8"))
    q3 = next(r for r in rag_17q if r["id"] == "Q3")
    holdout_items = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))

    test_items = [
        {"id": "Q3", "question": q3["question"], "rag_context_text": q3["rag_context_text"]}
    ]
    test_items += holdout_items

    results: dict = {
        "reproducibility_check": {},
        "prompt_sweep": {},
    }

    # --- 4I-2: Q3 ベースライン再現 (prompt A, v2, temp=0.3, 複数seed) ---
    print("=== 4I-2: Q3 baseline reproducibility (prompt A, v2, temp=0.3, seeds 42/43/44) ===")
    for seed in (42, 43, 44):
        gen = generate_reply(
            model, tokenizer, prompt_variants["A_current"], q3["rag_context_text"], q3["question"],
            use_adapter=True, temperature=0.3, seed=seed,
        )
        results["reproducibility_check"][f"seed_{seed}"] = gen
        print(f"  seed={seed}: {gen['text'][:60]}...")

    # --- 4I-3/4/5: prompt A/B/C/D x {base, v2} x {Q3 + P01-P10} (temp=0.3, seed=42固定) ---
    print("=== 4I-3/4/5: prompt sweep (A/B/C/D x base/v2 x Q3+P01-P10) ===")
    for item in test_items:
        item_id = item["id"]
        results["prompt_sweep"][item_id] = {"question": item["question"], "conditions": {}}
        for prompt_name, prompt_text in prompt_variants.items():
            for model_name, use_adapter in (("base", False), ("v2", True)):
                key = f"{prompt_name}__{model_name}"
                print(f"  {item_id} / {key}")
                gen = generate_reply(
                    model, tokenizer, prompt_text, item["rag_context_text"], item["question"],
                    use_adapter=use_adapter, temperature=0.3, seed=42,
                )
                results["prompt_sweep"][item_id]["conditions"][key] = gen

    out_path = EVAL_DIR / "phase4i_prompt_experiment_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # プロンプト全文もあわせて保存 (system promptファイル自体は変更していないことの証跡として)
    (EVAL_DIR / "phase4i_prompt_variants_used.json").write_text(
        json.dumps(prompt_variants, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
