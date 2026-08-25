"""Phase 4M-3/4/5/6/8/9: fresh-process生成・logits監査。

1プロセスにつき、ベースモデルをロードし、指定したadapterを1つだけ (または
adapterなし=base) ロードして、Q3・E36・Q11を生成する。他のadapterは一切
ロードしない (adapter切替コード自体の影響を完全に排除するため)。

同時に:
  - 生成直前のactive adapter名・peft_config登録一覧を記録する (診断)
  - 最初のassistant生成位置のlogits (top20) を保存する
  - do_sample=True (seed=42) と do_sample=False (greedy) の両方を実行する
  - --condition v4 の場合のみ、同一プロセス内でadapter ON/OFFのlogits差も追加で記録する
    (これは「adapter切替」ではなく「同一ロード済みadapterのON/OFF」を見るための、
    切替コードとは独立した直接検証)

使い方:
  python phase4m_fresh_process_generate.py --condition base
  python phase4m_fresh_process_generate.py --condition v2
  python phase4m_fresh_process_generate.py --condition v3
  python phase4m_fresh_process_generate.py --condition v4

出力: training/riru/reports/phase4m_fresh_<condition>.json

QLoRA/LoRA学習は行わない。adapterファイルは読み取り専用でロードするのみ。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parents[1]
EVAL_DIR = TRAINING_ROOT / "eval"
REPORTS_DIR = TRAINING_ROOT / "reports"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_PATHS = {
    "v1": str(TRAINING_ROOT / "lora-riru-qwen-v1"),
    "v2": str(TRAINING_ROOT / "lora-riru-qwen-v2"),
    "v3": str(TRAINING_ROOT / "lora-riru-qwen-v3"),
    "v4": str(TRAINING_ROOT / "lora-riru-qwen-v4"),
}

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
SEED = 42
TOP_K_LOGITS = 20


def load_base_and_tokenizer():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, quantization_config=quant_config, device_map="auto", trust_remote_code=True
    )
    return model, tokenizer


def diagnostics(model) -> dict:
    """generation直前のactive adapter状態を記録する。"""
    info = {}
    if hasattr(model, "active_adapters"):
        try:
            info["active_adapters"] = list(model.active_adapters)
        except Exception as exc:  # noqa: BLE001
            info["active_adapters_error"] = str(exc)
    if hasattr(model, "active_adapter"):
        try:
            info["active_adapter"] = model.active_adapter
        except Exception as exc:  # noqa: BLE001
            info["active_adapter_error"] = str(exc)
    if hasattr(model, "peft_config"):
        try:
            info["peft_config_registered_adapters"] = sorted(model.peft_config.keys())
        except Exception as exc:  # noqa: BLE001
            info["peft_config_error"] = str(exc)
    return info


def build_messages(system_prompt: str, rag_context: str | None, question: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    if rag_context:
        messages.append({"role": "system", "content": rag_context})
    messages.append({"role": "user", "content": question})
    return messages


def get_first_token_logits(model, tokenizer, messages: list[dict]) -> dict:
    """assistant生成開始位置での次トークン分布 (top20) を取得する。"""
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**encoded)
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, TOP_K_LOGITS)
    top_logits = logits[top_ids]
    tokens = [tokenizer.decode([tid]) for tid in top_ids.tolist()]
    return {
        "top_tokens": tokens,
        "top_token_ids": top_ids.tolist(),
        "top_logits": [round(x, 4) for x in top_logits.tolist()],
        "top_probs": [round(x, 6) for x in top_probs.tolist()],
        "full_logits_for_diff": logits.tolist(),  # 差分計算用 (レポート保存時は間引く)
    }


def generate(model, tokenizer, messages: list[dict], do_sample: bool, seed: int) -> dict:
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
    else:
        gen_kwargs.update(do_sample=False)
    with torch.no_grad():
        output_ids = model.generate(**encoded, **gen_kwargs)
    completion_ids = output_ids[0][prompt_len:]
    return {"text": tokenizer.decode(completion_ids, skip_special_tokens=True).strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["base", "v1", "v2", "v3", "v4"])
    args = parser.parse_args()

    print(f"Loading base model (condition={args.condition})...")
    model, tokenizer = load_base_and_tokenizer()

    adapter_loaded = None
    if args.condition != "base":
        adapter_path = ADAPTER_PATHS[args.condition]
        model = PeftModel.from_pretrained(model, adapter_path, adapter_name=args.condition)
        model.set_adapter(args.condition)
        adapter_loaded = args.condition
    model.eval()

    diag_at_load = diagnostics(model)
    print(f"diagnostics after load: {diag_at_load}")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    rag_17q = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    q3 = next(r for r in rag_17q if r["id"] == "Q3")

    eval_39 = [
        json.loads(line)
        for line in (EVAL_DIR / "riru_eval_set_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    e36 = next(x for x in eval_39 if x["id"] == "E36")
    q11_item = next(r for r in rag_17q if r["id"] == "Q11")

    result: dict = {
        "condition": args.condition,
        "adapter_loaded": adapter_loaded,
        "diagnostics_after_load": diag_at_load,
    }

    # --- Q3: sampled (seed=42, temp=0.3) ---
    q3_messages = build_messages(system_prompt, q3["rag_context_text"], q3["question"])
    diag_before_q3 = diagnostics(model)
    result["q3_sampled"] = {
        "diagnostics_before_generate": diag_before_q3,
        **generate(model, tokenizer, q3_messages, do_sample=True, seed=SEED),
    }

    # --- Q3: greedy (do_sample=False) ---
    result["q3_greedy"] = generate(model, tokenizer, q3_messages, do_sample=False, seed=SEED)

    # --- Q3: first-token logits (top20) ---
    logits_info = get_first_token_logits(model, tokenizer, q3_messages)
    full_logits = logits_info.pop("full_logits_for_diff")
    result["q3_first_token_logits_top20"] = logits_info
    # 差分計算用に生logits配列を別ファイルへ保存 (レポート本体を軽量に保つ)
    logits_path = REPORTS_DIR / f"phase4m_fresh_{args.condition}_full_logits.json"
    logits_path.write_text(json.dumps(full_logits), encoding="utf-8")
    result["full_logits_saved_to"] = str(logits_path)

    # --- E36 (character eval, no RAG context) ---
    e36_messages = build_messages(system_prompt, None, e36["prompt"])
    result["e36_sampled"] = generate(model, tokenizer, e36_messages, do_sample=True, seed=SEED)

    # --- Q11 (compound question) ---
    q11_messages = build_messages(system_prompt, q11_item["rag_context_text"], q11_item["question"])
    result["q11_sampled"] = generate(model, tokenizer, q11_messages, do_sample=True, seed=SEED)

    # --- v4限定: adapter ON/OFF直接比較 (切替コードを介さない直接検証) ---
    if args.condition == "v4":
        with model.disable_adapter():
            diag_off = diagnostics(model)
            off_sampled = generate(model, tokenizer, q3_messages, do_sample=True, seed=SEED)
            off_logits_info = get_first_token_logits(model, tokenizer, q3_messages)
            off_full_logits = off_logits_info.pop("full_logits_for_diff")
        result["adapter_on_off_check"] = {
            "diagnostics_with_adapter_disabled": diag_off,
            "q3_sampled_with_adapter_off": off_sampled,
            "q3_first_token_logits_top20_with_adapter_off": off_logits_info,
        }
        off_logits_path = REPORTS_DIR / "phase4m_fresh_v4_adapter_off_full_logits.json"
        off_logits_path.write_text(json.dumps(off_full_logits), encoding="utf-8")
        result["adapter_on_off_check"]["full_logits_off_saved_to"] = str(off_logits_path)

    out_path = REPORTS_DIR / f"phase4m_fresh_{args.condition}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
