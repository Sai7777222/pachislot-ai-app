"""Phase 4ZK Diagnostic F: ZJ teacher 26件に対するteacher-forced lossをZG/ZJで比較する。

train_qlora.pyのbuild_assistant_only_exampleと同一のassistant-only masking
ロジックを再利用し、forward passのみ(backwardなし)でlossを計算する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
PROJECT_ROOT = EVAL_DIR.parents[2]
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(TRAINING_ROOT))
REPORTS_DIR = TRAINING_ROOT / "reports"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTERS = {
    "zg": str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened"),
    "zj": str(TRAINING_ROOT / "lora-riru-qwen-phase4zj-instruction-override-hardened"),
    "m1": str(TRAINING_ROOT / "lora-riru-qwen-phase4zk-m1-diagnostic"),
}


def load_model(model_key: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTERS[model_key], adapter_name=model_key)
    model.eval()
    return model, tokenizer


def build_assistant_only_example(tokenizer, messages, max_seq_length=2048):
    """train_qlora.pyのbuild_assistant_only_exampleと同一ロジック(assistantターンのみlabel有効)。
    NOTE: apply_chat_template(tokenize=True)はBatchEncodingを返すため["input_ids"]で取り出す。"""
    full_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)["input_ids"]
    labels = [-100] * len(full_ids)

    running_messages = []
    prev_len = 0
    for msg in messages:
        running_messages.append(msg)
        ids_so_far = tokenizer.apply_chat_template(running_messages, add_generation_prompt=False, tokenize=True)["input_ids"]
        cur_len = len(ids_so_far)
        if msg["role"] == "assistant":
            start = prev_len
            end = cur_len
            if 0 <= start < end <= len(full_ids):
                labels[start:end] = full_ids[start:end]
        prev_len = cur_len

    if len(full_ids) > max_seq_length:
        full_ids = full_ids[:max_seq_length]
        labels = labels[:max_seq_length]

    return full_ids, labels


@torch.no_grad()
def compute_loss(model, tokenizer, messages) -> dict:
    input_ids, labels = build_assistant_only_example(tokenizer, messages)
    input_tensor = torch.tensor([input_ids]).to(model.device)
    label_tensor = torch.tensor([labels]).to(model.device)
    out = model(input_ids=input_tensor, labels=label_tensor)
    n_label_tokens = sum(1 for lbl in labels if lbl != -100)
    return {"loss": float(out.loss), "n_label_tokens": n_label_tokens, "n_total_tokens": len(input_ids)}


def main(model_key: str) -> int:
    from phase4zj_instruction_override_source_data import ALL_MULTITURN_RECORDS, ALL_SINGLE_TURN_RECORDS

    system_prompt = (PROJECT_ROOT / "config" / "prompts" / "system.jinja2").read_text(encoding="utf-8")
    model, tokenizer = load_model(model_key)

    results = {}
    for item in ALL_SINGLE_TURN_RECORDS:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": item["user"]},
                    {"role": "assistant", "content": item["assistant"]}]
        r = compute_loss(model, tokenizer, messages)
        results[item["id"]] = r

    for item in ALL_MULTITURN_RECORDS:
        messages = [{"role": "system", "content": system_prompt}] + item["turns"]
        r = compute_loss(model, tokenizer, messages)
        results[item["id"]] = r

    out = {"model": model_key, "n_teachers": len(results), "results": results}
    out_path = REPORTS_DIR / f"phase4zk_teacher_loss_{model_key}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    mean_loss = sum(r["loss"] for r in results.values()) / len(results)
    print(f"Saved -> {out_path}, mean_loss={mean_loss:.4f}")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["zg", "zj", "m1"])
    args = parser.parse_args()
    sys.exit(main(args.model))
