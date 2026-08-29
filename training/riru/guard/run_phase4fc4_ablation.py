# -*- coding: utf-8 -*-
"""Phase4FC4 Section10 (Stage A): P0/P1/P2 prompt ablation against personality_preference20。
RAG生成は行わない(system promptのみ、Section10の指示通り)。"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

import torch

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
PROJECT_ROOT = GUARD_DIR.parents[2]
REPORTS_DIR = TRAINING_ROOT / "reports"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ZG_ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

P1_PROMPT = (GUARD_DIR / "phase4zo_minimal_prompt.txt").read_text(encoding="utf-8").strip()
P2_PROMPT = (
    "あなたは「リル」という名前のAIキャラクターです。これは雑談・パーソナルな会話"
    "(挨拶、感情表現、趣味嗜好、日常の話など)です。データベースを参照したような言い方"
    "(「登録データにありません」「登録されていません」など)はせず、リル自身のキャラクターとして"
    "自然に答えてください。自分の好みや性格について聞かれた場合は、軽いキャラクター設定として"
    "自然に答えて構いません。これは事実データベースの主張ではなく、キャラクターとしての一意見です。"
    "パチスロの数値や機種名を創作しないでください。"
)

HEDGE_RE = re.compile(r"登録データ|データベース|データがない|登録されていない|情報がない|記録がない|確認できない")


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_fc4_ablation")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages, seed=42, max_new_tokens=200):
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    gen_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    elapsed = time.time() - gen_start
    text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
    return text, elapsed


def render_system_prompt() -> str:
    from jinja2 import Template
    return Template(SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")).render()


def main():
    sys.path.insert(0, str(PROJECT_ROOT / "training" / "riru" / "eval"))
    from phase4zn_unattended_probes import ALL_PROBES
    probes = [p for p in ALL_PROBES if p["category"] == "personality_preference"]
    print(f"n_probes: {len(probes)}")

    p0_system = render_system_prompt()
    model, tokenizer = load_model()
    print("model loaded")

    results = {"P0": [], "P1": [], "P2": []}
    for label, sys_prompt in [("P0", p0_system), ("P1", P1_PROMPT), ("P2", P2_PROMPT)]:
        hedge_count = 0
        for p in probes:
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": p["prompt"]}]
            text, elapsed = generate(model, tokenizer, messages)
            hedge = bool(HEDGE_RE.search(text))
            hedge_count += hedge
            results[label].append({"id": p["id"], "prompt": p["prompt"], "response": text, "hedge": hedge, "latency_sec": elapsed})
            print(f"[{label}] {p['id']} hedge={hedge} latency={elapsed:.1f}s")
        print(f"{label}: hedge {hedge_count}/{len(probes)}")

    out_path = REPORTS_DIR / "phase4fc4_ablation.json"
    summary = {label: {"n": len(rows), "hedge_count": sum(r["hedge"] for r in rows)} for label, rows in results.items()}
    out_path.write_text(json.dumps({"summary": summary, "detail": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary)
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
