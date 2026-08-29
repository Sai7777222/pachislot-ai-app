"""Phase4FW Stage A / E2: 既存モデル(Phase4ZG)にatomic claim分解「だけ」をさせる。
新しい事実を生成させない。構造化抽出タスクのみ。E1(deterministic)との比較サンプル用。"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

import torch

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
REPORTS_DIR = TRAINING_ROOT / "reports"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ZG_ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")

EXTRACTION_SYSTEM_PROMPT = """あなたは文章解析タスクを行うツールです。与えられた「回答文」を、独立した最小単位の事実主張(atomic claim)に分解してください。

厳守事項:
- 新しい情報を追加したり、事実の正誤を判断したりしないでください。これは分解タスクのみです。
- 1つのatomic claimは、1つのsubject(主語・対象)について、1つのpredicate(述べている内容)を持つ形にしてください。
- 「AはXで、BはYだ」のような文は、必ず「Aについての主張」と「Bについての主張」の2つに分けてください。
- 出力は必ず以下のJSON形式の配列だけにしてください。他の文章は一切書かないでください。

出力形式:
[{"subject": "...", "predicate": "..."}, ...]
"""

# サンプル対象probe ids(GPU budget節約のため全84件ではなく代表サンプルのみ)
SAMPLE_IDS = [
    "FU-D01", "FU-B05", "FU-A03", "FU-E02",  # known failures
    "FV-P02", "FV-P04", "FV-P09", "FV-P13", "FU-B03", "FU-F03", "FU-D05", "FV-P01",  # phantom (8)
    "FV-C01", "FV-C03", "FV-C06", "FU-D03", "FV-C08", "FV-C09",  # concept (6)
    "P02", "P04", "LC-08", "Q11", "Q1", "Q4",  # rag50 (6)
]


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_fw_e2")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system_prompt, user_text, seed=42, max_new_tokens=400):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    with torch.no_grad():
        output_ids = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


def try_parse_json(text):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    targets = json.loads((REPORTS_DIR / "phase4fw_target_responses.json").read_text(encoding="utf-8"))
    by_id = {t["id"]: t for t in targets}

    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    out = []
    t0 = time.time()
    for pid in SAMPLE_IDS:
        t = by_id[pid]
        user_text = f"回答文:\n{t['response']}"
        gen_start = time.time()
        raw = generate(model, tokenizer, EXTRACTION_SYSTEM_PROMPT, user_text, seed=42)
        elapsed = time.time() - gen_start
        parsed = try_parse_json(raw)
        out.append({"id": pid, "raw_output": raw, "parsed_claims": parsed,
                    "parse_success": parsed is not None, "latency_sec": elapsed})
        print(f"[E2] {pid} parse_success={parsed is not None} n_claims={len(parsed) if parsed else 0} t={elapsed:.1f}s")

    (REPORTS_DIR / "phase4fw_e2_extraction_raw.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"E2 DONE total_time={time.time()-t0:.1f}s n={len(out)}")


if __name__ == "__main__":
    main()
