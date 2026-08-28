"""Phase4ZS 追加確認: Phase4ZRの元の条件(context=None)を厳密に再現し、real contextを
与えた場合(Stage A-E)との対比を完結させる。10回greedy(異なるseed)。"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import torch

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
PROJECT_ROOT = GUARD_DIR.parents[2]
sys.path.insert(0, str(GUARD_DIR))
REPORTS_DIR = TRAINING_ROOT / "reports"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ZG_ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")
STRICT_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
Q6_QUERY = "GGとSGGの違いを初心者向けに説明して"
NUMERIC_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?パーセント|1/\d+")
CONTRADICTION_PATTERN = re.compile(
    r"(登録データ|情報)[^。！？]{0,15}(ない|なかった|見つから)[^。！？]{0,40}(だけど|けど|が|、)[^。！？]{0,60}"
    r"(\d+(?:\.\d+)?\s*%|1/\d+|\d+パーセント)"
)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_zs_zero")
    model.eval()

    strict_prompt = STRICT_PROMPT_PATH.read_text(encoding="utf-8")
    rows = []
    for seed in range(10):
        messages = [{"role": "system", "content": strict_prompt}, {"role": "user", "content": Q6_QUERY}]
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = encoded["input_ids"].shape[1]
        torch.manual_seed(seed)
        with torch.no_grad():
            output_ids = model.generate(**encoded, max_new_tokens=300, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
        numerics = NUMERIC_PATTERN.findall(text)
        rows.append({"seed": seed, "response": text, "numerics_in_output": numerics,
                     "has_numeric_claim": len(numerics) > 0,
                     "contradictory_self_awareness": bool(CONTRADICTION_PATTERN.search(text))})
        print(f"seed={seed} has_numeric={len(numerics)>0}")

    out = {
        "purpose": "Phase4ZR原条件(context=None、strict prompt)の厳密再現。Stage A-Eのreal-context結果との対比用。",
        "n": len(rows), "unsupported_numeric_count": sum(1 for r in rows if r["has_numeric_claim"]),
        "unsupported_numeric_rate": sum(1 for r in rows if r["has_numeric_claim"]) / len(rows),
        "contradiction_count": sum(1 for r in rows if r["contradictory_self_awareness"]),
        "rows": rows,
    }
    (REPORTS_DIR / "phase4zs_zero_context_confirmation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("summary:", out["unsupported_numeric_count"], "/", out["n"])


if __name__ == "__main__":
    main()
