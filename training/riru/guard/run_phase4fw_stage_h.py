# -*- coding: utf-8 -*-
"""Phase4FW Stage H: product action simulation。V1(破棄)/V2(unsafe claim除去)はoffline
logicのみ。V3(1回だけのcontext-grounded regeneration)のみ実際にGPU呼び出しを行う
(既存context・新規retrievalなし・最大1回)。"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
REPORTS_DIR = TRAINING_ROOT / "reports"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ZG_ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")
PRODUCTION_PROMPT_PATH = TRAINING_ROOT.parents[1] / "config" / "prompts" / "system.jinja2"

FALLBACK_MESSAGE = "登録情報だけでは正確に答えられない部分があるみたい。ごめんね。"

V3_REGEN_SAMPLE_IDS = ["FU-A03", "FU-B05", "FV-P04", "FV-P09"]


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_fw_regen")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system_prompt, user_text, seed=42, max_new_tokens=300):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    with torch.no_grad():
        output_ids = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


def v1_discard(claims, statuses):
    unsafe = any(s in ("UNSUPPORTED", "MISATTRIBUTED") for s in statuses)
    return FALLBACK_MESSAGE if unsafe else None


def v2_reconstruct(claims, statuses):
    supported_texts = [c["text"] for c, s in zip(claims, statuses) if s in ("SUPPORTED", "NON_FACTUAL")]
    if not supported_texts:
        return FALLBACK_MESSAGE
    return "。".join(supported_texts) + "。"


def main():
    gt = json.loads((REPORTS_DIR / "phase4fw_ground_truth.json").read_text(encoding="utf-8"))
    targets = json.loads((REPORTS_DIR / "phase4fw_target_responses.json").read_text(encoding="utf-8"))
    by_id = {t["id"]: t for t in targets}
    mod = json.loads((REPORTS_DIR / "phase4fw_model_verifier.json").read_text(encoding="utf-8"))
    det = json.loads((REPORTS_DIR / "phase4fw_deterministic_verifier.json").read_text(encoding="utf-8"))
    det_by_key = {(r["response_id"], r["claim_idx"]): r for r in det["rows"]}

    # hybrid predicted status per (id, claim_idx)
    hybrid_status = {}
    for r in mod["rows"]:
        key = (r["id"], r["claim_idx"])
        d = det_by_key.get(key)
        model_unsafe = r["predicted_status"] in ("UNSUPPORTED", "MISATTRIBUTED")
        det_unsafe = d and d["predicted_status"] in ("UNSUPPORTED", "MISATTRIBUTED")
        hybrid_status[key] = "UNSAFE" if (model_unsafe or det_unsafe) else "SAFE"

    v1_results = []
    v2_results = []
    for pid in [t["id"] for t in targets if t["category"] in ("known_failure", "phantom_entity", "concept_binding")]:
        claims = gt["claims_by_response"][pid]
        statuses = [("UNSUPPORTED" if hybrid_status.get((pid, i)) == "UNSAFE" else "SUPPORTED") for i in range(len(claims))]
        v1_out = v1_discard(claims, statuses)
        v2_out = v2_reconstruct(claims, statuses)
        original = by_id[pid]["response"]
        v1_results.append({"id": pid, "original": original, "v1_output": v1_out or original,
                            "v1_discarded": v1_out is not None})
        v2_results.append({"id": pid, "original": original, "v2_output": v2_out,
                            "v2_length_ratio": round(len(v2_out) / max(1, len(original)), 3)})

    v1_discard_rate = sum(1 for r in v1_results if r["v1_discarded"]) / len(v1_results)

    # ---- V3: 1回だけのcontext-grounded regeneration(実際にGPU使用、サンプル4件) ----
    production_prompt = PRODUCTION_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    v3_results = []
    for pid in V3_REGEN_SAMPLE_IDS:
        t = by_id[pid]
        claims = gt["claims_by_response"][pid]
        unsafe_claim_texts = [claims[i]["text"] for i in range(len(claims))
                               if hybrid_status.get((pid, i)) == "UNSAFE"]
        regen_prompt = (f"{t['prompt']}\n\n(注: 前回の回答のうち以下の内容は検索結果で裏付けが確認できませんでした。"
                         f"これらを含めず、確認できる内容だけで答え直してください: "
                         f"{'; '.join(unsafe_claim_texts)})")
        regen_text = generate(model, tokenizer, production_prompt, regen_prompt, seed=42)
        v3_results.append({"id": pid, "original": t["response"], "unsafe_claims_flagged": unsafe_claim_texts,
                            "v3_regenerated": regen_text})
        print(f"[V3] {pid} done")

    out = {
        "v1_discard_and_fallback": {"n": len(v1_results), "discard_rate": round(v1_discard_rate, 3), "rows": v1_results},
        "v2_reconstruct_from_supported": {"n": len(v2_results), "rows": v2_results},
        "v3_single_regeneration_sample": v3_results,
    }
    (REPORTS_DIR / "phase4fw_product_action_simulation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"V1 discard_rate={v1_discard_rate:.3f}")
    print("STAGE H DONE")


if __name__ == "__main__":
    main()
