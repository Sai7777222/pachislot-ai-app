# -*- coding: utf-8 -*-
"""Phase4FW Stage E: 既存モデル(Phase4ZG)による構造化claim verifier。
query/context/atomic claimsだけを渡し、「このclaimがcontextで支持されるか」だけを判定させる。
新しい事実を回答させない。temperature=0。1responseあたり1回のcallで、そのresponseの
全claimをまとめて判定させる(GPU呼び出し回数削減のため)。"""
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

VERIFIER_SYSTEM_PROMPT = """あなたは事実検証だけを行うツールです。以下の[検索結果]と、それに対する回答から
抽出された claim(事実主張)のリストが与えられます。各claimについて、[検索結果]の内容だけを根拠に、
そのclaimが支持されているかを判定してください。

厳守事項:
- 新しい事実を生成したり、claimの正誤についてあなた自身の知識を使ったりしないでください。
- 判定は[検索結果]に書かれている内容だけを根拠にしてください。
- claimのsubject(対象)が[検索結果]に一度も登場しない場合、それについての具体的な内容を述べているclaimは
  MISATTRIBUTEDです。
- claimのsubjectは[検索結果]に登場するが、claimの内容(述べられている数値・関係・性質)が[検索結果]の
  その対象についての記述と一致しない場合もMISATTRIBUTEDです。
- claimの数値・記号が[検索結果]に文字通り存在しない場合はUNSUPPORTEDです。
- claim自体が「情報が見つからない」という趣旨の場合、[検索結果]に本当に対応情報がなければSUPPORTEDです。
- 出力は必ず次のJSON配列の形式だけにしてください。他の文章は一切書かないでください。

出力形式:
[{"claim_index": 0, "status": "SUPPORTED"|"UNSUPPORTED"|"MISATTRIBUTED"|"AMBIGUOUS", "evidence": "根拠となった検索結果中の文字列(なければ空文字)", "reason": "一言で理由"}, ...]
"""


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_fw_verifier")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system_prompt, user_text, seed=42, max_new_tokens=600):
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


def build_user_text(context, claims):
    claim_lines = "\n".join(f'{i}: subject="{c["subject"]}", claim="{c["text"]}"' for i, c in enumerate(claims))
    return f"[検索結果]\n{context}\n\n[判定対象claims]\n{claim_lines}"


def main():
    gt = json.loads((REPORTS_DIR / "phase4fw_ground_truth.json").read_text(encoding="utf-8"))
    targets = json.loads((REPORTS_DIR / "phase4fw_target_responses.json").read_text(encoding="utf-8"))
    ctx_by_id = {t["id"]: t["context"] for t in targets}

    # 評価対象: known_failure + phantom_entity + concept_binding(34件、全件)
    # + RAG50から代表20件(FP測定用)
    critical_ids = [t["id"] for t in targets if t["category"] in ("known_failure", "phantom_entity", "concept_binding")]
    rag50_ids_all = [t["id"] for t in targets if t["category"] == "rag50"]
    rag50_sample = rag50_ids_all[::2][:20] if len(rag50_ids_all) >= 40 else rag50_ids_all[:20]
    target_ids = critical_ids + rag50_sample
    print(f"critical_ids={len(critical_ids)} rag50_sample={len(rag50_sample)} total={len(target_ids)}")

    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    out = []
    for pid in target_ids:
        claims = gt["claims_by_response"][pid]
        context = ctx_by_id[pid]
        user_text = build_user_text(context, claims)
        gen_start = time.time()
        raw = generate(model, tokenizer, VERIFIER_SYSTEM_PROMPT, user_text, seed=42)
        elapsed = time.time() - gen_start
        parsed = try_parse_json(raw)
        out.append({"id": pid, "n_claims": len(claims), "raw_output": raw, "parsed": parsed,
                     "parse_success": parsed is not None, "latency_sec": elapsed})
        print(f"[E-verify] {pid} n_claims={len(claims)} parse_success={parsed is not None} t={elapsed:.1f}s")

    (REPORTS_DIR / "phase4fw_model_verifier_raw.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MODEL VERIFIER DONE")


if __name__ == "__main__":
    main()
