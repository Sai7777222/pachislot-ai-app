# -*- coding: utf-8 -*-
"""Phase4FX: A0(現行、既存フェーズ4FU/4FVのデータを再利用)/A1(entity-grouped、新規generation)/
A2(query-bound-only、新規generation)の比較。production RAG promptは一切変更しない。"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

import torch

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
PROJECT_ROOT = TRAINING_ROOT.parents[1]
REPORTS_DIR = TRAINING_ROOT / "reports"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ZG_ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")
PRODUCTION_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

NUMERIC_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?パーセント|1/\d+(?:\.\d+)?|\d+G|\d+枚")
HEDGE_RE = re.compile("|".join(re.escape(p) for p in
    ["登録データ", "データベース", "データがない", "登録されていない", "情報がない", "記録がない",
     "確認できない", "見つかりません", "見つからない", "NO GROUNDED EVIDENCE"]))


def analyze(text):
    return {"response": text, "has_numeric_claim": bool(NUMERIC_PATTERN.search(text)),
            "abstain_or_hedge": bool(HEDGE_RE.search(text)), "length": len(text)}


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_fx")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system_prompt, user_text, context, seed=42, max_new_tokens=350):
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": f"[検索結果]\n{context}"})
    messages.append({"role": "user", "content": user_text})
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    with torch.no_grad():
        output_ids = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


# 既存フェーズのA0(production prompt, 現行embedding top-k)応答を再利用するためのマッピング
def load_a0_reuse_map():
    fu_stage_b = {r["id"]: r["response"] for r in
                  json.loads((REPORTS_DIR / "phase4fu_stage_b_base_generations.json").read_text(encoding="utf-8"))}
    fv_phantom = {r["id"]: r["P0"]["response"] for r in
                  json.loads((REPORTS_DIR / "phase4fv_phantom_entity.json").read_text(encoding="utf-8"))}
    fv_concept = {r["id"]: r["P0"]["response"] for r in
                  json.loads((REPORTS_DIR / "phase4fv_concept_binding.json").read_text(encoding="utf-8"))}
    fu_stage_f = json.loads((REPORTS_DIR / "phase4fu_query_style.json").read_text(encoding="utf-8"))
    rag50_baseline = {r["probe_id"]: r["response"] for r in
                       json.loads((REPORTS_DIR / "phase4zn_rag50_raw.json").read_text(encoding="utf-8"))}
    reuse: dict[str, str] = {}
    reuse.update(fu_stage_b)
    reuse.update(fv_phantom)
    reuse.update(fv_concept)
    reuse.update(rag50_baseline)
    # query_style: keyed by phrasing text, need id mapping done at call site
    reuse["_query_style_by_phrasing"] = {r["phrasing"]: r.get("response") for r in fu_stage_f}
    reuse["_query_style_by_phrasing"]["GGとSGGの違いを初心者向けに説明して(=Q6)"] = fu_stage_b.get("FU-D01")
    return reuse


# known_failure/concept_binding IDのA0再利用は、元のprobe id(FU-*/FV-*)を経由する
KF_ID_TO_SOURCE = {
    "FX-K01": "FU-D01", "FX-K02": "FU-D03", "FX-K03": "FU-E02", "FX-K06": "FV-C03",
    "FX-K07": "FU-A03", "FX-K08": "FU-B05", "FX-K05_alt": "FV-C05",
}
CB_ID_TO_SOURCE = {"CB-FU-D01": "FU-D01", "CB-FU-D03": "FU-D03", "CB-FU-D05": "FU-D05"}


def main():
    production_prompt = PRODUCTION_PROMPT_PATH.read_text(encoding="utf-8")
    assembled = json.loads((REPORTS_DIR / "phase4fx_context_assembly.json").read_text(encoding="utf-8"))
    reuse = load_a0_reuse_map()
    qs_by_phrasing = reuse.pop("_query_style_by_phrasing")

    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    out = []
    new_gen_count = 0
    for i, p in enumerate(assembled):
        pid, cat = p["id"], p["category"]
        row = {"id": pid, "category": cat, "prompt": p["prompt"], "query_entities": p.get("query_entities", [])}

        # ---- A0: 可能な限り既存データを再利用 ----
        a0_text = None
        if cat == "known_failure":
            src = KF_ID_TO_SOURCE.get(pid) or (KF_ID_TO_SOURCE.get(f"{pid}_alt"))
            if pid == "FX-K05":  # GG当選とSGG当選の違い = FV-C05
                a0_text = reuse.get("FV-C05")
            elif src:
                a0_text = reuse.get(src)
        elif cat == "concept_binding" and pid in CB_ID_TO_SOURCE:
            a0_text = reuse.get(CB_ID_TO_SOURCE[pid])
        elif cat == "query_style":
            a0_text = qs_by_phrasing.get(p["prompt"]) or reuse.get("FU-D01")
        else:
            a0_text = reuse.get(pid)

        if a0_text is not None:
            row["A0"] = {**analyze(a0_text), "latency_sec": None, "source": "reused"}
        else:
            text = generate(model, tokenizer, production_prompt, p["prompt"], context=p["A0"], seed=42)
            row["A0"] = {**analyze(text), "latency_sec": None, "source": "new"}
            new_gen_count += 1

        # ---- A1: known_failureのみ新規生成(3-way比較), 他はA2のみ ----
        if cat == "known_failure":
            t0 = time.time()
            text = generate(model, tokenizer, production_prompt, p["prompt"], context=p["A1"], seed=42)
            row["A1"] = {**analyze(text), "latency_sec": round(time.time() - t0, 2)}
            new_gen_count += 1

        # ---- A2: 常に新規生成 ----
        t0 = time.time()
        text = generate(model, tokenizer, production_prompt, p["prompt"], context=p["A2"], seed=42)
        row["A2"] = {**analyze(text), "latency_sec": round(time.time() - t0, 2)}
        new_gen_count += 1

        out.append(row)
        print(f"[{i+1}/{len(assembled)}] {pid} ({cat}) done, total_new_gen={new_gen_count}")

    (REPORTS_DIR / "phase4fx_generation_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"GENERATION DONE total_new_generations={new_gen_count}")


if __name__ == "__main__":
    main()
