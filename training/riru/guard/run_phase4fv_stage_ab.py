"""Phase4FV Stage A(known failure regression) + Stage B(query style regression)。
P0(現行)/P1(minimal grounding追加)/P2(explicit entity binding追加)の3候補を比較する。
P0はPhase4FUの既存生成データを可能な限り再利用し、新規はP1/P2のみ生成する。"""
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
P0_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
P1_PATH = GUARD_DIR / "phase4fv_prompts" / "p1_minimal_grounding.jinja2"
P2_PATH = GUARD_DIR / "phase4fv_prompts" / "p2_explicit_entity_binding.jinja2"

NUMERIC_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?パーセント|1/\d+")
HEDGE_RE = re.compile("|".join(re.escape(p) for p in
    ["登録データ", "データベース", "データがない", "登録されていない", "情報がない", "記録がない", "確認できない", "見つかりません", "見つからない"]))


def build_context_string(chunks):
    return "\n".join(f"[{c['title']}] {c['text']}" for c in chunks)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_fv")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system_prompt, user_text, context=None, seed=42,
             do_sample=False, temperature=None, max_new_tokens=350):
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": f"[検索結果]\n{context}"})
    messages.append({"role": "user", "content": user_text})
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample,
                  "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id}
    if do_sample and temperature is not None:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        output_ids = model.generate(**encoded, **gen_kwargs)
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


def analyze(text):
    return {"response": text, "has_numeric_claim": bool(NUMERIC_PATTERN.search(text)),
            "abstain_or_hedge": bool(HEDGE_RE.search(text)), "length": len(text)}


def main():
    p0 = P0_PATH.read_text(encoding="utf-8")
    p1 = P1_PATH.read_text(encoding="utf-8")
    p2 = P2_PATH.read_text(encoding="utf-8")

    fu_precomputed = json.loads((REPORTS_DIR / "phase4fu_precomputed_contexts.json").read_text(encoding="utf-8"))
    fu_by_id = {p["id"]: p for p in fu_precomputed}
    fu_stage_f = json.loads((REPORTS_DIR / "phase4fu_query_style.json").read_text(encoding="utf-8"))
    fu_stage_f_contexts = json.loads((REPORTS_DIR / "phase4fu_stage_f_contexts.json").read_text(encoding="utf-8"))

    # Stage A known failures: Q6=FU-D01, ZS-05=FU-B05, AT-F=FU-A03, gaiabell=FU-E02
    known_failure_ids = ["FU-D01", "FU-B05", "FU-A03", "FU-E02"]

    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    # ---- Stage A ----
    stage_a = []
    for pid in known_failure_ids:
        p = fu_by_id[pid]
        ctx = build_context_string(p["retrieved_chunks"])
        row = {"id": pid, "prompt": p["prompt"]}
        for cname, cprompt in (("P0", p0), ("P1", p1), ("P2", p2)):
            text = generate(model, tokenizer, cprompt, p["prompt"], context=ctx, seed=42, do_sample=False)
            row[cname] = analyze(text)
            print(f"[StageA] {pid}/{cname} done")
        stage_a.append(row)
    (REPORTS_DIR / "phase4fv_known_failures.json").write_text(
        json.dumps(stage_a, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Stage B: query style (5 phrasings, GG/SGG facts). P0 baseline reused from Phase4FU query_style.json ----
    d01 = fu_by_id["FU-D01"]
    d01_ctx = build_context_string(d01["retrieved_chunks"])
    phrasings = [("GGとSGGの違いを初心者向けに説明して(=Q6)", d01_ctx)]
    for sp in fu_stage_f_contexts:
        phrasings.append((sp["prompt"], build_context_string(sp["retrieved_chunks"])))

    stage_b = []
    for phrasing, ctx in phrasings:
        row = {"phrasing": phrasing}
        row["P0_reused_from_FU"] = next(
            (r for r in fu_stage_f if r["phrasing"] == phrasing), None)
        for cname, cprompt in (("P1", p1), ("P2", p2)):
            text = generate(model, tokenizer, cprompt, phrasing.split("(=")[0], context=ctx, seed=42, do_sample=False)
            row[cname] = analyze(text)
        stage_b.append(row)
        print(f"[StageB] {phrasing[:20]} done")
    (REPORTS_DIR / "phase4fv_query_style.json").write_text(
        json.dumps(stage_b, ensure_ascii=False, indent=2), encoding="utf-8")

    print("STAGE A/B DONE")


if __name__ == "__main__":
    main()
