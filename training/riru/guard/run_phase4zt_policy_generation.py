"""Phase4ZT Stage B(Q6 regression x3 policies)/C(UNKNOWN77 x3 policies)。
Phase4ZG read-only。conservative dispatchは呼ばない(既にUNKNOWN確定済みのprobeのみ対象)。"""
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
sys.path.insert(0, str(GUARD_DIR))
REPORTS_DIR = TRAINING_ROOT / "reports"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ZG_ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")
STRICT_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
CLARIFY_PROMPT_PATH = GUARD_DIR / "phase4zr_unknown_ux_prompt_b.txt"

from phase4zt_policy_c import decide_c1, decide_c2, decide_c3  # noqa: E402

Q6_QUERY = "GGとSGGの違いを初心者向けに説明して"
NUMERIC_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?パーセント|1/\d+")
CONTRADICTION_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?\s*%|1/\d+|\d+パーセント)[^。！？]{0,80}(登録データ|情報)[^。！？]{0,20}(ない|なかった|見つから)"
    r"|(登録データ|情報)[^。！？]{0,20}(ない|なかった|見つから)[^。！？]{0,80}(\d+(?:\.\d+)?\s*%|1/\d+|\d+パーセント)"
)
HEDGE_RE = re.compile("|".join(re.escape(p) for p in
    ["登録データ", "データベース", "データがない", "登録されていない", "情報がない", "記録がない", "確認できない"]))
BOUNDARY_RE = re.compile("|".join(re.escape(p) for p in ["専門外", "パチスロ", "スロット", "専門分野", "専門家"]))
PLACEHOLDER_RE = re.compile(r"パチスロ[〇○×××]|パチスロ[A-Z]{1,3}(?:機|台)?(?=[「」、。\s]|$)")


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_zt")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system_prompt, user_text, context=None, seed=42,
             do_sample=False, temperature=None, max_new_tokens=300):
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


def respond_via_policy(model, tokenizer, strict_prompt, clarify_prompt, variant, query, context, titles, texts,
                        seed=42, do_sample=False, temperature=None, max_new_tokens=300):
    if variant == "C1":
        decision = decide_c1(query, context)
    elif variant == "C2":
        decision = decide_c2(query, context)
    else:
        decision = decide_c3(query, context, titles, texts)

    if decision.selected_path == "rag_with_context":
        text = generate(model, tokenizer, strict_prompt, query, context=context, seed=seed,
                         do_sample=do_sample, temperature=temperature, max_new_tokens=max_new_tokens)
    else:
        text = generate(model, tokenizer, clarify_prompt, query, context=None, seed=seed,
                         do_sample=do_sample, temperature=temperature, max_new_tokens=max_new_tokens)
    return text, decision


def analyze(text):
    numerics = NUMERIC_PATTERN.findall(text)
    return {"response": text, "numerics_in_output": numerics, "has_numeric_claim": len(numerics) > 0,
            "contradictory_self_awareness": bool(CONTRADICTION_PATTERN.search(text)),
            "hedge": bool(HEDGE_RE.search(text)), "boundary_marker": bool(BOUNDARY_RE.search(text)),
            "placeholder_machine_name": bool(PLACEHOLDER_RE.search(text))}


def stage_b_q6_regression(model, tokenizer, strict_prompt, clarify_prompt, context, titles, texts):
    out = {}
    for variant in ("C1", "C2", "C3"):
        greedy = []
        for i in range(10):
            text, decision = respond_via_policy(model, tokenizer, strict_prompt, clarify_prompt, variant,
                                                  Q6_QUERY, context, titles, texts, seed=i, do_sample=False)
            greedy.append({"run": i, "path": decision.selected_path, **analyze(text)})
        prod = []
        for i in range(20):
            text, decision = respond_via_policy(model, tokenizer, strict_prompt, clarify_prompt, variant,
                                                  Q6_QUERY, context, titles, texts, seed=1000 + i,
                                                  do_sample=True, temperature=0.7, max_new_tokens=512)
            prod.append({"run": i, "path": decision.selected_path, **analyze(text)})

        def summarize(rows):
            n = len(rows)
            return {"n": n, "unsupported_numeric_count": sum(1 for r in rows if r["has_numeric_claim"]),
                    "contradiction_count": sum(1 for r in rows if r["contradictory_self_awareness"])}
        out[variant] = {"greedy": {"summary": summarize(greedy), "rows": greedy},
                         "production": {"summary": summarize(prod), "rows": prod}}
        print(f"Stage B {variant}: greedy={summarize(greedy)} prod={summarize(prod)}")
    (REPORTS_DIR / "phase4zt_q6_regression.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_c_unknown77(model, tokenizer, strict_prompt, clarify_prompt, precomputed):
    unk_items = {k: v for k, v in precomputed.items() if k != "ZS-Q6"}
    out = {}
    for variant in ("C1", "C2", "C3"):
        rows = []
        for pid, c in unk_items.items():
            text, decision = respond_via_policy(model, tokenizer, strict_prompt, clarify_prompt, variant,
                                                  c["prompt"], c["context"], c["titles"], c["texts"], seed=42)
            rows.append({"probe_id": pid, "expected_mode": c["expected_mode"], "prompt": c["prompt"],
                         "path": decision.selected_path, "lexical_overlap": decision.lexical_overlap_tokens,
                         **analyze(text)})
        out[variant] = rows
        n = len(rows)
        unsup = sum(1 for r in rows if r["has_numeric_claim"])
        print(f"Stage C {variant}: n={n} unsupported_numeric={unsup}")
    (REPORTS_DIR / "phase4zt_unknown77.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    strict_prompt = STRICT_PROMPT_PATH.read_text(encoding="utf-8")
    clarify_prompt = CLARIFY_PROMPT_PATH.read_text(encoding="utf-8")
    precomputed = json.loads((REPORTS_DIR / "phase4zt_precomputed_contexts.json").read_text(encoding="utf-8"))
    q6 = precomputed["ZS-Q6"]

    model, tokenizer = load_model()
    print(f"model loaded, t={time.time():.0f}")

    stage_b_q6_regression(model, tokenizer, strict_prompt, clarify_prompt, q6["context"], q6["titles"], q6["texts"])
    stage_c_unknown77(model, tokenizer, strict_prompt, clarify_prompt, precomputed)

    print("ALL STAGES DONE")


if __name__ == "__main__":
    main()
