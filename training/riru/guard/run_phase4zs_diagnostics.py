"""Phase4ZS Stage A/B(Q6 reproduction+sampling)/C(adapter attribution)/D(prompt attribution)/
E(context attribution) をまとめて実行する。Phase4ZG read-only、baseはPeftModel.disable_adapter()
で切り替える(base modelを別途ロードし直さない)。RAG DB/retriever/embeddingは読み取り専用。"""
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
sys.path.insert(0, str(TRAINING_ROOT / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
REPORTS_DIR = TRAINING_ROOT / "reports"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ZG_ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")
STRICT_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
MINIMAL_PROMPT_PATH = GUARD_DIR / "phase4zs_minimal_grounding_prompt.txt"

Q6_QUERY = "GGとSGGの違いを初心者向けに説明して"
NUMERIC_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?パーセント|1/\d+")
CONTRADICTION_PATTERN = re.compile(
    r"(登録データ|情報)[^。！？]{0,15}(ない|なかった|見つから)[^。！？]{0,40}(だけど|けど|が|、)[^。！？]{0,60}"
    r"(\d+(?:\.\d+)?\s*%|1/\d+|\d+パーセント)"
)


def load_zg_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_zs")
    model.eval()
    return model, tokenizer


def load_precomputed_contexts():
    # venv分離のため、retrieval(.venv側)はphase4zs_precompute_contexts.pyで事前計算済み。
    # このscript(.venv-qlora側、GPU generation用)はretrieverを直接importせず、そのJSONを読むだけ。
    return json.loads((REPORTS_DIR / "phase4zs_precomputed_contexts.json").read_text(encoding="utf-8"))


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
    text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
    return text


def analyze(text):
    numerics = NUMERIC_PATTERN.findall(text)
    contradiction = bool(CONTRADICTION_PATTERN.search(text))
    return {"response": text, "numerics_in_output": numerics, "has_numeric_claim": len(numerics) > 0,
            "contradictory_self_awareness": contradiction}


def stage_ab_q6_reproduction(model, tokenizer, strict_prompt, context):
    results_greedy = []
    for i in range(10):
        text = generate(model, tokenizer, strict_prompt, Q6_QUERY, context=context, seed=i, do_sample=False)
        results_greedy.append({"run": i, **analyze(text)})
    results_prod = []
    for i in range(30):
        text = generate(model, tokenizer, strict_prompt, Q6_QUERY, context=context, seed=1000 + i,
                         do_sample=True, temperature=0.7, max_new_tokens=512)
        results_prod.append({"run": i, **analyze(text)})

    def summarize(rows):
        n = len(rows)
        return {"n": n, "unsupported_numeric_count": sum(1 for r in rows if r["has_numeric_claim"]),
                "unsupported_numeric_rate": sum(1 for r in rows if r["has_numeric_claim"]) / n,
                "contradiction_count": sum(1 for r in rows if r["contradictory_self_awareness"]),
                "contradiction_rate": sum(1 for r in rows if r["contradictory_self_awareness"]) / n}

    out = {"purpose": "Stage A/B: Q6を実context付きで再現。greedy10回+production sampling30回。",
           "context_used": context, "greedy": {"summary": summarize(results_greedy), "rows": results_greedy},
           "production_sampling": {"summary": summarize(results_prod), "rows": results_prod}}
    (REPORTS_DIR / "phase4zs_q6_reproduction.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "phase4zs_sampling_comparison.json").write_text(json.dumps({
        "purpose": "Stage B: sampling効果の比較(greedy vs production temperature=0.7)。",
        "greedy_summary": summarize(results_greedy), "production_summary": summarize(results_prod),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Stage A/B done:", summarize(results_greedy), summarize(results_prod))


def stage_c_adapter_attribution(model, tokenizer, strict_prompt, probes_with_context):
    rows = []
    for p in probes_with_context:
        zg_text = generate(model, tokenizer, strict_prompt, p["prompt"], context=p["context"], seed=42)
        with model.disable_adapter():
            base_text = generate(model, tokenizer, strict_prompt, p["prompt"], context=p["context"], seed=42)
        rows.append({"probe_id": p["probe_id"], "prompt": p["prompt"],
                     "phase4zg": analyze(zg_text), "base_qwen": analyze(base_text)})

    def summarize(key):
        n = len(rows)
        cnt = sum(1 for r in rows if r[key]["has_numeric_claim"])
        return {"n": n, "unsupported_numeric_count": cnt, "unsupported_numeric_rate": cnt / n}

    out = {"purpose": "Stage C: Phase4ZG vs Base Qwen2.5-14B-Instruct、同一prompt/context/seedで比較。",
           "phase4zg_summary": summarize("phase4zg"), "base_qwen_summary": summarize("base_qwen"), "rows": rows}
    (REPORTS_DIR / "phase4zs_adapter_comparison.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Stage C done:", summarize("phase4zg"), summarize("base_qwen"))


def stage_d_prompt_attribution(model, tokenizer, strict_prompt, minimal_prompt, probes_with_context):
    rows = []
    for p in probes_with_context:
        strict_text = generate(model, tokenizer, strict_prompt, p["prompt"], context=p["context"], seed=42)
        minimal_text = generate(model, tokenizer, minimal_prompt, p["prompt"], context=p["context"], seed=42)
        rows.append({"probe_id": p["probe_id"], "prompt": p["prompt"],
                     "current_strict_prompt": analyze(strict_text), "minimal_grounding_prompt": analyze(minimal_text)})

    def summarize(key):
        n = len(rows)
        cnt = sum(1 for r in rows if r[key]["has_numeric_claim"])
        return {"n": n, "unsupported_numeric_count": cnt, "unsupported_numeric_rate": cnt / n}

    out = {"purpose": "Stage D: 既存strict RAG promptと、診断専用minimal grounding promptの比較"
                       "(productionプロンプトは変更していない、診断用コピーのみ使用)。",
           "current_prompt_summary": summarize("current_strict_prompt"),
           "minimal_prompt_summary": summarize("minimal_grounding_prompt"), "rows": rows}
    (REPORTS_DIR / "phase4zs_prompt_attribution.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Stage D done:", summarize("current_strict_prompt"), summarize("minimal_grounding_prompt"))


def stage_e_context_attribution(model, tokenizer, strict_prompt, titles, texts, full_context):
    text_only = "\n".join(texts)
    minimized = "\n".join(texts[:2])  # 最も関連度の高い2件のみ

    conditions = {
        "A_full_production_context": full_context,
        "B_structured_titles_only": "\n".join(titles),
        "C_text_only_no_titles": text_only,
        "D_manually_minimized_top2": minimized,
    }
    rows = {}
    for name, ctx in conditions.items():
        text = generate(model, tokenizer, strict_prompt, Q6_QUERY, context=ctx, seed=42)
        rows[name] = {"context_used": ctx, **analyze(text)}

    out = {"purpose": "Stage E: Q6についてcontext構造を切り分けた比較(RAG DB/retriever自体は変更せず、"
                       "offline diagnostic inputとしてcontextを切り分けたのみ)。",
           "rows": rows}
    (REPORTS_DIR / "phase4zs_context_attribution.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Stage E done:", {k: v["has_numeric_claim"] for k, v in rows.items()})


def main():
    strict_prompt = STRICT_PROMPT_PATH.read_text(encoding="utf-8")
    minimal_prompt = MINIMAL_PROMPT_PATH.read_text(encoding="utf-8")

    precomputed = load_precomputed_contexts()
    q6_context = precomputed["ZS-Q6"]["context"]
    q6_titles = precomputed["ZS-Q6"]["titles"]
    q6_texts = precomputed["ZS-Q6"]["texts"]

    gt = json.loads((REPORTS_DIR / "phase4zs_ground_truth.json").read_text(encoding="utf-8"))
    zs_new_rows = [r for r in gt["rows"] if r["source"] == "zs_new"]

    def with_context(rows):
        return [{"probe_id": r["probe_id"], "prompt": r["prompt"], "context": precomputed[r["probe_id"]]["context"]}
                for r in rows]

    stage_c_probes = zs_new_rows[:9]  # + Q6 = 10 total
    stage_d_probes = zs_new_rows  # + Q6 = 21 total

    stage_c_with_ctx = [{"probe_id": "ZS-Q6", "prompt": Q6_QUERY, "context": q6_context}] + with_context(stage_c_probes)
    stage_d_with_ctx = [{"probe_id": "ZS-Q6", "prompt": Q6_QUERY, "context": q6_context}] + with_context(stage_d_probes)

    model, tokenizer = load_zg_and_tokenizer()
    print(f"model loaded, t={time.time():.0f}")

    stage_ab_q6_reproduction(model, tokenizer, strict_prompt, q6_context)
    stage_c_adapter_attribution(model, tokenizer, strict_prompt, stage_c_with_ctx)
    stage_d_prompt_attribution(model, tokenizer, strict_prompt, minimal_prompt, stage_d_with_ctx)
    stage_e_context_attribution(model, tokenizer, strict_prompt, q6_titles, q6_texts, q6_context)

    print("ALL STAGES DONE")


if __name__ == "__main__":
    main()
