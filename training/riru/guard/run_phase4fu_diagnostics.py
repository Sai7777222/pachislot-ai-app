"""Phase4FU: Stage B(全31probeベース生成)/C(context構造ablation)/D(prompt比較)/
E(adapter比較)/F(query style比較)/G(insufficient-context stress test)。
Phase4ZG/dispatch/PolicyC3/RAG DB/本番prompt/generation configは一切変更しない(read-onlyのみ)。
GPU budget <=180 new generations。"""
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
STRICT_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
MINIMAL_PROMPT_PATH = GUARD_DIR / "phase4fu_minimal_grounding_prompt.txt"

GENERATION_COUNT = 0  # running tally, printed at the end for budget verification

# ---- Stage-selection lists (probe ids from phase4fu_ground_truth.json) ----
STAGE_C_PROBE_IDS = ["FU-D01", "FU-B05", "FU-A01", "FU-A02", "FU-A03", "FU-A04",
                      "FU-C01", "FU-C02", "FU-D02", "FU-E02", "FU-E04", "FU-F01"]
STAGE_E_CORE_IDS = ["FU-D01", "FU-B05"]  # Q6, ZS-05: >=10 reps each, both adapters
STAGE_E_EXTRA_IDS = ["FU-A03", "FU-B03", "FU-D05", "FU-F03"]  # 1 rep each, both adapters
STAGE_G_PROBE_IDS = ["FU-A02", "FU-A03", "FU-A05", "FU-B03", "FU-B04", "FU-B05",
                      "FU-C03", "FU-D05", "FU-F01", "FU-F02"]  # INSUFFICIENT/IRRELEVANT, 2 reps each


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
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_fu")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system_prompt, user_text, context=None, seed=42,
             do_sample=False, temperature=None, max_new_tokens=350):
    global GENERATION_COUNT
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
    GENERATION_COUNT += 1
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


# ---- context-structure ablation conditions (Stage C) ----

def condition_c1(chunks):
    """現在の本番実装と同じ: retrievalスコア順の全chunkをそのまま結合。"""
    return build_context_string(chunks)


def condition_c2(chunks, gt_row):
    """関連chunkのみ: GTのsupported_entitiesに言及があるchunkだけ残す。皆無ならC1と同一(フォールバック)。"""
    entities = gt_row.get("supported_entities", [])
    if not entities:
        return build_context_string(chunks)
    kept = [c for c in chunks if any(e and e in (c["title"] + c["text"]) for e in entities)]
    if not kept:
        return build_context_string(chunks)
    return build_context_string(kept)


def condition_c3(chunks, prompt):
    """entity-grouped: promptに登場する語を含むchunkを先頭にグルーピングし直す。"""
    def relevance(c):
        text = c["title"] + c["text"]
        return -sum(1 for token in re.findall(r"[一-龥ァ-ヶーA-Za-z0-9]{2,}", prompt) if token in text)
    ordered = sorted(chunks, key=relevance)
    return build_context_string(ordered)


NUMERIC_CHAR_RE = re.compile(r"\d")


def condition_c4(chunks):
    """structured-only: 数字・記号を含む(表・確率的)chunkのみ。無ければC1にフォールバック。"""
    kept = [c for c in chunks if NUMERIC_CHAR_RE.search(c["text"])]
    if not kept:
        return build_context_string(chunks)
    return build_context_string(kept)


def condition_c5(chunks):
    """explanation-only: 数字を含まない説明文寄りのchunkのみ。無ければC1にフォールバック。"""
    kept = [c for c in chunks if not NUMERIC_CHAR_RE.search(c["text"])]
    if not kept:
        return build_context_string(chunks)
    return build_context_string(kept)


NUMERIC_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?パーセント|1/\d+")
HEDGE_RE = re.compile("|".join(re.escape(p) for p in
    ["登録データ", "データベース", "データがない", "登録されていない", "情報がない", "記録がない", "確認できない", "見つかりません"]))


def analyze(text):
    return {"response": text, "has_numeric_claim": bool(NUMERIC_PATTERN.search(text)),
            "abstain_or_hedge": bool(HEDGE_RE.search(text)), "length": len(text)}


def main():
    strict_prompt = STRICT_PROMPT_PATH.read_text(encoding="utf-8")
    minimal_prompt = MINIMAL_PROMPT_PATH.read_text(encoding="utf-8")
    all_probes = json.loads((REPORTS_DIR / "phase4fu_precomputed_contexts.json").read_text(encoding="utf-8"))
    gt = json.loads((REPORTS_DIR / "phase4fu_ground_truth.json").read_text(encoding="utf-8"))
    gt_by_id = {r["id"]: r for r in gt["rows"]}
    probes_by_id = {p["id"]: p for p in all_probes}
    stage_f_probes = json.loads((REPORTS_DIR / "phase4fu_stage_f_contexts.json").read_text(encoding="utf-8"))

    model, tokenizer = load_model()
    print(f"model loaded t={time.time():.0f}")

    # ---------------- Stage B: base run, all 31 probes, production prompt, ZG, greedy ----------------
    stage_b = []
    for p in all_probes:
        ctx = build_context_string(p["retrieved_chunks"])
        text = generate(model, tokenizer, strict_prompt, p["prompt"], context=ctx, seed=42, do_sample=False)
        stage_b.append({"id": p["id"], "category": p["category"], "prompt": p["prompt"], **analyze(text)})
        print(f"[StageB] {p['id']} n_gen={GENERATION_COUNT} hedge={stage_b[-1]['abstain_or_hedge']}")
    (REPORTS_DIR / "phase4fu_stage_b_base_generations.json").write_text(
        json.dumps(stage_b, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- Stage C: context-structure ablation ----------------
    stage_c = []
    for pid in STAGE_C_PROBE_IDS:
        p = probes_by_id[pid]
        gt_row = gt_by_id[pid]
        chunks = p["retrieved_chunks"]
        conditions = {
            "C1_current_assembled": condition_c1(chunks),
            "C2_relevant_only": condition_c2(chunks, gt_row),
            "C3_entity_grouped": condition_c3(chunks, p["prompt"]),
            "C4_structured_only": condition_c4(chunks),
            "C5_explanation_only": condition_c5(chunks),
        }
        row = {"id": pid, "prompt": p["prompt"], "retrieval_sufficiency": gt_row["retrieval_sufficiency"], "conditions": {}}
        for cname, ctx in conditions.items():
            text = generate(model, tokenizer, strict_prompt, p["prompt"], context=ctx, seed=42, do_sample=False)
            row["conditions"][cname] = {"context_used": ctx, **analyze(text)}
            print(f"[StageC] {pid}/{cname} n_gen={GENERATION_COUNT}")
        stage_c.append(row)
    (REPORTS_DIR / "phase4fu_context_structure.json").write_text(
        json.dumps(stage_c, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- Stage D: prompt attribution (minimal-grounding vs production) ----------------
    stage_d = []
    for pid in STAGE_C_PROBE_IDS:
        p = probes_by_id[pid]
        ctx = build_context_string(p["retrieved_chunks"])
        production_text = next(r for r in stage_b if r["id"] == pid)  # reuse Stage B production result
        minimal_text = generate(model, tokenizer, minimal_prompt, p["prompt"], context=ctx, seed=42, do_sample=False)
        stage_d.append({"id": pid, "prompt": p["prompt"],
                         "production_prompt_result": {k: production_text[k] for k in ("response", "has_numeric_claim", "abstain_or_hedge")},
                         "minimal_prompt_result": analyze(minimal_text)})
        print(f"[StageD] {pid} n_gen={GENERATION_COUNT}")
    (REPORTS_DIR / "phase4fu_prompt_attribution.json").write_text(
        json.dumps(stage_d, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- Stage E: adapter attribution (ZG vs Base) ----------------
    stage_e = {}
    for pid in STAGE_E_CORE_IDS:
        p = probes_by_id[pid]
        ctx = build_context_string(p["retrieved_chunks"])
        zg_runs, base_runs = [], []
        for i in range(10):
            text = generate(model, tokenizer, strict_prompt, p["prompt"], context=ctx, seed=i,
                             do_sample=(i > 0), temperature=0.7 if i > 0 else None)
            zg_runs.append({"run": i, **analyze(text)})
        with model.disable_adapter():
            for i in range(10):
                text = generate(model, tokenizer, strict_prompt, p["prompt"], context=ctx, seed=i,
                                 do_sample=(i > 0), temperature=0.7 if i > 0 else None)
                base_runs.append({"run": i, **analyze(text)})
        stage_e[pid] = {"prompt": p["prompt"], "zg_adapter_runs": zg_runs, "base_qwen_runs": base_runs}
        print(f"[StageE-core] {pid} n_gen={GENERATION_COUNT}")

    for pid in STAGE_E_EXTRA_IDS:
        p = probes_by_id[pid]
        ctx = build_context_string(p["retrieved_chunks"])
        zg_text = generate(model, tokenizer, strict_prompt, p["prompt"], context=ctx, seed=42, do_sample=False)
        with model.disable_adapter():
            base_text = generate(model, tokenizer, strict_prompt, p["prompt"], context=ctx, seed=42, do_sample=False)
        stage_e[pid] = {"prompt": p["prompt"],
                         "zg_adapter_runs": [{"run": 0, **analyze(zg_text)}],
                         "base_qwen_runs": [{"run": 0, **analyze(base_text)}]}
        print(f"[StageE-extra] {pid} n_gen={GENERATION_COUNT}")
    (REPORTS_DIR / "phase4fu_adapter_attribution.json").write_text(
        json.dumps(stage_e, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- Stage F: query-style attribution (5 phrasings, same underlying facts) ----------------
    stage_f = []
    d01 = probes_by_id["FU-D01"]
    d01_ctx = build_context_string(d01["retrieved_chunks"])
    d01_text = next(r for r in stage_b if r["id"] == "FU-D01")
    stage_f.append({"phrasing": "GGとSGGの違いを初心者向けに説明して(=Q6/FU-D01)", **{k: d01_text[k] for k in ("response", "has_numeric_claim", "abstain_or_hedge")}})
    for sp in stage_f_probes:
        ctx = build_context_string(sp["retrieved_chunks"])
        text = generate(model, tokenizer, strict_prompt, sp["prompt"], context=ctx, seed=42, do_sample=False)
        stage_f.append({"phrasing": sp["prompt"], **analyze(text)})
        print(f"[StageF] {sp['id']} n_gen={GENERATION_COUNT}")
    (REPORTS_DIR / "phase4fu_query_style.json").write_text(
        json.dumps(stage_f, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- Stage G: insufficient-context stress test ----------------
    stage_g = []
    for pid in STAGE_G_PROBE_IDS:
        p = probes_by_id[pid]
        gt_row = gt_by_id[pid]
        ctx = build_context_string(p["retrieved_chunks"])
        for i, (sample, temp) in enumerate([(False, None), (True, 0.7)]):
            text = generate(model, tokenizer, strict_prompt, p["prompt"], context=ctx, seed=100 + i,
                             do_sample=sample, temperature=temp)
            stage_g.append({"id": pid, "prompt": p["prompt"], "retrieval_sufficiency": gt_row["retrieval_sufficiency"],
                             "run": i, **analyze(text)})
        print(f"[StageG] {pid} n_gen={GENERATION_COUNT}")
    (REPORTS_DIR / "phase4fu_insufficient_context_stress.json").write_text(
        json.dumps(stage_g, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"ALL STAGES DONE. total_new_generations={GENERATION_COUNT}")


if __name__ == "__main__":
    main()
