"""Phase 4S: complex multi-fact教師比率対照実験 本評価。

A=Base / B=v4 / C=ratio_mid(~5.7%) / D=ratio_high(~11.0%) の4条件で、
Phase4O/4P/4Qと同一のQ3/P01/P02/P04/Q9/Q11/E36/persona/structured17/character39を
評価する。generation設定はPhase4O/4P/4Qと完全一致 (max_new_tokens=300,
temperature=0.3, top_p=0.9, 5seed=42-46)。

QLoRA/LoRA学習は行わない。adapterは読み取り専用でロードするのみ。
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parents[1]
EVAL_DIR = Path(__file__).resolve().parent

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_V4_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v4")
ADAPTER_MID_PATH = str(TRAINING_ROOT / "lora-riru-qwen-ratio-mid")
ADAPTER_HIGH_PATH = str(TRAINING_ROOT / "lora-riru-qwen-ratio-high")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
SEED_DEFAULT = 42
SEEDS = (42, 43, 44, 45, 46)

CONDITIONS = ("A_base", "B_v4", "C_mid", "D_high")
ADAPTER_NAME_FOR_CONDITION = {"B_v4": "v4", "C_mid": "mid", "D_high": "high"}

Q3_KEY_FACTS = ["510G", "1000G", "1480G", "15.2%", "20.3%", "64.5%"]
Q9_CALC_PATTERN = re.compile(r"約\s*\d+(\.\d+)?\s*(倍|ポイント)")
Q11_YAMEDOKI_PATTERN = re.compile(r"ヤメ時|一旦ヤメ|止めるのが|ヤメる")
Q11_STRATEGY_PATTERN = re.compile(r"おすすめ|べきです|べきだ|戦略|コツ")
Q11_LOOPSTOCK_CAUSAL_PATTERN = re.compile(r"ループストック.{0,15}(ほど|により|によって)")
Q11_OTHER_CAUSAL_PATTERN = re.compile(r"(可能性が高くなり|なりやすく|傾向にあり).{0,10}(ため|から)")

WRONG_NAMES = [
    "リリ", "リサ", "リコ", "あいり", "あいこ", "ゆめぴょん", "ゆめちゃん",
    "ピコ", "ピッコロ", "ぴよこ", "パティ", "ココ", "あいだっち",
]
E36_PLACEHOLDER_PATTERN = re.compile(r"(私は|僕は|リルは)[〜ー]{1,3}(だよ|なんだ|だね)")
E36_AI_IDENTITY_PATTERN = re.compile(r"AI(アシスタント|です|モデル)")

PERSONA_ITEMS = ("E01", "E20", "E21", "E22")


def build_model_and_tokenizer():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, quantization_config=quant_config, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_V4_PATH, adapter_name="v4")
    model.load_adapter(ADAPTER_MID_PATH, adapter_name="mid")
    model.load_adapter(ADAPTER_HIGH_PATH, adapter_name="high")
    model.eval()
    return model, tokenizer


def generate_reply(model, tokenizer, messages, condition, seed, do_sample=True):
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
    else:
        gen_kwargs.update(do_sample=False)
    with torch.no_grad():
        if condition == "A_base":
            with model.disable_adapter():
                output_ids = model.generate(**encoded, **gen_kwargs)
        else:
            model.set_adapter(ADAPTER_NAME_FOR_CONDITION[condition])
            output_ids = model.generate(**encoded, **gen_kwargs)
    completion_ids = output_ids[0][prompt_len:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return {"text": text, "completion_tokens": len(completion_ids)}


def run_single(model, tokenizer, system_prompt, rag_context, question, cond, seed, do_sample=True):
    messages = [{"role": "system", "content": system_prompt}]
    if rag_context:
        messages.append({"role": "system", "content": rag_context})
    messages.append({"role": "user", "content": question})
    return generate_reply(model, tokenizer, messages, cond, seed, do_sample)


def run_multiturn(model, tokenizer, system_prompt, turns, cond, seed):
    messages = [{"role": "system", "content": system_prompt}]
    turn_log = []
    for i in range(0, len(turns), 2):
        user_text = turns[i]
        messages.append({"role": "user", "content": user_text})
        gen = generate_reply(model, tokenizer, messages, cond, seed)
        messages.append({"role": "assistant", "content": gen["text"]})
        turn_log.append({"user": user_text, "assistant": gen["text"]})
    return turn_log


def q3_fact_check(text: str) -> dict:
    found = [k for k in Q3_KEY_FACTS if k in text]
    pct_found = [k for k in Q3_KEY_FACTS if k.endswith("%") and k in text]
    game_found = [k for k in Q3_KEY_FACTS if k.endswith("G") and k in text]
    return {
        "text": text,
        "length": len(text),
        "key_facts_found": found,
        "recall_pct": round(len(found) / len(Q3_KEY_FACTS) * 100, 1),
        "all3_gamecounts": len(game_found) == 3,
        "all3_pcts": len(pct_found) == 3,
    }


def holdout_check(text: str, key_facts, irrelevant_markers) -> dict:
    found = [f for f in key_facts if f in text]
    leaked = [m for m in irrelevant_markers if m in text]
    return {
        "text": text,
        "length": len(text),
        "key_facts_found": found,
        "recall_pct": round(len(found) / len(key_facts) * 100, 1) if key_facts else None,
        "irrelevant_leaked": leaked,
    }


def q9_check(text: str) -> dict:
    return {"text": text, "has_derived_calc": bool(Q9_CALC_PATTERN.search(text))}


def q11_check(text: str) -> dict:
    return {
        "text": text,
        "yamedoki_advice": bool(Q11_YAMEDOKI_PATTERN.search(text)),
        "strategy_advice": bool(Q11_STRATEGY_PATTERN.search(text)),
        "loopstock_causal": bool(Q11_LOOPSTOCK_CAUSAL_PATTERN.search(text)),
        "other_causal": bool(Q11_OTHER_CAUSAL_PATTERN.search(text)),
    }


def e36_check(text: str) -> dict:
    wrong = [w for w in WRONG_NAMES if w in text]
    return {
        "text": text,
        "correct_name_riru": "リル" in text,
        "wrong_names_found": wrong,
        "has_wrong_name": len(wrong) > 0,
        "placeholder_or_unfinished": bool(E36_PLACEHOLDER_PATTERN.search(text)),
        "ai_base_identity": bool(E36_AI_IDENTITY_PATTERN.search(text)),
    }


def main() -> int:
    print("Loading base model + v4/ratio-mid/ratio-high adapters...")
    model, tokenizer = build_model_and_tokenizer()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    rag_17q = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rag_17q}
    q3, q9, q11 = by_id["Q3"], by_id["Q9"], by_id["Q11"]

    holdout_path = EVAL_DIR / "phase4i_holdout_omission_v2.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    p01 = next(r for r in holdout if r["id"] == "P01")
    p02 = next(r for r in holdout if r["id"] == "P02")
    p04 = next(r for r in holdout if r["id"] == "P04")

    eval_39 = [
        json.loads(line)
        for line in (EVAL_DIR / "riru_eval_set_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_39_by_id = {x["id"]: x for x in eval_39}

    results: dict = {
        "q3_greedy": {}, "q3_sampled": {}, "p01": {}, "p02": {}, "p04": {},
        "q9": {}, "q11": {}, "e36": {}, "persona_extra": {},
        "structured_17q": {}, "character_39": {},
    }

    t0 = time.perf_counter()
    for cond in CONDITIONS:
        print(f"=== condition={cond} ===")

        results["q3_greedy"][cond] = q3_fact_check(
            run_single(
                model, tokenizer, system_prompt, q3["rag_context_text"], q3["question"],
                cond, SEED_DEFAULT, do_sample=False,
            )["text"]
        )
        sampled = {}
        for seed in SEEDS:
            text = run_single(
                model, tokenizer, system_prompt, q3["rag_context_text"], q3["question"], cond, seed
            )["text"]
            sampled[str(seed)] = q3_fact_check(text)
        results["q3_sampled"][cond] = sampled

        for pid, item in (("p01", p01), ("p02", p02), ("p04", p04)):
            out = {}
            for seed in SEEDS:
                text = run_single(
                    model, tokenizer, system_prompt, item["rag_context_text"], item["question"],
                    cond, seed,
                )["text"]
                out[str(seed)] = holdout_check(
                    text, item["key_facts"], item.get("irrelevant_markers", [])
                )
            results[pid][cond] = out

        q9_out = {}
        for seed in SEEDS:
            text = run_single(
                model, tokenizer, system_prompt, q9["rag_context_text"], q9["question"], cond, seed
            )["text"]
            q9_out[str(seed)] = q9_check(text)
        results["q9"][cond] = q9_out

        q11_out = {}
        for seed in SEEDS:
            text = run_single(
                model, tokenizer, system_prompt,
                q11["rag_context_text"], q11["question"], cond, seed,
            )["text"]
            q11_out[str(seed)] = q11_check(text)
        results["q11"][cond] = q11_out

        e36_item = eval_39_by_id["E36"]
        e36_out = {}
        for seed in SEEDS:
            gen = run_single(model, tokenizer, system_prompt, None, e36_item["prompt"], cond, seed)
            e36_out[str(seed)] = e36_check(gen["text"])
        results["e36"][cond] = e36_out

        persona_out = {}
        for pid in PERSONA_ITEMS:
            item = eval_39_by_id[pid]
            if item["type"] == "single":
                text = run_single(model, tokenizer, system_prompt, None, item["prompt"], cond, 42)[
                    "text"
                ]
                persona_out[pid] = {"category": item["category"], "text": text}
            else:
                turns = run_multiturn(model, tokenizer, system_prompt, item["turns"], cond, 42)
                persona_out[pid] = {"category": item["category"], "turns": turns}
        results["persona_extra"][cond] = persona_out

        for item in rag_17q:
            results["structured_17q"].setdefault(
                item["id"], {"question": item["question"], "c": {}}
            )
            text = run_single(
                model, tokenizer, system_prompt, item["rag_context_text"], item["question"],
                cond, SEED_DEFAULT,
            )["text"]
            results["structured_17q"][item["id"]]["c"][cond] = {"text": text, "length": len(text)}

        for item in eval_39:
            results["character_39"].setdefault(
                item["id"], {"category": item["category"], "type": item["type"], "c": {}}
            )
            if item["type"] == "single":
                text = run_single(model, tokenizer, system_prompt, None, item["prompt"], cond, 42)[
                    "text"
                ]
                results["character_39"][item["id"]]["c"][cond] = {"text": text}
            else:
                turns = run_multiturn(model, tokenizer, system_prompt, item["turns"], cond, 42)
                joined = " ".join(t["assistant"] for t in turns)
                results["character_39"][item["id"]]["c"][cond] = {"text": joined, "turns": turns}

        print(f"  done ({time.perf_counter() - t0:.1f}s elapsed total)")

    out_path = EVAL_DIR / "phase4s_comprehensive_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
