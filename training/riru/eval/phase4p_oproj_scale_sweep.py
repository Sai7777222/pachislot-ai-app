"""Phase 4P: v4 o_proj-only LoRA scale sweep (推論時のみ、学習なし)。

q_proj/k_proj/v_proj のLoRA寄与は scale=1.0 固定のまま、o_projのLoRA寄与だけを
段階的に弱め (0.00/0.10/0.25/0.40/0.50/0.60/0.75/1.00)、Q3型省略の改善と
Q9/Q11/E36等の副作用のあいだにsweet spotが存在するかを調べる。

Phase 4N module ablationの実装ミス (module_types=None を「全layerに適用」と
誤解釈し、意図したfull_v4がscale=0.0相当になっていた) を再発させないため、
各条件の生成前に q/k/v_proj の scaling が 1.0 のまま・o_projのみ指定値で
あることを明示的にassertする。adapterファイルは一切変更しない。

QLoRA/LoRA学習は行わない。v5/v6等の新規学習も行わない。
"""

from __future__ import annotations

import json
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parents[1]
EVAL_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_V4_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v4")

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
SEEDS = (42, 43, 44, 45, 46)

O_SCALES = (0.00, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 1.00)

Q3_KEY_FACTS = ["510G", "1000G", "1480G", "15.2%", "20.3%", "64.5%"]
Q3_EXTRA_MARKERS = ["33.2%", "Z-ZONE"]
CEILING_REACH_MARKERS = ["天井に到達", "天井到達", "到達すると", "到達時"]
LOOPSTOCK_MARKERS = ["ループストック"]

Q9_CALC_PATTERN = re.compile(r"約\s*\d+(\.\d+)?\s*(倍|ポイント)")
Q11_YAMEDOKI_PATTERN = re.compile(r"ヤメ時|一旦ヤメ|止めるのが|ヤメる")
Q11_STRATEGY_PATTERN = re.compile(r"おすすめ|べきです|べきだ|戦略|コツ")
Q11_CAUSAL_PATTERN = re.compile(r"ループストック.{0,15}(ほど|により|によって)")

WRONG_NAMES = [
    "リリ", "リサ", "リコ", "あいり", "あいこ", "ゆめぴょん", "ゆめちゃん",
    "ピコ", "ピッコロ", "ぴよこ", "パティ", "ココ",
]
E36_CORRECT_NAME_PATTERN = re.compile(r"リル(だよ|です|なんだ|といいます|と申します|よ)")
E36_PLACEHOLDER_PATTERN = re.compile(r"(私は|僕は|リルは)[〜ー]{1,3}(だよ|なんだ|だね)")
E36_AI_IDENTITY_PATTERN = re.compile(r"AI(アシスタント|です|モデル)")

PERSONA_ITEMS = ("E01", "E20", "E21", "E22", "E36")


def find_lora_layers(model, adapter_name: str):
    results = []
    layer_pat = re.compile(r"\.layers\.(\d+)\.")
    module_types = ("q_proj", "k_proj", "v_proj", "o_proj")
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and adapter_name in scaling:
            m = layer_pat.search(name)
            layer_idx = int(m.group(1)) if m else None
            module_type = next((mt for mt in module_types if name.endswith(mt)), None)
            results.append((name, module, layer_idx, module_type))
    return results


@contextmanager
def oproj_only_scale(model, adapter_name: str, o_scale: float):
    """o_projのscalingだけをo_scale倍し、q/k/vは元の値(1.0)のまま保つ。
    終了時に必ず全moduleのscalingを元の値へ復元し、復元をassertする。
    """
    layers = find_lora_layers(model, adapter_name)
    originals: dict[str, float] = {}
    try:
        for name, module, layer_idx, module_type in layers:
            originals[name] = module.scaling[adapter_name]
            if module_type == "o_proj":
                module.scaling[adapter_name] = originals[name] * o_scale
            # q/k/v_proj は変更しない (=1.0のまま)

        # --- 実装ミス防止のための明示的検証 ---
        for name, module, layer_idx, module_type in layers:
            current = module.scaling[adapter_name]
            if module_type == "o_proj":
                expected = originals[name] * o_scale
                assert abs(current - expected) < 1e-9, f"o_proj scaling mismatch at {name}"
            else:
                assert current == originals[name], f"{module_type} scaling was modified at {name}"
        yield len(layers)
    finally:
        for name, module, layer_idx, module_type in layers:
            module.scaling[adapter_name] = originals[name]
        for name, module, layer_idx, module_type in layers:
            assert module.scaling[adapter_name] == originals[name], f"restore failed at {name}"


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
    model.set_adapter("v4")
    model.eval()
    return model, tokenizer


def build_messages(system_prompt, rag_context, question):
    messages = [{"role": "system", "content": system_prompt}]
    if rag_context:
        messages.append({"role": "system", "content": rag_context})
    messages.append({"role": "user", "content": question})
    return messages


def generate(model, tokenizer, messages, do_sample: bool, seed: int) -> str:
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
        output_ids = model.generate(**encoded, **gen_kwargs)
    completion_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def q3_fact_check(text: str) -> dict:
    found = [k for k in Q3_KEY_FACTS if k in text]
    extra = [k for k in Q3_EXTRA_MARKERS if k in text]
    pct_found = [k for k in Q3_KEY_FACTS if k.endswith("%") and k in text]
    game_found = [k for k in Q3_KEY_FACTS if k.endswith("G") and k in text]
    return {
        "text": text,
        "length": len(text),
        "key_facts_found": found,
        "recall_pct": round(len(found) / len(Q3_KEY_FACTS) * 100, 1),
        "extra_markers_found": extra,
        "all3_gamecounts": len(game_found) == 3,
        "all3_pcts": len(pct_found) == 3,
        "any_pct": len(pct_found) > 0,
        "ceiling_reach_mentioned": any(m in text for m in CEILING_REACH_MARKERS),
        "loopstock_mentioned": any(m in text for m in LOOPSTOCK_MARKERS),
    }


def holdout_fact_check(text: str, key_facts: list[str], irrelevant_markers: list[str]) -> dict:
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
    hits = Q9_CALC_PATTERN.findall(text)
    return {"text": text, "length": len(text), "has_derived_calc": len(hits) > 0}


def q11_check(text: str) -> dict:
    return {
        "text": text,
        "length": len(text),
        "yamedoki_advice": bool(Q11_YAMEDOKI_PATTERN.search(text)),
        "strategy_advice": bool(Q11_STRATEGY_PATTERN.search(text)),
        "loopstock_causal_fabrication": bool(Q11_CAUSAL_PATTERN.search(text)),
    }


def e36_check(text: str) -> dict:
    wrong = [w for w in WRONG_NAMES if w in text]
    return {
        "text": text,
        "length": len(text),
        "correct_name_riru": bool(E36_CORRECT_NAME_PATTERN.search(text)) or "リル" in text,
        "wrong_names_found": wrong,
        "has_wrong_name": len(wrong) > 0,
        "placeholder_or_unfinished": bool(E36_PLACEHOLDER_PATTERN.search(text)),
        "ai_base_identity": bool(E36_AI_IDENTITY_PATTERN.search(text)),
    }


def main() -> int:
    print("Loading base model + v4 adapter (o_proj-only scale sweep)...")
    model, tokenizer = build_model_and_tokenizer()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    rag_17q = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    q3 = next(r for r in rag_17q if r["id"] == "Q3")
    q9 = next(r for r in rag_17q if r["id"] == "Q9")
    q11 = next(r for r in rag_17q if r["id"] == "Q11")

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

    results: dict = {"meta": {"o_scales": O_SCALES, "seeds": SEEDS}, "by_scale": {}}

    t0 = time.perf_counter()
    for o_scale in O_SCALES:
        print(f"=== o_proj scale={o_scale} ===")
        scale_out: dict = {}
        with oproj_only_scale(model, "v4", o_scale) as n_layers:
            scale_out["n_lora_layers_total"] = n_layers

            q3_messages = build_messages(system_prompt, q3["rag_context_text"], q3["question"])
            greedy_text = generate(model, tokenizer, q3_messages, do_sample=False, seed=42)
            scale_out["q3_greedy"] = q3_fact_check(greedy_text)

            q3_sampled = {}
            for seed in SEEDS:
                text = generate(model, tokenizer, q3_messages, do_sample=True, seed=seed)
                q3_sampled[str(seed)] = q3_fact_check(text)
            scale_out["q3_sampled"] = q3_sampled
            recalls = [v["recall_pct"] for v in q3_sampled.values()]
            scale_out["q3_sampled_avg"] = round(sum(recalls) / len(recalls), 1)
            scale_out["q3_sampled_min"] = min(recalls)
            scale_out["q3_sampled_max"] = max(recalls)
            scale_out["q3_all3_gamecount_seeds"] = sum(
                1 for v in q3_sampled.values() if v["all3_gamecounts"]
            )
            scale_out["q3_all3_pct_seeds"] = sum(1 for v in q3_sampled.values() if v["all3_pcts"])
            scale_out["q3_any_pct_only_seeds"] = sum(
                1 for v in q3_sampled.values() if v["any_pct"] and not v["all3_pcts"]
            )
            scale_out["q3_ceiling_reach_seeds"] = sum(
                1 for v in q3_sampled.values() if v["ceiling_reach_mentioned"]
            )
            scale_out["q3_loopstock_seeds"] = sum(
                1 for v in q3_sampled.values() if v["loopstock_mentioned"]
            )

            for pid, pitem in (("P01", p01), ("P02", p02), ("P04", p04)):
                p_messages = build_messages(
                    system_prompt, pitem["rag_context_text"], pitem["question"]
                )
                p_out = {}
                for seed in SEEDS:
                    text = generate(model, tokenizer, p_messages, do_sample=True, seed=seed)
                    p_out[str(seed)] = holdout_fact_check(
                        text, pitem["key_facts"], pitem.get("irrelevant_markers", [])
                    )
                recalls_p = [v["recall_pct"] for v in p_out.values()]
                scale_out[pid.lower()] = {
                    "seeds": p_out,
                    "avg_recall": round(sum(recalls_p) / len(recalls_p), 1),
                }

            q9_messages = build_messages(system_prompt, q9["rag_context_text"], q9["question"])
            q9_out = {}
            for seed in SEEDS:
                text = generate(model, tokenizer, q9_messages, do_sample=True, seed=seed)
                q9_out[str(seed)] = q9_check(text)
            scale_out["q9"] = q9_out
            scale_out["q9_calc_hallucination_seeds"] = sum(
                1 for v in q9_out.values() if v["has_derived_calc"]
            )

            q11_messages = build_messages(system_prompt, q11["rag_context_text"], q11["question"])
            q11_out = {}
            for seed in SEEDS:
                text = generate(model, tokenizer, q11_messages, do_sample=True, seed=seed)
                q11_out[str(seed)] = q11_check(text)
            scale_out["q11"] = q11_out
            scale_out["q11_yamedoki_seeds"] = sum(
                1 for v in q11_out.values() if v["yamedoki_advice"]
            )
            scale_out["q11_strategy_seeds"] = sum(
                1 for v in q11_out.values() if v["strategy_advice"]
            )
            scale_out["q11_causal_fabrication_seeds"] = sum(
                1 for v in q11_out.values() if v["loopstock_causal_fabrication"]
            )

            e36_item = eval_39_by_id["E36"]
            e36_messages = build_messages(system_prompt, None, e36_item["prompt"])
            e36_out = {}
            for seed in SEEDS:
                text = generate(model, tokenizer, e36_messages, do_sample=True, seed=seed)
                e36_out[str(seed)] = e36_check(text)
            scale_out["e36"] = e36_out
            scale_out["e36_wrong_name_seeds"] = sum(
                1 for v in e36_out.values() if v["has_wrong_name"]
            )
            scale_out["e36_placeholder_seeds"] = sum(
                1 for v in e36_out.values() if v["placeholder_or_unfinished"]
            )
            scale_out["e36_correct_name_seeds"] = sum(
                1 for v in e36_out.values() if v["correct_name_riru"]
            )

            persona_out = {}
            for pid in PERSONA_ITEMS:
                if pid == "E36":
                    continue
                item = eval_39_by_id[pid]
                if item["type"] == "single":
                    p_messages = build_messages(system_prompt, None, item["prompt"])
                    p_text = generate(model, tokenizer, p_messages, do_sample=True, seed=42)
                else:
                    p_messages = [{"role": "system", "content": system_prompt}]
                    p_text = ""
                    for i in range(0, len(item["turns"]), 2):
                        p_messages.append({"role": "user", "content": item["turns"][i]})
                        p_text = generate(model, tokenizer, p_messages, do_sample=True, seed=42)
                        p_messages.append({"role": "assistant", "content": p_text})
                persona_out[pid] = {"category": item["category"], "text": p_text}
            scale_out["persona_extra"] = persona_out

        results["by_scale"][str(o_scale)] = scale_out
        print(f"  done ({time.perf_counter() - t0:.1f}s elapsed total)")

    out_path = EVAL_DIR / "phase4p_oproj_scale_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path} (total {time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
