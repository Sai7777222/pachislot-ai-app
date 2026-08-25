"""Phase 4N-6: layer別 LoRA ablation (v4)。

model.config.num_hidden_layers を実際に読み取り、前1/3・中1/3・後1/3に
分割したうえで、各グループのLoRA寄与のみを一時的に無効化(scale=0)して
評価する。all_on(通常v4)をbaselineとし、各groupをoffにした場合と、
各groupのみon(他2グループoff)にした場合の両方を見る。

adapterファイルは一切書き換えない。scaled_lora()の layer_indices引数を使う。

QLoRA/LoRA学習は行わない。v5 adapterは作成しない。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4n_lora_scale_experiment import (  # noqa: E402
    build_messages,
    generate,
    q3_fact_check,
    scaled_lora,
)

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parents[1]
EVAL_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_V4_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v4")

PERSONA_REP_IDS = ["E01", "E14", "E17", "E27", "E36"]


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
    return model, tokenizer, model.config.num_hidden_layers


def build_layer_groups(num_layers: int) -> dict:
    third = num_layers / 3.0
    front = list(range(0, round(third)))
    mid = list(range(round(third), round(2 * third)))
    back = list(range(round(2 * third), num_layers))
    return {"front": front, "mid": mid, "back": back}


def run_condition(model, tokenizer, layer_indices, q3_messages, eval_39_by_id, system_prompt):
    with scaled_lora(model, "v4", factor=0.0, layer_indices=layer_indices) as n_off:
        greedy_text = generate(model, tokenizer, q3_messages, do_sample=False, seed=42)
        sampled_text = generate(model, tokenizer, q3_messages, do_sample=True, seed=42)
        persona = {}
        for eid in PERSONA_REP_IDS:
            item = eval_39_by_id[eid]
            p_messages = build_messages(system_prompt, None, item["prompt"])
            p_text = generate(model, tokenizer, p_messages, do_sample=True, seed=42)
            persona[eid] = {"category": item["category"], "text": p_text}
    return {
        "n_layers_scaled_to_zero": n_off,
        "q3_greedy": q3_fact_check(greedy_text),
        "q3_sampled_seed42": q3_fact_check(sampled_text),
        "persona_representative": persona,
    }


def main() -> int:
    print("Loading base model + v4 adapter (layer ablation)...")
    model, tokenizer, num_layers = build_model_and_tokenizer()
    print(f"num_hidden_layers = {num_layers}")
    groups = build_layer_groups(num_layers)
    print(f"layer groups: front={groups['front']} mid={groups['mid']} back={groups['back']}")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    rag_17q = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    q3_item = next(r for r in rag_17q if r["id"] == "Q3")
    eval_39 = [
        json.loads(line)
        for line in (EVAL_DIR / "riru_eval_set_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_39_by_id = {x["id"]: x for x in eval_39}
    q3_messages = build_messages(system_prompt, q3_item["rag_context_text"], q3_item["question"])

    results = {"num_hidden_layers": num_layers, "layer_groups": groups, "conditions": {}}
    t0 = time.perf_counter()

    print("  condition=all_on (baseline)")
    results["conditions"]["all_on"] = run_condition(
        model, tokenizer, [], q3_messages, eval_39_by_id, system_prompt
    )

    for gname, indices in groups.items():
        cond = f"{gname}_off"
        print(f"  condition={cond} indices={indices}")
        results["conditions"][cond] = run_condition(
            model, tokenizer, indices, q3_messages, eval_39_by_id, system_prompt
        )

    for gname, indices in groups.items():
        other_indices = [i for i in range(num_layers) if i not in indices]
        cond = f"{gname}_only_on"
        print(f"  condition={cond} off_indices={other_indices}")
        results["conditions"][cond] = run_condition(
            model, tokenizer, other_indices, q3_messages, eval_39_by_id, system_prompt
        )

    out_path = EVAL_DIR / "phase4n_layer_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path} ({time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
