"""Phase 4N-3/4/5/7: LoRA適用強度(scale)実験。

PEFTの LoraLayer.scaling[adapter_name] (通常 lora_alpha/r で決まる係数、
forward時に `result + lora_B(lora_A(x)) * scaling` として使われる、
実装確認済み: peft==0.20.0) を、adapterファイルを一切書き換えずに
一時的にメモリ上でのみ乗算することで、推論時のLoRA適用強度を変更する。
実験終了後は必ず元の値へ復元する (contextmanagerで保証)。

対象: v4 (主, scale=0.00/0.10/0.25/0.50/0.75/1.00の6点)
      v2 (副, scale=0.00/0.50/1.00の3点、評価量を抑えるため)

各scaleについて:
  - Q3実物 greedy (do_sample=False)
  - Q3実物 sampled (temperature=0.3, seed=42,43,44,45,46)
  - persona代表14問 (seed=42, temperature=0.3)
  - Q3生成開始位置のlogits (top20)

QLoRA/LoRA学習は行わない。v5 adapterは作成しない。
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
REPORTS_DIR = TRAINING_ROOT / "reports"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ADAPTER_V2_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v2")
ADAPTER_V4_PATH = str(TRAINING_ROOT / "lora-riru-qwen-v4")

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
Q3_SEEDS = (42, 43, 44, 45, 46)
TOP_K_LOGITS = 20

V4_SCALES = (0.00, 0.10, 0.25, 0.50, 0.75, 1.00)
V2_SCALES = (0.00, 0.50, 1.00)

Q3_KEY_FACTS = ["510G", "1000G", "1480G", "15.2%", "20.3%", "64.5%"]
Q3_EXTRA_MARKERS = ["33.2%", "Z-ZONE"]

PERSONA_SCREEN_IDS = [
    "E01", "E02", "E07", "E08", "E14", "E17", "E23", "E24",
    "E27", "E30", "E35", "E36", "E37", "E39",
]

LAYER_INDEX_PATTERN = re.compile(r"\.layers\.(\d+)\.")
MODULE_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj")


def find_lora_layers(model, adapter_name: str) -> list[tuple[str, object, int | None, str | None]]:
    results = []
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and adapter_name in scaling:
            m = LAYER_INDEX_PATTERN.search(name)
            layer_idx = int(m.group(1)) if m else None
            module_type = next((mt for mt in MODULE_TYPES if name.endswith(mt)), None)
            results.append((name, module, layer_idx, module_type))
    return results


@contextmanager
def scaled_lora(
    model, adapter_name: str, factor: float = 1.0, module_types=None, layer_indices=None
):
    """マッチするLoraLayerの scaling[adapter_name] を一時的に factor倍する。
    module_types/layer_indices を指定した場合はマッチする層のみ変更し、
    それ以外は元の値のまま (一部moduleだけablationする用途)。
    終了時に必ず元の値へ復元する。ファイルへの書き込みは一切行わない。
    """
    layers = find_lora_layers(model, adapter_name)
    originals: dict[str, float] = {}
    try:
        for name, module, layer_idx, module_type in layers:
            if module_types is not None and module_type not in module_types:
                continue
            if layer_indices is not None and layer_idx not in layer_indices:
                continue
            originals[name] = module.scaling[adapter_name]
            module.scaling[adapter_name] = originals[name] * factor
        yield len(originals)
    finally:
        for name, module, layer_idx, module_type in layers:
            if name in originals:
                module.scaling[adapter_name] = originals[name]


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
    model = PeftModel.from_pretrained(base_model, ADAPTER_V2_PATH, adapter_name="v2")
    model.load_adapter(ADAPTER_V4_PATH, adapter_name="v4")
    model.eval()
    num_layers = model.config.num_hidden_layers
    return model, tokenizer, num_layers


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


def get_top_logits(model, tokenizer, messages) -> dict:
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**encoded)
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, TOP_K_LOGITS)
    tokens = [tokenizer.decode([tid]) for tid in top_ids.tolist()]
    return {
        "top_tokens": tokens,
        "top_probs": [round(x, 6) for x in top_probs.tolist()],
    }


def q3_fact_check(text: str) -> dict:
    found = [k for k in Q3_KEY_FACTS if k in text]
    extra = [k for k in Q3_EXTRA_MARKERS if k in text]
    return {
        "text": text,
        "length": len(text),
        "key_facts_found": found,
        "recall_pct": round(len(found) / len(Q3_KEY_FACTS) * 100, 1),
        "extra_markers_found": extra,
        "has_510G": "510G" in text,
        "has_1000G": "1000G" in text,
        "has_1480G": "1480G" in text,
        "has_15.2pct": "15.2%" in text,
        "has_20.3pct": "20.3%" in text,
        "has_64.5pct": "64.5%" in text,
        "has_33.2pct": "33.2%" in text,
        "has_ZZONE": "Z-ZONE" in text,
    }


def run_scale_sweep(
    model, tokenizer, adapter_name: str, scales: tuple, system_prompt, q3_item, eval_39_by_id
):
    out = {}
    for scale in scales:
        print(f"  [{adapter_name}] scale={scale}")
        with scaled_lora(model, adapter_name, factor=scale) as n_matched:
            model.set_adapter(adapter_name)
            q3_messages = build_messages(
                system_prompt, q3_item["rag_context_text"], q3_item["question"]
            )
            greedy_text = generate(model, tokenizer, q3_messages, do_sample=False, seed=42)
            sampled = {}
            for seed in Q3_SEEDS:
                text = generate(model, tokenizer, q3_messages, do_sample=True, seed=seed)
                sampled[str(seed)] = q3_fact_check(text)
            logits_info = get_top_logits(model, tokenizer, q3_messages)

            persona = {}
            for eid in PERSONA_SCREEN_IDS:
                item = eval_39_by_id[eid]
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
                persona[eid] = {"category": item["category"], "text": p_text}

        out[str(scale)] = {
            "n_lora_layers_matched": n_matched,
            "q3_greedy": q3_fact_check(greedy_text),
            "q3_sampled": sampled,
            "q3_avg_recall_sampled": round(
                sum(v["recall_pct"] for v in sampled.values()) / len(sampled), 1
            ),
            "q3_first_token_top_logits": logits_info,
            "persona_screen": persona,
        }
    return out


def main() -> int:
    print("Loading base model + v2/v4 adapters...")
    model, tokenizer, num_layers = build_model_and_tokenizer()
    print(f"num_hidden_layers = {num_layers}")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    rag_17q = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    q3_item = next(r for r in rag_17q if r["id"] == "Q3")
    eval_39 = [
        json.loads(line)
        for line in (EVAL_DIR / "riru_eval_set_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_39_by_id = {x["id"]: x for x in eval_39}

    t0 = time.perf_counter()
    print("=== v4 scale sweep ===")
    v4_results = run_scale_sweep(
        model, tokenizer, "v4", V4_SCALES, system_prompt, q3_item, eval_39_by_id
    )
    (EVAL_DIR / "phase4n_scale_results_v4.json").write_text(
        json.dumps(v4_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"v4 sweep done in {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    print("=== v2 scale sweep (reduced) ===")
    v2_results = run_scale_sweep(
        model, tokenizer, "v2", V2_SCALES, system_prompt, q3_item, eval_39_by_id
    )
    (EVAL_DIR / "phase4n_scale_results_v2.json").write_text(
        json.dumps(v2_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"v2 sweep done in {time.perf_counter() - t1:.1f}s")

    meta = {"num_hidden_layers": num_layers, "v4_scales": V4_SCALES, "v2_scales": V2_SCALES}
    (EVAL_DIR / "phase4n_scale_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
