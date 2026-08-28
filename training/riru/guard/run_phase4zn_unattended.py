"""Phase4ZN: 2時間無人実行の生成ドライバ。Phase4ZG read-only、raw出力のみ保存。
時間予算(1h45m-1h50m)を超えたら新規generationを停止し、その時点までの結果を保存する。
20件ごとにcheckpoint JSONを書き出す。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
PROJECT_ROOT = GUARD_DIR.parents[2]
sys.path.insert(0, str(GUARD_DIR))
sys.path.insert(0, str(TRAINING_ROOT))
sys.path.insert(0, str(TRAINING_ROOT / "eval"))
REPORTS_DIR = TRAINING_ROOT / "reports"

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
ZG_ADAPTER_PATH = str(TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

START_EPOCH = 1787899376.0
HARD_STOP_SECONDS = 6600  # 1h50m
SOFT_STOP_SECONDS = 6300  # 1h45m target
CHECKPOINT_EVERY = 20


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, ZG_ADAPTER_PATH, adapter_name="zg_zn")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system_prompt, user_text, extra_context=None, seed=42):
    messages = [{"role": "system", "content": system_prompt}]
    if extra_context:
        messages.append({"role": "system", "content": extra_context})
    messages.append({"role": "user", "content": user_text})
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **encoded, max_new_tokens=300, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    dt = time.perf_counter() - t0
    completion_ids = output_ids[0][prompt_len:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return text, dt


def elapsed():
    return time.time() - START_EPOCH


def build_task_list():
    from phase4zn_unattended_probes import probes_in_priority_order
    from phase4zi_ood_sanity_probes import ALL_PROBES as ZI_OOD24
    from phase4zf_rag_stress_eval import load_rag_probe_pool

    new120_ordered = probes_in_priority_order()
    # split by category to interleave per Section11's exact sequence:
    # C -> A/B/D -> [ZI-OOD24 inserted here] -> G -> H -> E -> F -> [RAG50 inserted here]
    by_cat = {}
    for p in new120_ordered:
        by_cat.setdefault(p["category"], []).append(p)

    rag_pool = load_rag_probe_pool()
    required_ids = {"P02", "LC-08", "Q11", "AD-04"}
    required = [p for p in rag_pool if p["id"] in required_ids]
    extra = [p for p in rag_pool if p["id"] not in required_ids][:46]
    rag50 = required + extra

    tasks = []
    for p in by_cat.get("personality_preference", []):
        tasks.append({"dataset": "new120", **p})
    for cat in ("greeting_farewell", "emotional_casual", "social_small_talk"):
        for p in by_cat.get(cat, []):
            tasks.append({"dataset": "new120", **p})
    for p in ZI_OOD24:
        tasks.append({"dataset": "zi_ood24", "id": p["id"], "category": p.get("category", ""),
                       "prompt": p["prompt"], "expected_mode": "OOD_FACTUAL_OR_SMALL_TALK",
                       "rag_expected": False, "specialist_refusal_expected": False})
    for cat in ("ood_factual", "ambiguous_boundary", "pachislot_factual", "pachislot_conversational"):
        for p in by_cat.get(cat, []):
            tasks.append({"dataset": "new120", **p})
    for p in rag50:
        tasks.append({"dataset": "rag50", "id": p["id"], "category": p.get("set", "rag"),
                       "prompt": p["question"], "context": p.get("context"),
                       "expected_mode": "PACHISLOT", "rag_expected": True,
                       "specialist_refusal_expected": False})
    return tasks


def main():
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    tasks = build_task_list()
    print(f"total tasks queued: {len(tasks)}", flush=True)

    model, tokenizer = load_model()
    print(f"model loaded at t={elapsed():.1f}s", flush=True)

    results = []
    crash_count = 0
    stopped_reason = None

    for i, task in enumerate(tasks):
        if elapsed() > SOFT_STOP_SECONDS:
            stopped_reason = f"soft_stop_time_budget_reached (elapsed={elapsed():.0f}s > {SOFT_STOP_SECONDS}s)"
            print(stopped_reason, flush=True)
            break
        try:
            raw, dt = generate(model, tokenizer, system_prompt, task["prompt"],
                                extra_context=task.get("context"))
            crash_count = 0
        except torch.cuda.OutOfMemoryError as e:  # noqa: BLE001
            crash_count += 1
            print(f"CUDA OOM on {task['id']}: {e}", flush=True)
            if crash_count >= 2:
                stopped_reason = "repeated_cuda_error_stop"
                break
            torch.cuda.empty_cache()
            continue
        except RuntimeError as e:  # noqa: BLE001
            if "CUDA" in str(e):
                crash_count += 1
                print(f"CUDA error on {task['id']}: {e}", flush=True)
                if crash_count >= 2:
                    stopped_reason = "repeated_cuda_error_stop"
                    break
                torch.cuda.empty_cache()
                continue
            raise

        results.append({
            "probe_id": task["id"], "dataset": task["dataset"], "category": task["category"],
            "prompt": task["prompt"], "expected_mode": task.get("expected_mode"),
            "rag_expected": task.get("rag_expected"), "specialist_refusal_expected": task.get("specialist_refusal_expected"),
            "response": raw, "generation_time_sec": dt, "backend": "phase4zg_eager_hf",
            "generation_settings": {"do_sample": False, "max_new_tokens": 300, "seed": 42},
            "elapsed_at_generation_sec": elapsed(),
        })

        if len(results) % CHECKPOINT_EVERY == 0:
            ckpt_path = REPORTS_DIR / "phase4zn_unattended_generations.json"
            ckpt_path.write_text(json.dumps({"n_done": len(results), "n_total_planned": len(tasks),
                                              "stopped_reason": None, "results": results},
                                             ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"checkpoint: {len(results)}/{len(tasks)} done, t={elapsed():.1f}s", flush=True)

    final_path = REPORTS_DIR / "phase4zn_unattended_generations.json"
    final_path.write_text(json.dumps({"n_done": len(results), "n_total_planned": len(tasks),
                                       "stopped_reason": stopped_reason, "results": results},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE: {len(results)}/{len(tasks)} generated. stopped_reason={stopped_reason}. "
          f"total_elapsed={elapsed():.1f}s", flush=True)

    # split out per-dataset raw files too, per Section15 artifact list
    by_dataset = {}
    for r in results:
        by_dataset.setdefault(r["dataset"], []).append(r)
    (REPORTS_DIR / "phase4zn_phase4zi_reproduction_raw.json").write_text(
        json.dumps(by_dataset.get("zi_ood24", []), ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "phase4zn_rag50_raw.json").write_text(
        json.dumps(by_dataset.get("rag50", []), ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote split per-dataset raw files.", flush=True)


if __name__ == "__main__":
    main()
