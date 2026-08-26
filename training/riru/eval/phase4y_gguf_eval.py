"""Phase 4Y-R: GGUF候補(BF16/Q8_0/Q5_K_M)の代表probe評価。

llama-cpp-pythonでモデルを1回ロードし、phase4y_representative_probes.pyの
12代表probeをgreedy+seed42/43/44で生成する。A_lora_final/B_merged_hfと
同じ構造のJSONを出力し、直接比較できるようにする。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from llama_cpp import Llama

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parents[2]

sys.path.insert(0, str(EVAL_DIR))
from phase4y_representative_probes import load_probes  # noqa: E402

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
SEEDS = (42, 43, 44)


def run_single(llm: Llama, system_prompt: str, rag_context, question: str, seed: int,
                do_sample: bool) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if rag_context:
        messages.append({"role": "system", "content": rag_context})
    messages.append({"role": "user", "content": question})
    llm.set_seed(seed)
    kwargs = dict(
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
    )
    if do_sample:
        kwargs.update(temperature=TEMPERATURE, top_p=TOP_P)
    else:
        kwargs.update(temperature=0.0)
    out = llm.create_chat_completion(**kwargs)
    return out["choices"][0]["message"]["content"].strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--n-ctx", type=int, default=4096)
    args = parser.parse_args()

    print(f"Loading GGUF ({args.label}): {args.gguf_path}")
    t0 = time.perf_counter()
    llm = Llama(
        model_path=args.gguf_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        verbose=False,
    )
    load_time = time.perf_counter() - t0
    print(f"  loaded in {load_time:.1f}s")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    probes = load_probes()

    results = {}
    t1 = time.perf_counter()
    for pid, probe in probes.items():
        out = {
            "greedy": run_single(llm, system_prompt, probe["context"], probe["question"], 42,
                                  do_sample=False)
        }
        out["sampled"] = {
            str(s): run_single(llm, system_prompt, probe["context"], probe["question"], s,
                                do_sample=True)
            for s in SEEDS
        }
        results[pid] = out
        print(f"  {pid} done ({time.perf_counter() - t1:.1f}s)")

    gen_time = time.perf_counter() - t1
    meta = {"label": args.label, "gguf_path": args.gguf_path, "load_time_sec": round(load_time, 1),
            "generation_time_sec": round(gen_time, 1)}

    out_path = Path(args.out_json)
    out_path.write_text(
        json.dumps({"_meta": meta, **results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
