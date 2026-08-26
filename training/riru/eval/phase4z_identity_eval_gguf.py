"""Phase 4Z: C_gguf_bf16 (BF16 GGUF) 大規模identity診断評価。

Probe Set A(Phase4W naming stress) / B(Phase4X held-out naming) /
C(E36 original+paraphrase) / D(E02 original+paraphrase) を greedy+seed101-N で
評価し、Scope(PT-01〜22)・RAG safety sanity・E36 greedy独立5回再現性を実施する。
学習・再merge・再GGUF変換は一切行わない。BF16 GGUFのみを対象とする
(Q8_0/Q5_K_Mは今回の主評価には使用しない)。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from llama_cpp import Llama

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parents[2]
TRAINING_ROOT = EVAL_DIR.parents[0]

sys.path.insert(0, str(EVAL_DIR))
from phase4t_probes import P04_PROBES  # noqa: E402
from phase4z_probes import PROBE_SET_A, PROBE_SET_B, PROBE_SET_C, PROBE_SET_D  # noqa: E402

GGUF_PATH = str(TRAINING_ROOT / "gguf" / "riru-qwen-final-bf16.gguf")
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9

SEEDS_20 = tuple(range(101, 121))
SEEDS_30 = tuple(range(101, 131))
SEEDS_10 = tuple(range(101, 111))
SEEDS_3 = (101, 102, 103)


def run_single(llm: Llama, system_prompt: str, rag_context, question: str, seed: int,
                do_sample: bool = True) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if rag_context:
        messages.append({"role": "system", "content": rag_context})
    messages.append({"role": "user", "content": question})
    llm.set_seed(seed)
    kwargs = dict(messages=messages, max_tokens=MAX_NEW_TOKENS)
    if do_sample:
        kwargs.update(temperature=TEMPERATURE, top_p=TOP_P)
    else:
        kwargs.update(temperature=0.0)
    out = llm.create_chat_completion(**kwargs)
    return out["choices"][0]["message"]["content"].strip()


def sweep(llm, system_prompt, context, question, seeds, greedy=True):
    out = {}
    if greedy:
        out["greedy"] = run_single(llm, system_prompt, context, question, 42, do_sample=False)
    out["sampled"] = {str(s): run_single(llm, system_prompt, context, question, s) for s in seeds}
    return out


def main() -> int:
    print(f"Loading BF16 GGUF: {GGUF_PATH}")
    t0 = time.perf_counter()
    llm = Llama(model_path=GGUF_PATH, n_gpu_layers=99, n_ctx=2048, verbose=False)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    results: dict = {"_meta": {"label": "C_gguf_bf16", "gguf_path": GGUF_PATH,
                                "temperature": TEMPERATURE, "top_p": TOP_P,
                                "max_tokens": MAX_NEW_TOKENS}}
    t1 = time.perf_counter()

    print("=== Probe Set A: Phase4W naming stress (20 x greedy+20seed) ===")
    results["set_a_naming_stress"] = {}
    for p in PROBE_SET_A:
        results["set_a_naming_stress"][p["id"]] = sweep(llm, system_prompt, None, p["prompt"],
                                                          SEEDS_20)
    print(f"  done ({time.perf_counter() - t1:.1f}s)")

    print("=== Probe Set B: Phase4X held-out naming (24 x greedy+20seed) ===")
    results["set_b_heldout_naming"] = {}
    for p in PROBE_SET_B:
        results["set_b_heldout_naming"][p["id"]] = sweep(llm, system_prompt, None, p["prompt"],
                                                           SEEDS_20)
    print(f"  done ({time.perf_counter() - t1:.1f}s)")

    print("=== Probe Set C: E36 original+paraphrase (17 x greedy+30seed) ===")
    results["set_c_e36"] = {}
    for p in PROBE_SET_C:
        results["set_c_e36"][p["id"]] = sweep(llm, system_prompt, None, p["prompt"], SEEDS_30)
    print(f"  done ({time.perf_counter() - t1:.1f}s)")

    print("=== Probe Set D: E02 original+paraphrase (16 x greedy+20seed) ===")
    results["set_d_e02"] = {}
    for p in PROBE_SET_D:
        results["set_d_e02"][p["id"]] = sweep(llm, system_prompt, None, p["prompt"], SEEDS_20)
    print(f"  done ({time.perf_counter() - t1:.1f}s)")

    print("=== Scope: PT-01..22 (22 x greedy+10seed) ===")
    results["scope"] = {}
    for p in P04_PROBES:
        results["scope"][p["id"]] = sweep(llm, system_prompt, p["context"], p["question"],
                                           SEEDS_10)
    print(f"  done ({time.perf_counter() - t1:.1f}s)")

    print("=== RAG safety sanity (6 probes x greedy+3seed) ===")
    rag17_path = EVAL_DIR / "structured_rag_17q_context.json"
    rag17 = json.loads(rag17_path.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rag17}
    from phase4w_probes import ADVERSARIAL_PROBES, CONFLICTING_PROBES, LONGCONTEXT_PROBES

    ad01 = next(p for p in ADVERSARIAL_PROBES if p["id"] == "AD-01")
    cf01 = next(p for p in CONFLICTING_PROBES if p["id"] == "CF-01")
    lc01 = next(p for p in LONGCONTEXT_PROBES if p["id"] == "LC-01")
    results["rag_safety"] = {
        "Q3": sweep(llm, system_prompt, by_id["Q3"]["rag_context_text"], by_id["Q3"]["question"],
                    SEEDS_3),
        "Q9": sweep(llm, system_prompt, by_id["Q9"]["rag_context_text"], by_id["Q9"]["question"],
                    SEEDS_3),
        "Q11": sweep(llm, system_prompt, by_id["Q11"]["rag_context_text"],
                     by_id["Q11"]["question"], SEEDS_3),
        "AD-01": sweep(llm, system_prompt, ad01["context"], ad01["question"], SEEDS_3),
        "CF-01": sweep(llm, system_prompt, cf01["context"], cf01["question"], SEEDS_3),
        "LC-01": sweep(llm, system_prompt, lc01["context"], lc01["question"], SEEDS_3),
    }
    print(f"  done ({time.perf_counter() - t1:.1f}s)")

    out_path = EVAL_DIR / "phase4z_identity_results_gguf.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path} (total {time.perf_counter() - t1:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
