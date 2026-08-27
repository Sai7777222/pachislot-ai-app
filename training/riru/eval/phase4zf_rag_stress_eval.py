"""Phase 4ZF Section16: RAG Regression Stress Gate。

Probe pool: structured_rag_17q(17) + holdout P01-P10(10) + Scope PT-01~22(22) +
Broad(36) + Adversarial(20) + Conflicting(10) + Long-context(10) = 125probes、greedyのみ。
既存の評価資産(structured_rag_17q_context.json, phase4i_holdout_omission_v2.json,
phase4t_probes.P04_PROBES, phase4v_probes.PROBES, phase4w_probes.*)をそのまま再利用する。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
PROJECT_ROOT = EVAL_DIR.parents[2]
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(TRAINING_ROOT))
REPORTS_DIR = TRAINING_ROOT / "reports"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"


def load_rag_probe_pool() -> list[dict]:
    probes = []
    rag17 = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    for r in rag17:
        probes.append({"set": "structured_17q", "id": r["id"], "context": r["rag_context_text"],
                        "question": r["question"]})

    holdout = json.loads((EVAL_DIR / "phase4i_holdout_omission_v2.json").read_text(encoding="utf-8"))
    for r in holdout:
        probes.append({"set": "holdout_p", "id": r["id"], "context": r["rag_context_text"],
                        "question": r["question"]})

    from phase4t_probes import P04_PROBES
    for p in P04_PROBES:
        probes.append({"set": "scope_pt", "id": p["id"], "context": p["context"], "question": p["question"]})

    from phase4v_probes import PROBES as BROAD_PROBES
    for p in BROAD_PROBES:
        probes.append({"set": "broad", "id": p["id"], "context": p.get("context"), "question": p["question"]})

    from phase4w_probes import ADVERSARIAL_PROBES, CONFLICTING_PROBES, LONGCONTEXT_PROBES
    for p in ADVERSARIAL_PROBES:
        probes.append({"set": "adversarial", "id": p["id"], "context": p.get("context"), "question": p["question"]})
    for p in CONFLICTING_PROBES:
        probes.append({"set": "conflicting", "id": p["id"], "context": p.get("context"), "question": p["question"]})
    for p in LONGCONTEXT_PROBES:
        probes.append({"set": "longcontext", "id": p["id"], "context": p.get("context"), "question": p["question"]})

    return probes


def run_hf(attn_impl: str) -> int:
    from phase4ze_identity_eval import load_model, generate_reply

    probes = load_rag_probe_pool()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = load_model(attn_impl)

    results = {}
    t0 = time.time()
    for p in probes:
        messages = [{"role": "system", "content": system_prompt}]
        if p.get("context"):
            messages.append({"role": "system", "content": p["context"]})
        messages.append({"role": "user", "content": p["question"]})
        greedy = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
        results[p["id"]] = {"set": p["set"], "greedy": greedy}
        if len(results) % 20 == 0:
            print(f"{len(results)}/{len(probes)} done ({time.time()-t0:.1f}s)")

    out = {"attn_impl": attn_impl, "n_probes": len(probes), "results": results}
    out_path = REPORTS_DIR / f"phase4zf_rag_{attn_impl}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


def run_llama() -> int:
    from jinja2 import Environment

    SERVER_URL = "http://127.0.0.1:8712"

    def get_chat_template() -> str:
        with urllib.request.urlopen(f"{SERVER_URL}/props", timeout=30) as resp:
            return json.loads(resp.read())["chat_template"]

    def render(env, tmpl_str, system_prompt, context, question):
        messages = [{"role": "system", "content": system_prompt}]
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": question})
        tmpl = env.from_string(tmpl_str)
        return tmpl.render(messages=messages, add_generation_prompt=True, tools=None,
                            bos_token="", eos_token="<|im_end|>")

    def gen(rendered):
        body = {"prompt": rendered, "n_predict": 300, "temperature": 0.0}
        req = urllib.request.Request(f"{SERVER_URL}/completion", data=json.dumps(body).encode("utf-8"),
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())["content"].strip()

    probes = load_rag_probe_pool()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl_str = get_chat_template()

    results = {}
    t0 = time.time()
    for p in probes:
        rendered = render(env, tmpl_str, system_prompt, p.get("context"), p["question"])
        greedy = gen(rendered)
        results[p["id"]] = {"set": p["set"], "greedy": greedy}
        if len(results) % 20 == 0:
            print(f"{len(results)}/{len(probes)} done ({time.time()-t0:.1f}s)")

    out = {"engine": "llama.cpp CPU (BF16 GGUF, phase4ze)", "n_probes": len(probes), "results": results}
    out_path = REPORTS_DIR / "phase4zf_rag_llamacpp.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=["hf", "llama"])
    parser.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args()
    if args.engine == "hf":
        sys.exit(run_hf(args.attn_impl))
    else:
        sys.exit(run_llama())
