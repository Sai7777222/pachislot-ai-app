"""Phase 4ZD Section18: 最小限のRAG/broadスモークテスト(Q3/P01/Q9/Q11/PT-01/AD-01/CF-01/LC-01)。
B_HF_BF16_EAGER と D_LLAMA_BF16_CPU のみ、greedy + seed101-103の4生成/probeで確認する。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import torch
from jinja2 import Environment

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
PROJECT_ROOT = EVAL_DIR.parents[2]
sys.path.insert(0, str(EVAL_DIR))
REPORTS_DIR = TRAINING_ROOT / "reports"

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
SEEDS_3 = (101, 102, 103)
SERVER_URL = "http://127.0.0.1:8712"


def load_probes() -> dict:
    rag17 = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rag17}
    q3, q9, q11 = by_id["Q3"], by_id["Q9"], by_id["Q11"]

    holdout = json.loads((EVAL_DIR / "phase4i_holdout_omission_v2.json").read_text(encoding="utf-8"))
    p01 = next(r for r in holdout if r["id"] == "P01")

    from phase4t_probes import P04_PROBES
    pt01 = next(p for p in P04_PROBES if p["id"] == "PT-01")

    from phase4w_probes import ADVERSARIAL_PROBES, CONFLICTING_PROBES, LONGCONTEXT_PROBES
    ad01 = next(p for p in ADVERSARIAL_PROBES if p["id"] == "AD-01")
    cf01 = next(p for p in CONFLICTING_PROBES if p["id"] == "CF-01")
    lc01 = next(p for p in LONGCONTEXT_PROBES if p["id"] == "LC-01")

    return {
        "Q3": {"context": q3["rag_context_text"], "question": q3["question"]},
        "P01": {"context": p01["rag_context_text"], "question": p01["question"]},
        "Q9": {"context": q9["rag_context_text"], "question": q9["question"]},
        "Q11": {"context": q11["rag_context_text"], "question": q11["question"]},
        "PT-01": {"context": pt01["context"], "question": pt01["question"]},
        "AD-01": {"context": ad01["context"], "question": ad01["question"]},
        "CF-01": {"context": cf01["context"], "question": cf01["question"]},
        "LC-01": {"context": lc01["context"], "question": lc01["question"]},
    }


def run_hf(condition: str) -> int:
    from phase4zd_identity_eval import build_model_and_tokenizer, generate_reply

    probes = load_probes()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    model, tokenizer = build_model_and_tokenizer(condition)

    results = {}
    for pid, p in probes.items():
        messages = [{"role": "system", "content": system_prompt}]
        if p["context"]:
            messages.append({"role": "system", "content": p["context"]})
        messages.append({"role": "user", "content": p["question"]})
        greedy = generate_reply(model, tokenizer, messages, seed=42, do_sample=False)
        sampled = {str(s): generate_reply(model, tokenizer, messages, seed=s, do_sample=True) for s in SEEDS_3}
        results[pid] = {"greedy": greedy, "sampled": sampled}
        print(f"{pid} done")

    out = {"condition": condition, "results": results}
    out_path = REPORTS_DIR / f"phase4zd_rag_smoke_{condition}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


def run_llama() -> int:
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

    def gen(rendered, seed, do_sample):
        body = {"prompt": rendered, "n_predict": 300}
        if do_sample:
            body.update(temperature=0.3, top_p=0.9, seed=seed)
        else:
            body.update(temperature=0.0)
        req = urllib.request.Request(f"{SERVER_URL}/completion", data=json.dumps(body).encode("utf-8"),
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())["content"].strip()

    probes = load_probes()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl_str = get_chat_template()

    results = {}
    for pid, p in probes.items():
        rendered = render(env, tmpl_str, system_prompt, p["context"], p["question"])
        greedy = gen(rendered, None, False)
        sampled = {str(s): gen(rendered, s, True) for s in SEEDS_3}
        results[pid] = {"greedy": greedy, "sampled": sampled}
        print(f"{pid} done")

    out = {"condition": "D_LLAMA_BF16_CPU", "results": results}
    out_path = REPORTS_DIR / "phase4zd_rag_smoke_D_LLAMA_BF16_CPU.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=["hf", "llama"])
    parser.add_argument("--condition", default="B_HF_BF16_EAGER")
    args = parser.parse_args()
    if args.engine == "hf":
        sys.exit(run_hf(args.condition))
    else:
        sys.exit(run_llama())
