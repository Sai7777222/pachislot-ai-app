"""Phase 4ZE Section22: llama.cpp(CPU-only, BF16 GGUF)でのFinal Identity Gate評価。

--mode margin    : E36 forced-prefixでの「リ」「ル」top20 logits/margin
--mode identity  : Phase4ZE holdout(27) + Phase4W naming_stress(20) + Phase4X held-out
                   naming(24) + E36 family(17) + E02 family(16) = 104probes、greedyのみ
--mode regression: Q3/P01/Q9/Q11/PT-01/AD-01/CF-01/LC-01の8問、greedyのみ
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

from jinja2 import Environment

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
PROJECT_ROOT = EVAL_DIR.parents[2]
sys.path.insert(0, str(EVAL_DIR))
REPORTS_DIR = TRAINING_ROOT / "reports"

SERVER_URL = "http://127.0.0.1:8712"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
MAX_NEW_TOKENS = 300
FORCED_PREFIX = "こんにちは〜！私はパチスロの専門アシスタントの"
RI_ID, RU_ID = 36723, 32610


def get_chat_template() -> str:
    with urllib.request.urlopen(f"{SERVER_URL}/props", timeout=30) as resp:
        return json.loads(resp.read())["chat_template"]


def render_prompt(env, tmpl_str, system_prompt, question, context=None) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": question})
    tmpl = env.from_string(tmpl_str)
    return tmpl.render(messages=messages, add_generation_prompt=True, tools=None,
                        bos_token="", eos_token="<|im_end|>")


def generate_greedy(rendered_prompt: str) -> str:
    body = {"prompt": rendered_prompt, "n_predict": MAX_NEW_TOKENS, "temperature": 0.0}
    req = urllib.request.Request(f"{SERVER_URL}/completion", data=json.dumps(body).encode("utf-8"),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["content"].strip()


def mode_margin() -> int:
    import numpy as np
    from transformers import AutoTokenizer
    from phase4z_probes import PROBE_SET_C

    tokenizer = AutoTokenizer.from_pretrained(
        str(TRAINING_ROOT / "merged" / "riru-phase4ze-identity-margin-hf"), trust_remote_code=True)
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    e36_original = PROBE_SET_C[0]["prompt"]
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": e36_original}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    full_text = prompt_text + FORCED_PREFIX

    body = {"prompt": full_text, "n_predict": 1, "temperature": 0.0, "n_probs": 20,
            "post_sampling_probs": False}
    req = urllib.request.Request(f"{SERVER_URL}/completion", data=json.dumps(body).encode("utf-8"),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        out = json.loads(resp.read())
    top_logprobs = out["completion_probabilities"][0]["top_logprobs"]
    top_list = []
    ri_logprob = ru_logprob = None
    for i, item in enumerate(top_logprobs):
        tok_id = item.get("id")
        logprob = item.get("logprob")
        top_list.append({"rank": i + 1, "token_id": tok_id, "token_str": item.get("token", ""),
                          "logprob": logprob})
        if tok_id == RI_ID:
            ri_logprob = logprob
        if tok_id == RU_ID:
            ru_logprob = logprob

    result = {
        "engine": "llama.cpp CPU (BF16 GGUF, phase4ze)",
        "ri_logprob": ri_logprob, "ru_logprob": ru_logprob,
        "winner": ("リ" if (ri_logprob or -999) > (ru_logprob or -999)
                   else ("ル" if (ru_logprob or -999) > (ri_logprob or -999) else "TIE/N-A")),
        "margin_logprob": (ri_logprob - ru_logprob) if (ri_logprob is not None and ru_logprob is not None) else None,
        "top20": top_list,
    }
    out_path = REPORTS_DIR / "phase4ze_gguf_margin.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(result)
    return 0


def load_identity_probes() -> list[dict]:
    from phase4ze_holdout_probes import ALL_PROBES as ZE_HOLDOUT
    from phase4z_probes import PROBE_SET_A, PROBE_SET_B, PROBE_SET_C, PROBE_SET_D

    probes = []
    for p in ZE_HOLDOUT:
        probes.append({"set": "phase4ze_holdout", "id": p["id"], "prompt": p["prompt"]})
    for p in PROBE_SET_A:
        probes.append({"set": "phase4w_naming_stress", "id": p["id"], "prompt": p["prompt"]})
    for p in PROBE_SET_B:
        probes.append({"set": "phase4x_heldout_naming", "id": p["id"], "prompt": p["prompt"]})
    for p in PROBE_SET_C:
        probes.append({"set": "e36_family", "id": p["id"], "prompt": p["prompt"]})
    for p in PROBE_SET_D:
        probes.append({"set": "e02_family", "id": p["id"], "prompt": p["prompt"]})
    return probes


def mode_identity() -> int:
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl_str = get_chat_template()
    probes = load_identity_probes()

    results = {}
    t0 = time.time()
    for p in probes:
        rendered = render_prompt(env, tmpl_str, system_prompt, p["prompt"])
        greedy = generate_greedy(rendered)
        results[p["id"]] = {"set": p["set"], "greedy": greedy}
        if len(results) % 10 == 0:
            print(f"{len(results)}/{len(probes)} done ({time.time()-t0:.1f}s)")

    out = {"engine": "llama.cpp CPU (BF16 GGUF, phase4ze)", "n_probes": len(probes), "results": results}
    out_path = REPORTS_DIR / "phase4ze_gguf_identity.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


def load_regression_probes() -> dict:
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


def mode_regression() -> int:
    probes = load_regression_probes()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl_str = get_chat_template()

    results = {}
    for pid, p in probes.items():
        rendered = render_prompt(env, tmpl_str, system_prompt, p["question"], context=p["context"])
        greedy = generate_greedy(rendered)
        results[pid] = {"greedy": greedy}
        print(f"{pid} done")

    out = {"engine": "llama.cpp CPU (BF16 GGUF, phase4ze)", "results": results}
    out_path = REPORTS_DIR / "phase4ze_gguf_regression.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["margin", "identity", "regression"])
    args = parser.parse_args()
    fn = {"margin": mode_margin, "identity": mode_identity, "regression": mode_regression}[args.mode]
    sys.exit(fn())
