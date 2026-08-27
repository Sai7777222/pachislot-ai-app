"""Phase 4ZG Section23-24: llama.cpp CPU(Phase4ZG candidate BF16 GGUF)でのidentity/margin/RAG評価。

Probe pool = phase4zg_identity_eval.pyと同一の171probe。
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
TEMPERATURE = 0.3
TOP_P = 0.9
SEEDS = (101, 102, 103)
FORCED_PREFIX = "こんにちは〜！私はパチスロの専門アシスタントの"
RI_ID, RU_ID = 36723, 32610

MERGED_HF_PATH = str(TRAINING_ROOT / "merged" / "riru-phase4zg-identity-hardened-hf")


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


def generate(rendered_prompt: str, seed, do_sample: bool) -> str:
    body = {"prompt": rendered_prompt, "n_predict": MAX_NEW_TOKENS}
    if do_sample:
        body.update(temperature=TEMPERATURE, top_p=TOP_P, seed=seed)
    else:
        body.update(temperature=0.0)
    req = urllib.request.Request(f"{SERVER_URL}/completion", data=json.dumps(body).encode("utf-8"),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["content"].strip()


def load_probe_pool() -> list[dict]:
    from phase4ze_holdout_probes import ALL_PROBES as ZE_HOLDOUT
    from phase4z_probes import PROBE_SET_A, PROBE_SET_B, PROBE_SET_C, PROBE_SET_D
    from phase4zf_stress_probes import WRONG_NAME_INDUCTION, ROLE_NAME_CONFUSION, IDENTITY_CORRECTION_STRESS
    from phase4zg_holdout_probes import ALL_PROBES as ZGH

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
    for p in WRONG_NAME_INDUCTION:
        probes.append({"set": "zf_wrong_name_induction", "id": p["id"], "prompt": p["prompt"]})
    for p in ROLE_NAME_CONFUSION:
        probes.append({"set": "zf_role_name_confusion", "id": p["id"], "prompt": p["prompt"]})
    for p in IDENTITY_CORRECTION_STRESS:
        probes.append({"set": "zf_identity_correction_stress", "id": p["id"], "prompt": p["prompt"]})
    for p in ZGH:
        probes.append({"set": "zg_holdout", "id": p["id"], "prompt": p["prompt"]})
    return probes


def mode_margin() -> int:
    from transformers import AutoTokenizer
    from phase4z_probes import PROBE_SET_C

    tokenizer = AutoTokenizer.from_pretrained(MERGED_HF_PATH, trust_remote_code=True)
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    e36_original = PROBE_SET_C[0]["prompt"]
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": e36_original}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    full_text = prompt_text + FORCED_PREFIX

    body = {"prompt": full_text, "n_predict": 1, "temperature": 0.0, "n_probs": 20, "post_sampling_probs": False}
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
        top_list.append({"rank": i + 1, "token_id": tok_id, "token_str": item.get("token", ""), "logprob": logprob})
        if tok_id == RI_ID:
            ri_logprob = logprob
        if tok_id == RU_ID:
            ru_logprob = logprob
    result = {
        "engine": "llama.cpp CPU (BF16 GGUF, phase4zg)", "ri_logprob": ri_logprob, "ru_logprob": ru_logprob,
        "winner": ("リ" if (ri_logprob or -999) > (ru_logprob or -999)
                   else ("ル" if (ru_logprob or -999) > (ri_logprob or -999) else "TIE/N-A")),
        "margin_logprob": (ri_logprob - ru_logprob) if (ri_logprob is not None and ru_logprob is not None) else None,
        "top20": top_list,
    }
    out_path = REPORTS_DIR / "phase4zg_gguf_margin.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(result)
    return 0


def mode_identity() -> int:
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl_str = get_chat_template()
    probes = load_probe_pool()

    results = {}
    t0 = time.time()
    for p in probes:
        rendered = render_prompt(env, tmpl_str, system_prompt, p["prompt"])
        greedy = generate(rendered, seed=None, do_sample=False)
        sampled = {str(s): generate(rendered, seed=s, do_sample=True) for s in SEEDS}
        results[p["id"]] = {"set": p["set"], "greedy": greedy, "sampled": sampled}
        if len(results) % 10 == 0:
            elapsed = time.time() - t0
            print(f"{len(results)}/{len(probes)} done ({elapsed:.1f}s, "
                  f"eta {(elapsed/len(results))*(len(probes)-len(results)):.0f}s)")

    out = {"engine": "llama.cpp CPU (BF16 GGUF, phase4zg)", "n_probes": len(probes),
           "seeds": list(SEEDS), "n_generations": len(probes) * (1 + len(SEEDS)), "results": results}
    out_path = REPORTS_DIR / "phase4zg_identity_llamacpp.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(f"total_generations={out['n_generations']}")
    return 0


def mode_regression() -> int:
    from phase4zf_rag_stress_eval import load_rag_probe_pool
    probes = load_rag_probe_pool()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl_str = get_chat_template()

    results = {}
    t0 = time.time()
    for p in probes:
        rendered = render_prompt(env, tmpl_str, system_prompt, p["question"], context=p.get("context"))
        greedy = generate(rendered, seed=None, do_sample=False)
        results[p["id"]] = {"set": p["set"], "greedy": greedy}
        if len(results) % 20 == 0:
            print(f"{len(results)}/{len(probes)} done ({time.time()-t0:.1f}s)")

    out = {"engine": "llama.cpp CPU (BF16 GGUF, phase4zg)", "n_probes": len(probes), "results": results}
    out_path = REPORTS_DIR / "phase4zg_rag_llamacpp.json"
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
