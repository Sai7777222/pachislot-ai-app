"""Phase 4ZH Section30-31: llama.cpp CPU(Phase4ZH candidate BF16 GGUF)でのidentity/margin/RAG/multiturn評価。

HF Gateが完全PASSした場合のみ実施(Section27-29の規定)。
Probe pool = phase4zh_identity_eval.load_probe_pool_stage2()と同一の214probe。
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

MERGED_HF_PATH = str(TRAINING_ROOT / "merged" / "riru-phase4zh-structural-hardened-hf")


def get_chat_template() -> str:
    with urllib.request.urlopen(f"{SERVER_URL}/props", timeout=30) as resp:
        return json.loads(resp.read())["chat_template"]


def render_prompt(env, tmpl_str, messages) -> str:
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
    from phase4zh_identity_eval import load_probe_pool_stage2
    return load_probe_pool_stage2()


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
        "engine": "llama.cpp CPU (BF16 GGUF, phase4zh)", "ri_logprob": ri_logprob, "ru_logprob": ru_logprob,
        "winner": ("リ" if (ri_logprob or -999) > (ru_logprob or -999)
                   else ("ル" if (ru_logprob or -999) > (ri_logprob or -999) else "TIE/N-A")),
        "margin_logprob": (ri_logprob - ru_logprob) if (ri_logprob is not None and ru_logprob is not None) else None,
        "top20": top_list,
    }
    out_path = REPORTS_DIR / "phase4zh_gguf_margin.json"
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
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": p["prompt"]}]
        rendered = render_prompt(env, tmpl_str, messages)
        greedy = generate(rendered, seed=None, do_sample=False)
        sampled = {str(s): generate(rendered, seed=s, do_sample=True) for s in SEEDS}
        results[p["id"]] = {"set": p["set"], "greedy": greedy, "sampled": sampled}
        if len(results) % 10 == 0:
            elapsed = time.time() - t0
            print(f"{len(results)}/{len(probes)} done ({elapsed:.1f}s, "
                  f"eta {(elapsed/len(results))*(len(probes)-len(results)):.0f}s)")

    out = {"engine": "llama.cpp CPU (BF16 GGUF, phase4zh)", "n_probes": len(probes),
           "seeds": list(SEEDS), "n_generations": len(probes) * (1 + len(SEEDS)), "results": results}
    out_path = REPORTS_DIR / "phase4zh_identity_llamacpp.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(f"total_generations={out['n_generations']}")
    return 0


def mode_multiturn() -> int:
    from phase4zh_holdout_probes import MULTITURN_SCENARIOS

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl_str = get_chat_template()

    results = {}
    for sc in MULTITURN_SCENARIOS:
        messages = [{"role": "system", "content": system_prompt}]
        turn_log = []
        for i, user_turn in enumerate(sc["turns"]):
            messages.append({"role": "user", "content": user_turn})
            rendered = render_prompt(env, tmpl_str, messages)
            reply = generate(rendered, seed=None, do_sample=False)
            messages.append({"role": "assistant", "content": reply})
            turn_log.append({"turn": i + 1, "user": user_turn, "assistant": reply})
        results[sc["id"]] = {"category": sc["category"], "turns": turn_log}
        print(f"{sc['id']} ({sc['category']}) done, {len(sc['turns'])} turns")

    out = {"engine": "llama.cpp CPU (BF16 GGUF, phase4zh)", "n_scenarios": len(MULTITURN_SCENARIOS), "results": results}
    out_path = REPORTS_DIR / "phase4zh_multiturn_llamacpp.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
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
        messages = [{"role": "system", "content": system_prompt}]
        if p.get("context"):
            messages.append({"role": "system", "content": p["context"]})
        messages.append({"role": "user", "content": p["question"]})
        rendered = render_prompt(env, tmpl_str, messages)
        greedy = generate(rendered, seed=None, do_sample=False)
        results[p["id"]] = {"set": p["set"], "greedy": greedy}
        if len(results) % 20 == 0:
            print(f"{len(results)}/{len(probes)} done ({time.time()-t0:.1f}s)")

    out = {"engine": "llama.cpp CPU (BF16 GGUF, phase4zh)", "n_probes": len(probes), "results": results}
    out_path = REPORTS_DIR / "phase4zh_rag_llamacpp.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["margin", "identity", "multiturn", "regression"])
    args = parser.parse_args()
    fn = {"margin": mode_margin, "identity": mode_identity, "multiturn": mode_multiturn,
          "regression": mode_regression}[args.mode]
    sys.exit(fn())
