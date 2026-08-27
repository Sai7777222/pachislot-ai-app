"""Phase 4ZF Section7-9: overnight stress evaluation (llama.cpp CPU, ZE candidate BF16 GGUF frozen).

Probe pool = same 144-probe suite as phase4zf_identity_eval.py.
Sampling: greedy + seed101-103 (4/probe) = 576 generations.
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


def get_chat_template() -> str:
    with urllib.request.urlopen(f"{SERVER_URL}/props", timeout=30) as resp:
        return json.loads(resp.read())["chat_template"]


def render_prompt(env, tmpl_str, system_prompt, question) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
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
    return probes


def main() -> int:
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

    out = {"engine": "llama.cpp CPU (BF16 GGUF, phase4ze)", "n_probes": len(probes),
           "seeds": list(SEEDS), "n_generations": len(probes) * (1 + len(SEEDS)), "results": results}
    out_path = REPORTS_DIR / "phase4zf_identity_llamacpp.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(f"total_generations={out['n_generations']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
