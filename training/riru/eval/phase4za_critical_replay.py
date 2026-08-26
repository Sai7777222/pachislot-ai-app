"""Phase 4ZA Section11: Phase4Zで発見された49件のcritical paired regressionから
選んだ代表20件を、CPU-only BF16 GGUF(llama-server経由)でgreedy replayする。

時間制約上、Section10の許可(「CPU実行時間が極端に長い場合はgreedyを最優先」)に
従い、oritinal seedでのsampled再現ではなくgreedyでのreplayに統一する
(greedyは決定論的でありCPU/CUDA比較の目的に直接資する)。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

from jinja2 import Environment

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parents[2]
REPORTS_DIR = EVAL_DIR.parents[0] / "reports"
sys.path.insert(0, str(EVAL_DIR))

from phase4z_probes import PROBE_SET_A, PROBE_SET_B, PROBE_SET_C, PROBE_SET_D  # noqa: E402

SERVER_URL = "http://127.0.0.1:8712"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
MAX_NEW_TOKENS = 300

PROBE_BY_ID = {}
for pset in (PROBE_SET_A, PROBE_SET_B, PROBE_SET_C, PROBE_SET_D):
    for p in pset:
        PROBE_BY_ID[p["id"]] = p["prompt"]


def get_chat_template() -> str:
    with urllib.request.urlopen(f"{SERVER_URL}/props", timeout=30) as resp:
        props = json.loads(resp.read())
    return props["chat_template"]


def render_prompt(env: Environment, tmpl_str: str, system_prompt: str, question: str) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    tmpl = env.from_string(tmpl_str)
    return tmpl.render(messages=messages, add_generation_prompt=True, tools=None,
                        bos_token="", eos_token="<|im_end|>")


def generate_greedy(rendered_prompt: str) -> str:
    req_body = json.dumps({
        "prompt": rendered_prompt, "n_predict": MAX_NEW_TOKENS, "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(f"{SERVER_URL}/completion", data=req_body,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read())
    return out["content"].strip()


def main() -> int:
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl_str = get_chat_template()

    selected = json.loads(
        (EVAL_DIR / "_phase4za_selected20_cases.json").read_text(encoding="utf-8")
    )

    results = []
    t0 = time.perf_counter()
    for i, case in enumerate(selected):
        probe_id = case["probe"]
        question = PROBE_BY_ID[probe_id]
        rendered = render_prompt(env, tmpl_str, system_prompt, question)
        cpu_text = generate_greedy(rendered)
        results.append({
            "block": case["block"], "probe": probe_id, "original_seed": case["seed"],
            "b_cat_original": case["b_cat"], "c_cat_original_cuda": case["c_cat"],
            "b_text_original": case["b_text"], "c_text_original_cuda": case["c_text"],
            "cpu_text_greedy_replay": cpu_text,
        })
        print(f"  [{i + 1}/{len(selected)}] {probe_id} done ({time.perf_counter() - t0:.1f}s)")

    out_path = EVAL_DIR / "phase4za_critical_replay_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path} (total {time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
