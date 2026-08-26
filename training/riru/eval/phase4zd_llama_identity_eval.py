"""Phase 4ZD Section10-12: D_LLAMA_BF16_CPU条件でのidentity生成評価(llama-server経由)。

--mode repro5      : E36 original greedyを5回(独立HTTP呼び出し)実行し決定論性を確認 (Section10)
--mode paraphrase8 : Phase4ZAの8問(無改変)をgreedyで評価 (Section11)
--mode naming220   : Phase4W naming stress 20問 x (greedy + seed101-110) = 220生成 (Section12 Stage1)
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

from phase4z_probes import PROBE_SET_A, PROBE_SET_B, PROBE_SET_C  # noqa: E402

SEEDS_3 = (101, 102, 103)

SERVER_URL = "http://127.0.0.1:8712"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3
TOP_P = 0.9
SEEDS_10 = tuple(range(101, 111))

PARAPHRASE8_IDS = ["E36_ORIGINAL", "PZ36-12", "PZ36-06", "PZ36-14", "PZ36-15",
                   "PZ36-01", "PZ36-02", "PZ36-03"]


def get_chat_template() -> str:
    with urllib.request.urlopen(f"{SERVER_URL}/props", timeout=30) as resp:
        props = json.loads(resp.read())
    return props["chat_template"]


def render_prompt(env: Environment, tmpl_str: str, system_prompt: str, question: str) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    tmpl = env.from_string(tmpl_str)
    return tmpl.render(messages=messages, add_generation_prompt=True, tools=None,
                        bos_token="", eos_token="<|im_end|>")


def generate(rendered_prompt: str, seed: int | None, do_sample: bool) -> str:
    body = {"prompt": rendered_prompt, "n_predict": MAX_NEW_TOKENS}
    if do_sample:
        body.update(temperature=TEMPERATURE, top_p=TOP_P, seed=seed)
    else:
        body.update(temperature=0.0)
    req = urllib.request.Request(f"{SERVER_URL}/completion", data=json.dumps(body).encode("utf-8"),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        out = json.loads(resp.read())
    return out["content"].strip()


def mode_repro5() -> int:
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl_str = get_chat_template()
    e36_original = PROBE_SET_C[0]["prompt"]
    rendered = render_prompt(env, tmpl_str, system_prompt, e36_original)

    results = []
    for i in range(5):
        t0 = time.time()
        text = generate(rendered, seed=None, do_sample=False)
        results.append({"run": i + 1, "text": text, "elapsed_sec": round(time.time() - t0, 2)})
        print(f"run {i+1}/5 done")

    all_identical = len({r["text"] for r in results}) == 1
    out = {"condition": "D_LLAMA_BF16_CPU", "mode": "repro5", "runs": results,
           "all_identical": all_identical}
    out_path = REPORTS_DIR / "phase4zd_repro5_D_LLAMA_BF16_CPU.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}, all_identical={all_identical}")
    return 0


def mode_paraphrase8() -> int:
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl_str = get_chat_template()
    probe_by_id = {p["id"]: p["prompt"] for p in PROBE_SET_C}

    results = {}
    for pid in PARAPHRASE8_IDS:
        rendered = render_prompt(env, tmpl_str, system_prompt, probe_by_id[pid])
        text = generate(rendered, seed=None, do_sample=False)
        results[pid] = {"greedy": text}
        print(f"{pid} done")

    out = {"condition": "D_LLAMA_BF16_CPU", "mode": "paraphrase8", "results": results}
    out_path = REPORTS_DIR / "phase4zd_paraphrase8_D_LLAMA_BF16_CPU.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


def mode_naming220() -> int:
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl_str = get_chat_template()

    results = {}
    t0 = time.time()
    for p in PROBE_SET_A:
        rendered = render_prompt(env, tmpl_str, system_prompt, p["prompt"])
        greedy = generate(rendered, seed=None, do_sample=False)
        sampled = {}
        for s in SEEDS_10:
            sampled[str(s)] = generate(rendered, seed=s, do_sample=True)
        results[p["id"]] = {"greedy": greedy, "sampled": sampled}
        print(f"{p['id']} done ({time.time()-t0:.1f}s elapsed)")

    out = {"condition": "D_LLAMA_BF16_CPU", "mode": "naming220", "n_probes": len(PROBE_SET_A),
           "seeds": list(SEEDS_10), "temperature": TEMPERATURE, "top_p": TOP_P,
           "results": results}
    out_path = REPORTS_DIR / "phase4zd_naming220_D_LLAMA_BF16_CPU.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


def mode_stage2() -> int:
    """Section14 Stage2: PROBE_SET_B(Phase4X held-out naming 24問) x (greedy+seed101-103)。"""
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl_str = get_chat_template()

    results = {}
    t0 = time.time()
    for p in PROBE_SET_B:
        rendered = render_prompt(env, tmpl_str, system_prompt, p["prompt"])
        greedy = generate(rendered, seed=None, do_sample=False)
        sampled = {}
        for s in SEEDS_3:
            sampled[str(s)] = generate(rendered, seed=s, do_sample=True)
        results[p["id"]] = {"greedy": greedy, "sampled": sampled}
        print(f"{p['id']} done ({time.time()-t0:.1f}s elapsed)")

    out = {"condition": "D_LLAMA_BF16_CPU", "mode": "stage2", "n_probes": len(PROBE_SET_B),
           "seeds": list(SEEDS_3), "temperature": TEMPERATURE, "top_p": TOP_P,
           "results": results}
    out_path = REPORTS_DIR / "phase4zd_stage2_D_LLAMA_BF16_CPU.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                         choices=["repro5", "paraphrase8", "naming220", "stage2"])
    args = parser.parse_args()
    fn = {"repro5": mode_repro5, "paraphrase8": mode_paraphrase8, "naming220": mode_naming220,
          "stage2": mode_stage2}[args.mode]
    sys.exit(fn())
