"""Phase 4ZA Section10: E36 paraphrase代表8問(lossyなもの5問+control 3問)を
CPU-only BF16 GGUF(llama-server経由)でgreedy評価する。Phase4Zの既存probeを
一切変更せず再利用する。
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
sys.path.insert(0, str(EVAL_DIR))

from phase4z_probes import PROBE_SET_C  # noqa: E402

SERVER_URL = "http://127.0.0.1:8712"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
MAX_NEW_TOKENS = 300

SELECTED_IDS = ["E36_ORIGINAL", "PZ36-12", "PZ36-06", "PZ36-14", "PZ36-15",
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

    probe_by_id = {p["id"]: p["prompt"] for p in PROBE_SET_C}
    results = {}
    t0 = time.perf_counter()
    for pid in SELECTED_IDS:
        rendered = render_prompt(env, tmpl_str, system_prompt, probe_by_id[pid])
        text = generate_greedy(rendered)
        results[pid] = {"greedy": text}
        print(f"  {pid} done ({time.perf_counter() - t0:.1f}s)")

    out_path = EVAL_DIR / "phase4za_e36_paraphrase_cpu_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
