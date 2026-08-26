"""Phase 4ZA Section18: CPU-only backendでのRAG sanity smoke test。
Q3/Q9/Q11/Adversarial/Conflicting/Long-contextを各1代表probe、greedyのみで
確認する。目的はCPU化によって推論自体が異常になっていないことの確認であり、
Phase4X/4ZのRAG full Gate再実行は行わない。
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

SERVER_URL = "http://127.0.0.1:8712"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
MAX_NEW_TOKENS = 300


def get_chat_template() -> str:
    with urllib.request.urlopen(f"{SERVER_URL}/props", timeout=30) as resp:
        props = json.loads(resp.read())
    return props["chat_template"]


def render_prompt(env: Environment, tmpl_str: str, system_prompt: str, context,
                   question: str) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": question})
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

    rag17 = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rag17}
    from phase4w_probes import ADVERSARIAL_PROBES, CONFLICTING_PROBES, LONGCONTEXT_PROBES

    ad01 = next(p for p in ADVERSARIAL_PROBES if p["id"] == "AD-01")
    cf01 = next(p for p in CONFLICTING_PROBES if p["id"] == "CF-01")
    lc01 = next(p for p in LONGCONTEXT_PROBES if p["id"] == "LC-01")

    probes = {
        "Q3": (by_id["Q3"]["rag_context_text"], by_id["Q3"]["question"]),
        "Q9": (by_id["Q9"]["rag_context_text"], by_id["Q9"]["question"]),
        "Q11": (by_id["Q11"]["rag_context_text"], by_id["Q11"]["question"]),
        "AD-01": (ad01["context"], ad01["question"]),
        "CF-01": (cf01["context"], cf01["question"]),
        "LC-01": (lc01["context"], lc01["question"]),
    }

    results = {}
    t0 = time.perf_counter()
    for pid, (context, question) in probes.items():
        rendered = render_prompt(env, tmpl_str, system_prompt, context, question)
        text = generate_greedy(rendered)
        results[pid] = {"greedy": text}
        print(f"  {pid} done ({time.perf_counter() - t0:.1f}s)")

    out_path = EVAL_DIR / "phase4za_rag_smoke_cpu_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
