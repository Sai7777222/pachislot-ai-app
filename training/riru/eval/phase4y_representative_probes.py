"""Phase 4Y: LoRA/merged HF/GGUF の同等性検証用の代表probe一式。

Final Candidate freeze前のA_lora_final基準出力と、merge後のB_merged_hf・
GGUF化後のC/D条件を同一promptで比較するための最小代表セット。
Section10/12/18で指定された各カテゴリ(Q3/P01/P02/scope/Q9/Q11/E02/E36/
naming/Broad/Adversarial/Long-context)を最低1問ずつ含む。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))


def load_probes() -> dict:
    rag17 = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rag17}
    q3, q9, q11 = by_id["Q3"], by_id["Q9"], by_id["Q11"]

    holdout = json.loads(
        (EVAL_DIR / "phase4i_holdout_omission_v2.json").read_text(encoding="utf-8")
    )
    p01 = next(r for r in holdout if r["id"] == "P01")
    p02 = next(r for r in holdout if r["id"] == "P02")

    eval_39 = [
        json.loads(line)
        for line in (EVAL_DIR / "riru_eval_set_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_39_by_id = {x["id"]: x for x in eval_39}
    e02_item = eval_39_by_id["E02"]
    e36_item = eval_39_by_id["E36"]

    from phase4t_probes import P04_PROBES

    pt01 = next(p for p in P04_PROBES if p["id"] == "PT-01")

    from phase4v_probes import PROBES as BROAD_PROBES

    v1a = next(p for p in BROAD_PROBES if p["id"] == "V1-A")

    from phase4w_probes import ADVERSARIAL_PROBES, LONGCONTEXT_PROBES, NAMING_STRESS_PROBES

    ad01 = next(p for p in ADVERSARIAL_PROBES if p["id"] == "AD-01")
    lc01 = next(p for p in LONGCONTEXT_PROBES if p["id"] == "LC-01")
    nw01 = next(p for p in NAMING_STRESS_PROBES if p["id"] == "NW-01")

    return {
        "Q3": {"context": q3["rag_context_text"], "question": q3["question"]},
        "P01": {"context": p01["rag_context_text"], "question": p01["question"]},
        "P02": {"context": p02["rag_context_text"], "question": p02["question"]},
        "PT-01_scope": {"context": pt01["context"], "question": pt01["question"]},
        "Q9": {"context": q9["rag_context_text"], "question": q9["question"]},
        "Q11": {"context": q11["rag_context_text"], "question": q11["question"]},
        "E02": {"context": None, "question": e02_item["prompt"]},
        "E36": {"context": None, "question": e36_item["prompt"]},
        "NW-01_naming": {"context": None, "question": nw01["prompt"]},
        "V1-A_broad": {"context": v1a["context"], "question": v1a["question"]},
        "AD-01_adversarial": {"context": ad01["context"], "question": ad01["question"]},
        "LC-01_longcontext": {"context": lc01["context"], "question": lc01["question"]},
    }
