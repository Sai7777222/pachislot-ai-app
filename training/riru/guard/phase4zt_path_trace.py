"""Phase4ZT Stage A: 全260probeについてfinal pathを記録し、mandatory invariantを検証する。
generation不要(dispatch結果は既にphase4zr_dispatch_results.jsonにある)。"""
from __future__ import annotations
import json
import sys
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GUARD_DIR))
REPORTS_DIR = GUARD_DIR.parent / "reports"

from phase4zt_policy_c import decide_c1, decide_c2, decide_c3  # noqa: E402


def main():
    dispatch_results = json.loads((REPORTS_DIR / "phase4zr_dispatch_results.json").read_text(encoding="utf-8"))["rows"]
    precomputed = json.loads((REPORTS_DIR / "phase4zt_precomputed_contexts.json").read_text(encoding="utf-8"))

    traces = {"C1": [], "C2": [], "C3": []}
    for r in dispatch_results:
        is_unknown = r["dispatched_mode"] == "UNKNOWN"
        for variant in ("C1", "C2", "C3"):
            if not is_unknown:
                # 既存の高確信mode: dispatchの時点で確定、Policy Cは一切関与しない。
                # 既存(Phase4ZP)の per-mode prompt を使う経路であり、strict RAGへ行くのは
                # PACHISLOT_FACTUALのみで、必ずcontext(existing RAG pipeline経由)を伴う。
                trace = {"probe_id": r["probe_id"], "dispatch_mode": r["dispatched_mode"], "is_unknown": False,
                         "retrieval_called": r["dispatched_mode"] == "PACHISLOT_FACTUAL",
                         "context_injected": r["dispatched_mode"] == "PACHISLOT_FACTUAL",
                         "selected_policy": "existing_mode_specific_prompt (Phase4ZP, unchanged)",
                         "final_path": "strict_rag_with_context" if r["dispatched_mode"] == "PACHISLOT_FACTUAL"
                                        else "mode_specific_lightweight_prompt"}
            else:
                c = precomputed.get(r["probe_id"])
                if c is None:
                    trace = {"probe_id": r["probe_id"], "dispatch_mode": "UNKNOWN", "is_unknown": True,
                             "error": "context not precomputed"}
                else:
                    if variant == "C1":
                        d = decide_c1(c["prompt"], c["context"])
                    elif variant == "C2":
                        d = decide_c2(c["prompt"], c["context"])
                    else:
                        d = decide_c3(c["prompt"], c["context"], c["titles"], c["texts"])
                    trace = {"probe_id": r["probe_id"], "dispatch_mode": "UNKNOWN", "is_unknown": True,
                             "retrieval_called": d.retrieval_called, "context_injected": d.context_injected,
                             "selected_policy": f"PolicyC_{variant}",
                             "final_path": "strict_rag_with_context" if d.selected_path == "rag_with_context"
                                            else "clarification_no_context"}
            traces[variant].append(trace)

    # mandatory invariant check: UNKNOWN + strict_RAG + context_absent = 0
    violations = {}
    for variant in ("C1", "C2", "C3"):
        v = [t for t in traces[variant] if t.get("is_unknown") and t.get("final_path") == "strict_rag_with_context"
             and not t.get("context_injected", True)]
        violations[variant] = v

    out = {
        "purpose": "Section6/Mandatory Invariant: UNKNOWN + strict_RAG_generation + context_absent = 0件 を検証する。",
        "total_probes": len(dispatch_results),
        "unknown_count": sum(1 for r in dispatch_results if r["dispatched_mode"] == "UNKNOWN"),
        "mandatory_invariant_violations_by_variant": {k: len(v) for k, v in violations.items()},
        "mandatory_invariant_satisfied": all(len(v) == 0 for v in violations.values()),
        "traces": traces,
    }
    out_path = REPORTS_DIR / "phase4zt_path_trace.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"total={out['total_probes']} unknown={out['unknown_count']}")
    print(f"violations: {out['mandatory_invariant_violations_by_variant']}")
    print(f"invariant satisfied: {out['mandatory_invariant_satisfied']}")


if __name__ == "__main__":
    main()
