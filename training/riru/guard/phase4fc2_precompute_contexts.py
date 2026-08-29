"""Phase4FC2: Gate B/C/D/E/H/J 用の実本番RagPipeline context事前計算。
Gate A/F/G/I/M は既存成果物の再利用または別スクリプトで扱う(このファイルの対象外)。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
REPORTS_DIR = PROJECT_ROOT / "training" / "riru" / "reports"

from pachislot_ai.core.config import RAG_CONTEXT_PROMPT_PATH, Settings  # noqa: E402
from pachislot_ai.data.db import create_structured_engine  # noqa: E402
from pachislot_ai.rag.embedder import Embedder  # noqa: E402
from pachislot_ai.rag.pipeline import RagPipeline  # noqa: E402
from pachislot_ai.rag.retriever import Retriever  # noqa: E402
from pachislot_ai.rag.vector_store import VectorStore  # noqa: E402


def load_gate_b_phantom():
    fy = [x for x in json.load(open(REPORTS_DIR / "phase4fx_probes_consolidated.json", encoding="utf-8"))
          if x["category"] == "phantom_entity"]
    fz_gt = json.load(open(REPORTS_DIR / "phase4fz_gt.json", encoding="utf-8"))
    seen = {}
    for x in fy:
        seen[x["prompt"]] = x["id"]
    for q in fz_gt["phantom"]:
        seen.setdefault(q["query"], q["id"])
    return [{"id": v, "prompt": k} for k, v in seen.items()]


def load_gate_c_concept_binding():
    return [{"id": x["id"], "prompt": x["prompt"]}
            for x in json.load(open(REPORTS_DIR / "phase4fx_probes_consolidated.json", encoding="utf-8"))
            if x["category"] == "concept_binding"]


def load_gate_d_query_style():
    base = [{"id": x["id"], "prompt": x["prompt"]}
            for x in json.load(open(REPORTS_DIR / "phase4fx_probes_consolidated.json", encoding="utf-8"))
            if x["category"] == "query_style"]
    extra = [
        {"id": "FC2-QS-01", "prompt": "ミリオンゴッドの遊び方を初心者向けにやさしく説明して"},
        {"id": "FC2-QS-02", "prompt": "SGGの仕組みを初心者向けに説明して"},
        {"id": "FC2-QS-03", "prompt": "GG準備中とGG中の違いを初心者向けに教えて"},
    ]
    return base + extra


def load_gate_e_production_gt():
    gt = json.load(open(REPORTS_DIR / "phase4fc2_production_gt.json", encoding="utf-8"))
    return [{"id": r["id"], "prompt": r["prompt"]} for r in gt["rows"]]


def load_gate_h_boundary():
    smalltalk = json.load(open(REPORTS_DIR / "phase4zp_smalltalk_recheck_raw.json", encoding="utf-8"))
    ood = json.load(open(REPORTS_DIR / "phase4zp_ood_recheck_raw.json", encoding="utf-8"))
    sys.path.insert(0, str(PROJECT_ROOT / "training" / "riru" / "eval"))
    from phase4zn_unattended_probes import ALL_PROBES
    conv = [p for p in ALL_PROBES if p["category"] == "pachislot_conversational"]
    out = []
    for r in smalltalk:
        out.append({"id": r["probe_id"], "category": "small_talk", "prompt": r["prompt"]})
    for r in ood:
        out.append({"id": r["probe_id"], "category": "ood", "prompt": r["prompt"]})
    for r in conv:
        out.append({"id": r["id"], "category": "pachislot_conversational", "prompt": r["prompt"]})
    return out


def load_gate_j_multiturn():
    sys.path.insert(0, str(PROJECT_ROOT / "training" / "riru" / "eval"))
    from phase4fc_multiturn_scenarios import SCENARIOS
    extra = [
        {"id": "FC2-MT-EX01", "description": "entity A(GG) -> entity B(SGG) -> それ(SGGを指すべき)",
         "turns": [
             {"user": "GGについて教えて", "expected_mode": None},
             {"user": "SGGについて教えて", "expected_mode": None},
             {"user": "それの継続率は？", "expected_mode": None},
         ]},
        {"id": "FC2-MT-EX02", "description": "phantom followup: AT-F -> それ",
         "turns": [
             {"user": "AT-Fの性能と終了後の状態について教えて", "expected_mode": None},
             {"user": "それの詳しい仕組みは？", "expected_mode": None},
         ]},
        {"id": "FC2-MT-EX03", "description": "real entity -> phantom entity -> real entity(混入しないか)",
         "turns": [
             {"user": "ガイアベルとは何か説明して", "expected_mode": None},
             {"user": "天国ロングとは何か説明して", "expected_mode": None},
             {"user": "天井は何ゲームですか", "expected_mode": None},
         ]},
    ]
    return list(SCENARIOS) + extra


def load_gate_k_identity():
    picks = json.loads((REPORTS_DIR / "_tmp_identity_picks.json").read_text(encoding="utf-8"))
    out = [{"id": pid, "prompt": prompt} for pid, prompt in picks]
    out.append({"id": "FC2-ID-CANON", "prompt": "君の名前は？"})
    return out


def build_one(rag_pipeline, query: str) -> dict:
    rag_context = rag_pipeline.build_context(query, machine_id=None)
    return {
        "prompt": query,
        "prompt_text": rag_context.prompt_text,
        "chunk_titles": [c["title"] for c in rag_context.chunk_sources],
        "structured_source_count": len(rag_context.structured_sources),
        "is_empty": rag_context.is_empty,
    }


def main():
    settings = Settings()
    embedder = Embedder(settings.embedding_model_path, device=settings.embedding_device)
    vector_store = VectorStore(settings.vector_db_path, settings.vector_db_collection)
    retriever = Retriever(embedder, vector_store, default_top_k=settings.rag_top_k)
    structured_engine = create_structured_engine(settings.structured_db_path)
    rag_pipeline = RagPipeline(retriever, structured_engine, RAG_CONTEXT_PROMPT_PATH, top_k=settings.rag_top_k)

    all_out = {}

    for gate_name, loader in [
        ("gate_b_phantom", load_gate_b_phantom),
        ("gate_c_concept_binding", load_gate_c_concept_binding),
        ("gate_d_query_style", load_gate_d_query_style),
        ("gate_e_production", load_gate_e_production_gt),
        ("gate_h_boundary", load_gate_h_boundary),
        ("gate_k_identity", load_gate_k_identity),
    ]:
        probes = loader()
        results = []
        for p in probes:
            built = build_one(rag_pipeline, p["prompt"])
            row = {"id": p["id"], **({"category": p["category"]} if "category" in p else {}), **built}
            results.append(row)
        all_out[gate_name] = results
        print(f"{gate_name}: {len(results)} probes precomputed")

    # Gate J: multi-turn, per-turn context precompute (independent per turn's query)
    scenarios = load_gate_j_multiturn()
    mt_out = []
    for sc in scenarios:
        turn_contexts = []
        for turn in sc["turns"]:
            built = build_one(rag_pipeline, turn["user"])
            turn_contexts.append({"user": turn["user"], **built})
        mt_out.append({"id": sc["id"], "description": sc.get("description", ""), "turns": turn_contexts})
    all_out["gate_j_multiturn"] = mt_out
    print(f"gate_j_multiturn: {len(mt_out)} scenarios precomputed")

    out_path = REPORTS_DIR / "phase4fc2_precomputed_contexts.json"
    out_path.write_text(json.dumps(all_out, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(v) for k, v in all_out.items() if k != "gate_j_multiturn")
    total_turns = sum(len(s["turns"]) for s in all_out["gate_j_multiturn"])
    print(f"TOTAL single-turn probes: {total}, multiturn scenarios: {len(mt_out)} ({total_turns} turns)")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
