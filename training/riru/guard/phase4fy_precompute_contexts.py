"""Phase4FY Stage B-J precompute: 実際の本番 RagPipeline(entity-attribution統合後)を
使い、各probeの rag_context.prompt_text を事前計算する。GPU generation(.venv-qlora側)は
このJSONを読むだけで、retriever/entity_attributionを直接importしない
(phase4zt_precompute_contexts.py と同じ既存パターンを踏襲)。

比較用に、entity-attribution適用前(素のembedding top-k)のcontextも同時に記録し、
production_diffの根拠として使う。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
REPORTS_DIR = PROJECT_ROOT / "training" / "riru" / "reports"

from pachislot_ai.core.config import RAG_CONTEXT_PROMPT_PATH, Settings  # noqa: E402
from pachislot_ai.data.db import create_structured_engine  # noqa: E402
from pachislot_ai.rag.context_builder import build_rag_context  # noqa: E402
from pachislot_ai.rag.embedder import Embedder  # noqa: E402
from pachislot_ai.rag.entity_attribution import select_grounded_chunks  # noqa: E402
from pachislot_ai.rag.pipeline import RagPipeline  # noqa: E402
from pachislot_ai.rag.retriever import Retriever  # noqa: E402
from pachislot_ai.rag.structured_lookup import find_relevant_structured_facts  # noqa: E402
from pachislot_ai.data.repositories import machine_repository as mrepo  # noqa: E402
from pachislot_ai.data.db import open_session  # noqa: E402
from pachislot_ai.data.models.structured import Machine  # noqa: E402
from sqlalchemy import select  # noqa: E402


def load_probes() -> list[dict]:
    """既存の全ソースからprobeを収集する(新規probeの作成は行わない、既存資産の再利用のみ)。"""
    probes: list[dict] = []

    fx = json.loads((REPORTS_DIR / "phase4fx_probes_consolidated.json").read_text(encoding="utf-8"))
    for x in fx:
        probes.append({"id": x["id"], "stage": _stage_for_category(x["category"]),
                        "category": x["category"], "prompt": x["prompt"]})

    # Stage J (Section22): mandatory direct retrieval-miss test + recall比較用query
    retrieval_recall_queries = [
        ("RJ-01", "GG中とはどんな状態か教えて"),  # mandatory
        ("RJ-02", "GG中の状態について"),
        ("RJ-03", "GGゲーム中はどんな感じ？"),
        ("RJ-04", "引き戻しについて教えて"),
        ("RJ-05", "引き戻しとは何ですか"),
        ("RJ-06", "SGGゲーム数の振り分けを教えて"),
        ("RJ-07", "青7が連続したときのGG当選率は？"),
        ("RJ-08", "天井ゲーム数を教えて"),
        ("RJ-09", "天井は何ゲームですか"),
        ("RJ-10", "ヤメ時の目安を教えて"),
        ("RJ-11", "設定判別の要素を教えて"),
        ("RJ-12", "LV5について教えて"),
        ("RJ-13", "GG継続の条件は？"),
        ("RJ-14", "Z-ZONE昇格率について教えて"),
        ("RJ-15", "初当りの契機を教えて"),
        ("RJ-16", "小役確率について教えて"),
        ("RJ-17", "設定変更時の挙動を教えて"),
        ("RJ-18", "ミリオンゴッドの機種の特徴を教えて"),
        ("RJ-19", "GG準備中とは何ですか"),
        ("RJ-20", "継続契機について教えて"),
        ("RJ-21", "GGストックの仕組みを教えて"),
    ]
    for pid, q in retrieval_recall_queries:
        probes.append({"id": pid, "stage": "stage_j_retrieval_recall", "category": "retrieval_recall", "prompt": q})

    return probes


def _stage_for_category(cat: str) -> str:
    return {
        "known_failure": "stage_b_known_failure",
        "phantom_entity": "stage_d_phantom_entity",
        "concept_binding": "stage_e_concept_binding",
        "query_style": "stage_c_query_style",
        "rag50": "stage_f_rag50",
    }[cat]


def load_multiturn_scenarios() -> list[dict]:
    """Stage I(Section21): FC/FUの既存multi-turnシナリオを再利用する。"""
    path = PROJECT_ROOT / "training" / "riru" / "eval" / "phase4fc_multiturn_scenarios.py"
    if not path.is_file():
        return []
    ns: dict = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)  # noqa: S102
    for key in ("SCENARIOS", "MULTITURN_SCENARIOS", "scenarios"):
        if key in ns:
            return ns[key]
    return []


BOUNDARY_PROBES = [
    ("BD-ST-01", "おはよう！", "small_talk"),
    ("BD-ST-02", "今日は天気がいいね", "small_talk"),
    ("BD-ST-03", "ありがとう、助かったよ", "small_talk"),
    ("BD-ST-04", "疲れたから少し休憩しようかな", "small_talk"),
    ("BD-ST-05", "最近ハマってることある？", "small_talk"),
    ("BD-ST-06", "それじゃあ、また今度話そうね、バイバイ", "small_talk"),
    ("BD-OOD-01", "今日の東京の最高気温は？", "ood"),
    ("BD-OOD-02", "おすすめのラーメン屋教えて", "ood"),
    ("BD-OOD-03", "円安はいつまで続きますか", "ood"),
    ("BD-OOD-04", "次のオリンピックはどこで開催されますか", "ood"),
    ("BD-CONV-01", "今日パチスロ打ちに行こうと思うんだ", "pachislot_conversational"),
    ("BD-CONV-02", "パチスロで一番好きな瞬間は？", "pachislot_conversational"),
    ("BD-CONV-03", "スロット初めてでちょっと緊張してる", "pachislot_conversational"),
]


def build_one(rag_pipeline, retriever, structured_engine, settings, query: str) -> dict:
    from pachislot_ai.rag.entity_attribution import extract_query_entities

    # 実本番経路: RagPipeline.build_context() (entity-attribution込み)
    rag_context = rag_pipeline.build_context(query, machine_id=None)

    # 比較用: entity-attribution適用前の素のembedding top-k (A0相当)
    raw_chunks = retriever.search(query, machine_id=None, top_k=settings.rag_top_k)
    effective_machine_id = raw_chunks[0].machine_id if raw_chunks else rag_pipeline._sole_machine_id()
    with open_session(structured_engine) as session:
        machine = mrepo.get_machine(session, effective_machine_id) if effective_machine_id else None
        machine_name = machine.name if machine else effective_machine_id
        structured_findings = (
            find_relevant_structured_facts(session, effective_machine_id, query)
            if effective_machine_id else []
        )
    refined_raw = retriever.search(query, machine_id=effective_machine_id, top_k=settings.rag_top_k) if effective_machine_id else raw_chunks
    if refined_raw:
        raw_chunks = refined_raw
    a0_context = build_rag_context(RAG_CONTEXT_PROMPT_PATH, structured_findings=structured_findings,
                                    chunks=raw_chunks, machine_name=machine_name)

    query_entities = extract_query_entities(query)

    return {
        "prompt": query,
        "query_entities": query_entities,
        "raw_embedding_titles": [c.title for c in raw_chunks],
        "selected_titles_after_entity_attribution": [c["title"] for c in rag_context.chunk_sources],
        "a0_prompt_text": a0_context.prompt_text,
        "fy_prompt_text": rag_context.prompt_text,
        "fy_is_empty": rag_context.is_empty,
        "a0_is_empty": a0_context.is_empty,
        "structured_source_count": len(rag_context.structured_sources),
    }


def main():
    settings = Settings()
    embedder = Embedder(settings.embedding_model_path, device=settings.embedding_device)
    from pachislot_ai.rag.vector_store import VectorStore
    vector_store = VectorStore(settings.vector_db_path, settings.vector_db_collection)
    retriever = Retriever(embedder, vector_store, default_top_k=settings.rag_top_k)
    structured_engine = create_structured_engine(settings.structured_db_path)
    rag_pipeline = RagPipeline(retriever, structured_engine, RAG_CONTEXT_PROMPT_PATH, top_k=settings.rag_top_k)

    probes = load_probes()
    print(f"loaded {len(probes)} single-turn probes")

    results = []
    for p in probes:
        built = build_one(rag_pipeline, retriever, structured_engine, settings, p["prompt"])
        results.append({"id": p["id"], "stage": p["stage"], "category": p["category"], **built})
        print(f"[{p['stage']}] {p['id']}: raw={len(built['raw_embedding_titles'])} "
              f"selected={len(built['selected_titles_after_entity_attribution'])} entities={built['query_entities']}")

    out_path = REPORTS_DIR / "phase4fy_precomputed_contexts.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(results)} precomputed contexts -> {out_path}")

    # Stage H(Section20)境界サンプル
    boundary_results = []
    for pid, query, category in BOUNDARY_PROBES:
        built = build_one(rag_pipeline, retriever, structured_engine, settings, query)
        boundary_results.append({"id": pid, "category": category, **built})
        print(f"[boundary:{category}] {pid}: selected={len(built['selected_titles_after_entity_attribution'])} "
              f"is_empty={built['fy_is_empty']}")
    (REPORTS_DIR / "phase4fy_boundary_contexts.json").write_text(
        json.dumps(boundary_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(boundary_results)} boundary contexts")

    # Stage I(Section21) multi-turn: 各turnのqueryだけを独立にcontext化する
    # (ChatServiceは常に user_messages[-1].content だけを検索queryに使うため、
    # 会話履歴自体はGPU側で逐次組み立てる)
    scenarios = load_multiturn_scenarios()
    multiturn_results = []
    for sc in scenarios:
        turn_contexts = []
        for turn in sc["turns"]:
            built = build_one(rag_pipeline, retriever, structured_engine, settings, turn["user"])
            turn_contexts.append({"user": turn["user"], "expected_mode": turn.get("expected_mode"), **built})
        multiturn_results.append({"id": sc["id"], "description": sc.get("description", ""), "turns": turn_contexts})
        print(f"[multiturn] {sc['id']}: {len(turn_contexts)} turns precomputed")
    (REPORTS_DIR / "phase4fy_multiturn_contexts.json").write_text(
        json.dumps(multiturn_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(multiturn_results)} multiturn scenarios")


if __name__ == "__main__":
    main()
