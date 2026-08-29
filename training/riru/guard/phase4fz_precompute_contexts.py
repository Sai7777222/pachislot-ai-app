"""Phase4FZ Section9-14: 実本番RagPipeline(entity_attribution + 修正済みstructured_lookup)
を使い、各probeのrag_context.prompt_textを事前計算する(GPU generation用)。"""
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

KNOWN_FAILURE_8 = [
    ("FZ-K01", "Q6_GG_SGG", "GGとSGGの違いを初心者向けに説明して"),
    ("FZ-K02", "SGG_GG準備中", "SGGの仕組みを分かりやすく説明して"),
    ("FZ-K03", "ガイアベル", "ガイアベルとは何か説明して"),
    ("FZ-K04", "SU4", "SU4について教えて"),
    ("FZ-K05", "GG当選", "GG当選とSGG当選の違いを教えて"),
    ("FZ-K06", "loop_gg_stock", "ループストックとGGストックの違いを教えて"),
    ("FZ-K07", "AT-F", "AT-Fの性能と終了後の状態について教えて"),
    ("FZ-K08", "RT-A_RT-B", "RT-AとRT-Bの違いを要約して"),
]

RAG50_MANDATORY_8 = [
    ("P02", "ボーナス確率について教えて"),
    ("P04", "最低設定と最高設定の機械割の差を教えて"),
    ("LC-08", "AT-Fの性能と終了後の状態を教えて"),
    ("Q6", "GGとSGGの違いを初心者向けに説明して"),
    ("Q11", "天井とヤメ時を合わせて初心者向けに説明して"),
    ("Q15", "ミリオンゴッドの遊び方を初心者向けにやさしく説明して"),
    ("Q17", "ミリオンゴッドの遊び方を少し詳しく説明して"),
    ("AD-04", "ヤメ時はいつがいい？"),
]

FZ_SPECIFIC_MANDATORY = [
    ("FZ-M01", "天国ロングとは何か説明して"),
    ("FZ-M02", "天国について教えて"),
    ("FZ-M03", "GG継続の条件は？"),
]


def load_gt_queries():
    gt = json.loads((REPORTS_DIR / "phase4fz_gt.json").read_text(encoding="utf-8"))
    out = []
    for q in gt["phantom"]:
        out.append((q["id"], "phantom", q["query"]))
    for q in gt["real"]:
        out.append((q["id"], "real", q["query"]))
    for q in gt["close_concept"]:
        out.append((q["id"], "close_concept", q["query"]))
    return out


def main():
    settings = Settings()
    embedder = Embedder(settings.embedding_model_path, device=settings.embedding_device)
    vector_store = VectorStore(settings.vector_db_path, settings.vector_db_collection)
    retriever = Retriever(embedder, vector_store, default_top_k=settings.rag_top_k)
    structured_engine = create_structured_engine(settings.structured_db_path)
    rag_pipeline = RagPipeline(retriever, structured_engine, RAG_CONTEXT_PROMPT_PATH, top_k=settings.rag_top_k)

    probes = []
    for pid, label, q in KNOWN_FAILURE_8:
        probes.append({"id": pid, "stage": "stage9_known_failure", "label": label, "prompt": q})
    for pid, q in RAG50_MANDATORY_8:
        probes.append({"id": pid, "stage": "stage14_rag50_mandatory", "label": "", "prompt": q})
    for pid, q in FZ_SPECIFIC_MANDATORY:
        probes.append({"id": pid, "stage": "stage9_fz_mandatory", "label": "", "prompt": q})
    for pid, cat, q in load_gt_queries():
        probes.append({"id": pid, "stage": f"stage_gt_{cat}", "label": "", "prompt": q})

    print(f"total probes: {len(probes)}")

    results = []
    for p in probes:
        query = p["prompt"]
        rag_context = rag_pipeline.build_context(query, machine_id=None)
        results.append({
            "id": p["id"], "stage": p["stage"], "label": p["label"], "prompt": query,
            "structured_source_count": len(rag_context.structured_sources),
            "chunk_source_count": len(rag_context.chunk_sources),
            "chunk_titles": [c["title"] for c in rag_context.chunk_sources],
            "prompt_text": rag_context.prompt_text,
            "is_empty": rag_context.is_empty,
        })
        print(f"[{p['stage']}] {p['id']}: chunks={len(rag_context.chunk_sources)} "
              f"structured={len(rag_context.structured_sources)} is_empty={rag_context.is_empty}")

    out_path = REPORTS_DIR / "phase4fz_precomputed_contexts.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(results)} -> {out_path}")


if __name__ == "__main__":
    main()
