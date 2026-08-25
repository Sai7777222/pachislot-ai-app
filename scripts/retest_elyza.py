"""ELYZA専用改善プロンプトの再テスト (Phase 3.8)。

Qwen/Swallow/LLM-jpは一切呼び出さない (負荷・時間を抑えるため、かつ他モデルに
一切影響を与えないことを構造的に保証する)。ELYZAのみを対象に、

1. 既存17問 (scripts/compare_llms.py の QUESTIONS を無変更でそのままimport)
2. 新規5問 (今回追加。汎化性能の確認用。正解はstructured.db/rag_store.dbの
   実データを事前に直接確認して用意したものであり、LLM自身の回答を正解とはしない)

を、改善後の system prompt (`config/prompts/system_elyza.jinja2`) で実行する。

structured.db / rag_store.db / Vector DB / RagPipeline の実装は一切変更しない。
"""

from __future__ import annotations

import asyncio
import gc
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compare_llms import (  # noqa: E402
    MACHINE_ID,
    MAX_TOKENS,
    QUESTIONS,
    TEMPERATURE,
    GpuSampler,
    _nvidia_smi_snapshot,
)

from pachislot_ai.core.config import RAG_CONTEXT_PROMPT_PATH, get_settings  # noqa: E402
from pachislot_ai.data.db import create_structured_engine  # noqa: E402
from pachislot_ai.llm.base import ChatMessage  # noqa: E402
from pachislot_ai.llm.local_llama_cpp import LocalLlamaCppProvider  # noqa: E402
from pachislot_ai.llm.model_registry import MODEL_REGISTRY  # noqa: E402
from pachislot_ai.rag.embedder import Embedder  # noqa: E402
from pachislot_ai.rag.pipeline import RagPipeline  # noqa: E402
from pachislot_ai.rag.retriever import Retriever  # noqa: E402
from pachislot_ai.rag.vector_store import VectorStore  # noqa: E402
from pachislot_ai.services.chat_service import ChatService  # noqa: E402

# 新規5問。既存17問の言い換えではなく、Phase 3.7で確認された
# 「別ゾーンからの数値転用」「未登録数値の創作」パターンを別の切り口で検出する。
# 正解は structured.db / rag_store.db の実データを直接確認して記載したもの
# (LLM自身の回答を正解として採用してはいない)。
NEW_QUESTIONS: list[dict] = [
    {
        "id": "N1",
        "group": "複数ゾーン比較(数値混同防止)",
        "text": "GGとPGG（プレミアムゴッドゲーム）の獲得枚数の違いを教えてください。",
        "expected_answer_note": (
            "正解: GGの平均獲得枚数は350枚。PGGの期待枚数は3000枚以上"
            "（GOD揃い後に突入、最低でもGG4セット以上を獲得できる）。"
            "zones.GG.attributes_json['平均獲得']=350枚、"
            "zones.プレミアムゴッドゲーム(PGG).attributes_json['期待枚数']=3000枚以上 "
            "（structured.db zonesテーブルより確認）。"
            "2つの数値を混同・入れ替えないことが焦点。"
        ),
    },
    {
        "id": "N2",
        "group": "片方にしか存在しない数値の比較",
        "text": "GGとSGGそれぞれの継続率を教えてください。",
        "expected_answer_note": (
            "正解: SGGの継続率は75%以上と登録されている"
            "（zones.SGG / zones.スーパーゴッドゲーム(SGG).attributes_json['継続率']）。"
            "GGには継続率という数値は登録データに存在しない"
            "（zones.GG / zones.ゴッドゲーム(GG)にはループストック最大80%という"
            "別概念の数値のみ登録、継続率フィールドなし）。"
            "SGGの75%以上をGGにも適用するのは誤り。"
        ),
    },
    {
        "id": "N3",
        "group": "DBに存在しない数値",
        "text": "ゼウスモード中の純増枚数（約何枚/G）を教えてください。",
        "expected_answer_note": (
            "正解: 登録データにありません。"
            "zones.ゼウスモード.attributes_jsonには「突入契機」「滞在中の恩恵」"
            "「継続抽選」「ゼウスモード解説」はあるが、純増枚数の数値は含まれていない"
            "（structured.db zonesテーブルより確認）。"
            "GG/PGG/Z-ZONE等の「約7枚/G」を転用してはならない。"
        ),
    },
    {
        "id": "N4",
        "group": "RAGに説明はあるが数値がない",
        "text": "非ガイアステージ中のZ-ZONE昇格率は何パーセントですか？",
        "expected_answer_note": (
            "正解: 登録データにありません（数値の記載なし）。"
            "rag_store.dbの該当チャンク（タイトル『昇格解説』、"
            "セクション『Z-ZONE昇格率 (非ガイアステージ)』）の本文は"
            "「GG当選時に前兆ゲーム数が決まったタイミングでZ-ZONEへの昇格を抽選する。」"
            "のみで、具体的な%数値は記載されていない。"
        ),
    },
    {
        "id": "N5",
        "group": "存在しない機能",
        "text": "この機種にはスイカ小役がありますか？あるとすれば確率を教えてください。",
        "expected_answer_note": (
            "正解: 登録データにありません。"
            "rag_store.db・structured.dbを全文検索しても「スイカ」という語は"
            "この機種のデータに一切出現しない（0件）。"
        ),
    },
]


async def run_elyza(rag_pipeline: RagPipeline, questions: list[dict], label: str) -> dict:
    spec = MODEL_REGISTRY["elyza"]
    settings = get_settings()

    print(f"\n{'=' * 70}\nelyza ({label}): {spec.display_name}\n{'=' * 70}")
    print(f"system_prompt={spec.system_prompt_path}")

    vram_before = _nvidia_smi_snapshot()
    t0 = time.perf_counter()
    provider = LocalLlamaCppProvider(
        model_path=spec.path,
        n_gpu_layers=settings.llm_n_gpu_layers,
        n_ctx=settings.llm_context_size,
        default_max_tokens=MAX_TOKENS,
        default_temperature=TEMPERATURE,
        chat_format=spec.chat_format,
    )
    load_time_sec = time.perf_counter() - t0
    vram_after_load = _nvidia_smi_snapshot()
    print(f"Loaded in {load_time_sec:.2f}s. VRAM after load: {vram_after_load}")

    chat_service = ChatService(provider, spec.system_prompt_path, rag_pipeline)

    answers = []
    with GpuSampler(interval_sec=0.3) as sampler:
        for q in questions:
            t0 = time.perf_counter()
            result = await chat_service.chat(
                [ChatMessage(role="user", content=q["text"])],
                machine_id=MACHINE_ID,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            elapsed = time.perf_counter() - t0
            tok_per_sec = (
                result.completion_tokens / elapsed
                if elapsed > 0 and result.completion_tokens
                else None
            )
            print(f"  [{q['id']}] {elapsed:.2f}s, {result.completion_tokens} tok")
            answers.append(
                {
                    "id": q["id"],
                    "group": q["group"],
                    "question": q["text"],
                    "answer": result.content,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "elapsed_sec": round(elapsed, 3),
                    "tokens_per_sec": round(tok_per_sec, 2) if tok_per_sec else None,
                    "structured_sources": len(result.sources.get("structured_sources", [])),
                    "chunk_sources": len(result.sources.get("chunk_sources", [])),
                }
            )

    vram_peak = sampler.max_vram_mib
    gpu_util_peak = sampler.max_util_pct

    del provider
    del chat_service
    gc.collect()
    time.sleep(1.0)
    vram_after_unload = _nvidia_smi_snapshot()
    print(f"VRAM after unload: {vram_after_unload}")

    return {
        "label": label,
        "model_key": "elyza",
        "display_name": spec.display_name,
        "system_prompt_path": str(spec.system_prompt_path),
        "load_time_sec": round(load_time_sec, 2),
        "vram_before_mib": vram_before.get("vram_used_mib"),
        "vram_after_load_mib": vram_after_load.get("vram_used_mib"),
        "vram_peak_during_generation_mib": vram_peak,
        "vram_after_unload_mib": vram_after_unload.get("vram_used_mib"),
        "gpu_util_peak_pct": gpu_util_peak,
        "answers": answers,
    }


async def main() -> int:
    settings = get_settings()

    print("Building shared RAG pipeline (Retriever + structured.db)...")
    embedder = Embedder(settings.embedding_model_path, device=settings.embedding_device)
    vector_store = VectorStore(settings.vector_db_path, settings.vector_db_collection)
    retriever = Retriever(embedder, vector_store, default_top_k=settings.rag_top_k)
    structured_engine = create_structured_engine(settings.structured_db_path)
    rag_pipeline = RagPipeline(
        retriever, structured_engine, RAG_CONTEXT_PROMPT_PATH, top_k=settings.rag_top_k
    )

    results = {
        "machine_id": MACHINE_ID,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "rag_top_k": settings.rag_top_k,
        "runs": [],
    }

    existing17_result = await run_elyza(rag_pipeline, QUESTIONS, "existing_17_questions")
    results["runs"].append(existing17_result)

    new5_result = await run_elyza(rag_pipeline, NEW_QUESTIONS, "new_5_questions")
    # 正解メモ (LLMの回答ではなく、事前にDBを確認して用意したもの) を併記して保存
    for a, q in zip(new5_result["answers"], NEW_QUESTIONS, strict=True):
        a["expected_answer_note"] = q["expected_answer_note"]
    results["runs"].append(new5_result)

    out_path = PROJECT_ROOT / "scripts" / "_elyza_retest_raw.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRaw results written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
