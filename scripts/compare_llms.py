"""Qwen2.5-14B-Instruct vs Llama-3.1-Swallow-8B-Instruct-v0.5 の A/B比較 (Phase 3.5)。

同一の RagPipeline (Retriever + structured.db検索) を両モデルで使い回し、
LLMProvider だけを差し替えて 17 問すべてに回答させる。結果は JSON に保存し、
docs/llm_comparison.md 作成の元データとする。

このスクリプトはチャット回答処理の実装 (ChatService/RagPipeline) をそのまま
使うため、本番の /v1/chat と全く同じ経路 (RAG検索・プロンプト組立) で比較できる。
"""

from __future__ import annotations

import asyncio
import gc
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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

MACHINE_ID = "smart_million_god_kamigami_no_kiseki"
MAX_TOKENS = 300
TEMPERATURE = 0.3  # 両モデルで完全に同一の値を使う (再現性重視で低め)

QUESTIONS: list[dict] = [
    {"id": "Q1", "group": "構造化データ", "text": "設定6の機械割は？"},
    {"id": "Q2", "group": "構造化データ", "text": "設定6の初当りは？"},
    {"id": "Q3", "group": "構造化データ", "text": "天井は？"},
    {"id": "Q4", "group": "構造化データ", "text": "ガイアベルの確率は？"},
    {"id": "Q5", "group": "RAG文章理解", "text": "Z-ZONEって何？"},
    {"id": "Q6", "group": "RAG文章理解", "text": "GGとSGGの違いを初心者向けに説明して"},
    {"id": "Q7", "group": "RAG文章理解", "text": "この機種のヤメ時を説明して"},
    {"id": "Q8", "group": "RAG文章理解", "text": "ガイアステージについて教えて"},
    {"id": "Q9", "group": "複合質問", "text": "設定6の初当りと機械割を設定1と比較して"},
    {"id": "Q10", "group": "複合質問", "text": "Z-ZONEとGGの関係を説明して"},
    {"id": "Q11", "group": "複合質問", "text": "天井とヤメ時を合わせて初心者向けに説明して"},
    {"id": "Q12", "group": "ハルシネーション耐性", "text": "銀河系ボーナスの発生率は？"},
    {
        "id": "Q13",
        "group": "ハルシネーション耐性",
        "text": "登録データにない架空の設定7の機械割は？",
    },
    {
        "id": "Q14",
        "group": "ハルシネーション耐性",
        "text": "このミリオンゴッドにはボーナス後のリプレイタイム(RT)機能がありますか？",
    },
    {
        "id": "Q15",
        "group": "日本語品質",
        "text": "ミリオンゴッドの遊び方を初心者向けにやさしく説明して",
    },
    {"id": "Q16", "group": "日本語品質", "text": "ミリオンゴッドの遊び方を簡潔に説明して"},
    {"id": "Q17", "group": "日本語品質", "text": "ミリオンゴッドの遊び方を少し詳しく説明して"},
]


def _nvidia_smi_snapshot() -> dict:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        mem_used, mem_total, util = (x.strip() for x in out.stdout.strip().split(","))
        return {
            "vram_used_mib": int(mem_used),
            "vram_total_mib": int(mem_total),
            "gpu_util_pct": int(util),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


class GpuSampler:
    """バックグラウンドで nvidia-smi を定期ポーリングし、期間中の最大VRAM/使用率を記録する。"""

    def __init__(self, interval_sec: float = 0.5) -> None:
        self._interval = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.max_vram_mib = 0
        self.max_util_pct = 0
        self.samples = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            snap = _nvidia_smi_snapshot()
            if "vram_used_mib" in snap:
                self.max_vram_mib = max(self.max_vram_mib, snap["vram_used_mib"])
                self.max_util_pct = max(self.max_util_pct, snap["gpu_util_pct"])
                self.samples += 1
            self._stop.wait(self._interval)

    def __enter__(self) -> GpuSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


async def run_model(model_key: str, rag_pipeline: RagPipeline) -> dict:
    spec = MODEL_REGISTRY[model_key]
    settings = get_settings()

    print(f"\n{'=' * 70}\n{model_key}: {spec.display_name}\n{'=' * 70}")
    print(f"path={spec.path}")
    print(f"size={spec.path.stat().st_size / (1024**3):.2f} GB")
    print(f"system_prompt={spec.system_prompt_path}")

    vram_before = _nvidia_smi_snapshot()
    print(f"VRAM before load: {vram_before}")

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
        for q in QUESTIONS:
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
            print(f"  [{q['id']}] {elapsed:.2f}s, {result.completion_tokens} tok, "
                  f"{tok_per_sec:.1f} tok/s" if tok_per_sec else f"  [{q['id']}] {elapsed:.2f}s")
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

    vram_peak_during_generation = sampler.max_vram_mib
    gpu_util_peak = sampler.max_util_pct

    del provider
    del chat_service
    gc.collect()
    time.sleep(1.0)
    vram_after_unload = _nvidia_smi_snapshot()
    print(f"VRAM after unload: {vram_after_unload}")

    return {
        "model_key": model_key,
        "display_name": spec.display_name,
        "model_path": str(spec.path),
        "model_file_size_gb": round(spec.path.stat().st_size / (1024**3), 3),
        "license_summary": spec.license_summary,
        "load_time_sec": round(load_time_sec, 2),
        "vram_before_mib": vram_before.get("vram_used_mib"),
        "vram_after_load_mib": vram_after_load.get("vram_used_mib"),
        "vram_peak_during_generation_mib": vram_peak_during_generation,
        "vram_after_unload_mib": vram_after_unload.get("vram_used_mib"),
        "gpu_util_peak_pct": gpu_util_peak,
        "gpu_total_vram_mib": vram_before.get("vram_total_mib"),
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

    # RAG検索結果自体がモデル間で変わらないことの確認用に、代表質問のコンテキストを記録
    sample_context = rag_pipeline.build_context("Z-ZONEって何？", machine_id=MACHINE_ID)
    print(f"Sample RAG context chars: {len(sample_context.prompt_text)}")

    results = {
        "machine_id": MACHINE_ID,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "rag_top_k": settings.rag_top_k,
        "sample_rag_context_char_count": len(sample_context.prompt_text),
        "models": [],
    }

    for model_key in ["qwen", "swallow", "llm-jp", "elyza"]:
        model_result = await run_model(model_key, rag_pipeline)
        results["models"].append(model_result)

    out_path = PROJECT_ROOT / "scripts" / "_llm_comparison_raw.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRaw results written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
