"""LLM-jp-3-13B-Instruct (Q4_K_M GGUF) の単体動作確認 (Phase 3.6)。

RTX 5090 への GPU offload 可否、モデルロード時間、VRAM使用量、生成速度、
チャットテンプレートの自動検出、日本語出力の健全性を確認する。
Qwen/Swallow用の chat_format を流用せず、chat_format=None (GGUF埋め込み
テンプレートを自動使用) で検証する。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pachislot_ai.core.config import get_settings  # noqa: E402
from pachislot_ai.llm.base import ChatMessage  # noqa: E402
from pachislot_ai.llm.local_llama_cpp import LocalLlamaCppProvider  # noqa: E402
from pachislot_ai.llm.model_registry import MODEL_REGISTRY  # noqa: E402


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


def main() -> int:
    spec = MODEL_REGISTRY["llm-jp"]
    settings = get_settings()

    print(f"model: {spec.display_name}")
    print(f"path: {spec.path}")
    print(f"size: {spec.path.stat().st_size / (1024**3):.2f} GB")

    vram_before = _nvidia_smi_snapshot()
    print(f"VRAM before load: {vram_before}")

    t0 = time.perf_counter()
    provider = LocalLlamaCppProvider(
        model_path=spec.path,
        n_gpu_layers=settings.llm_n_gpu_layers,
        n_ctx=settings.llm_context_size,
        default_max_tokens=200,
        default_temperature=0.3,
        chat_format=spec.chat_format,
    )
    load_time_sec = time.perf_counter() - t0
    vram_after_load = _nvidia_smi_snapshot()
    print(f"Loaded in {load_time_sec:.2f}s")
    print(f"VRAM after load: {vram_after_load}")

    import asyncio

    async def _chat(llm_provider: LocalLlamaCppProvider, text: str) -> None:
        t0 = time.perf_counter()
        result = await llm_provider.chat(
            [
                ChatMessage(
                    role="system",
                    content="あなたは親切な日本語アシスタントです。簡潔に日本語で回答してください。",
                ),
                ChatMessage(role="user", content=text),
            ],
            max_tokens=150,
            temperature=0.3,
        )
        elapsed = time.perf_counter() - t0
        tok_per_sec = result.completion_tokens / elapsed if elapsed > 0 else 0
        print(f"\nQ: {text}")
        print(f"A: {result.content}")
        print(
            f"[{elapsed:.2f}s, {result.completion_tokens} tok, {tok_per_sec:.1f} tok/s, "
            f"prompt_tokens={result.prompt_tokens}]"
        )

    asyncio.run(_chat(provider, "こんにちは。自己紹介してください。"))
    asyncio.run(_chat(provider, "日本の首都はどこですか？一言で答えてください。"))
    asyncio.run(
        _chat(provider, "パチスロにおける「天井」とは何か、初心者向けに2文で説明してください。")
    )

    vram_during = _nvidia_smi_snapshot()
    print(f"\nVRAM during/after generation: {vram_during}")

    del provider
    import gc

    gc.collect()
    time.sleep(1.0)
    vram_after_unload = _nvidia_smi_snapshot()
    print(f"VRAM after unload: {vram_after_unload}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
