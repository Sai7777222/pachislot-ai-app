"""設定 (`LLM_PROVIDER`) に応じて LLMProvider 実装を生成する。"""

from __future__ import annotations

from pachislot_ai.core.config import Settings
from pachislot_ai.llm.base import LLMProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.lower()

    if provider == "local_llama_cpp":
        from pachislot_ai.llm.local_llama_cpp import LocalLlamaCppProvider

        return LocalLlamaCppProvider(
            model_path=settings.resolved_llm_model_path,
            n_gpu_layers=settings.llm_n_gpu_layers,
            n_ctx=settings.llm_context_size,
            default_max_tokens=settings.llm_max_tokens,
            default_temperature=settings.llm_temperature,
            chat_format=settings.resolved_llm_chat_format,
        )

    # 将来: "cloud_openai" 等をここに追加 (LLMProvider の実装差し替えのみで対応)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
