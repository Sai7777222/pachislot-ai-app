"""LocalLlamaCppProvider の単体テスト。実モデルをロードするため重い/GPU 必須。"""

from __future__ import annotations

import pytest

from pachislot_ai.core.config import get_settings
from pachislot_ai.llm.base import ChatMessage
from pachislot_ai.llm.local_llama_cpp import LocalLlamaCppProvider

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def provider() -> LocalLlamaCppProvider:
    settings = get_settings()
    return LocalLlamaCppProvider(
        model_path=settings.llm_model_path,
        n_gpu_layers=settings.llm_n_gpu_layers,
        n_ctx=settings.llm_context_size,
        default_max_tokens=64,
        default_temperature=0.7,
    )


async def test_health_check(provider: LocalLlamaCppProvider) -> None:
    assert await provider.health_check() is True


async def test_chat_returns_nonempty_content(provider: LocalLlamaCppProvider) -> None:
    result = await provider.chat([ChatMessage(role="user", content="こんにちは")])
    assert result.content.strip() != ""
    assert result.completion_tokens is not None and result.completion_tokens > 0


async def test_chat_stream_yields_tokens(provider: LocalLlamaCppProvider) -> None:
    chunks = [
        c async for c in provider.chat_stream([ChatMessage(role="user", content="1+1=?")])
    ]
    assert len(chunks) > 0
    assert "".join(chunks).strip() != ""
