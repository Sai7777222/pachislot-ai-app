"""ChatService の単体テスト（実モデルを使わないフェイク LLMProvider）。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from pachislot_ai.core.config import SYSTEM_PROMPT_PATH
from pachislot_ai.llm.base import ChatCompletionResult, ChatMessage, LLMProvider
from pachislot_ai.services.chat_service import ChatService


class FakeLLMProvider(LLMProvider):
    def __init__(self) -> None:
        self.received_messages: list[ChatMessage] | None = None

    async def chat(  # noqa: ANN001
        self, messages, *, max_tokens=None, temperature=None
    ) -> ChatCompletionResult:
        self.received_messages = messages
        return ChatCompletionResult(
            content="fake response", prompt_tokens=1, completion_tokens=2
        )

    async def chat_stream(  # noqa: ANN001
        self, messages, *, max_tokens=None, temperature=None
    ) -> AsyncIterator[str]:
        self.received_messages = messages
        for token in ["fake", " ", "stream"]:
            yield token

    async def health_check(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "fake-model"


async def test_chat_prepends_system_prompt() -> None:
    provider = FakeLLMProvider()
    service = ChatService(provider, SYSTEM_PROMPT_PATH)

    result = await service.chat([ChatMessage(role="user", content="こんにちは")])

    assert result.content == "fake response"
    assert provider.received_messages is not None
    assert provider.received_messages[0].role == "system"
    assert provider.received_messages[-1] == ChatMessage(role="user", content="こんにちは")


async def test_chat_stream_yields_tokens() -> None:
    provider = FakeLLMProvider()
    service = ChatService(provider, SYSTEM_PROMPT_PATH)

    tokens = [t async for t in service.chat_stream([ChatMessage(role="user", content="hi")])]

    assert tokens == ["fake", " ", "stream"]


async def test_health_check_delegates_to_provider() -> None:
    provider = FakeLLMProvider()
    service = ChatService(provider, SYSTEM_PROMPT_PATH)

    assert await service.health_check() is True
