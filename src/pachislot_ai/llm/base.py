"""LLMProvider 抽象インターフェース。

DESIGN.md の「4. ローカル LLM と RAG の接続方法」に対応。
RAG パイプライン・API は常にこの抽象クラス経由で推論し、具体的な
推論エンジン (llama.cpp / vLLM / クラウド API) を知らない。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class ChatCompletionResult:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMProvider(ABC):
    """推論エンジンの抽象インターフェース。"""

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletionResult:
        """非ストリーミングで応答全体を生成する。"""

    @abstractmethod
    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """トークン（テキスト断片）をストリーミングで生成する。"""

    @abstractmethod
    async def health_check(self) -> bool:
        """モデルが応答可能な状態かどうか。"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """クライアントに返すモデル識別子。"""
