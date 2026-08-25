"""チャットフロー統合。

Phase 3: パチスロ向けシステムプロンプト + RAGコンテキスト (構造化DBの数値 +
Vector DB の解説文章) + ユーザー入力を LLMProvider に渡す。RAG検索は
`RagPipeline` に委譲し、ChatService 自体は「注入する」役割に徹する。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template

from pachislot_ai.llm.base import ChatCompletionResult, ChatMessage, LLMProvider
from pachislot_ai.rag.context_builder import RagContext
from pachislot_ai.rag.pipeline import RagPipeline

logger = logging.getLogger(__name__)

_NOT_GIVEN = object()


def _load_system_prompt(path: Path) -> str:
    template = Template(path.read_text(encoding="utf-8"))
    return template.render()


@dataclass(frozen=True, slots=True)
class ChatAnswer:
    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
    sources: dict  # {"structured_sources": [...], "chunk_sources": [...]}


def _sources_dict(rag_context: RagContext | None) -> dict:
    if rag_context is None:
        return {"structured_sources": [], "chunk_sources": []}
    return {
        "structured_sources": rag_context.structured_sources,
        "chunk_sources": rag_context.chunk_sources,
    }


class ChatService:
    def __init__(
        self,
        llm: LLMProvider,
        system_prompt_path: Path,
        rag_pipeline: RagPipeline | None = None,
    ) -> None:
        self._llm = llm
        self._system_prompt = _load_system_prompt(system_prompt_path)
        self._rag_pipeline = rag_pipeline

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    @property
    def llm(self) -> LLMProvider:
        """/v1/health でプロバイダー詳細を表示するための読み取り専用アクセサ。"""
        return self._llm

    @property
    def rag_enabled(self) -> bool:
        return self._rag_pipeline is not None

    def build_rag_context(
        self, user_messages: list[ChatMessage], machine_id: str | None
    ) -> RagContext | None:
        """検索を実行しコンテキストを組み立てる。RAG未構成時は None。

        API層が /v1/chat/stream で先に出典情報 (sources イベント) を送るため、
        chat()/chat_stream() より前に単独で呼び出せるよう公開している。
        """
        if self._rag_pipeline is None or not user_messages:
            return None
        query = user_messages[-1].content
        try:
            return self._rag_pipeline.build_context(query, machine_id=machine_id)
        except Exception:  # noqa: BLE001 - RAG障害でチャット自体は継続させる
            logger.exception("RAG context retrieval failed; continuing without RAG context")
            return None

    def _build_messages(
        self, user_messages: list[ChatMessage], rag_context: RagContext | None
    ) -> list[ChatMessage]:
        messages = [ChatMessage(role="system", content=self._system_prompt)]
        if rag_context is not None and rag_context.prompt_text:
            # is_empty (該当データなし) の場合もテンプレート側で「登録されていません」
            # という文言をレンダリングしているため、常に注入する
            # (そうしないと LLM が検索を試みたことすら分からず、内部知識で補完しがちになる)
            messages.append(ChatMessage(role="system", content=rag_context.prompt_text))
        messages.extend(user_messages)
        return messages

    async def chat(
        self,
        user_messages: list[ChatMessage],
        *,
        machine_id: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        rag_context: RagContext | None | object = _NOT_GIVEN,
    ) -> ChatAnswer:
        if rag_context is _NOT_GIVEN:
            rag_context = self.build_rag_context(user_messages, machine_id)
        messages = self._build_messages(user_messages, rag_context)  # type: ignore[arg-type]
        result: ChatCompletionResult = await self._llm.chat(
            messages, max_tokens=max_tokens, temperature=temperature
        )
        return ChatAnswer(
            content=result.content,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            sources=_sources_dict(rag_context),  # type: ignore[arg-type]
        )

    def chat_stream(
        self,
        user_messages: list[ChatMessage],
        *,
        machine_id: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        rag_context: RagContext | None | object = _NOT_GIVEN,
    ) -> AsyncIterator[str]:
        if rag_context is _NOT_GIVEN:
            rag_context = self.build_rag_context(user_messages, machine_id)
        messages = self._build_messages(user_messages, rag_context)  # type: ignore[arg-type]
        return self._llm.chat_stream(messages, max_tokens=max_tokens, temperature=temperature)

    async def health_check(self) -> bool:
        return await self._llm.health_check()
