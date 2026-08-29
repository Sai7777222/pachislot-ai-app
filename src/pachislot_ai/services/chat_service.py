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

from pachislot_ai.dispatch import dispatch
from pachislot_ai.dispatch.conservative_dispatch import (
    IDENTITY_PERSONA,
    OOD_FACTUAL,
    SMALL_TALK,
)
from pachislot_ai.llm.base import ChatCompletionResult, ChatMessage, LLMProvider
from pachislot_ai.rag.context_builder import RagContext
from pachislot_ai.rag.pipeline import RagPipeline

# Phase4FC3: これらのモードはRAG context system messageを一切注入しない
# (雑談・自己紹介・専門外の質問に「登録データにありません」という不自然な
# 断り書きが混入するのを防ぐ、FC2 Gate H/Kで確認された regression の修正)。
_NO_RAG_CONTEXT_MODES = frozenset({SMALL_TALK, IDENTITY_PERSONA, OOD_FACTUAL})

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

        # Phase4FC3 Section2/6-11: production dispatch。SMALL_TALK/IDENTITY_PERSONA/
        # OOD_FACTUALと**確信を持って**判定された場合のみretrieval自体を行わず
        # RAG contextをNoneのまま返す(「登録データにありません」が雑談・自己紹介に
        # 混入する既存アーキテクチャギャップを解消する、FC2 Gate H/Kで確認済みの
        # regression修正)。PACHISLOT_FACTUAL/PACHISLOT_CONVERSATIONALは既存のRAG
        # pipelineへそのまま委譲する(entity-aware chunk binding・structured facts
        # binding・title補完検索、いずれも無変更)。
        #
        # UNKNOWN(確信が持てない発話)は、必ずRAG pipelineを通す(＝空contextでも
        # 明示的なfallback文言を注入する、既存の安全側デフォルトを維持する)。
        # これは意図的な設計判断: 「GGプラスとは何か説明して」等のphantomなパチスロ
        # 固有名詞クエリは、GENERAL_PACHISLOT_TERMSのような一般語彙に一致しないため
        # dispatch()の結果はUNKNOWNになる。ここでcontextの注入を省略すると、
        # Phase4ZGが「検索結果:該当なし」という明示的signalを失い、自身の内部知識
        # から自信満々に架空の説明を創作してしまうことをablation testで実証した
        # (「GGプラス」で確認: contextありなら正しくdecline、contextを完全に
        # 省略すると具体的な架空説明を生成した)。したがってUNKNOWN + is_empty を
        # small-talk同様に「contextを省略してよい」ケースとして扱ってはならない
        # (Section11の『obvious non-RAG conversation』はSMALL_TALK/IDENTITY_PERSONA/
        # OOD_FACTUALのような確信の持てるケースのみを指すと解釈する)。
        dispatch_result = dispatch(query)
        if dispatch_result.mode in _NO_RAG_CONTEXT_MODES:
            return None

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
