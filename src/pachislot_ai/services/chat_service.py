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

from pachislot_ai.core.config import (
    IDENTITY_PERSONA_PROMPT_PATH,
    MODERATION_POLICY_PATH,
    OOD_BOUNDARY_PROMPT_PATH,
    SMALL_TALK_PROMPT_PATH,
)
from pachislot_ai.dispatch import dispatch
from pachislot_ai.dispatch.conservative_dispatch import (
    IDENTITY_PERSONA,
    OOD_FACTUAL,
    SMALL_TALK,
)
from pachislot_ai.llm.base import ChatCompletionResult, ChatMessage, LLMProvider
from pachislot_ai.moderation import ModerationEngine, ModerationResult
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
        *,
        small_talk_prompt_path: Path = SMALL_TALK_PROMPT_PATH,
        identity_persona_prompt_path: Path = IDENTITY_PERSONA_PROMPT_PATH,
        ood_boundary_prompt_path: Path = OOD_BOUNDARY_PROMPT_PATH,
        moderation_policy_path: Path = MODERATION_POLICY_PATH,
    ) -> None:
        self._llm = llm
        self._system_prompt = _load_system_prompt(system_prompt_path)
        self._rag_pipeline = rag_pipeline
        # Phase4FM: 決定的モデレーション層。RAG/dispatch/生成より前の入力チェックと、
        # 生成後・ユーザー表示前の出力チェックの両方に、同一エンジンを使う。
        self._moderation = ModerationEngine.from_yaml(moderation_policy_path)
        # Phase4FC4: SMALL_TALK/IDENTITY_PERSONA/OOD_FACTUALと確信を持って判定された
        # 場合、事実RAG用system.jinja2(数値の厳密な取り扱いを繰り返し指示する内容で、
        # 雑談文脈にまで過度な慎重さを持ち込む一因になっていた)の代わりに、この短い
        # mode-specific promptで**置き換える**(積み増しではない、1リクエストにつき
        # 有効なsystem policyは常に1つ)。
        self._mode_system_prompts = {
            SMALL_TALK: _load_system_prompt(small_talk_prompt_path),
            IDENTITY_PERSONA: _load_system_prompt(identity_persona_prompt_path),
            OOD_FACTUAL: _load_system_prompt(ood_boundary_prompt_path),
        }

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

    def check_input(self, user_messages: list[ChatMessage]) -> ModerationResult:
        """Section9: dispatch/RAG/生成より前に呼ぶ、入力側モデレーション判定。
        API層(/v1/chat/stream)がRAG検索より前に単独で呼べるよう公開している
        (build_rag_context()と同じ設計方針)。"""
        if not user_messages:
            return self._moderation.check_input("")
        return self._moderation.check_input(user_messages[-1].content)

    def check_output(self, text: str) -> ModerationResult:
        """Section11: 生成後・ユーザー表示前に呼ぶ、出力側モデレーション判定。"""
        return self._moderation.check_output(text)

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

    def _select_system_prompt(self, user_messages: list[ChatMessage]) -> str:
        """Phase4FC4: SMALL_TALK/IDENTITY_PERSONA/OOD_FACTUALと確信を持って判定
        された場合はmode-specific promptへ**置き換える**(system.jinja2は使わない)。
        それ以外(PACHISLOT_FACTUAL/CONVERSATIONAL/UNKNOWN)は既存のsystem.jinja2
        のまま変更しない。build_rag_context()と同じdispatch()呼び出しを独立に行う
        (dispatchは純粋関数で計算コストも無視できるため、二重呼び出しの問題はない)。"""
        if not user_messages:
            return self._system_prompt
        dispatch_result = dispatch(user_messages[-1].content)
        return self._mode_system_prompts.get(dispatch_result.mode, self._system_prompt)

    def _build_messages(
        self, user_messages: list[ChatMessage], rag_context: RagContext | None
    ) -> list[ChatMessage]:
        system_prompt = self._select_system_prompt(user_messages)
        messages = [ChatMessage(role="system", content=system_prompt)]
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
        # Section9: HARD_BLOCK_INPUT対象なら、dispatch/RAG/生成を一切呼ばずに
        # 安全な代替応答を即座に返す。
        input_mod = self.check_input(user_messages)
        if not input_mod.allowed:
            return ChatAnswer(
                content=input_mod.safe_response or "",
                prompt_tokens=None,
                completion_tokens=None,
                sources=_sources_dict(None),
            )

        if rag_context is _NOT_GIVEN:
            rag_context = self.build_rag_context(user_messages, machine_id)
        messages = self._build_messages(user_messages, rag_context)  # type: ignore[arg-type]
        result: ChatCompletionResult = await self._llm.chat(
            messages, max_tokens=max_tokens, temperature=temperature
        )

        # Section11: 生成後・ユーザー表示前の出力チェック。ブロックされた場合、
        # ユーザーに見えるcontentのみを安全な代替文へ差し替える(sources/使用量は
        # 検索メタデータであり、禁止表現そのものではないためそのまま保持する)。
        output_mod = self.check_output(result.content)
        final_content = output_mod.safe_response if not output_mod.allowed else result.content

        return ChatAnswer(
            content=final_content or "",
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            sources=_sources_dict(rag_context),  # type: ignore[arg-type]
        )

    async def chat_stream(
        self,
        user_messages: list[ChatMessage],
        *,
        machine_id: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        rag_context: RagContext | None | object = _NOT_GIVEN,
    ) -> AsyncIterator[str]:
        # Section9: 入力ブロック時はdispatch/RAG/生成を一切呼ばない。
        input_mod = self.check_input(user_messages)
        if not input_mod.allowed:
            yield input_mod.safe_response or ""
            return

        if rag_context is _NOT_GIVEN:
            rag_context = self.build_rag_context(user_messages, machine_id)
        messages = self._build_messages(user_messages, rag_context)  # type: ignore[arg-type]

        # Section12: streamingは現状トークン即時送出のため、生成後チェックだけでは
        # 既に禁止表現がクライアントへ届いてしまう恐れがある。このPhaseでは
        # 「生成が完了するまでバッファリングし、モデレーション判定後に送出する」
        # 方針を採用する(複雑な逐次検閲は行わない、Section12の明示的な指示通り)。
        buffer: list[str] = []
        async for delta in self._llm.chat_stream(messages, max_tokens=max_tokens, temperature=temperature):
            buffer.append(delta)
        full_text = "".join(buffer)

        output_mod = self.check_output(full_text)
        yield output_mod.safe_response if not output_mod.allowed else full_text

    async def health_check(self) -> bool:
        return await self._llm.health_check()
