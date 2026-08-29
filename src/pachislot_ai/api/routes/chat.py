from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from pachislot_ai.api.deps import get_chat_service
from pachislot_ai.api.schemas.chat import (
    ChatMessageOut,
    ChatRequest,
    ChatResponse,
    ChunkSourceRef,
    SourcesInfo,
    StructuredSourceRef,
    UsageInfo,
)
from pachislot_ai.core.exceptions import LLMGenerationError
from pachislot_ai.llm.base import ChatMessage
from pachislot_ai.rag.context_builder import RagContext
from pachislot_ai.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter()


def _to_domain_messages(request: ChatRequest) -> list[ChatMessage]:
    return [ChatMessage(role=m.role, content=m.content) for m in request.messages]


def _sources_info(sources: dict) -> SourcesInfo:
    return SourcesInfo(
        structured_sources=[
            StructuredSourceRef(**s) for s in sources.get("structured_sources", [])
        ],
        chunk_sources=[ChunkSourceRef(**c) for c in sources.get("chunk_sources", [])],
    )


@router.post("/chat", response_model=ChatResponse)
async def post_chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    result = await service.chat(
        _to_domain_messages(request),
        machine_id=request.machine_id,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    logger.info(
        "chat: machine_id=%s structured_sources=%d chunk_sources=%d",
        request.machine_id,
        len(result.sources.get("structured_sources", [])),
        len(result.sources.get("chunk_sources", [])),
    )
    return ChatResponse(
        message=ChatMessageOut(content=result.content),
        model=service.model_name,
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        ),
        sources=_sources_info(result.sources),
    )


@router.post("/chat/stream")
async def post_chat_stream(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    messages = _to_domain_messages(request)

    # Phase4FM Section9: HARD_BLOCK_INPUT対象ならRAG検索(build_rag_context)自体を
    # 一切呼ばない。これはChatService.chat_stream()内部の入力チェックより前段の
    # 防御であり、production streamingパス自体でRAG_called=0を保証する
    # (Section17のmandatory gate)。
    input_mod = service.check_input(messages)

    async def event_generator():
        if not input_mod.allowed:
            empty_sources = json.dumps(_sources_info({}).model_dump(), ensure_ascii=False)
            yield f"event: sources\ndata: {empty_sources}\n\n"
            payload = json.dumps({"delta": input_mod.safe_response or ""}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        # RAG検索は事前に1回だけ実行し、sourcesイベントとLLM入力コンテキストの両方に使い回す
        rag_context: RagContext | None = service.build_rag_context(messages, request.machine_id)
        sources_payload = json.dumps(
            _sources_info(
                {
                    "structured_sources": rag_context.structured_sources if rag_context else [],
                    "chunk_sources": rag_context.chunk_sources if rag_context else [],
                }
            ).model_dump(),
            ensure_ascii=False,
        )
        yield f"event: sources\ndata: {sources_payload}\n\n"

        try:
            async for delta in service.chat_stream(
                messages,
                machine_id=request.machine_id,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                rag_context=rag_context,
            ):
                payload = json.dumps({"delta": delta}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            yield "event: done\ndata: {}\n\n"
        except LLMGenerationError as exc:
            logger.exception("LLM generation failed during streaming")
            payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
