from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessageIn] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    # Phase 3: 任意指定。指定時はその機種のRAG/構造化DBを優先検索する。
    # 未指定時は Vector DB 全体から検索し、ヒットした機種の構造化DBを自動的に使う。
    machine_id: str | None = Field(default=None, max_length=128)


class ChatMessageOut(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class UsageInfo(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class StructuredSourceRef(BaseModel):
    source_id: int
    url: str
    label: str | None = None
    data_source_type: str


class ChunkSourceRef(BaseModel):
    chunk_id: str
    title: str
    source_url: str
    source_label: str | None = None
    score: float


class SourcesInfo(BaseModel):
    structured_sources: list[StructuredSourceRef] = Field(default_factory=list)
    chunk_sources: list[ChunkSourceRef] = Field(default_factory=list)


class ChatResponse(BaseModel):
    message: ChatMessageOut
    model: str
    usage: UsageInfo | None = None
    # Phase 3: RAGで参照した出典 (機種情報が未登録/RAG未初期化の場合は空リスト)
    sources: SourcesInfo = Field(default_factory=SourcesInfo)
