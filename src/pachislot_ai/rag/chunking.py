"""RagDocument (rag_store.db) を意味のまとまり (段落 → 文) でチャンク化する。

Phase 2/2.5 の時点で RagDocument は既にブロック単位 (機種概要、ゾーン解説、
演出法則の解説など) に分かれており、多くは十分に短い。長い文書だけを
段落単位、それでも長い場合は文単位でさらに分割する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT_RE = re.compile(r"(?<=。)")


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str
    chunk_index: int


def chunk_text(text: str, *, max_chars: int = 500) -> list[TextChunk]:
    """段落 (改行区切り) を基本単位とし、必要な場合のみ文単位・強制分割で細分化する。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [TextChunk(text=text, chunk_index=0)]

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    pieces: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            pieces.append(current)
            current = ""

        if len(para) <= max_chars:
            current = para
        else:
            pieces.extend(_split_long_text(para, max_chars))

    if current:
        pieces.append(current)

    return [TextChunk(text=p, chunk_index=i) for i, p in enumerate(pieces)]


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """1段落が長すぎる場合、文単位 (「。」区切り) で分割する。それでも長い文は強制分割する。"""
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s]
    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = current + sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = sentence

    if current:
        pieces.append(current)

    # 1文だけでも max_chars を超える場合の最終手段
    result: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            result.append(piece)
        else:
            result.extend(piece[i : i + max_chars] for i in range(0, len(piece), max_chars))
    return result


def chunk_rag_document(doc, *, max_chars: int = 500) -> list[dict]:
    """RagDocument (ORM オブジェクト) から Vector DB 投入用のチャンクレコードを作る。

    出典 (source_url / data_source_type / retrieved_at 等) はすべての
    チャンクに引き継ぐ (チャンク単位でも出典を追跡できるようにする)。
    """
    text_chunks = chunk_text(doc.body_text, max_chars=max_chars)
    records = []
    for tc in text_chunks:
        records.append(
            {
                "chunk_id": f"{doc.doc_id}::chunk{tc.chunk_index}",
                "doc_id": doc.doc_id,
                "machine_id": doc.machine_id,
                "category": doc.category,
                "title": doc.title,
                "text": tc.text,
                "chunk_index": tc.chunk_index,
                "chunk_count": len(text_chunks),
                "source_url": doc.source_url,
                "source_label": doc.source_label,
                "data_source_type": doc.data_source_type,
                "retrieved_at": doc.retrieved_at.isoformat() if doc.retrieved_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                "review_status": doc.review_status,
            }
        )
    return records
