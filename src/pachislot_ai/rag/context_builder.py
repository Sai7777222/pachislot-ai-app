"""structured.db の検索結果 + Vector DB のチャンクを、LLM へ渡す
コンテキスト文字列に組み立てる (`config/prompts/rag_context.jinja2`)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template

from pachislot_ai.rag.retriever import RetrievedChunk
from pachislot_ai.rag.structured_lookup import StructuredFinding


@dataclass(frozen=True, slots=True)
class RagContext:
    """LLM へ渡す文字列 + 内部トレース用の出典情報 (source_id / source_url) の両方を持つ。"""

    prompt_text: str
    structured_source_ids: list[int]
    structured_sources: list[dict]  # [{"source_id", "url", "label", "data_source_type"}]
    chunk_sources: list[dict]  # [{"title", "source_url", "source_label", "chunk_id", "score"}]
    is_empty: bool


def build_rag_context(
    template_path: Path,
    *,
    structured_findings: list[StructuredFinding],
    chunks: list[RetrievedChunk],
    machine_name: str | None = None,
) -> RagContext:
    template = Template(template_path.read_text(encoding="utf-8"))

    structured_lines = [f.detail for f in structured_findings]
    chunk_dicts = [
        {"title": c.title, "category": c.category, "text": c.text} for c in chunks
    ]

    # 機種名を明示しないと、LLM が学習知識から別の機種名を補完してしまう
    # (ハルシネーション) ことがあるため、常に対象機種を明記する。
    prompt_text = template.render(
        machine_name=machine_name, structured_facts=structured_lines, chunks=chunk_dicts
    ).strip()

    structured_source_ids = sorted(
        {f.source_id for f in structured_findings if f.source_id is not None}
    )
    chunk_sources = [
        {
            "chunk_id": c.chunk_id,
            "title": c.title,
            "source_url": c.source_url,
            "source_label": c.source_label,
            "score": c.score,
        }
        for c in chunks
    ]

    return RagContext(
        prompt_text=prompt_text,
        structured_source_ids=structured_source_ids,
        structured_sources=[],  # RagPipeline 側で source 解決後に差し替える
        chunk_sources=chunk_sources,
        is_empty=not structured_findings and not chunks,
    )
