"""RAG 全体オーケストレーション (Retriever + 構造化DB検索 + コンテキスト組立)。

DESIGN.md の RAGPipeline に対応。ChatService から呼び出される唯一の入口。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from sqlalchemy import Engine, select

from pachislot_ai.data.db import open_session
from pachislot_ai.data.models.structured import Machine
from pachislot_ai.data.repositories import machine_repository as mrepo
from pachislot_ai.rag.context_builder import RagContext, build_rag_context
from pachislot_ai.rag.entity_attribution import select_grounded_chunks
from pachislot_ai.rag.evidence_arbitration import arbitrate
from pachislot_ai.rag.retriever import Retriever
from pachislot_ai.rag.structured_lookup import find_relevant_structured_facts


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        structured_engine: Engine,
        rag_context_template_path: Path,
        *,
        top_k: int = 6,
    ) -> None:
        self._retriever = retriever
        self._structured_engine = structured_engine
        self._template_path = rag_context_template_path
        self._top_k = top_k

    def _sole_machine_id(self) -> str | None:
        with open_session(self._structured_engine) as session:
            machine_ids = list(session.scalars(select(Machine.machine_id)))
        return machine_ids[0] if len(machine_ids) == 1 else None

    def build_context(self, query: str, *, machine_id: str | None = None) -> RagContext:
        chunks = self._retriever.search(query, machine_id=machine_id, top_k=self._top_k)

        effective_machine_id = machine_id
        if effective_machine_id is None and chunks:
            # machine_id 未指定の場合、検索でヒットした上位チャンクの機種を
            # 構造化DB検索にも使う (機種が1つしか登録されていない現段階では常に一致する)
            effective_machine_id = chunks[0].machine_id or None
        if effective_machine_id is None:
            effective_machine_id = self._sole_machine_id()

        structured_findings = []
        structured_sources: list[dict] = []
        machine_name: str | None = None
        if effective_machine_id:
            with open_session(self._structured_engine) as session:
                machine = mrepo.get_machine(session, effective_machine_id)
                machine_name = machine.name if machine else effective_machine_id
                structured_findings = find_relevant_structured_facts(
                    session, effective_machine_id, query
                )
                source_ids = sorted(
                    {f.source_id for f in structured_findings if f.source_id is not None}
                )
                for source_id in source_ids:
                    src = mrepo.get_source(session, source_id)
                    if src is not None:
                        structured_sources.append(
                            {
                                "source_id": src.id,
                                "url": src.url,
                                "label": src.label,
                                "data_source_type": src.data_source_type,
                            }
                        )
            if machine_id is None:
                # 機種が確定したら、その機種に絞ってチャンクを取り直す (他機種のノイズを減らす)
                refined = self._retriever.search(
                    query, machine_id=effective_machine_id, top_k=self._top_k
                )
                if refined:
                    chunks = refined

        # entity-aware context assembly (Phase4FX/4FY): query entityとretrieved chunkを
        # titleメタデータでbindingし、query-boundなchunkのみに絞り込む。他機種混入や
        # クロスエンティティ誤帰属を防ぐため、all_chunksは常にeffective_machine_idで
        # スコープする(新しい検索器・新しいembeddingは使わない)。
        if chunks:
            all_chunks = self._retriever.get_all_chunks(machine_id=effective_machine_id)
            chunks = select_grounded_chunks(query, chunks, all_chunks)

        # evidence arbitration (Phase4FC3): chunk側のno-evidence合成マーカーが、
        # 独立したstructured facts側の実データと矛盾しないようにする。
        # entity_attribution.py・structured_lookup.py 双方の内部ロジックは無変更。
        chunks = arbitrate(chunks, structured_findings)

        context = build_rag_context(
            self._template_path,
            structured_findings=structured_findings,
            chunks=chunks,
            machine_name=machine_name,
        )
        return replace(context, structured_sources=structured_sources)
