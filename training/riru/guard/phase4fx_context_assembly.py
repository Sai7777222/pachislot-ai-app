# -*- coding: utf-8 -*-
"""Phase4FX Stage: context assembly候補(A0/A1/A2)。"""
from __future__ import annotations


def build_context_a0(embedding_chunks: list[dict]) -> str:
    """A0: 現行のcontext組み立て(embedding top-kをそのまま結合)。baseline。"""
    return "\n".join(f"[{c['title']}] {c['text']}" for c in embedding_chunks)


def build_context_a1(query_entities: list[str], binding: dict) -> str:
    """A1: entity-grouped。各query entityごとにevidenceをグループ化して提示し、
    bindingされたevidenceが0件のentityには明示的に「NO GROUNDED EVIDENCE」を記載する。
    UNBOUND(embedding top-kのうちどのentityにもtitle一致しなかったchunk)は
    「その他の関連情報」として末尾に残す(completeness維持のため)。"""
    parts = []
    for entity in query_entities:
        chunks = binding.get(entity, [])
        parts.append(f"[ENTITY: {entity}]")
        if not chunks:
            parts.append("NO GROUNDED EVIDENCE（この対象については検索結果に直接の情報が見つかりませんでした）")
        else:
            for c in chunks:
                parts.append(f"- [{c['title']}] {c['text']}")
    unbound = binding.get("UNBOUND", [])
    if unbound:
        parts.append("[OTHER / UNBOUND（各対象に直接紐付かなかった関連情報）]")
        for c in unbound:
            parts.append(f"- [{c['title']}] {c['text']}")
    return "\n".join(parts)


def build_context_a2(query_entities: list[str], binding: dict) -> str:
    """A2: query-bound only。query対象entityにbindingされたevidenceだけを渡す。
    bindingされなかったembedding chunk(UNBOUND)はcontextから完全に除外する
    (ただしdebug artifactには別途保存)。evidenceが0件のentityは
    NO GROUNDED EVIDENCEとして明示する。"""
    parts = []
    for entity in query_entities:
        chunks = binding.get(entity, [])
        parts.append(f"[ENTITY: {entity}]")
        if not chunks:
            parts.append("NO GROUNDED EVIDENCE（この対象については検索結果に直接の情報が見つかりませんでした）")
        else:
            for c in chunks:
                parts.append(f"- [{c['title']}] {c['text']}")
    return "\n".join(parts)
