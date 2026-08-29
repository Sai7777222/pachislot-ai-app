"""Phase4FC3 Section12-13: chunk側とstructured facts側、独立した2つのevidence源を
統合した最終evidence状態を決定する(production integration、新しいbinding研究ではない)。

Phase4FX/FYのentity_attribution.py(chunk側の`__no_evidence__`合成チャンク)と
Phase4FZのstructured_lookup.py(構造化facts側の独立した検索)は、どちらも
**それ単体としては正しく動作している**。しかし2つが組み合わさると、
「chunk側にはevidenceが無い」という否定シグナルと「structured側には実データが
ある」という肯定シグナルが同一プロンプト内に矛盾した形で共存し、モデルが
否定シグナルを優先して不要なdeclineをしてしまうことがFC2で判明した
(例: 「天井とヤメ時の関係」で、実在するmetric_key「ヤメ時」のstructured dataが
あるにもかかわらず、chunk側の「「ヤメ時」について: 見つかりませんでした」という
合成チャンクにモデルが引きずられて完全decline)。

このモジュールは、entity_attribution.py・structured_lookup.py のどちらの
内部ロジック(title_match_score・extract_query_entities・structured facts検索
そのもの)にも一切変更を加えず、両者の**出力を組み合わせる統合層**としてのみ
機能する。`__no_evidence__`の意味論を「chunk側で0件だった」から「combined
evidence pipeline全体で0件だった」へ修正する(Phase4FC3 Section13)。
"""

from __future__ import annotations

from pachislot_ai.rag.retriever import RetrievedChunk
from pachislot_ai.rag.structured_lookup import (
    StructuredFinding,
    _value_matches_query_with_boundary,
)

_NO_EVIDENCE_CATEGORY = "__no_evidence__"
_NO_EVIDENCE_ID_PREFIX = "__no_evidence__::"


def _entity_from_no_evidence_chunk(chunk: RetrievedChunk) -> str:
    if chunk.chunk_id.startswith(_NO_EVIDENCE_ID_PREFIX):
        return chunk.chunk_id[len(_NO_EVIDENCE_ID_PREFIX):]
    return ""


# structured factsのdetail文字列は「[低確A・低確B・天国準備滞在時]」のように
# 「・」(U+30FB, 中点)でラベルを区切ることが多い。しかし
# structured_lookup._value_matches_query_with_boundary()の境界チェックは
# 「[ァ-ー]」(U+30A1-U+30FC)というカタカナ範囲の正規表現を使っており、この範囲には
# U+30FBの中点も数値的に含まれてしまうため、「天国準備」の直前の「・」を誤って
# 「単語が継続している」と判定し、境界安全な一致を妨げてしまうことが判明した
# (FX-CB06は一致したがPG-CP06は一致しなかった原因、生クエリに対する元の用途では
# 「・」が出現しにくいため顕在化していなかった)。structured_lookup.py自体は
# Section26の指示により変更しないため、ここではこの新しい呼び出しパターン
# (entity vs. structured finding detail文字列)専用に、中点をスペースへ置換して
# from渡す(FZの共有関数のロジック自体には一切手を加えない)。
def _normalize_detail_separators(detail: str) -> str:
    return detail.replace("・", " ")


def _structured_evidence_covers_entity(entity: str, structured_findings: list[StructuredFinding]) -> bool:
    if not entity:
        return False
    return any(
        _value_matches_query_with_boundary(entity, _normalize_detail_separators(finding.detail))
        for finding in structured_findings
    )


def arbitrate(
    chunks: list[RetrievedChunk], structured_findings: list[StructuredFinding]
) -> list[RetrievedChunk]:
    """chunk側のno-evidence合成チャンクのうち、structured facts側に同一entityの
    実データが存在するものを除去する(Section12のCASE B/C: 一方の否定シグナルが
    もう一方の肯定的evidenceと矛盾する状態を解消する)。

    それ以外のchunk(実evidence・他entityのno-evidenceマーカー)には一切触れない。
    entity_attribution.select_grounded_chunks()・structured_lookup.py の
    どちらも呼び出し側は変更しない(この関数は両者の出力を受け取るだけ)。
    """
    if not structured_findings:
        return chunks

    result: list[RetrievedChunk] = []
    for chunk in chunks:
        if chunk.category == _NO_EVIDENCE_CATEGORY:
            entity = _entity_from_no_evidence_chunk(chunk)
            if _structured_evidence_covers_entity(entity, structured_findings):
                # structured側に実データがあるため、矛盾する否定マーカーは注入しない
                continue
        result.append(chunk)
    return result
