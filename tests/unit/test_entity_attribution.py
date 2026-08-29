"""Phase4FY Section13: entity-aware context assembly の unit tests。

Phase4FX で発見した3つのバグの回帰テストを最優先で含む:
  1. compound entity truncation (「GGプラス」が「GG」に切り詰められない)
  2. substring collision (「GG」が「SGG」titleへ誤ってbindingされない)
  3. over-filtering (Phase4FYで発見: chunkベースの選別にRAG50由来の
     retention-ratio fallbackを誤って適用しない = 少数の真に関連するchunkへの
     絞り込みが正しく機能する)
"""

from __future__ import annotations

from pachislot_ai.rag.entity_attribution import (
    bind_entities_to_evidence,
    extract_query_entities,
    select_grounded_chunks,
    title_match_score,
)
from pachislot_ai.rag.retriever import RetrievedChunk


def _chunk(title: str, text: str = "dummy text", chunk_id: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or f"chunk::{title}",
        text=text,
        doc_id=f"doc::{title}",
        machine_id="test_machine",
        category="test_category",
        title=title,
        source_url="",
        source_label=None,
        data_source_type="unknown",
        score=0.8,
    )


# --- Test 1: compound entity truncation ---


def test_compound_entity_not_truncated_gg_plus():
    entities = extract_query_entities("GGプラスとは何か説明して")
    assert entities == ["GGプラス"], "「GGプラス」が「GG」に切り詰められてはいけない"


def test_compound_entity_not_truncated_gaia_stage_max():
    entities = extract_query_entities("ガイアステージMAXについて教えて")
    assert entities == ["ガイアステージMAX"]


def test_compound_entity_not_truncated_neo():
    entities = extract_query_entities("確定役ネオとは何か説明して")
    assert entities == ["確定役ネオ"]


def test_compound_entity_with_genuine_particle_boundary_still_splits():
    # 「AT-Fの性能と終了後の状態」のような、助詞を挟む複合名詞句は先頭のcore名詞まで切り出す
    entities = extract_query_entities("AT-Fの性能と終了後の状態について教えて")
    assert entities == ["AT-F"]


# --- Phase4FY統合テストで発見した回帰: stopwordが隣接漢字と融合して抽出をすり抜けるバグ ---


def test_qualifier_suffix_not_extracted_as_spurious_second_entity_beginner():
    # 「初心者向け」の「初心者」(stopword)+「向」が「初心者向」という複合トークンに
    # 融合し、完全一致のstopwordフィルタをすり抜けてentity化していたバグの回帰テスト。
    # このバグにより、機種概要chunkが取得できているにもかかわらず、存在しない
    # 「初心者向」entityのno-evidenceマーカーにモデルが引きずられ、Stage F必須probe
    # (Q15相当)で不要な全面declineが発生していた。
    entities = extract_query_entities("ミリオンゴッドの遊び方を初心者向けにやさしく説明して")
    assert entities == ["ミリオンゴッド"], f"「初心者向」が誤って第2entityとして抽出されてはいけない: {entities}"


def test_qualifier_suffix_not_extracted_as_spurious_second_entity_concise():
    entities = extract_query_entities("ミリオンゴッドの遊び方を簡潔に説明して")
    assert entities == ["ミリオンゴッド"], f"「簡潔」が誤って第2entityとして抽出されてはいけない: {entities}"


def test_qualifier_suffix_not_extracted_as_spurious_second_entity_detailed():
    entities = extract_query_entities("ミリオンゴッドの遊び方を少し詳しく説明して")
    assert entities == ["ミリオンゴッド"]


# --- Test 2: substring collision (GG != SGG) ---


def test_gg_does_not_match_sgg_title():
    assert title_match_score("GG", "SGGゲーム数抽選概要") == 0.0


def test_sgg_matches_its_own_title():
    assert title_match_score("SGG", "SGGゲーム数抽選概要") > 0.0


def test_gg_matches_gg_naka_kaisetsu():
    # 「GG中解説」のように、GGの直後が非ASCII文字(境界)の場合は正しく一致する
    assert title_match_score("GG", "GG中解説") > 0.0


def test_binding_comparison_query_no_cross_contamination():
    embedding_chunks = [
        _chunk("青7連続時GG当選率"),
        _chunk("GG中解説"),
        _chunk("SGGゲーム数抽選概要"),
        _chunk("準備中解説"),
    ]
    all_chunks = embedding_chunks
    binding = bind_entities_to_evidence(["GG", "SGG"], embedding_chunks, all_chunks)
    gg_titles = {c.title for c in binding["GG"]}
    sgg_titles = {c.title for c in binding["SGG"]}
    assert "SGGゲーム数抽選概要" not in gg_titles, "SGGのchunkがGGにbindingされてはいけない"
    assert "SGGゲーム数抽選概要" in sgg_titles
    assert "GG中解説" in gg_titles


def test_binding_single_entity_query_no_cross_contamination():
    # SGGが co-query entityとして存在しない場合でも、「GG」が「SGG」chunkへ
    # 誤ってbindingされないことを確認する(Phase4FXで発見・修正した実バグ)
    embedding_chunks = [
        _chunk("青7連続時GG当選率"),
        _chunk("GG中解説"),
        _chunk("SGGゲーム数抽選概要"),
    ]
    binding = bind_entities_to_evidence(["GG"], embedding_chunks, embedding_chunks)
    gg_titles = {c.title for c in binding["GG"]}
    assert "SGGゲーム数抽選概要" not in gg_titles


# --- Test 3: chunk-based selection does not over-filter-then-fallback incorrectly ---


def test_select_grounded_chunks_narrows_to_relevant_subset_no_incorrect_fallback():
    # 6件中1件だけが真に関連する場合、正しく1件に絞り込まれ、絞り込み前の6件へ
    # フォールバックしてはいけない(Phase4FY統合時に発見・修正した実バグ)
    embedding_chunks = [
        _chunk("準備中解説"),
        _chunk("裏モード概要"),
        _chunk("効果"),
        _chunk("なし"),
        _chunk("表モード概要"),
        _chunk("GG中解説"),
    ]
    selected = select_grounded_chunks("GG中とはどんな状態か教えて", embedding_chunks, embedding_chunks)
    titles = [c.title for c in selected]
    assert titles == ["GG中解説"], f"6件中1件の関連chunkへ正しく絞り込まれるべき: {titles}"


# --- Phantom entity behavior ---


def test_phantom_entity_returns_empty_no_evidence_borrowed():
    embedding_chunks = [
        _chunk("天井突入条件"),
        _chunk("LV5（GG継続濃厚）"),
        _chunk("効果"),
    ]
    selected = select_grounded_chunks("AT-Fの性能について教えて", embedding_chunks, embedding_chunks)
    assert selected == [], "存在しないentityに他chunkの情報を流用してはいけない"


def test_multi_entity_one_phantom_one_real_gets_no_evidence_marker():
    embedding_chunks = [
        _chunk("青7連続時GG当選率"),
        _chunk("SGGゲーム数抽選概要"),
    ]
    selected = select_grounded_chunks("GG当選とSGG当選の違いを教えて", embedding_chunks, embedding_chunks)
    titles = [c.title for c in selected]
    assert "青7連続時GG当選率" in titles
    # SGG当選(存在しない)については、SGGゲーム数抽選概要を誤って流用せず、
    # 情報不足を示す合成チャンクが追加されているべき
    assert "SGGゲーム数抽選概要" not in titles or any(
        c.category == "__no_evidence__" for c in selected
    )


# --- Exact title match / multiple entities / duplicate chunks / bound-unbound separation ---


def test_exact_title_match_scores_highest():
    assert title_match_score("GG中解説", "GG中解説") == 1.0


def test_duplicate_chunks_deduplicated_in_selection():
    dup = _chunk("GG中解説", chunk_id="same_id")
    embedding_chunks = [dup, dup]
    selected = select_grounded_chunks("GG中とはどんな状態か教えて", embedding_chunks, embedding_chunks)
    assert len(selected) == 1


def test_unbound_chunks_separated_from_bound():
    embedding_chunks = [_chunk("GG中解説"), _chunk("裏モード概要")]
    binding = bind_entities_to_evidence(["GG中"], embedding_chunks, embedding_chunks)
    assert "裏モード概要" in {c.title for c in binding["UNBOUND"]}
    assert "裏モード概要" not in {c.title for c in binding["GG中"]}


def test_no_embedding_chunks_returns_empty_unchanged():
    assert select_grounded_chunks("何か教えて", [], []) == []
