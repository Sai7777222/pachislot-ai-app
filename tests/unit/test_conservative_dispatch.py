"""Phase4FC3 Section14: production dispatch(conservative_dispatch)の unit tests。"""

from __future__ import annotations

from pachislot_ai.dispatch.conservative_dispatch import (
    IDENTITY_PERSONA,
    OOD_FACTUAL,
    PACHISLOT_CONVERSATIONAL,
    PACHISLOT_FACTUAL,
    SMALL_TALK,
    UNKNOWN,
    dispatch,
)


# 1. greeting -> no RAG context (SMALL_TALK)
def test_greeting_dispatches_small_talk():
    assert dispatch("おはよう！").mode == SMALL_TALK


# 2. preference -> no RAG context (SMALL_TALK)
def test_preference_question_dispatches_small_talk():
    assert dispatch("犬派？猫派？").mode == SMALL_TALK


# 3. identity -> no RAG context (IDENTITY_PERSONA)
def test_identity_question_dispatches_identity_persona():
    assert dispatch("君の名前は？").mode == IDENTITY_PERSONA


def test_self_introduction_dispatches_identity_persona():
    assert dispatch("自己紹介して").mode == IDENTITY_PERSONA


# 4. OOD factual -> no RAG context (OOD_FACTUAL)
def test_ood_factual_dispatches_ood():
    assert dispatch("犬の平均寿命は？").mode == OOD_FACTUAL
    assert dispatch("今日の天気を教えて").mode == OOD_FACTUAL


# 5. clear pachislot factual -> RAG enabled (PACHISLOT_FACTUAL)
def test_clear_pachislot_factual_keyword_dispatches_factual():
    assert dispatch("天井は何ゲームですか").mode == PACHISLOT_FACTUAL
    assert dispatch("ヤメ時はいつがいい？").mode == PACHISLOT_FACTUAL


# 6. unknown does not automatically become small-talk
def test_ambiguous_machine_specific_query_is_unknown_not_small_talk():
    result = dispatch("GGとSGGの違いを初心者向けに説明して")
    assert result.mode == UNKNOWN
    assert result.mode != SMALL_TALK


def test_short_ambiguous_text_is_unknown():
    result = dispatch("あ")
    assert result.mode == UNKNOWN
    assert result.confident is False


# 7. possible factual query never dangerously routed OOD
def test_pachislot_factual_keyword_never_routes_to_ood():
    # 「平均」はSTRONG_FACTUAL_MARKERSに含まれるが、天井キーワードが優先されるべき
    result = dispatch("天井到達までの平均ゲーム数は？")
    assert result.mode == PACHISLOT_FACTUAL
    assert result.mode != OOD_FACTUAL


def test_general_pachislot_term_never_routes_to_ood_even_with_weather_word():
    # GENERAL_PACHISLOT_TERMSがOOD判定より優先されることを確認
    result = dispatch("パチスロと天気、どっちが好き？")
    assert result.mode != OOD_FACTUAL


# 8-11. evidence arbitration tests (chunk/structured combinations)
def test_evidence_arbitration_chunk_yes_structured_no_no_negative_marker():
    from pachislot_ai.rag.evidence_arbitration import arbitrate
    from pachislot_ai.rag.retriever import RetrievedChunk

    real_chunk = RetrievedChunk(
        chunk_id="c1", text="text", doc_id="d1", machine_id="m1", category="cat",
        title="実在チャンク", source_url="", source_label=None, data_source_type="unknown", score=0.8,
    )
    result = arbitrate([real_chunk], structured_findings=[])
    assert result == [real_chunk]


def test_evidence_arbitration_chunk_no_structured_yes_removes_negative_marker():
    from pachislot_ai.rag.evidence_arbitration import arbitrate
    from pachislot_ai.rag.entity_attribution import _no_evidence_chunk
    from pachislot_ai.rag.structured_lookup import StructuredFinding

    marker = _no_evidence_chunk("ヤメ時")
    finding = StructuredFinding("metric_fact", "[ヤメ時] : GG終了後...G-ZONE終了後、32G消化", None)
    result = arbitrate([marker], structured_findings=[finding])
    assert result == []  # 矛盾する否定マーカーは除去される


def test_evidence_arbitration_handles_middle_dot_separated_structured_labels():
    # 実DBのstructured factsは「[低確A・低確B・天国準備滞在時]」のように中点(・)で
    # ラベルを区切ることが多い。structured_lookup._value_matches_query_with_boundary()
    # の境界チェック(カタカナ範囲[ァ-ー])は中点(U+30FB)を数値的に含んでしまうため、
    # 前処理なしでは「天国準備」の前の中点を誤って「単語継続」と判定してしまっていた
    # (FC3で発見、evidence_arbitration.py内でのみ中点をスペースへ正規化して対処、
    # structured_lookup.py自体は無変更)。
    from pachislot_ai.rag.evidence_arbitration import arbitrate
    from pachislot_ai.rag.entity_attribution import _no_evidence_chunk
    from pachislot_ai.rag.structured_lookup import StructuredFinding

    marker = _no_evidence_chunk("天国準備")
    finding = StructuredFinding("metric_fact", "[天国準備] group=その他・下段黄7・設定=1: 0.0001", None)
    result = arbitrate([marker], structured_findings=[finding])
    assert result == []


def test_evidence_arbitration_known_limitation_compound_suffix_label_not_matched():
    # 既知の残存制約: 実DBのラベルが「天国準備滞在時」のように、クエリentity
    # 「天国準備」に別の語(滞在時)が直接融合した複合語である場合、境界安全チェックは
    # (意図的に)これを「天国準備」単体の一致とはみなさない — 複合語誤衝突を防ぐ
    # という設計原則(FZで確立)を優先した結果であり、バグではない。この場合、
    # 否定マーカーは安全側(decline)のまま残る(fabricationにはならない)。
    from pachislot_ai.rag.evidence_arbitration import arbitrate
    from pachislot_ai.rag.entity_attribution import _no_evidence_chunk
    from pachislot_ai.rag.structured_lookup import StructuredFinding

    marker = _no_evidence_chunk("天国準備")
    finding = StructuredFinding(
        "metric_fact", "[【低確A・低確B・天国準備滞在時】] group=その他・下段黄7・設定=1: 0.0001", None
    )
    result = arbitrate([marker], structured_findings=[finding])
    assert result == [marker]  # 既知の制約: マーカーは残る(安全側、fabricationではない)


def test_evidence_arbitration_both_no_valid_no_evidence_state():
    from pachislot_ai.rag.evidence_arbitration import arbitrate
    from pachislot_ai.rag.entity_attribution import _no_evidence_chunk

    marker = _no_evidence_chunk("天国ロング")
    result = arbitrate([marker], structured_findings=[])
    assert result == [marker]  # structured側にも無いので、マーカーはそのまま残す


def test_evidence_arbitration_both_yes_retains_both():
    from pachislot_ai.rag.evidence_arbitration import arbitrate
    from pachislot_ai.rag.retriever import RetrievedChunk
    from pachislot_ai.rag.structured_lookup import StructuredFinding

    real_chunk = RetrievedChunk(
        chunk_id="c1", text="text", doc_id="d1", machine_id="m1", category="cat",
        title="実在チャンク", source_url="", source_label=None, data_source_type="unknown", score=0.8,
    )
    finding = StructuredFinding("metric_fact", "[何か] : データ", None)
    result = arbitrate([real_chunk], structured_findings=[finding])
    assert result == [real_chunk]  # no-evidenceマーカーではないので無変更


def test_evidence_arbitration_unrelated_marker_not_removed_by_unrelated_structured_fact():
    from pachislot_ai.rag.evidence_arbitration import arbitrate
    from pachislot_ai.rag.entity_attribution import _no_evidence_chunk
    from pachislot_ai.rag.structured_lookup import StructuredFinding

    marker = _no_evidence_chunk("RT-A")
    finding = StructuredFinding("metric_fact", "[ヤメ時] : GG終了後...32G消化", None)
    result = arbitrate([marker], structured_findings=[finding])
    assert result == [marker]  # RT-Aとヤメ時は無関係、マーカーは残るべき


# 12. identity response path unaffected by RAG (dispatch does not depend on rag pipeline)
def test_identity_dispatch_independent_of_rag_pipeline_state():
    # dispatchはpure functionであり、RAG pipelineの状態に一切依存しない
    r1 = dispatch("君の名前は？")
    r2 = dispatch("君の名前は？")
    assert r1.mode == r2.mode == IDENTITY_PERSONA


# 13. empty retrieval does not automatically create RAG system message for small-talk
def test_small_talk_never_reaches_rag_pipeline_call():
    # SMALL_TALKと判定された場合、build_rag_context内でdispatch段階でNoneを返し
    # RagPipeline.build_context()が呼ばれないことをChatServiceの統合テスト側で確認
    # (tests/integration側でカバー)。ここではdispatch結果自体を確認する。
    result = dispatch("疲れたから少し休憩しようかな")
    assert result.mode in (SMALL_TALK, UNKNOWN)  # 明確な挨拶語彙が無いためUNKNOWNもあり得るが、
    # いずれにせよPACHISLOT_FACTUAL/CONVERSATIONALではないことを確認
    assert result.mode not in (PACHISLOT_FACTUAL, PACHISLOT_CONVERSATIONAL)


# 14. GG/SGG factual routing remains factual (via UNKNOWN + evidence-found path, integration-level;
#     here confirm dispatch itself does not misroute it to a skip-mode)
def test_gg_sgg_query_not_misrouted_to_skip_mode():
    result = dispatch("GGとSGGの違いを初心者向けに説明して")
    assert result.mode not in (SMALL_TALK, IDENTITY_PERSONA, OOD_FACTUAL)


# 15. phantom pachislot entity still enters safe factual/unknown evidence path rather than
#     casual answer (dispatch level: must not be classified as SMALL_TALK/IDENTITY/OOD)
def test_phantom_entity_query_not_misrouted_to_skip_mode():
    for q in ["GGプラスとは何か説明して", "天国ロングとは何か説明して", "AT-Fの性能について教えて"]:
        result = dispatch(q)
        assert result.mode not in (SMALL_TALK, IDENTITY_PERSONA, OOD_FACTUAL), q


# Regression test: an earlier draft of the small-talk hedge-reduction fix used a bare
# 「の？」 sentence-ending pattern for SMALL_TALK detection, which dangerously misrouted
# machine-specific factual questions like "GG中はどんな状態なの？" (no FACTUAL_METRIC_KEYWORDS/
# GENERAL_PACHISLOT_TERMS match, since "GG"/"GG中" are machine-specific vocabulary
# intentionally excluded from those small closed lists). Caught via manual dispatch testing
# before finalizing Stage B, fixed by removing the bare「の」branch (kept ある/してる/した only).
def test_machine_specific_factual_question_with_no_da_ending_not_misrouted_to_small_talk():
    for q in ["GG中はどんな状態なの？", "GG準備中とはどんな状態なの？"]:
        result = dispatch(q)
        assert result.mode not in (SMALL_TALK, IDENTITY_PERSONA, OOD_FACTUAL), q
