"""Phase4FZ Section16: structured facts entity-binding の unit tests。

find_relevant_structured_facts() は実DBセッションが必要なため、この単体テストでは
boundary判定の純粋関数 (_value_matches_query_with_boundary) を直接テストする
(RULE: DB接続を要する統合テストは tests/integration 側の既存カバレッジに委ねる)。
"""

from __future__ import annotations

from pachislot_ai.rag.structured_lookup import _value_matches_query_with_boundary


# --- 1. 天国ロング != 天国 ---


def test_tenkoku_does_not_match_tenkoku_long():
    assert _value_matches_query_with_boundary("天国", "天国ロングとは何か説明して") is False


def test_tenkoku_matches_standalone_query():
    assert _value_matches_query_with_boundary("天国", "天国について教えて") is True


# --- 2. GG != SGG (境界安全なsubstring) ---


def test_gg_does_not_match_sgg_query():
    assert _value_matches_query_with_boundary("GG", "SGGについて教えて") is False


def test_sgg_matches_its_own_query():
    assert _value_matches_query_with_boundary("SGG", "SGGについて教えて") is True


# --- 3. 実在entity + suffix のphantom (SGG-EX, ガイアステージMAX, Z-ZONE極) ---


def test_sgg_does_not_match_sgg_ex_phantom():
    assert _value_matches_query_with_boundary("SGG", "SGG-EXとは何か説明して") is False


def test_gaia_stage_does_not_match_phantom_suffix():
    assert _value_matches_query_with_boundary("ガイアステージ", "ガイアステージMAXについて教えて") is False


def test_z_zone_does_not_match_phantom_suffix():
    assert _value_matches_query_with_boundary("Z-ZONE", "Z-ZONE極について教えて") is False


# --- 4. 実在entityの完全一致は保持される ---


def test_gaiabelle_exact_entity_preserved():
    assert _value_matches_query_with_boundary("ガイアベル", "ガイアベルとは何か説明して") is True


def test_gg_junbi_chu_metric_key_style_value_preserved():
    assert _value_matches_query_with_boundary("GG準備中", "GG準備中とは何ですか") is True


# --- 5. 終了(汎用ラベル) が AT-F の「終了後」に偶然一致しないこと ---


def test_generic_label_shuuryou_does_not_leak_into_at_f_query_via_end_boundary():
    # 「終了後」の「終了」がgroup値「終了」と偶然一致する経路を境界チェックで防ぐ
    # (「終了後」の「後」は継続文字ではないため、素朴なsubstringなら誤ってTrueになるが、
    # 「後」は漢字であり単語継続とみなされ、境界チェックでFalseになる)
    assert _value_matches_query_with_boundary("終了", "AT-Fの性能と終了後の状態について教えて") is False


def test_generic_label_shuuryou_matches_when_standalone():
    # 「終了状態」は複合語であり「終了」単体の一致ではない(意図的に境界チェックで除外
    # される)。「終了」が真に単独の語として現れる場合(直後が助詞)は一致するべき。
    assert _value_matches_query_with_boundary("終了", "終了について教えて") is True


def test_generic_label_compound_word_correctly_rejected_as_non_standalone():
    # 「終了状態」は「終了」の直後が漢字(状)で継続しているため、単語境界チェックにより
    # 「終了」単体としての一致とはみなされない(意図した挙動)。
    assert _value_matches_query_with_boundary("終了", "終了状態について教えて") is False


# --- 6. 複数の一致候補(複数箇所に出現する場合、いずれかが境界安全ならTrue) ---


def test_multiple_occurrences_one_safe_one_unsafe():
    # 「GG」が「GGプラス」(継続、unsafe)と単独の「GG」(safe)の両方に出現する場合
    assert _value_matches_query_with_boundary("GG", "GGプラスとGGの違いを教えて") is True


# --- 7. 汎用metricの値が境界チェックなしでentity不一致を上書きしないこと(概念確認) ---


def test_hyphen_treated_as_continuation():
    assert _value_matches_query_with_boundary("AT", "AT-Fについて教えて") is False


# --- 8. AT-F 回帰(既知failure) ---


def test_at_f_phantom_no_boundary_match_for_generic_labels():
    for generic in ("終了", "継続", "通常", "天国"):
        assert _value_matches_query_with_boundary(generic, "AT-Fの性能と終了後の状態について教えて") is False, generic
