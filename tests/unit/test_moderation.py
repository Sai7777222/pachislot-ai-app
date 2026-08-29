"""Phase4FM Section16: モデレーションエンジンの unit tests(engine/matcher/normalize)。"""

from __future__ import annotations

from pachislot_ai.core.config import MODERATION_POLICY_PATH
from pachislot_ai.moderation import ModerationEngine
from pachislot_ai.moderation.normalize import normalize_text


def _engine() -> ModerationEngine:
    return ModerationEngine.from_yaml(MODERATION_POLICY_PATH)


# 1. exact input block (メッセージ全体が完全一致する場合のみ検知するexactモード)
def test_exact_input_block():
    engine = _engine()
    result = engine.check_input("EXACT_ONLY_TEST_TERM")
    assert result.allowed is False
    assert result.rule_id == "FM-TEST-A03"
    assert result.policy_action == "HARD_BLOCK_INPUT"

    # exactモードは部分一致では発火しない(token_boundaryとの違いを明示)
    partial = engine.check_input("これはEXACT_ONLY_TEST_TERMを含む文です")
    assert partial.allowed is True


# token_boundary: 文中に埋め込まれていても検知する(現実的なHARD_BLOCK_INPUT想定)
def test_token_boundary_input_block_embedded_in_sentence():
    engine = _engine()
    result = engine.check_input("TEST_BLOCK_INPUT_Aだよ、元気？")
    assert result.allowed is False
    assert result.rule_id == "FM-TEST-A01"


# 2. NFKC-normalized block (全角英数字が半角と同一視される)
def test_nfkc_normalized_block():
    engine = _engine()
    fullwidth = "ＴＥＳＴ＿ＢＬＯＣＫ＿ＩＮＰＵＴ＿Ａ"
    result = engine.check_input(fullwidth)
    assert result.allowed is False
    assert result.rule_id == "FM-TEST-A01"


# 3. whitespace-obfuscated block
def test_whitespace_obfuscated_block():
    engine = _engine()
    result = engine.check_input("禁 止 語 テ ス ト")
    assert result.allowed is False
    assert result.rule_id == "FM-TEST-A02"


# 4. punctuation-obfuscated block
def test_punctuation_obfuscated_block():
    engine = _engine()
    result = engine.check_input("禁・止・語・テ・スト")
    assert result.allowed is False
    assert result.rule_id == "FM-TEST-A02"


# 5. benign near-match allowed (禁止語テストの一部だけを含むが完全一致しない)
def test_benign_near_match_allowed():
    engine = _engine()
    result = engine.check_input("禁止事項について教えて")
    assert result.allowed is True


# 6. pachislot benign term allowed
def test_pachislot_benign_terms_allowed():
    engine = _engine()
    for term in ["GGとSGGの違いを教えて", "天井は何ゲームですか", "天国ロングとは何か説明して", "ガイアベルとは"]:
        result = engine.check_input(term)
        assert result.allowed is True, f"{term} should not be blocked"


# 7. disabled rule ignored
def test_disabled_rule_ignored():
    engine = _engine()
    result = engine.check_input("TEST_DISABLED_TERM")
    assert result.allowed is True


# 11. SUPPRESS_ECHO: allowed in input, blocked in output
def test_suppress_echo_allowed_in_input_blocked_in_output():
    engine = _engine()
    input_result = engine.check_input("TEST_SUPPRESS_ECHO_A")
    assert input_result.allowed is True
    output_result = engine.check_output("TEST_SUPPRESS_ECHO_A")
    assert output_result.allowed is False
    assert output_result.policy_action == "HARD_BLOCK_OUTPUT"


# 12. HARD_BLOCK_OUTPUT detected
def test_hard_block_output_detected():
    engine = _engine()
    result = engine.check_output("これはTEST_BLOCK_OUTPUT_Aを含む応答です")
    assert result.allowed is False
    assert result.rule_id == "FM-TEST-E01"


# 13. blocked output replaced before display (safe_responseが存在し、元のテキストと異なる)
def test_blocked_output_has_replacement_text():
    engine = _engine()
    original = "TEST_BLOCK_OUTPUT_A"
    result = engine.check_output(original)
    assert result.allowed is False
    assert result.safe_response is not None
    assert result.safe_response != original
    assert "TEST_BLOCK_OUTPUT_A" not in result.safe_response


# 14. blocked output does not expose matched term/rule_id/category to the safe_response text
def test_blocked_output_does_not_expose_internal_details():
    engine = _engine()
    result = engine.check_output("TEST_BLOCK_OUTPUT_A")
    assert result.rule_id not in result.safe_response
    assert result.category not in result.safe_response
    assert "FM-TEST" not in result.safe_response


# 15. safe output unchanged
def test_safe_output_unchanged():
    engine = _engine()
    result = engine.check_output("こんにちは、リルだよ！")
    assert result.allowed is True
    assert result.safe_response is None


# 16. multiple rules deterministic (同じ入力に対して複数回呼んでも同じ結果)
def test_multiple_rules_deterministic():
    engine = _engine()
    results = {engine.check_input("TEST_BLOCK_INPUT_A").rule_id for _ in range(5)}
    assert results == {"FM-TEST-A01"}


# 17. input/output policies independent (SUPPRESS_ECHO語のinputとoutputで判定が違う)
def test_input_output_policies_independent():
    engine = _engine()
    assert engine.check_input("TEST_SUPPRESS_ECHO_A").allowed != engine.check_output("TEST_SUPPRESS_ECHO_A").allowed


# 19. normalization does not mutate user-visible safe content (normalize_textは
#     判定にのみ使われ、ユーザーに見える最終応答は変更しないことを確認する)
def test_normalization_used_only_for_matching_not_for_display():
    original = "  こんにちは　　リル  "
    normalized = normalize_text(original)
    assert normalized != original  # 判定用には正規化される
    engine = _engine()
    result = engine.check_input(original)
    assert result.allowed is True  # 安全なテキストなのでブロックされない
    assert result.safe_response is None  # 応答の書き換えは発生しない(元のcontentは呼び出し側がそのまま使う)


# 20. empty input/output safe handling
def test_empty_input_output_safe():
    engine = _engine()
    assert engine.check_input("").allowed is True
    assert engine.check_output("").allowed is True


# 境界安全性(Section8): 短い禁止語が長い合成語の一部として連続する場合は誤検知しない
def test_boundary_safety_no_false_positive_on_word_continuation():
    engine = _engine()
    # 「アカン語」の直後に漢字が連続する架空の複合語 → 継続とみなし誤検知しない
    result = engine.check_input("アカン語源風の説明をお願いします")
    assert result.allowed is True


def test_boundary_safety_matches_when_standalone():
    engine = _engine()
    result = engine.check_input("アカン語を使わないでください")
    assert result.allowed is False
    assert result.rule_id == "FM-TEST-C01"
