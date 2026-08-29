"""Phase4FM: ルール単位のマッチング判定。

Section8: 短い禁止語が、無関係な長いベニンワードの一部として偶然出現しただけで
誤検知しないよう、境界安全性チェックを持つ。この技術はPhase4FZ/structured_lookup.py
の`_value_matches_query_with_boundary()`が確立した「隣接文字が同じ文字種(漢字/
カタカナ)かどうかで単語の継続を判定する」という考え方を、この新しいモデレーション
モジュール用に再実装したものであり、structured_lookup.py自体は一切変更していない。
"""

from __future__ import annotations

import re

from pachislot_ai.moderation.normalize import normalize_text, strip_obfuscation_separators
from pachislot_ai.moderation.policy import MatchForm, ModerationRule

# 漢字・カタカナは「単語が継続している可能性が高い」文字種として扱う
# (structured_lookup.pyの_WORD_CONTINUATION_REと同じ考え方の再実装)。
_WORD_CONTINUATION_RE = re.compile(r"[一-鿿ァ-ー]")


def _is_word_continuation(ch: str) -> bool:
    if _WORD_CONTINUATION_RE.match(ch):
        return True
    return ch.isascii() and ch.isalnum()


def _has_safe_boundary(haystack: str, term: str, idx: int) -> bool:
    before_ok = idx == 0 or not _is_word_continuation(haystack[idx - 1])
    after_idx = idx + len(term)
    after_ok = after_idx >= len(haystack) or not _is_word_continuation(haystack[after_idx])
    return before_ok and after_ok


def _token_boundary_match(haystack: str, term: str) -> bool:
    idx = haystack.find(term)
    while idx != -1:
        if _has_safe_boundary(haystack, term, idx):
            return True
        idx = haystack.find(term, idx + 1)
    return False


def term_matches(text: str, term: str, match_form: MatchForm) -> bool:
    """1つのterm(既に正規化前提の生文字列)が、textの中に指定match_formで
    存在するかを判定する。textは呼び出し側で正規化済みでなくてよい
    (この関数の内部で必要な正規化を行う)。"""
    normalized_text = normalize_text(text)
    normalized_term = normalize_text(term)
    if not normalized_term:
        return False

    if match_form is MatchForm.EXACT:
        return normalized_text == normalized_term

    if match_form is MatchForm.TOKEN_BOUNDARY:
        return _token_boundary_match(normalized_text, normalized_term)

    if match_form is MatchForm.NORMALIZED_SEQUENCE:
        stripped_text = strip_obfuscation_separators(normalized_text)
        stripped_term = strip_obfuscation_separators(normalized_term)
        if not stripped_term:
            return False
        return stripped_term in stripped_text

    return False  # pragma: no cover - MatchFormは列挙型で網羅済み


def rule_matches(text: str, rule: ModerationRule) -> bool:
    return any(term_matches(text, term, rule.match_form) for term in rule.terms)
