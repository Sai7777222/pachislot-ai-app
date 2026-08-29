"""Phase4FM: モデレーション用の決定的テキスト正規化。

意味解析やLLM呼び出しは一切行わない、有界(bounded)な文字列変換のみ。
Section7の指示通り、無限に続く敵対的regex軍拡競争にはしない
(ファジー・セマンティック分類器なし、編集距離計算なし)。
"""

from __future__ import annotations

import unicodedata

# Section7: 区切り文字/難読化目的で挿入されやすい記号のうち、明示的にopt-inされた
# ルール(normalized_sequence)でのみ除去対象とする、境界を持つ固定集合。
# 新しい記号を無制限に追加する運用は想定しない(Section7の禁止事項)。
_OBFUSCATION_SEPARATORS = frozenset(
    " \t\n\r　"  # 半角/全角スペース類
    "・･"  # 中黒
    "-_/.。、,，."  # ハイフン・アンダースコア・スラッシュ・句読点類
)


def normalize_text(text: str) -> str:
    """NFKC正規化(全角/半角統一を含む) + 前後空白除去 + 内部連続空白の単一化 +
    ASCII部分の小文字化。exact/token_boundaryマッチの基礎となる、意味を変えない
    決定的正規化のみを行う。"""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.lower()  # ASCII文字のみに影響し、日本語には影響しない
    # 内部の連続空白(NFKC後は半角スペースに統一される)を単一スペースへ
    normalized = " ".join(normalized.split())
    return normalized


def strip_obfuscation_separators(text: str) -> str:
    """normalize_text()した文字列から、区切り/難読化記号を除去する。
    「禁止語」「禁 止 語」「禁・止・語」を同一視するためのopt-in変換であり、
    normalized_sequence match_formでのみ使用する(exact/token_boundaryには使わない)。"""
    return "".join(ch for ch in text if ch not in _OBFUSCATION_SEPARATORS)
