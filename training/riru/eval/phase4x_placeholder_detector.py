"""Phase 4X: 改良版 placeholder detector。

旧detector(`phase4t_wrongname_detector.PLACEHOLDER_PATTERN = re.compile(r"[〜ー]{2,}")`)は
「〜」等が2文字以上連続する場合のみ検出しており、
「私は〜だよ」のような「名前スロットが単一の〜のまま埋まっていない」生成を見逃していた。

新detectorは「名前が要求される構文 (name cue) の直後に、
実質的な名前トークンが存在せず、プレースホルダー的な記号・空白のみが
入っている」ケースを、name cueとpredicateの位置関係から検出する。

自然な語尾の「〜」(例:「だよ〜」「ね〜」)は、predicateの後ろに来るため
本detectorの対象にはならない(cueとpredicateの"間"のみを見る設計)。
"""

from __future__ import annotations

import re

# name cueの直後に来る「名前スロット」を表す文字クラス。
# 全角チルダ・長音記号・三点リーダー・省略記号・空白(半角/全角)のみで構成される場合、
# 実質的に名前が埋まっていないとみなす。
_SLOT = r"[〜ー…\.\s　]{1,6}"

_PREDICATE = r"(だよ|だね|なんだ|です|でーす|よ|なんだよ|って|んだ)"

NAME_SLOT_EMPTY_PATTERNS = [
    re.compile(rf"(私は|僕は|わたしは|ぼくは){_SLOT}{_PREDICATE}"),
    re.compile(rf"(私の名前は|僕の名前は|名前は){_SLOT}{_PREDICATE}?"),
    re.compile(rf"{_SLOT}って呼んで"),
    re.compile(rf"(登録名は|呼び名は)[「『]?{_SLOT}[」』]?(って|だよ|です)?"),
]

# 旧detector(参照用・比較のため残す)
LEGACY_PLACEHOLDER_PATTERN = re.compile(r"[〜ー]{2,}")


def classify_placeholder(text: str) -> dict:
    """textにname-slot placeholderが存在するか判定する。

    Returns: {"is_placeholder": bool, "matched_pattern_idx": int|None, "matched_text": str|None}
    """
    for idx, pat in enumerate(NAME_SLOT_EMPTY_PATTERNS):
        m = pat.search(text)
        if m:
            return {
                "is_placeholder": True,
                "matched_pattern_idx": idx,
                "matched_text": m.group(0),
            }
    return {"is_placeholder": False, "matched_pattern_idx": None, "matched_text": None}


def legacy_classify_placeholder(text: str) -> bool:
    return bool(LEGACY_PLACEHOLDER_PATTERN.search(text))
