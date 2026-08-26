"""Phase 4Z: 拡張placeholder detector。

Phase4X detector(`phase4x_placeholder_detector.py`、変更しない)の設計を踏襲しつつ、
Phase4Y-Rで新たに発見された「○○」型のname-slot placeholderを検出対象に追加する。

検出対象:
  - 単一「〜」 (Phase4X detectorと同じ)
  - 連続「〜〜」
  - ○○ / 〇〇
  - XX / xx
  - [名前] / <名前> / {name} / [NAME] / <NAME>
  - その他の空白・記号のみのname slot

name cueとの構文的位置関係を優先し、通常文章中の自然な「〜」(語尾の装飾)は
対象としない。
"""

from __future__ import annotations

import re

_SLOT = r"[〜ー…\.\s　○〇XxＸｘ]{1,6}|\[名前\]|<名前>|\{name\}|\[NAME\]|<NAME>|\[name\]"
_PREDICATE = r"(だよ|だね|なんだ|です|でーす|よ|なんだよ|って|んだ)"

NAME_SLOT_EMPTY_PATTERNS = [
    re.compile(rf"(私は|僕は|わたしは|ぼくは)({_SLOT}){_PREDICATE}"),
    re.compile(rf"(私の名前は|僕の名前は|名前は)({_SLOT}){_PREDICATE}?"),
    re.compile(rf"({_SLOT})って呼んで"),
    re.compile(rf"(登録名は|呼び名は)[「『]?({_SLOT})[」』]?(って|だよ|です)?"),
    # 「アシスタントの○○です」「アシスタントの○○だよ」型
    re.compile(rf"(アシスタント|案内役)の({_SLOT}){_PREDICATE}"),
]

LEGACY_PLACEHOLDER_PATTERN = re.compile(r"[〜ー]{2,}")


def classify_placeholder(text: str) -> dict:
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
