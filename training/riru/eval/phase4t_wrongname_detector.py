"""Phase 4T-6: 誤名乗り検出器の改善版。

固定wrong-name listだけに依存せず、「私は」「僕は」「名前は」「〜って呼んで」等の
名乗りcueの直後から名詞句候補を抽出し、正しい名前「リル」以外を全てreview_required
としてフラグする。完全自動で誤名と断定せず、目視確認を前提とする
(false negative最小化を優先し、false positiveは目視で除外する設計)。
"""

from __future__ import annotations

import re

CORRECT_NAME = "リル"

# 一般名詞・記号などをストップワードとして除外 (名前ではないと確信できるもの)
STOPWORDS = {
    "パチスロ", "アシスタント", "AI", "キャラクター", "キュート", "元気", "笑顔",
    "とっても", "ちょっと", "みんな", "詳しく", "得意", "専門", "情報", "登録",
    "データ", "データベース", "何でも", "今日", "私", "僕", "その", "この",
}

NAME_CUE_PATTERNS = [
    re.compile(r"私は([^\s、。！？♪〜\-]{1,12})(?:だよ|なんだ|です|といいます|と申します|よ)"),
    re.compile(r"僕は([^\s、。！？♪〜\-]{1,12})(?:だよ|なんだ|です)"),
    re.compile(r"名前は([^\s、。！？♪〜\-]{1,12})(?:だよ|なんだ|です|といいます|よ)?"),
    re.compile(r"([^\s、。！？♪〜\-]{1,12})って呼んで"),
    re.compile(r"([^\s、。！？♪〜\-]{1,12})と申します"),
    re.compile(r"([^\s、。！？♪〜\-]{1,12})といいます"),
    re.compile(r"アシスタントの([^\s、。！？♪〜\-]{1,12})(?:だよ|です|よ)"),
    re.compile(r"呼び名は([^\s、。！？♪〜\-]{1,12})"),
]

PLACEHOLDER_PATTERN = re.compile(r"[〜ー]{2,}")
CHATML_PATTERN = re.compile(r"<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]")
AI_IDENTITY_PATTERN = re.compile(r"AI(アシスタント|です|モデル)")


def extract_name_candidates(text: str) -> list[str]:
    """名乗りcueの直後の候補文字列を全て抽出する (重複除去)。"""
    candidates = []
    for pat in NAME_CUE_PATTERNS:
        for m in pat.finditer(text):
            cand = m.group(1).strip()
            if cand and cand not in candidates:
                candidates.append(cand)
    return candidates


def classify_naming(text: str) -> dict:
    candidates = extract_name_candidates(text)
    correct_hit = any(c == CORRECT_NAME or c.startswith(CORRECT_NAME) for c in candidates)
    review_required = [
        c for c in candidates
        if c != CORRECT_NAME and not c.startswith(CORRECT_NAME) and c not in STOPWORDS
    ]
    has_placeholder = bool(PLACEHOLDER_PATTERN.search(text))
    has_chatml = bool(CHATML_PATTERN.search(text))
    has_ai_identity = bool(AI_IDENTITY_PATTERN.search(text))
    is_empty = len(text.strip()) == 0
    # 極端な途中終了の簡易ヒューリスティック: 句点・感嘆符・疑問符で終わっていない
    # かつ一定以上の長さがある場合を「途中終了の疑い」とする
    ends_abruptly = (
        len(text.strip()) > 10
        and text.strip()[-1] not in "。！？♪♫~〜～」』"
        and not has_placeholder
    )
    return {
        "text": text,
        "name_candidates": candidates,
        "correct_name_used": correct_hit,
        "review_required_candidates": review_required,
        "has_review_required": len(review_required) > 0,
        "placeholder_or_unfinished": has_placeholder,
        "chatml_residue": has_chatml,
        "ai_identity": has_ai_identity,
        "empty": is_empty,
        "ends_abruptly": ends_abruptly,
    }
