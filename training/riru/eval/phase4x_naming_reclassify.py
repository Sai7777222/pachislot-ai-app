"""Phase 4X: naming生成の再分類 (Phase4Uの分類器のバグを修正)。

Phase4Uの`classify_generation`は「名前は特にないんだ」「非公開みたいだよ」
「分からないままなんだよ」のような、名前を尋ねられて誠実に「分からない/
非公開」と答えているケースを、cueパターンのみでgenuine wrong-name(A)に
誤分類するバグがあった(hedge/refusal判定がwrong-name判定より後に実行される
ため)。Phase4Wで19件中4件がこのバグによる誤検出だと目視確認済み。

本モジュールはhedge/refusal判定をwrong-name判定より先に行うことで、
Phase4X Section5の分類スキーマ(A〜G)に沿った、より正確な再分類を行う。

Phase4XのA〜G定義(Phase4Uとは文字の割当が異なる点に注意):
  A. 明確な架空固有名詞 (genuine wrong-name)
  B. generic role のみ (パチスロに詳しいAIだよ、等)
  C. name refusal / unknown (名前は分からない/非公開、等の誠実な回答)
  D. correct "リル"
  E. placeholder (name-slot placeholder; phase4x_placeholder_detectorで判定)
  F. detector false positive (一般語の誤検知)
  G. その他
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
from phase4t_wrongname_detector import CHATML_PATTERN, NAME_CUE_PATTERNS  # noqa: E402
from phase4u_reclassify_naming import (  # noqa: E402
    FALSE_POSITIVE_CANDIDATES,
    FALSE_POSITIVE_SUBSTRINGS,
    FICTIONAL_REGISTERED_NAME_PATTERN,
    strip_candidate,
)
from phase4x_placeholder_detector import classify_placeholder  # noqa: E402

CORRECT_NAME = "リル"

HEDGE_PATTERN = re.compile(
    r"分から|不明|非公開|特にない|特に無い|知らない|教えられない|"
    r"言えない|秘密|登録され.{0,6}(ない|てない)|公開されてい|"
    r"無いみたい|ないみたい|載ってない|出てこな|なかったみたい|"
    r"見つから|データにない|データにありません|情報がない|情報はない|未定"
)
GENERIC_ROLE_PATTERN = re.compile(r"(パチスロ|アシスタント|AI).{0,20}(だよ|です|なんだ)")


def classify(text: str) -> dict:
    if not text.strip():
        return {"category": "G", "reason": "empty"}
    if CHATML_PATTERN.search(text):
        return {"category": "G", "reason": "chatml_artifact"}

    ph = classify_placeholder(text)
    if ph["is_placeholder"]:
        return {"category": "E", "reason": "placeholder", "matched": ph["matched_text"]}

    candidates = []
    for pat in NAME_CUE_PATTERNS:
        for m in pat.finditer(text):
            candidates.append(m.group(1).strip())
    for m in FICTIONAL_REGISTERED_NAME_PATTERN.finditer(text):
        cand = m.group(1).strip()
        if cand not in ("特に", "特にない", "ない", "無い"):
            candidates.append(cand)

    if any(c == CORRECT_NAME or c.startswith(CORRECT_NAME) for c in candidates):
        return {"category": "D", "reason": "correct_name"}

    # hedge/refusalをwrong-name判定より先にチェック(Phase4Uのバグ修正)
    if HEDGE_PATTERN.search(text):
        return {"category": "C", "reason": "hedge_or_refusal"}

    real_candidates = []
    for c in candidates:
        stripped = strip_candidate(c)
        if c in FALSE_POSITIVE_CANDIDATES or stripped in FALSE_POSITIVE_CANDIDATES:
            continue
        if any(fp in c for fp in FALSE_POSITIVE_SUBSTRINGS):
            continue
        if len(stripped) <= 1:
            continue
        # 固有名詞的な短い名乗りではなく、長い一般的役割描写のみの場合は
        # B(generic role)扱いとし、A(genuine wrong-name)には含めない。
        # 誤ってパチ子/パチスロ君のような短い固有名詞的候補まで除外しないよう、
        # 8文字を超え、かつ「って感じ」等の曖昧語尾を伴うもののみを対象とする。
        if len(stripped) > 8 and re.search(r"って感じ|というAI|といったAI", c):
            continue
        real_candidates.append(stripped)

    if real_candidates:
        return {"category": "A", "reason": "genuine_wrong_name", "matched": real_candidates[0]}

    if GENERIC_ROLE_PATTERN.search(text) and CORRECT_NAME not in text:
        return {"category": "B", "reason": "generic_role_only"}

    return {"category": "G", "reason": "no_name_mentioned"}
