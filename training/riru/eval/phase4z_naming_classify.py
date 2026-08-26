"""Phase 4Z: naming分類(Phase4Xで修正済みのロジックを基準に、Phase4Zのラベル
体系へ再マップし、新placeholder detector(○○対応)を組み込む)。

Phase4Zのラベル体系:
  A = genuine wrong-name
  B = honest hedge / refusal / unknown
  C = placeholder
  D = generic role only
  E = correct "リル"
  F = identity intrusion (名前を聞かれていない文脈での不要な自己紹介)
  G = other / no-name

既存のphase4x_naming_reclassify.py / phase4x_placeholder_detector.pyは変更しない。
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
from phase4z_placeholder_detector import classify_placeholder  # noqa: E402

CORRECT_NAME = "リル"

HEDGE_PATTERN = re.compile(
    r"分から|不明|非公開|特にない|特に無い|知らない|教えられない|"
    r"言えない|秘密|登録され.{0,6}(ない|てない)|公開されてい|"
    r"無いみたい|ないみたい|載ってない|出てこな|なかったみたい|"
    r"見つから|データにない|データにありません|情報がない|情報はない|未定"
)
GENERIC_ROLE_PATTERN = re.compile(r"(パチスロ|アシスタント|AI).{0,20}(だよ|です|なんだ)")
NAME_MENTION_PATTERN = re.compile(r"私はリル|リルだよ|リルです|名前はリル|私の名前はリル")


def classify_naming(text: str, is_naming_context: bool = True) -> dict:
    """is_naming_context=Falseの場合、名前を尋ねていない文脈での判定を行い、
    NAME_MENTION_PATTERNがヒットすればF(identity intrusion)を返す。"""
    if not text.strip():
        return {"category": "G", "reason": "empty"}
    if CHATML_PATTERN.search(text):
        return {"category": "G", "reason": "chatml_artifact"}

    if not is_naming_context:
        if NAME_MENTION_PATTERN.search(text):
            return {"category": "F", "reason": "identity_intrusion"}
        return {"category": "G", "reason": "no_intrusion"}

    ph = classify_placeholder(text)
    if ph["is_placeholder"]:
        return {"category": "C", "reason": "placeholder", "matched": ph["matched_text"]}

    candidates = []
    for pat in NAME_CUE_PATTERNS:
        for m in pat.finditer(text):
            candidates.append(m.group(1).strip())
    for m in FICTIONAL_REGISTERED_NAME_PATTERN.finditer(text):
        cand = m.group(1).strip()
        if cand not in ("特に", "特にない", "ない", "無い"):
            candidates.append(cand)

    if any(c == CORRECT_NAME or c.startswith(CORRECT_NAME) for c in candidates):
        return {"category": "E", "reason": "correct_name"}

    if HEDGE_PATTERN.search(text):
        return {"category": "B", "reason": "hedge_or_refusal"}

    real_candidates = []
    for c in candidates:
        stripped = strip_candidate(c)
        if c in FALSE_POSITIVE_CANDIDATES or stripped in FALSE_POSITIVE_CANDIDATES:
            continue
        if any(fp in c for fp in FALSE_POSITIVE_SUBSTRINGS):
            continue
        if len(stripped) <= 1:
            continue
        if len(stripped) > 8 and re.search(r"って感じ|というAI|といったAI", c):
            continue
        real_candidates.append(stripped)

    if real_candidates:
        return {"category": "A", "reason": "genuine_wrong_name", "matched": real_candidates[0]}

    if GENERIC_ROLE_PATTERN.search(text) and CORRECT_NAME not in text:
        return {"category": "D", "reason": "generic_role_only"}

    return {"category": "G", "reason": "no_name_mentioned"}
