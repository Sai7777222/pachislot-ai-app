"""Phase4ZR Section3-4: Conservative dispatch。

新しいrouterを研究しない(Section3)。Phase4ZPの既存keyword category
(FACTUAL_METRIC_KEYWORDS/GENERAL_PACHISLOT_TERMS/STRONG_FACTUAL_MARKERS/
OOD_TOPIC_KEYWORDS)を無編集のまま再利用し、決定木の**終端(default)を
「無理にSMALL_TALKと推測する」から「UNKNOWN」へ変更する**、という1点だけが
ZPからの変更点。

追加した2つの高precision signal(挨拶語彙・好み質問suffix)は、いずれも
閉じた小さな一般語彙セットであり、Section4が禁止する「機種固有名詞・AT名・
CZ名・演出名」のようなopen-vocabulary辞書ではない。
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4zp_router import (  # noqa: E402
    _FACTUAL_METRIC_RE, _GENERAL_PACHISLOT_RE, _OOD_TOPIC_RE,
    PACHISLOT_FACTUAL, PACHISLOT_CONVERSATIONAL, OOD_FACTUAL,
)

# Section4/Stage D: 「PACHISLOT質問をOODへ送ることを最重要で防止する」。ZPのSTRONG_FACTUAL_MARKERS
# には裸の「確率」「一番」が含まれていたが、これらはFACTUAL_METRIC_KEYWORDSから既に除外していた
# のと同じ理由(降水確率/一番好き等、パチスロ以外でも極めて一般的)で曖昧。実RAG50データ
# (「ガイアベルの確率は？」「モードの中で滞在率が一番高いものと低いものの差は？」等、機種固有語彙で
# GENERAL_PACHISLOT_TERMSと一致しないケース)に対しdangerous misrouteを引き起こすことが判明したため、
# この2語を「確信の持てるOOD signal」から除外する(新しい辞書の追加ではなく、既存の曖昧語を除去する、
# Phase4ZM/ZPで既に確立した安全側への一般化可能な修正パターンと同じ)。
_STRONG_FACTUAL_RE = re.compile(
    "|".join(re.escape(k) for k in ["平均", "レート", "気温", "降水", "寿命", "為替", "定理"])
)

SMALL_TALK = "SMALL_TALK"
UNKNOWN = "UNKNOWN"

# Section4: 「明確な挨拶」。閉じた一般語彙(machine-specific noun dictionaryではない)。
_GREETING_RE = re.compile(
    "|".join(re.escape(w) for w in [
        "おはよう", "こんにちは", "こんにちわ", "こんばんは", "ただいま", "おかえり",
        "バイバイ", "さようなら", "またね", "おやすみ", "ありがとう", "どういたしまして",
        "元気にしてた", "元気？", "元気?",
    ])
)
# Section4: 「明確なリル自身への好み/性格質問」。文末suffix patternのみ(固有名詞なし)。
_PREFERENCE_QUESTION_RE = re.compile(r"(好き|派|性格|理想の一日|得意|苦手|モットー)[？?、]|(ある|してる|した)[？?]$")


@dataclass
class DispatchResult:
    mode: str
    confident: bool
    matched_rule: str
    matched_keyword: str | None = None


def dispatch(text: str) -> DispatchResult:
    m = _FACTUAL_METRIC_RE.search(text)
    if m:
        return DispatchResult(PACHISLOT_FACTUAL, True, "factual_metric_keyword", m.group(0))
    m = _GENERAL_PACHISLOT_RE.search(text)
    if m:
        return DispatchResult(PACHISLOT_CONVERSATIONAL, True, "general_pachislot_term", m.group(0))
    m = _STRONG_FACTUAL_RE.search(text)
    if m:
        return DispatchResult(OOD_FACTUAL, True, "strong_factual_marker", m.group(0))
    m = _OOD_TOPIC_RE.search(text)
    if m:
        return DispatchResult(OOD_FACTUAL, True, "ood_topic_keyword", m.group(0))
    m = _GREETING_RE.search(text)
    if m:
        return DispatchResult(SMALL_TALK, True, "greeting_word", m.group(0))
    m = _PREFERENCE_QUESTION_RE.search(text)
    if m:
        return DispatchResult(SMALL_TALK, True, "preference_question_suffix", m.group(0))
    # Section3の核心: 確信が持てない場合はSMALL_TALKへ推測しない。UNKNOWNとする。
    return DispatchResult(UNKNOWN, False, "unknown_no_confident_signal", None)
