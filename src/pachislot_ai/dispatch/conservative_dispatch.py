"""Phase4FC3 Section3-5: production conservative dispatch。

Phase4ZR/ZPで検証済みのconservative dispatch決定木(training/riru/guard/
phase4zr_conservative_dispatch.py, phase4zp_router.py)の**最小限の意味論**を
本番コードへ移植したもの。診断ハーネスのコードをそのままコピーしたのではなく、
以下の設計原則のみを踏襲する:

- 巨大なregex辞書・機種固有名詞辞書は作らない。全てのkeywordリストは小さく
  閉じた一般語彙集合(Phase4ZP/ZRの監査コメントで個別に正当化済みのものを
  そのまま再利用)。
- retrieval scoreをtopic classifierとして使わない。
- 第2のLLM classifierは使わない。
- 決定順序は「安全側」優先: パチスロ固有の事実キーワードを最初に判定し、
  それ以外の全ての判定(雑談・自己紹介・専門外)より優先する
  (dangerous factual -> SMALL_TALK/OOD misroute = 0 を担保するため)。
- 確信が持てない場合は無理にどれかへ分類せず UNKNOWN とする(Phase4ZRの核心原則)。

Phase4FC3で新規追加したのは IDENTITY_PERSONA カテゴリのみ(小さな閉じた
語彙集合、機種固有名詞は一切含まない)。既存のFACTUAL_METRIC_KEYWORDS等の
判定順序・語彙は一切変更していない。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PACHISLOT_FACTUAL = "PACHISLOT_FACTUAL"
PACHISLOT_CONVERSATIONAL = "PACHISLOT_CONVERSATIONAL"
OOD_FACTUAL = "OOD_FACTUAL"
SMALL_TALK = "SMALL_TALK"
IDENTITY_PERSONA = "IDENTITY_PERSONA"
UNKNOWN = "UNKNOWN"

# --- Phase4ZP由来、無編集で再利用(意味論のみ移植、値は完全一致) ---
FACTUAL_METRIC_KEYWORDS = [
    "天井", "設定差", "機械割", "ゾーン", "初当り", "初当たり", "導入日",
    "やめ時", "ヤメ時", "期待値", "出玉率", "有利区間", "前兆", "継続率",
    "差枚数", "スルー回数", "リプレイ確率", "天井狙い", "ゾーン狙い", "獲得枚数",
    "AT初当", "ボーナス確率",
]
GENERAL_PACHISLOT_TERMS = [
    "パチスロ", "スロット", "パチンコ", "GOD", "ゴッド", "ミリオンゴッド", "設定",
    "機種", "実践", "出玉", "打ちに行", "遊技", "台", "打つ", "打って",
]
# Phase4ZR由来: 「平均/確率/一番」等は降水確率・一番好き等パチスロ外でも極めて
# 一般的なため、確信の持てるOOD signalから除外済み(Phase4ZRの監査結果をそのまま踏襲)。
_STRONG_FACTUAL_MARKERS = ["レート", "気温", "降水", "寿命", "為替", "定理"]
OOD_TOPIC_KEYWORDS = [
    "天気", "レシピ", "料理の作り方", "株式投資", "投資のコツ", "プログラミング",
    "python", "Python", "数学", "スマホ", "ダイエット", "睡眠", "ニュース",
    "観光地", "英語の勉強法", "ラーメン屋", "おすすめのお店", "為替レート",
]
_GREETING_WORDS = [
    "おはよう", "こんにちは", "こんにちわ", "こんばんは", "ただいま", "おかえり",
    "バイバイ", "さようなら", "またね", "おやすみ", "ありがとう", "どういたしまして",
    "元気にしてた", "元気？", "元気?", "久しぶり",
]

# Phase4FC3新規: IDENTITY_PERSONA(自己紹介・名前・性格の質問)。
# 機種固有名詞は一切含まない、閉じた一般語彙集合のみ。
_IDENTITY_PATTERNS = [
    r"(君|きみ|あなた)の?名前",
    r"名前(は|なんだっけ|教えて|って)",
    r"自己紹介",
    r"性格",
    r"って呼(んで|べば)",
    r"て呼んでもいい",
    r"何者",
    r"長所と短所",
]

# Phase4FC3 Section15の実測(FC3自身の生成結果)により判明した、UNKNOWNへ落ちて
# 不要なRAG context注入(ひいてはhedge)を招いていた一般的な雑談語彙。
# 全て機種固有名詞ではない、日常会話で広く使われる閉じた語彙集合。
_PERSONAL_TOPIC_WORDS = [
    "趣味", "休みの日", "朝型", "夜型", "幸せを感じる", "相談していい",
    "面白いこと", "新しいこと始めたい", "イライラする",
]

_FACTUAL_METRIC_RE = re.compile("|".join(re.escape(k) for k in FACTUAL_METRIC_KEYWORDS))
_GENERAL_PACHISLOT_RE = re.compile("|".join(re.escape(k) for k in GENERAL_PACHISLOT_TERMS), re.IGNORECASE)
_STRONG_FACTUAL_RE = re.compile("|".join(re.escape(k) for k in _STRONG_FACTUAL_MARKERS))
_OOD_TOPIC_RE = re.compile("|".join(re.escape(k) for k in OOD_TOPIC_KEYWORDS), re.IGNORECASE)
_GREETING_RE = re.compile("|".join(re.escape(w) for w in _GREETING_WORDS))
# 「好き」「派」等は直前直後に？が無くても(「好きな食べ物は？」等)一般に好み質問の
# signalとして十分安全なため、隣接要求を外す。文末が「ある/してる/した」+？の
# casual疑問文もあわせて検出する(「の？」単独は「GG中はどんな状態なの？」のような
# 機種固有の事実質問にも一致してしまい危険なため、意図的に含めない)。
_PREFERENCE_QUESTION_RE = re.compile(
    r"好き|派[？?、]|理想の一日|得意|苦手|モットー|"
    + "|".join(re.escape(w) for w in _PERSONAL_TOPIC_WORDS)
    + r"|(ある|してる|した)[？?]$"
)
_IDENTITY_RE = re.compile("|".join(_IDENTITY_PATTERNS))


@dataclass(frozen=True, slots=True)
class DispatchResult:
    mode: str
    confident: bool
    matched_rule: str
    matched_keyword: str | None = None


def dispatch(text: str) -> DispatchResult:
    """会話モードを判定する。優先順位(安全側から順):

    1. FACTUAL_METRIC_KEYWORDS -> PACHISLOT_FACTUAL(最優先。他分野とほぼ衝突しない
       パチスロ固有語彙のため、dangerous factual misrouteを避けるにはここを最初に置く必要がある)
    2. GENERAL_PACHISLOT_TERMS -> PACHISLOT_CONVERSATIONAL
    3. IDENTITY_PERSONA(名前・自己紹介・性格) -> IDENTITY_PERSONA
    4. STRONG_FACTUAL_MARKERS / OOD_TOPIC_KEYWORDS -> OOD_FACTUAL
    5. 挨拶語彙 / 好み質問suffix -> SMALL_TALK
    6. それ以外(確信が持てない) -> UNKNOWN(無理にSMALL_TALKへ推測しない)
    """
    m = _FACTUAL_METRIC_RE.search(text)
    if m:
        return DispatchResult(PACHISLOT_FACTUAL, True, "factual_metric_keyword", m.group(0))
    m = _GENERAL_PACHISLOT_RE.search(text)
    if m:
        return DispatchResult(PACHISLOT_CONVERSATIONAL, True, "general_pachislot_term", m.group(0))
    m = _IDENTITY_RE.search(text)
    if m:
        return DispatchResult(IDENTITY_PERSONA, True, "identity_persona_pattern", m.group(0))
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
    return DispatchResult(UNKNOWN, False, "unknown_no_confident_signal", None)
