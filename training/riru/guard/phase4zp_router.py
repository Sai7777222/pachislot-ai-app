"""Phase4ZP Section3-5: Lightweight deterministic mode router。

second LLM/embedding classifier/新規ML classifierは使わない(Section4)。
巨大なregex辞書も作らない — 目的は100%のintent分類ではなく、明確な
queryだけを安全に分ける決定木(Section4)。

決定順序(優先度が高い順):
1. FACTUAL_METRIC_KEYWORDS(パチスロ固有の数値・仕様用語)にmatch
   -> PACHISLOT_FACTUAL (最も安全側に倒す: この語彙は他分野とほぼ衝突しない)
2. GENERAL_PACHISLOT_TERMS(パチスロという話題そのもの)にmatch
   -> PACHISLOT_CONVERSATIONAL
3. STRONG_FACTUAL_MARKERS(一番/平均/確率/気温等、外部世界の事実を明確に
   要求する語)にmatch -> OOD_FACTUAL
4. OOD_TOPIC_KEYWORDS(天気/レシピ/株/プログラミング等の話題語)にmatch
   -> OOD_FACTUAL
5. それ以外(曖昧なqueryを含む) -> SMALL_TALK (デフォルト、安全側)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SMALL_TALK = "SMALL_TALK"
PACHISLOT_FACTUAL = "PACHISLOT_FACTUAL"
PACHISLOT_CONVERSATIONAL = "PACHISLOT_CONVERSATIONAL"
OOD_FACTUAL = "OOD_FACTUAL"

FACTUAL_METRIC_KEYWORDS = [
    "天井", "設定差", "機械割", "ゾーン", "初当り", "初当たり", "導入日",
    "やめ時", "ヤメ時", "期待値", "出玉率", "有利区間", "前兆", "継続率",
    "差枚数", "スルー回数", "リプレイ確率", "天井狙い", "ゾーン狙い", "獲得枚数",
    "AT初当", "ボーナス確率",
]
# 注意1: 「スペック」は「この機種のスペック」(PACHISLOT_FACTUAL)と「スマホの
# スペック」(OOD_FACTUAL)の両方で使われうる曖昧語のため、単独では
# FACTUAL_METRIC_KEYWORDSに含めない。パチスロ文脈は他のGENERAL_PACHISLOT_TERMS
# (「機種」「台」等)との共起で自然にPACHISLOT側へ回収される設計とする。
# 注意2: 裸の「確率」「解析」「リセット」は、パチスロ以外の文脈(降水確率、
# データ解析、パスワードリセット等)でも一般的に使われるため単独では含めない。
# パチスロ固有の確率表現は「リプレイ確率」「AT初当」「ボーナス確率」等の
# 複合語として個別に列挙し、それらだけでカバーする。
GENERAL_PACHISLOT_TERMS = [
    "パチスロ", "スロット", "パチンコ", "GOD", "ゴッド", "ミリオンゴッド", "設定",
    "機種", "実践", "出玉", "打ちに行", "遊技", "台", "打つ", "打って",
]
STRONG_FACTUAL_MARKERS = [
    "一番", "平均", "確率", "レート", "気温", "降水", "寿命", "為替", "定理",
]
OOD_TOPIC_KEYWORDS = [
    "天気", "レシピ", "料理の作り方", "株式投資", "投資のコツ", "プログラミング",
    "python", "Python", "数学", "スマホ", "ダイエット", "睡眠", "ニュース",
    "観光地", "英語の勉強法", "ラーメン屋", "おすすめのお店", "為替レート",
]
# 注意: 「映画」「アニメ」「ドラマ」「本」「スポーツ」等の娯楽トピック語は、
# 「映画好き？」(SMALL_TALK: 好みの質問)と「今年一番売れた映画は？」
# (OOD_FACTUAL: 事実の質問)の両方に自然に出現し、topic語だけでは区別できない
# (Section6の必須distinction test)。この種のtopicは、topic語単独ではなく
# STRONG_FACTUAL_MARKERS(一番/平均/確率等)との共起で判定する設計とし、
# OOD_TOPIC_KEYWORDSには含めない(誤ってSMALL_TALKをOOD_FACTUALへ誤routeする
# 危険を避けるため)。

_FACTUAL_METRIC_RE = re.compile("|".join(re.escape(k) for k in FACTUAL_METRIC_KEYWORDS))
_GENERAL_PACHISLOT_RE = re.compile("|".join(re.escape(k) for k in GENERAL_PACHISLOT_TERMS), re.IGNORECASE)
_STRONG_FACTUAL_RE = re.compile("|".join(re.escape(k) for k in STRONG_FACTUAL_MARKERS))
_OOD_TOPIC_RE = re.compile("|".join(re.escape(k) for k in OOD_TOPIC_KEYWORDS), re.IGNORECASE)

# Section13: 「最近〜した？」型のexperience questionは、topic markerがなければ
# SMALL_TALKのデフォルトへ自然に落ちる。特別扱いする必要はない
# (パチスロ/OOD語彙が含まれない限りSMALL_TALKになる決定木の構造上、
# 自動的にカバーされる)。


@dataclass
class RouteResult:
    mode: str
    matched_rule: str
    matched_keyword: str | None = None


def route(text: str) -> RouteResult:
    m = _FACTUAL_METRIC_RE.search(text)
    if m:
        return RouteResult(PACHISLOT_FACTUAL, "factual_metric_keyword", m.group(0))
    m = _GENERAL_PACHISLOT_RE.search(text)
    if m:
        return RouteResult(PACHISLOT_CONVERSATIONAL, "general_pachislot_term", m.group(0))
    m = _STRONG_FACTUAL_RE.search(text)
    if m:
        return RouteResult(OOD_FACTUAL, "strong_factual_marker", m.group(0))
    m = _OOD_TOPIC_RE.search(text)
    if m:
        return RouteResult(OOD_FACTUAL, "ood_topic_keyword", m.group(0))
    return RouteResult(SMALL_TALK, "default_fallback", None)
