"""ラベル文字列からカテゴリ・示唆確度・示唆設定を推定するキーワードルール。

すべてベストエフォートの推定であり、断定できない場合はニュートラルな
既定値 (その他/不明) を返す。数値そのものを推測することはしない。
"""

from __future__ import annotations

import re

from pachislot_ai.data.enums import (
    ConfidenceLevel,
    HintCategory,
    HintScene,
    MetricCategory,
    RagCategory,
)

_SETTING_AT_LEAST_RE = re.compile(r"設定\s*(\d)\s*以上")
_SETTING_EXACT_RE = re.compile(r"設定\s*(\d)\s*(?:濃厚|確定)")


def infer_metric_category(label: str, section_title: str | None) -> str:
    text = f"{section_title or ''} {label}"
    if "小役" in text or ("役" in text and "確率" in text):
        return MetricCategory.SMALL_WIN_PROBABILITY
    if "天井" in text:
        return MetricCategory.CEILING
    if "ガイアステージ" in text or "CZ" in text or "前兆" in text:
        return MetricCategory.CZ_PROBABILITY
    if any(k in text for k in ("GG", "AT", "ストック", "当選率")):
        return MetricCategory.AT_PROBABILITY
    if any(k in text for k in ("ZONE", "ゾーン", "モード", "ステージ", "昇格")):
        return MetricCategory.ZONE
    if "示唆" in text or "演出" in text:
        return MetricCategory.EFFECT
    return MetricCategory.OTHER


def infer_rag_category(label: str, section_title: str | None) -> str:
    text = f"{section_title or ''} {label}"
    if "概要" in label:
        return RagCategory.OVERVIEW
    if "攻略" in text or "打ち方" in text or "ヤメ時" in text:
        return RagCategory.STRATEGY
    if "演出" in text:
        return RagCategory.EFFECT_LORE
    if any(k in text for k in ("GG", "SGG", "PGG", "AT", "ストック")):
        return RagCategory.AT_MECHANISM
    if any(k in text for k in ("ZONE", "ゾーン", "モード", "ステージ")):
        return RagCategory.ZONE_EXPLANATION
    if "示唆" in text:
        return RagCategory.HINT_EXPLANATION
    if "解説" in text:
        return RagCategory.GAME_MECHANISM
    return RagCategory.OTHER


def parse_confidence_level(text: str) -> str:
    if "確定" in text:
        return ConfidenceLevel.CONFIRMED
    if "濃厚" in text:
        return ConfidenceLevel.ALMOST_CONFIRMED
    if "期待大" in text or "期待" in text:
        return ConfidenceLevel.EXPECTED
    if "示唆" in text:
        return ConfidenceLevel.SUGGESTED
    return ConfidenceLevel.UNKNOWN


def parse_implied_settings(text: str) -> list[int] | None:
    settings: set[int] = set()
    for m in _SETTING_AT_LEAST_RE.finditer(text):
        n = int(m.group(1))
        settings.update(range(n, 7))
    for m in _SETTING_EXACT_RE.finditer(text):
        settings.add(int(m.group(1)))
    return sorted(settings) if settings else None


def infer_hint_category(text: str) -> str:
    if "設定" in text:
        return HintCategory.SETTING
    if "ストック" in text or "継続" in text:
        return HintCategory.STOCK
    if "モード" in text:
        return HintCategory.MODE
    if any(k in text for k in ("GG", "AT", "前兆")):
        return HintCategory.AT
    return HintCategory.OTHER


def infer_hint_scene(label: str, section_title: str | None) -> str:
    text = f"{section_title or ''} {label}"
    if "終了画面" in text:
        return HintScene.ENDING_SCREEN
    if "SGG" in text:
        return HintScene.SGG
    if "GG" in text:
        return HintScene.GG
    if "ガイア" in text:
        return HintScene.GAIA
    if "通常" in text:
        return HintScene.NORMAL
    return HintScene.UNKNOWN
