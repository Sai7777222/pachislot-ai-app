"""セル値の正規化。

方針: どんな場合も元の表記 (display_raw) は失わない。パースに成功した場合のみ
value / value_type / odds_denominator を付与し、失敗した場合は
parse_status="failed" として display_raw だけを保持する（推測で埋めない）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

from pachislot_ai.data.enums import ParseStatus, ValueType

_CITATION_RE = re.compile(r"\s*\[cite:\s*[\d,\s]+\]\s*")
_PERCENT_RE = re.compile(r"^(約)?\s*(-?\d+(?:\.\d+)?)\s*%$")
_PERCENT_RANGE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*%\s*[~〜]\s*(\d+(?:\.\d+)?)\s*%$")
_FRACTION_RE = re.compile(r"^1\s*/\s*(\d+(?:\.\d+)?)$")
_GAME_COUNT_RE = re.compile(r"^(約)?\s*(\d+(?:\.\d+)?)\s*[GgＧ]$")
_COIN_PER_GAME_RE = re.compile(r"^約?\s*(\d+(?:\.\d+)?)\s*枚\s*/\s*[GgＧ]$")
_PLAIN_INT_RE = re.compile(r"^-?\d+$")
_PLAIN_FLOAT_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_COMPOUND_SEP = "・"

_SENTENCE_MAX_LEN = 40


def strip_citation(text: str) -> tuple[str, bool]:
    """末尾等に付与された `[cite: 253]` のような引用マーカーを除去する。

    引用マーカー自体は捨てず、呼び出し側で "had_citation" として記録できるようにする。
    値の意味には影響しないため normalize では常に除去してからパースする。
    """
    cleaned, count = _CITATION_RE.subn("", text)
    return cleaned.strip(), count > 0


_MARKETING_MARKERS = ("●", "★", "☆", "♪", "!!", "!?")


def looks_like_sentence(text: str) -> bool:
    """長文・説明文らしいかどうかの簡易判定（RAG行 判定に使用）。"""
    if "。" in text:
        return True
    if len(text) > _SENTENCE_MAX_LEN:
        return True
    if any(marker in text for marker in _MARKETING_MARKERS):
        return True
    return False


def cell_is_numeric_like(value: object) -> bool:
    """テーブルのヘッダー行かデータ行かを判定するための簡易数値判定。"""
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    if isinstance(value, str):
        cleaned, _ = strip_citation(value)
        return bool(re.match(r"^-?\d", cleaned.strip()))
    return False


def looks_like_pure_number(value: object) -> bool:
    """実際に数値として解釈できるかどうかの厳密判定 (ヘッダー行の妥当性チェック専用)。

    `cell_is_numeric_like` は "1G目" のように数字で始まる列位置ラベルも数値扱い
    してしまい、ヘッダー行の誤判定を招く。一方で "1/37.6" のような分数表記は
    数字で始まらないため見た目の判定だけでは見抜けない。
    ここでは実際に `normalize_cell` でパースを試み、値が得られた場合のみ
    「本物の数値データ」とみなす (= ヘッダー候補として不適格と判定する)。
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    if isinstance(value, str):
        return normalize_cell(value).value is not None
    return False


@dataclass(frozen=True, slots=True)
class NormalizedValue:
    display_raw: str
    value: float | None
    value_type: str
    odds_denominator: float | None
    parse_status: str
    had_citation: bool = False
    is_approximate: bool = False


def normalize_cell(raw_value: object) -> NormalizedValue:
    """Excelセル値 (str / int / float / datetime いずれか) を正規化する。"""
    if raw_value is None:
        return NormalizedValue("", None, ValueType.TEXT, None, ParseStatus.NOT_APPLICABLE)

    if isinstance(raw_value, bool):  # bool is a subclass of int; excel 上ほぼ出現しない
        return NormalizedValue(
            str(raw_value), None, ValueType.TEXT, None, ParseStatus.NOT_APPLICABLE
        )

    if isinstance(raw_value, int | float):
        return _normalize_numeric_cell(raw_value)

    if isinstance(raw_value, datetime | date):
        return NormalizedValue(
            raw_value.isoformat(), None, ValueType.TEXT, None, ParseStatus.NOT_APPLICABLE
        )

    text = str(raw_value).strip()
    cleaned, had_citation = strip_citation(text)
    return _normalize_text_cell(display_raw=text, cleaned=cleaned, had_citation=had_citation)


def _normalize_numeric_cell(raw_value: float) -> NormalizedValue:
    # openpyxl は「97.2%」のようなパーセント書式セルを 0.972 の float として返す。
    # 0-1 の範囲なら確率とみなす。それ以外 (設定番号1-6等) はそのまま数値として保持する。
    display_raw = repr(raw_value) if isinstance(raw_value, float) else str(raw_value)
    if isinstance(raw_value, float) and 0.0 <= raw_value <= 1.0:
        return NormalizedValue(display_raw, raw_value, ValueType.PROBABILITY, None, ParseStatus.OK)
    return NormalizedValue(display_raw, float(raw_value), ValueType.COUNT, None, ParseStatus.OK)


def _normalize_text_cell(*, display_raw: str, cleaned: str, had_citation: bool) -> NormalizedValue:
    if cleaned in ("", "-", "ー", "‐"):
        return NormalizedValue(
            display_raw, None, ValueType.TEXT, None, ParseStatus.NOT_APPLICABLE, had_citation
        )

    if m := _PERCENT_RE.match(cleaned):
        is_approx = m.group(1) is not None
        pct = float(m.group(2))
        return NormalizedValue(
            display_raw,
            pct / 100.0,
            ValueType.PROBABILITY,
            None,
            ParseStatus.OK,
            had_citation,
            is_approx,
        )

    if m := _FRACTION_RE.match(cleaned):
        denom = float(m.group(1))
        prob = (1.0 / denom) if denom != 0 else None
        status = ParseStatus.OK if prob is not None else ParseStatus.FAILED
        return NormalizedValue(
            display_raw, prob, ValueType.PROBABILITY, denom, status, had_citation
        )

    if m := _GAME_COUNT_RE.match(cleaned):
        is_approx = m.group(1) is not None
        return NormalizedValue(
            display_raw,
            float(m.group(2)),
            ValueType.GAME_COUNT,
            None,
            ParseStatus.OK,
            had_citation,
            is_approx,
        )

    if m := _COIN_PER_GAME_RE.match(cleaned):
        return NormalizedValue(
            display_raw, float(m.group(1)), ValueType.COIN_COUNT, None, ParseStatus.OK, had_citation
        )

    if _PLAIN_FLOAT_RE.match(cleaned):
        return NormalizedValue(
            display_raw, float(cleaned), ValueType.COUNT, None, ParseStatus.OK, had_citation
        )

    # 数値らしき記号 (%, /, 数字) を含むのにどのパターンにも一致しない -> パース失敗として記録
    if looks_like_sentence(cleaned):
        return NormalizedValue(
            display_raw, None, ValueType.TEXT, None, ParseStatus.NOT_APPLICABLE, had_citation
        )

    if re.search(r"\d", cleaned) and re.search(r"[%/]", cleaned):
        return NormalizedValue(
            display_raw, None, ValueType.TEXT, None, ParseStatus.FAILED, had_citation
        )

    # 短いが数値パターンに一致しないテキスト（役名・アイテム名等） -> そのままテキスト値として保持
    return NormalizedValue(display_raw, None, ValueType.TEXT, None, ParseStatus.OK, had_citation)


def split_compound(header_label: str | None, value_text: str) -> list[tuple[str, str]] | None:
    """「右上がり黄7・中段黄7」+「0.7%・7.5%」のように、見出しと値がともに
    「・」区切りで同じ個数に分割できる場合のみ、対応する (サブ見出し, サブ値) の
    リストへ展開する。対応関係が不明確な場合は None を返し、呼び出し側で
    未分類として保持させる（推測で割り当てない）。
    """
    if not header_label or _COMPOUND_SEP not in value_text:
        return None
    if _COMPOUND_SEP not in header_label:
        return None

    label_parts = [p.strip() for p in header_label.split(_COMPOUND_SEP)]
    value_parts = [p.strip() for p in value_text.split(_COMPOUND_SEP)]
    if len(label_parts) != len(value_parts) or len(label_parts) < 2:
        return None
    if any(not p for p in label_parts) or any(not p for p in value_parts):
        return None
    return list(zip(label_parts, value_parts, strict=True))


def parse_payout_range(text: str) -> tuple[float, float] | None:
    """機械割レンジ「97.2%~114.6%」を (min, max) に変換する。"""
    cleaned, _ = strip_citation(text)
    if m := _PERCENT_RANGE_RE.match(cleaned):
        return float(m.group(1)) / 100.0, float(m.group(2)) / 100.0
    return None


_JP_DATE_RE = re.compile(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日")


def parse_japanese_date(value: object) -> tuple[date | None, str]:
    """和暦風の「2026年04月20日(月)」等を date に変換する。原文は常に返す。"""
    if isinstance(value, datetime):
        return value.date(), value.isoformat()
    if isinstance(value, date):
        return value, value.isoformat()

    text = str(value).strip()
    if m := _JP_DATE_RE.search(text):
        year, month, day = (int(g) for g in m.groups())
        try:
            return date(year, month, day), text
        except ValueError:
            return None, text
    return None, text


def extract_setting_number(value: object) -> int | None:
    """行の先頭セルなどに現れる設定番号 (1-6) を抽出する。整数以外は None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1 <= value <= 6 else None
    if isinstance(value, float) and value.is_integer():
        v = int(value)
        return v if 1 <= v <= 6 else None
    if isinstance(value, str) and _PLAIN_INT_RE.match(value.strip()):
        v = int(value.strip())
        return v if 1 <= v <= 6 else None
    return None
