from __future__ import annotations

from pachislot_ai.data.enums import ParseStatus, ValueType
from pachislot_ai.ingestion.value_parsing import (
    normalize_cell,
    parse_japanese_date,
    parse_payout_range,
    strip_citation,
)


def test_fraction_is_normalized_and_raw_preserved() -> None:
    result = normalize_cell("1/533")
    assert result.display_raw == "1/533"
    assert result.odds_denominator == 533.0
    assert result.value == 1 / 533
    assert result.value_type == ValueType.PROBABILITY
    assert result.parse_status == ParseStatus.OK


def test_percentage_string_is_normalized() -> None:
    result = normalize_cell("1.2% [cite: 295]")
    assert result.display_raw == "1.2% [cite: 295]"  # 原文は citation 込みでそのまま保持
    assert result.value == 0.012
    assert result.had_citation is True


def test_percentage_float_cell_from_excel_is_kept_as_probability() -> None:
    # openpyxl は 97.2% 書式セルを 0.972 の float として返す
    result = normalize_cell(0.972)
    assert result.value == 0.972
    assert result.value_type == ValueType.PROBABILITY


def test_game_count_notation() -> None:
    result = normalize_cell("1480G")
    assert result.value == 1480.0
    assert result.value_type == ValueType.GAME_COUNT
    assert result.display_raw == "1480G"


def test_coin_per_game_notation() -> None:
    result = normalize_cell("約7枚/G")
    assert result.value == 7.0
    assert result.value_type == ValueType.COIN_COUNT
    assert result.display_raw == "約7枚/G"


def test_unparseable_numeric_like_text_is_marked_failed_not_dropped() -> None:
    result = normalize_cell("0.7%・7.5%")
    assert result.parse_status == ParseStatus.FAILED
    assert result.value is None
    assert result.display_raw == "0.7%・7.5%"  # 失敗時も原文は失われない


def test_dash_is_not_applicable() -> None:
    result = normalize_cell("-")
    assert result.parse_status == ParseStatus.NOT_APPLICABLE
    assert result.value is None


def test_payout_range_parsing() -> None:
    rng = parse_payout_range("97.2%~114.6%")
    assert rng == (0.972, 1.146)


def test_japanese_date_parsing() -> None:
    parsed, raw = parse_japanese_date("2026年04月20日(月)")
    assert parsed is not None
    assert parsed.isoformat() == "2026-04-20"
    assert raw == "2026年04月20日(月)"


def test_strip_citation() -> None:
    cleaned, had = strip_citation("12.5% [cite: 289, 290]")
    assert cleaned == "12.5%"
    assert had is True
