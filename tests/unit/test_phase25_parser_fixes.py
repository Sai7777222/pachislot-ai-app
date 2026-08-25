"""Phase 2.5: パーサ修正の単体テスト（合成データ、実Excel非依存で高速に検証）。

RawRow の並びは (row_idx, a, b, c, d, e, f, g)。`a` 列はブロックラベル用に
予約されており、`non_key_cells()` は b〜g の6セルを返す。テストでは
`a` にダミー値 (None) を入れ、実データは b から詰める。
"""

from __future__ import annotations

import json

from pachislot_ai.data.enums import DataSourceType
from pachislot_ai.ingestion.classifier import BlockKind, classify_block
from pachislot_ai.ingestion.pipeline import (
    IngestResult,
    PatternIndexCounter,
    _handle_metric_rows,
    _handle_paired_columns_table,
    _handle_pattern_table,
    _handle_unclassified,
)
from pachislot_ai.ingestion.raw_table_reader import RawBlock, RawRow


def _row(row_idx: int, *cells: object) -> RawRow:
    """b〜g (最大6セル) を渡して RawRow を組み立てるテスト用ヘルパー。"""
    padded = list(cells) + [None] * (6 - len(cells))
    return RawRow(row_idx, None, *padded)


def _make_result() -> IngestResult:
    return IngestResult(
        machine_id="test_machine",
        excel_path="dummy.xlsx",
        source_url="file:///dummy.xlsx",
        source_label=None,
        data_source_type=DataSourceType.UNKNOWN,
    )


# --- 1. 2列ペア横並びテーブル ------------------------------------------------------


def test_paired_columns_table_is_classified_correctly() -> None:
    rows = [
        _row(1, "前兆", "振り分け", "前兆", "振り分け"),
        _row(2, "2G", "0.02%", "17G", "11.9%"),
    ]
    block = RawBlock(label="前兆振り分けテスト", rows=rows, section_title="前兆振り分けテスト")
    result = classify_block(block)
    assert result.kind == BlockKind.PAIRED_COLUMNS_TABLE


def test_paired_columns_table_splits_into_independent_facts_with_correct_pairing() -> None:
    rows = [
        _row(1, "前兆", "振り分け", "前兆", "振り分け"),
        _row(2, "2G", "0.02%", "17G", "11.9%"),
        _row(3, "3G", "0.02%", "18G", "0.2%"),
    ]
    block = RawBlock(label="前兆振り分けテスト", rows=rows, section_title="前兆振り分けテスト")
    result = _make_result()
    _handle_paired_columns_table(block, 0, result)

    assert len(result.metric_facts) == 4  # 2行 x 2ペア = 4件の独立したfact

    by_dims = {
        json.dumps(json.loads(f["dimensions_json"]), sort_keys=True, ensure_ascii=False): f
        for f in result.metric_facts
    }
    f_2g = by_dims[json.dumps({"前兆": "2G"}, ensure_ascii=False)]
    f_17g = by_dims[json.dumps({"前兆": "17G"}, ensure_ascii=False)]
    f_3g = by_dims[json.dumps({"前兆": "3G"}, ensure_ascii=False)]
    f_18g = by_dims[json.dumps({"前兆": "18G"}, ensure_ascii=False)]

    assert f_2g["display_raw"] == "0.02%"
    assert abs(f_2g["value"] - 0.0002) < 1e-9
    # 誤って 2G に 11.9% が紐付かないこと (修正前のバグの再発防止)
    assert f_17g["display_raw"] == "11.9%"
    assert abs(f_17g["value"] - 0.119) < 1e-9
    assert f_3g["display_raw"] == "0.02%"
    assert f_18g["display_raw"] == "0.2%"


# --- 2/3/4. 多列組み合わせ表 (PatternFact) -----------------------------------------


def test_pattern_table_is_classified_correctly_when_key_repeats() -> None:
    rows = [
        _row(1, "1G目", "2G目", "3G目", "示唆"),
        _row(2, 5, 5, 5, "継続に期待"),
        _row(3, 1, 3, 5, "継続濃厚"),
        _row(4, 5, "V", 0, "継続かつVor0揃いに期待"),  # key=5 が再登場
    ]
    block = RawBlock(label="テストパターン表", rows=rows, section_title="テストパターン表")
    result = classify_block(block)
    assert result.kind == BlockKind.PATTERN_TABLE


def test_pattern_table_preserves_all_columns_without_loss() -> None:
    rows = [
        _row(1, "1G目", "2G目", "3G目", "示唆"),
        _row(2, 5, 5, 5, "継続に期待"),
        _row(3, 1, 3, 5, "継続濃厚"),
        _row(4, 5, "V", 0, "継続かつVor0揃いに期待"),
    ]
    block = RawBlock(label="テストパターン表", rows=rows, section_title="テストパターン表")
    result = _make_result()
    counter = PatternIndexCounter()
    _handle_pattern_table(block, 0, result, counter)

    assert len(result.pattern_facts) == 3  # データ行3件 (ヘッダー行を除く)
    for pf in result.pattern_facts:
        cols = json.loads(pf["columns_json"])
        # 2G目・3G目・示唆の3列すべてが保持されている (3列目以降を切り捨てない)
        assert set(cols.keys()) == {"2G目", "3G目", "示唆"}


def test_pattern_table_or_alternatives_use_pattern_index_without_collision() -> None:
    rows = [
        _row(1, "1G目", "2G目", "3G目", "示唆"),
        _row(2, 5, 5, 5, "継続に期待"),
        _row(3, 5, "V", 0, "継続かつVor0揃いに期待"),  # key=5 の2つ目の代替パターン
    ]
    block = RawBlock(label="テストパターン表", rows=rows, section_title="テストパターン表")
    result = _make_result()
    counter = PatternIndexCounter()
    _handle_pattern_table(block, 0, result, counter)

    label5 = [p for p in result.pattern_facts if p["pattern_label"] == "5"]
    assert len(label5) == 2
    # pattern_index で区別され、互いに上書きされていないこと
    assert {p["pattern_index"] for p in label5} == {0, 1}
    cols0 = json.loads(label5[0]["columns_json"])
    cols1 = json.loads(label5[1]["columns_json"])
    assert cols0 != cols1


# --- 4. METRIC_ROWS フォールバックでも3列目以降を切り捨てない ------------------------


def test_metric_rows_with_three_plus_cells_is_not_silently_dropped() -> None:
    rows = [_row(1, "項目A", "値1", "値2", "値3")]
    block = RawBlock(label="テスト単発多列", rows=rows, section_title=None)
    result = _make_result()
    counter = PatternIndexCounter()
    _handle_metric_rows(block, result, counter)

    # 通常の1値factにはならず、全セルを保持する PatternFact になる
    assert len(result.metric_facts) == 0
    assert len(result.pattern_facts) == 1
    cols = json.loads(result.pattern_facts[0]["columns_json"])
    raws = {v["display_raw"] for v in cols.values()}
    assert raws == {"項目A", "値1", "値2", "値3"}


def test_metric_rows_with_two_cells_still_uses_simple_fact() -> None:
    rows = [_row(1, "型式名", "L/テスト機/X")]
    block = RawBlock(label="基本情報テスト", rows=rows, section_title=None)
    result = _make_result()
    counter = PatternIndexCounter()
    _handle_metric_rows(block, result, counter)

    assert len(result.pattern_facts) == 0
    assert len(result.metric_facts) == 1
    assert result.metric_facts[0]["display_raw"] == "L/テスト機/X"


# --- 5. 箇条書き解説は RAG (TEXT) へ分類 ------------------------------------------


def test_multi_row_single_cell_block_is_classified_as_text_not_metric() -> None:
    rows = [
        _row(1, "裏モードのポイント"),
        _row(2, "裏天国中のGG当選で複数セット獲得の期待大"),
        _row(3, "裏天国中の下段黄7は大チャンス!"),
        _row(4, "アルテミスの矢演出で青7を否定すればチャンス"),
    ]
    block = RawBlock(label="裏モードのポイント（テーブル）", rows=rows, section_title=None)
    result = classify_block(block)
    assert result.kind == BlockKind.TEXT


def test_single_row_single_cell_block_still_classified_as_metric_rows() -> None:
    # 単発行 (例:「コイン持ち」) は従来どおり METRIC_ROWS のまま
    rows = [_row(1, "50枚あたりのゲーム数：約30.8G(設定1)")]
    block = RawBlock(label="コイン持ち", rows=rows, section_title=None)
    result = classify_block(block)
    assert result.kind == BlockKind.METRIC_ROWS


# --- 6. display_raw / raw_cells_json は常に保持される -------------------------------


def test_unclassified_item_retains_raw_cells_json() -> None:
    rows = [_row(1, "012 / 333 / 000")]
    block = RawBlock(label="液晶出目", rows=rows, section_title="テストセクション")
    result = _make_result()
    _handle_unclassified(block, "test reason", result)

    assert len(result.unclassified) == 1
    item = result.unclassified[0]
    assert item["reason"] == "test reason"
    assert item["label"] == "液晶出目"
    assert "012 / 333 / 000" in item["raw_cells_json"]


def test_paired_columns_facts_retain_display_raw() -> None:
    rows = [
        _row(1, "前兆", "振り分け", "前兆", "振り分け"),
        _row(2, "2G", "0.02%", "17G", "11.9%"),
    ]
    block = RawBlock(label="前兆振り分けテスト", rows=rows, section_title="前兆振り分けテスト")
    result = _make_result()
    _handle_paired_columns_table(block, 0, result)
    assert all(f["display_raw"] for f in result.metric_facts)


# --- 軽微なparse改善 (approximate / compound) ---------------------------------------


def test_approximate_percentage_is_flagged() -> None:
    from pachislot_ai.ingestion.value_parsing import normalize_cell

    n = normalize_cell("約80%")
    assert n.value == 0.8
    assert n.is_approximate is True
    assert n.display_raw == "約80%"


def test_compound_value_splits_only_when_unambiguous() -> None:
    from pachislot_ai.ingestion.value_parsing import split_compound

    ok = split_compound("右上がり黄7・中段黄7", "0.7%・7.5%")
    assert ok == [("右上がり黄7", "0.7%"), ("中段黄7", "7.5%")]

    # 見出し側が複合でない場合は対応関係が不明確なので None (推測しない)
    ambiguous = split_compound("役", "0.7%・7.5%")
    assert ambiguous is None
