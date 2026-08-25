"""RawBlock を「構造化DB向け」「RAG文章向け」「未分類」に分類する。

判断できないものは推測で分類せず UNCLASSIFIED に落とす方針
（Phase 2 要件「データ品質」節）。

Phase 2.5 で追加した表構造:
- PAIRED_COLUMNS_TABLE: 1行に同じ意味の列ペアが複数並ぶ表
  (例:「前兆|振り分け|前兆|振り分け」)。ヘッダーラベルの繰り返し周期を検出して分割する。
- PATTERN_TABLE: 先頭列(キー列)の値が複数データ行で重複する表
  (例: G-ZONEのLV1が複数の代替パターン行を持つ)。1キー=1値という
  EAVモデルに当てはまらないため、行全体を1パターンとして保持する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from pachislot_ai.ingestion.raw_table_reader import RawBlock
from pachislot_ai.ingestion.value_parsing import looks_like_pure_number, looks_like_sentence

_TABLE_SUFFIX_RE = re.compile(r"[（(]テーブル\d*[）)]$")
_SETTING_COL_RE = re.compile(r"^設定\s*(\d)$")

# 機種基本情報として扱う既知フィールド（機種名/掲載元など、明示的にマッピングする）
MACHINE_META_LABELS = {"機種名", "掲載元", "最終更新日"}
BASIC_INFO_BLOCK_LABEL = "基本情報（テーブル）"
BASIC_INFO_FIELD_MAP = {
    "型式名": "model_name",
    "メーカー名": "maker_name",
    "機械割": "payout_rate_range",
    "導入開始日": "release_date",
}
# 機種概要はテキスト扱い（プロセスの都合上 basic info テーブルの中に紛れ込むため個別判定）
OVERVIEW_LABEL = "機種概要"

HINT_HEADER_KEYWORDS = {"示唆", "示唆内容", "期待度", "継続期待度", "当選期待度"}
ZONE_LABEL_SUFFIXES = ("性能", "基本性能")


class BlockKind(StrEnum):
    MACHINE_META = "machine_meta"
    MACHINE_BASIC_INFO = "machine_basic_info"
    SPEC_TABLE = "spec_table"  # 設定別 初当り/機械割 -> SettingCoreSpec
    HINT_TABLE = "hint_table"
    ZONE_ATTRIBUTES = "zone_attributes"
    PAIRED_COLUMNS_TABLE = "paired_columns_table"  # 1行に複数の(キー,値)組が並ぶ表
    PATTERN_TABLE = "pattern_table"  # キー列の値が複数行で重複する多列パターン表
    METRIC_TABLE = "metric_table"  # ヘッダー行 + 複数データ行の汎用数値テーブル
    METRIC_ROWS = "metric_rows"  # ヘッダー行のない key-value 形式（各行が1メトリック）
    TEXT = "text"
    MIXED = "mixed"  # 行ごとに構造化/テキストが混在
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    kind: BlockKind
    reason: str
    header_row_index: int | None = None  # 表系 kind 用: rows内でのヘッダー行インデックス


def strip_table_suffix(label: str) -> str:
    return _TABLE_SUFFIX_RE.sub("", label).strip()


def parse_setting_column_label(label: str) -> int | None:
    if isinstance(label, str) and (m := _SETTING_COL_RE.match(label.strip())):
        return int(m.group(1))
    return None


def _detect_header_row(block: RawBlock) -> int | None:
    """先頭行が「短いラベルの並び (ヘッダー)」で、以降の行が同じ列構成で埋まっているか判定。

    数値データに限定すると、値が文章的でない列挙 (ユニバプレートの示唆内容等) を
    含む表を取りこぼすため、値の数値性ではなく「列構成の一貫性」で判定する。
    ヘッダー候補が1列しかない場合は、単なる同一ラベル繰り返しの文章ブロック
    (例:「裏モードのポイント」) と区別できないため対象外とする。

    ヘッダーの数値性チェックには `looks_like_pure_number` を使う (「1G目」等の
    列位置ラベルを誤って数値データと判定し、ヘッダー検出を失敗させないため)。
    """
    if len(block.rows) < 2:
        return None

    header = block.rows[0].non_key_cells()
    header_cols = [i for i, v in enumerate(header) if v is not None]
    if len(header_cols) < 2:
        return None

    if any(
        isinstance(header[i], str) and looks_like_sentence(header[i]) for i in header_cols
    ):
        return None
    if any(looks_like_pure_number(header[i]) for i in header_cols):
        return None

    data_rows = block.rows[1:]
    consistent_shape = all(
        all(dr.non_key_cells()[i] is not None for i in header_cols) for dr in data_rows
    )
    return 0 if consistent_shape else None


def detect_repeating_column_groups(
    header_cells: list[object], header_cols: list[int]
) -> int | None:
    """ヘッダーラベルが周期的に繰り返されているか判定する (「前兆,振り分け,前兆,振り分け」等)。

    見つかった場合、その周期 (1グループあたりの列数) を返す。
    """
    labels = [str(header_cells[i]).strip() for i in header_cols]
    n = len(labels)
    for period in range(1, n // 2 + 1):
        if n % period != 0:
            continue
        if not all(labels[i] == labels[i % period] for i in range(n)):
            continue
        if len(set(labels[:period])) != period:
            continue
        return period
    return None


def key_values_repeat(block: RawBlock, header_idx: int, key_col: int) -> bool:
    """ヘッダーテーブルのキー列 (先頭列) の値が、複数データ行で重複しているか判定する。

    重複している場合、1キー=1値の EAV モデルには収まらない
    (例: G-ZONEの「LV1」が複数の代替パターン行を持つ) と判断する。
    """
    seen: set[str] = set()
    for r in block.rows[header_idx + 1 :]:
        key_val = r.non_key_cells()[key_col]
        if key_val is None:
            continue
        key_repr = str(key_val)
        if key_repr in seen:
            return True
        seen.add(key_repr)
    return False


def classify_block(block: RawBlock) -> ClassificationResult:
    label = block.label

    if label in MACHINE_META_LABELS:
        return ClassificationResult(BlockKind.MACHINE_META, f"known machine field: {label}")

    if label == BASIC_INFO_BLOCK_LABEL:
        return ClassificationResult(BlockKind.MACHINE_BASIC_INFO, "basic info block")

    stripped = strip_table_suffix(label)
    if stripped == "スペック":
        return ClassificationResult(BlockKind.SPEC_TABLE, "setting spec table (hit rate/payout)")

    # ブロック自身のラベルが「○○性能」の場合はゾーン/モードの複数行スペックとして最優先で扱う
    # (突入契機/継続ゲーム数/純増... の2列 key-value も、見かけ上はヘッダー付きテーブルに
    # 似てしまうため、ヘッダー判定より前に確定させる)
    if stripped.endswith(ZONE_LABEL_SUFFIXES):
        return ClassificationResult(BlockKind.ZONE_ATTRIBUTES, "zone/mode spec block (multi-row)")

    header_idx = _detect_header_row(block)
    if header_idx is not None:
        header_cells = block.rows[header_idx].non_key_cells()
        header_cols = [i for i, v in enumerate(header_cells) if v is not None]
        header_labels = [str(header_cells[i]).strip() for i in header_cols]

        period = detect_repeating_column_groups(header_cells, header_cols)
        if period is not None:
            return ClassificationResult(
                BlockKind.PAIRED_COLUMNS_TABLE,
                f"header labels repeat with period={period}: {header_labels}",
                header_idx,
            )

        key_col = header_cols[0]
        if len(header_cols) >= 2 and key_values_repeat(block, header_idx, key_col):
            return ClassificationResult(
                BlockKind.PATTERN_TABLE,
                f"key column values repeat across rows (headers={header_labels})",
                header_idx,
            )

        if any(h in HINT_HEADER_KEYWORDS for h in header_labels):
            return ClassificationResult(
                BlockKind.HINT_TABLE, f"header contains hint keyword: {header_labels}", header_idx
            )
        return ClassificationResult(
            BlockKind.METRIC_TABLE, f"header-row table, headers={header_labels}", header_idx
        )

    if block.section_title is not None and block.section_title.endswith(ZONE_LABEL_SUFFIXES):
        return ClassificationResult(BlockKind.ZONE_ATTRIBUTES, "zone/mode spec block (single-row)")

    # ヘッダー行なしの key-value / 単発行ブロック: 行ごとに構造化/テキスト/未分類を判定
    row_kinds = [classify_row_value(r.non_key_cells()) for r in block.rows]
    if all(k == "text" for k in row_kinds):
        return ClassificationResult(BlockKind.TEXT, "all rows look like prose")

    if all(k == "structured" for k in row_kinds):
        # 複数行あるのに全行が1セルしか値を持たない場合、行を区別するキーが存在しない
        # (= 箇条書き/短い解説文の羅列である可能性が高い) ため、構造化データとして
        # 無理に確定させず RAG 文章として扱う (例:「裏モードのポイント」)
        if len(block.rows) > 1 and all(
            len([c for c in r.non_key_cells() if c is not None]) == 1 for r in block.rows
        ):
            return ClassificationResult(
                BlockKind.TEXT,
                "multiple rows share no distinguishing key column; treated as a bullet list",
            )
        return ClassificationResult(BlockKind.METRIC_ROWS, "all rows look like short kv facts")

    if all(k == "unclassified" for k in row_kinds):
        return ClassificationResult(BlockKind.UNCLASSIFIED, "ambiguous value shape")
    return ClassificationResult(BlockKind.MIXED, f"row kinds mixed: {row_kinds}")


def classify_row_value(cells: list[object]) -> str:
    """flat な行の「値らしきセル」1つを見て structured/text/unclassified を判定する。"""
    from pachislot_ai.data.enums import ParseStatus
    from pachislot_ai.ingestion.value_parsing import normalize_cell

    values = [v for v in cells if v is not None]
    if not values:
        return "unclassified"

    text_repr = " / ".join(str(v) for v in values)
    if looks_like_sentence(text_repr):
        return "text"

    # 複数セルに値がある行（例: パターン/移行先/示唆の3列）は末尾セルを主値とみなす
    normalized = normalize_cell(values[-1])
    if normalized.parse_status == ParseStatus.FAILED:
        return "unclassified"
    return "structured"
