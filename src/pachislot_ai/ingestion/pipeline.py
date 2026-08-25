"""Excel(フラット行形式) -> 構造化DB向けレコード / RAGドキュメント / 未分類データ への変換。

このモジュールは純粋な変換ロジックのみを持ち、DBへの永続化は
`persist_result()` で行う（変換とDB書き込みを分離し、テストしやすくする）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pachislot_ai.data.enums import (
    DataSourceType,
    ParseStatus,
    ValueType,
)
from pachislot_ai.ingestion.anomalies import (
    Anomaly,
    check_date_contradiction,
    check_payout_rate_range,
    check_probability_range,
    find_duplicate_metric_facts,
)
from pachislot_ai.ingestion.category_rules import (
    infer_hint_category,
    infer_hint_scene,
    infer_metric_category,
    infer_rag_category,
    parse_confidence_level,
    parse_implied_settings,
)
from pachislot_ai.ingestion.classifier import (
    BASIC_INFO_FIELD_MAP,
    OVERVIEW_LABEL,
    BlockKind,
    classify_block,
    detect_repeating_column_groups,
    parse_setting_column_label,
    strip_table_suffix,
)
from pachislot_ai.ingestion.ids import make_id, strip_zone_suffix
from pachislot_ai.ingestion.raw_table_reader import RawBlock, read_blocks
from pachislot_ai.ingestion.value_parsing import (
    extract_setting_number,
    normalize_cell,
    parse_japanese_date,
    parse_payout_range,
    split_compound,
    strip_citation,
)

_INDEPENDENT_RESEARCH_MARKER = "独自調査"
_ZONE_ABBREV_RE = re.compile(r"[（(]([A-Za-zＡ-Ｚａ-ｚ0-9\-]+)[）)]")


def _now() -> datetime:
    return datetime.now(UTC)


def _dim_value(raw: object) -> object:
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        return raw
    text, _ = strip_citation(str(raw))
    return text.strip()


def _row_source_type_override(cells: list[object], default: str) -> str:
    joined = " ".join(str(c) for c in cells if c is not None)
    if _INDEPENDENT_RESEARCH_MARKER in joined:
        return DataSourceType.COMMUNITY_RESEARCH
    return default


def _column_entry(value: object) -> dict:
    """PatternFact の1列分 (原文値 + 正規化値) を辞書化する。"""
    n = normalize_cell(value)
    return {
        "display_raw": n.display_raw,
        "value": n.value,
        "value_type": n.value_type,
        "parse_status": n.parse_status,
        "is_approximate": n.is_approximate,
    }


class PatternIndexCounter:
    """(metric_key, pattern_label) ごとの連番。同じラベルの複数パターンが
    互いを上書きしないよう pattern_index を発行する。
    """

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = {}

    def next(self, metric_key: str, pattern_label: str) -> int:
        key = (metric_key, pattern_label)
        idx = self._counts.get(key, 0)
        self._counts[key] = idx + 1
        return idx


@dataclass(slots=True)
class IngestResult:
    machine_id: str
    excel_path: str
    source_url: str
    source_label: str | None
    data_source_type: str

    total_rows: int = 0
    total_blocks: int = 0
    block_kind_counts: dict[str, int] = field(default_factory=dict)

    machine: dict = field(default_factory=dict)
    setting_core_specs: list[dict] = field(default_factory=list)
    metric_facts: list[dict] = field(default_factory=list)
    pattern_facts: list[dict] = field(default_factory=list)
    zones: list[dict] = field(default_factory=list)
    hints: list[dict] = field(default_factory=list)
    rag_documents: list[dict] = field(default_factory=list)
    unclassified: list[dict] = field(default_factory=list)
    anomalies: list[dict] = field(default_factory=list)
    metric_definitions: dict[str, dict] = field(default_factory=dict)

    @property
    def parse_failed_count(self) -> int:
        return sum(1 for f in self.metric_facts if f["parse_status"] == ParseStatus.FAILED)

    def summary(self) -> dict:
        return {
            "machine_id": self.machine_id,
            "excel_path": self.excel_path,
            "total_rows": self.total_rows,
            "total_blocks": self.total_blocks,
            "block_kind_counts": self.block_kind_counts,
            "structured_counts": {
                "machines": 1,
                "setting_core_specs": len(self.setting_core_specs),
                "metric_facts": len(self.metric_facts),
                "pattern_facts": len(self.pattern_facts),
                "zones": len(self.zones),
                "hints": len(self.hints),
                "metric_definitions": len(self.metric_definitions),
            },
            "rag_document_count": len(self.rag_documents),
            "unclassified_count": len(self.unclassified),
            "parse_failed_count": self.parse_failed_count,
            "anomaly_count": len(self.anomalies),
            "anomalies_by_type": _count_by(self.anomalies, "anomaly_type"),
        }


def _count_by(items: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item[key]] = counts.get(item[key], 0) + 1
    return counts


def ingest_excel(
    excel_path: Path,
    *,
    machine_id: str,
    source_url: str,
    data_source_type: str = DataSourceType.UNKNOWN,
    source_label: str | None = None,
    sheet_name: str | None = None,
) -> IngestResult:
    blocks = read_blocks(excel_path, sheet_name=sheet_name)
    result = IngestResult(
        machine_id=machine_id,
        excel_path=str(excel_path),
        source_url=source_url,
        source_label=source_label,
        data_source_type=data_source_type,
    )
    result.total_blocks = len(blocks)
    result.total_rows = sum(len(b.rows) for b in blocks)
    result.machine = {"machine_id": machine_id, "overview_texts": []}

    overview_texts: list[str] = []
    zones_acc: dict[str, dict] = {}
    pattern_counter = PatternIndexCounter()

    for block in blocks:
        classification = classify_block(block)
        kind = classification.kind
        result.block_kind_counts[kind] = result.block_kind_counts.get(kind, 0) + 1
        header_idx = classification.header_row_index or 0

        if kind == BlockKind.MACHINE_META:
            _handle_machine_meta(block, result)
        elif kind == BlockKind.MACHINE_BASIC_INFO:
            _handle_basic_info(block, result, overview_texts)
        elif kind == BlockKind.SPEC_TABLE:
            _handle_spec_table(block, result)
        elif kind == BlockKind.HINT_TABLE:
            _handle_hint_table(block, header_idx, result)
        elif kind == BlockKind.ZONE_ATTRIBUTES:
            _handle_zone_attributes(block, zones_acc)
        elif kind == BlockKind.PAIRED_COLUMNS_TABLE:
            _handle_paired_columns_table(block, header_idx, result)
        elif kind == BlockKind.PATTERN_TABLE:
            _handle_pattern_table(block, header_idx, result, pattern_counter)
        elif kind == BlockKind.METRIC_TABLE:
            _handle_metric_table(block, header_idx, result)
        elif kind == BlockKind.METRIC_ROWS:
            _handle_metric_rows(block, result, pattern_counter)
        elif kind == BlockKind.TEXT:
            _handle_text_block(block, result)
        elif kind == BlockKind.MIXED:
            _handle_mixed_block(block, result, pattern_counter)
        else:  # UNCLASSIFIED
            _handle_unclassified(block, "ambiguous value shape", result)

    _finalize_zones(zones_acc, result)
    _finalize_machine(result, overview_texts)
    _run_anomaly_checks(result)
    _register_metric_definitions(result)
    return result


# --- block handlers -----------------------------------------------------------------


def _handle_machine_meta(block: RawBlock, result: IngestResult) -> None:
    label = block.label
    value = block.rows[0].non_key_cells()[0]
    if label == "機種名":
        result.machine["name"] = str(value).strip() if value is not None else None
    elif label == "掲載元":
        result.machine["source_breadcrumb"] = str(value).strip() if value is not None else None
        if not result.source_label:
            result.source_label = str(value).strip() if value is not None else None
    elif label == "最終更新日":
        parsed, raw = parse_japanese_date(value)
        result.machine["source_page_last_updated_raw"] = raw
        result.machine["source_page_last_updated"] = parsed


def _handle_basic_info(block: RawBlock, result: IngestResult, overview_texts: list[str]) -> None:
    for row in block.rows:
        cells = row.non_key_cells()
        sub_label = cells[0]
        value = cells[1] if len(cells) > 1 else None
        if sub_label is None:
            continue
        sub_label = str(sub_label).strip()

        if sub_label == OVERVIEW_LABEL:
            if value is not None:
                overview_texts.append(str(value).strip())
            continue

        field_name = BASIC_INFO_FIELD_MAP.get(sub_label)
        if field_name is None:
            _handle_unclassified(
                block, f"unknown basic-info field: {sub_label}", result, rows=[row]
            )
            continue

        if field_name == "model_name":
            result.machine["model_name"] = str(value).strip() if value is not None else None
        elif field_name == "maker_name":
            result.machine["maker_name"] = str(value).strip() if value is not None else None
        elif field_name == "release_date":
            parsed, raw = parse_japanese_date(value)
            result.machine["release_date_display_raw"] = raw
            result.machine["release_date"] = parsed
        elif field_name == "payout_rate_range":
            raw_text = str(value).strip() if value is not None else ""
            result.machine["payout_rate_display_raw"] = raw_text
            rng = parse_payout_range(raw_text)
            if rng is not None:
                result.machine["payout_rate_min"], result.machine["payout_rate_max"] = rng
            else:
                _handle_unclassified(
                    block, f"could not parse payout rate range: {raw_text!r}", result, rows=[row]
                )


def _handle_spec_table(block: RawBlock, result: IngestResult) -> None:
    header_cells = block.rows[0].non_key_cells()
    header_index = {
        str(v).strip(): i for i, v in enumerate(header_cells) if v is not None
    }
    setting_idx = header_index.get("設定")
    hit_idx = header_index.get("初当り")
    payout_idx = header_index.get("機械割")

    if setting_idx is None:
        _handle_unclassified(block, "spec table missing '設定' column", result)
        return

    for row in block.rows[1:]:
        cells = row.non_key_cells()
        setting = extract_setting_number(cells[setting_idx])
        if setting is None:
            _handle_unclassified(
                block, f"could not parse setting number: {cells[setting_idx]!r}", result, rows=[row]
            )
            continue

        hit_norm = normalize_cell(cells[hit_idx]) if hit_idx is not None else None
        payout_norm = normalize_cell(cells[payout_idx]) if payout_idx is not None else None

        source_type = _row_source_type_override(cells, result.data_source_type)
        spec = {
            "machine_id": result.machine_id,
            "setting": setting,
            "hit_rate_display_raw": hit_norm.display_raw if hit_norm else None,
            "hit_rate_odds_denominator": hit_norm.odds_denominator if hit_norm else None,
            "hit_rate_probability": hit_norm.value if hit_norm else None,
            "payout_rate_display_raw": payout_norm.display_raw if payout_norm else None,
            "payout_rate": payout_norm.value if payout_norm else None,
            "data_source_type": source_type,
            "row_ref": str(row.row_idx),
            "section_title": block.section_title,
        }
        result.setting_core_specs.append(spec)


def _handle_hint_table(block: RawBlock, header_idx: int, result: IngestResult) -> None:
    header_cells = block.rows[header_idx].non_key_cells()
    hint_keywords = {"示唆", "示唆内容", "期待度", "継続期待度", "当選期待度"}
    hint_col = next(
        (
            i
            for i, v in enumerate(header_cells)
            if v is not None and str(v).strip() in hint_keywords
        ),
        None,
    )
    if hint_col is None:
        _handle_unclassified(block, "hint table without hint column", result)
        return

    for row in block.rows[header_idx + 1 :]:
        cells = row.non_key_cells()
        hint_value = cells[hint_col]
        if hint_value is None:
            continue
        raw_text = str(hint_value).strip()
        clean_text, _ = strip_citation(raw_text)

        other_parts = [
            str(v).strip() for i, v in enumerate(cells) if i != hint_col and v is not None
        ]
        trigger_pattern = " / ".join(other_parts) if other_parts else block.label

        source_type = _row_source_type_override(cells, result.data_source_type)
        hint = {
            "machine_id": result.machine_id,
            "hint_category": infer_hint_category(clean_text + trigger_pattern),
            "hint_scene": infer_hint_scene(block.label, block.section_title),
            "trigger_pattern": trigger_pattern,
            "implied_settings_json": json.dumps(
                parse_implied_settings(clean_text), ensure_ascii=False
            ),
            "confidence_level": parse_confidence_level(clean_text),
            "raw_text": raw_text,
            "data_source_type": source_type,
            "row_ref": str(row.row_idx),
            "section_title": block.section_title,
        }
        result.hints.append(hint)


def _handle_zone_attributes(block: RawBlock, zones_acc: dict[str, dict]) -> None:
    is_multi_row_zone_block = strip_table_suffix(block.label).endswith(("性能", "基本性能"))
    zone_source_label = (
        strip_table_suffix(block.label)
        if is_multi_row_zone_block
        else (block.section_title or block.label)
    )
    zone_key = strip_zone_suffix(zone_source_label)
    acc = zones_acc.setdefault(
        zone_key,
        {
            "name": zone_key,
            "attributes": {},
            "first_row_ref": block.row_ref,
            "section_title": block.section_title,
        },
    )

    if is_multi_row_zone_block:
        for row in block.rows:
            cells = row.non_key_cells()
            attr_name = cells[0]
            attr_value = cells[1] if len(cells) > 1 else None
            if attr_name is None:
                continue
            acc["attributes"][str(attr_name).strip()] = (
                str(attr_value).strip() if attr_value is not None else None
            )
    else:
        # 単発ラベル行 (例: "突入契機" | 値) がセクション見出し配下に並ぶケース
        value = block.rows[0].non_key_cells()[0]
        acc["attributes"][block.label] = str(value).strip() if value is not None else None


def _detect_zone_canonical_keys(zones_acc: dict[str, dict]) -> dict[str, str]:
    """表記ゆれの軽量な別名検出 (Phase 2.5)。

    ゾーン名に含まれる括弧内の略称 (例:「ゴッドゲーム(GG)」の"GG") が
    他のゾーンの zone_key と完全一致する場合、それらを同じ canonical_zone_key
    にまとめる。データそのものは統合・削除せず、グルーピング情報だけを付与する。
    """
    zone_keys = set(zones_acc.keys())
    canonical: dict[str, str] = {k: k for k in zone_keys}

    for zone_key, acc in zones_acc.items():
        m = _ZONE_ABBREV_RE.search(acc["name"])
        if not m:
            continue
        abbrev = m.group(1)
        if abbrev in zone_keys and abbrev != zone_key:
            # 短縮形 (略称) を代表キーとする
            canonical[zone_key] = abbrev
            canonical[abbrev] = abbrev

    return canonical


def _finalize_zones(zones_acc: dict[str, dict], result: IngestResult) -> None:
    canonical_map = _detect_zone_canonical_keys(zones_acc)

    groups: dict[str, list[str]] = {}
    for zone_key, canonical_key in canonical_map.items():
        groups.setdefault(canonical_key, []).append(zone_key)

    for zone_key, acc in zones_acc.items():
        canonical_key = canonical_map[zone_key]
        aliases = [k for k in groups.get(canonical_key, []) if k != zone_key]
        result.zones.append(
            {
                "machine_id": result.machine_id,
                "zone_key": zone_key,
                "name": acc["name"],
                "attributes_json": json.dumps(acc["attributes"], ensure_ascii=False),
                "canonical_zone_key": canonical_key,
                "aliases_json": json.dumps(aliases, ensure_ascii=False),
                "data_source_type": result.data_source_type,
                "row_ref": acc["first_row_ref"],
                "section_title": acc["section_title"],
            }
        )


def _append_metric_fact(
    result: IngestResult,
    *,
    metric_key: str,
    metric_label_ja: str,
    category: str,
    dims: dict[str, object],
    normalized,  # NormalizedValue
    source_type: str,
    row_ref: str,
    section_title: str | None,
) -> None:
    result.metric_facts.append(
        {
            "machine_id": result.machine_id,
            "metric_key": metric_key,
            "metric_label_ja": metric_label_ja,
            "category": category,
            "dimensions_json": json.dumps(dims, ensure_ascii=False, sort_keys=True),
            "display_raw": normalized.display_raw,
            "value": normalized.value,
            "value_type": normalized.value_type,
            "odds_denominator": normalized.odds_denominator,
            "parse_status": normalized.parse_status,
            "is_approximate": normalized.is_approximate,
            "data_source_type": source_type,
            "row_ref": row_ref,
            "section_title": section_title,
        }
    )


def _handle_metric_table(block: RawBlock, header_idx: int, result: IngestResult) -> None:
    header_cells = block.rows[header_idx].non_key_cells()
    header_cols = [i for i, v in enumerate(header_cells) if v is not None]
    if len(header_cols) < 2:
        _handle_unclassified(block, "metric table with <2 header columns", result)
        return

    key_col = header_cols[0]
    other_cols = header_cols[1:]
    key_dim_name = str(header_cells[key_col]).strip()
    metric_key = strip_table_suffix(block.label)
    category = infer_metric_category(block.label, block.section_title)

    for row in block.rows[header_idx + 1 :]:
        cells = row.non_key_cells()
        row_key_raw = cells[key_col]
        if row_key_raw is None:
            continue
        source_type = _row_source_type_override(cells, result.data_source_type)

        for col in other_cols:
            value_cell = cells[col]
            if value_cell is None:
                continue
            col_label = str(header_cells[col]).strip()
            dims: dict[str, object] = {key_dim_name: _dim_value(row_key_raw)}
            setting_from_col = parse_setting_column_label(col_label)
            if setting_from_col is not None:
                dims["設定"] = setting_from_col
            elif len(other_cols) > 1:
                dims["group"] = col_label

            # 「右上がり黄7・中段黄7」+「0.7%・7.5%」のような複合見出し+複合値は
            # 対応関係が明確な場合のみ分割し、不明確なら推測せず未分類として保持する
            if isinstance(value_cell, str) and "・" in value_cell:
                compound = split_compound(col_label, value_cell)
                if compound is not None:
                    for sub_label, sub_value in compound:
                        sub_dims = dict(dims)
                        sub_dims["group"] = sub_label
                        normalized = normalize_cell(sub_value)
                        _append_metric_fact(
                            result,
                            metric_key=metric_key,
                            metric_label_ja=block.label,
                            category=category,
                            dims=sub_dims,
                            normalized=normalized,
                            source_type=source_type,
                            row_ref=f"{row.row_idx}:{col}",
                            section_title=block.section_title,
                        )
                    continue
                _handle_unclassified(
                    block,
                    f"compound value with unclear header correspondence: "
                    f"header={col_label!r} value={value_cell!r}",
                    result,
                    rows=[row],
                )
                continue

            normalized = normalize_cell(value_cell)
            _append_metric_fact(
                result,
                metric_key=metric_key,
                metric_label_ja=block.label,
                category=category,
                dims=dims,
                normalized=normalized,
                source_type=source_type,
                row_ref=f"{row.row_idx}:{col}",
                section_title=block.section_title,
            )


def _handle_paired_columns_table(block: RawBlock, header_idx: int, result: IngestResult) -> None:
    """1行に同じ意味の列ペアが複数並ぶ表 (例:「前兆|振り分け|前兆|振り分け」) を、
    ペアごとに独立したfactへ分解する。
    """
    header_cells = block.rows[header_idx].non_key_cells()
    header_cols = [i for i, v in enumerate(header_cells) if v is not None]
    period = detect_repeating_column_groups(header_cells, header_cols)
    if period is None or period < 2:
        _handle_unclassified(block, "paired-columns table without a valid period", result)
        return

    groups = [header_cols[i : i + period] for i in range(0, len(header_cols), period)]
    metric_key = strip_table_suffix(block.label)
    category = infer_metric_category(block.label, block.section_title)

    for row in block.rows[header_idx + 1 :]:
        cells = row.non_key_cells()
        source_type = _row_source_type_override(cells, result.data_source_type)

        for group in groups:
            key_col = group[0]
            row_key_raw = cells[key_col]
            if row_key_raw is None:
                continue
            key_dim_name = str(header_cells[key_col]).strip()

            for col in group[1:]:
                value_cell = cells[col]
                if value_cell is None:
                    continue
                dims: dict[str, object] = {key_dim_name: _dim_value(row_key_raw)}
                if len(group) > 2:
                    dims["group"] = str(header_cells[col]).strip()

                normalized = normalize_cell(value_cell)
                _append_metric_fact(
                    result,
                    metric_key=metric_key,
                    metric_label_ja=block.label,
                    category=category,
                    dims=dims,
                    normalized=normalized,
                    source_type=source_type,
                    row_ref=f"{row.row_idx}:{col}",
                    section_title=block.section_title,
                )


def _handle_pattern_table(
    block: RawBlock, header_idx: int, result: IngestResult, pattern_counter: PatternIndexCounter
) -> None:
    """多列組み合わせ表 (キー列の値が複数行で重複=代替パターン) を、
    行全体を1パターンとして pattern_facts に保存する。列を切り捨てない。
    """
    header_cells = block.rows[header_idx].non_key_cells()
    header_cols = [i for i, v in enumerate(header_cells) if v is not None]
    if len(header_cols) < 2:
        _handle_unclassified(block, "pattern table with <2 header columns", result)
        return

    key_col = header_cols[0]
    value_cols = header_cols[1:]
    header_labels = {i: str(header_cells[i]).strip() for i in header_cols}
    metric_key = strip_table_suffix(block.label)

    for row in block.rows[header_idx + 1 :]:
        cells = row.non_key_cells()
        row_key_raw = cells[key_col]
        if row_key_raw is None:
            continue
        pattern_label = str(_dim_value(row_key_raw))

        columns: dict[str, dict] = {}
        for col in value_cols:
            value_cell = cells[col]
            if value_cell is None:
                continue
            columns[header_labels[col]] = _column_entry(value_cell)

        if not columns:
            continue

        source_type = _row_source_type_override(cells, result.data_source_type)
        pattern_index = pattern_counter.next(metric_key, pattern_label)
        result.pattern_facts.append(
            {
                "machine_id": result.machine_id,
                "metric_key": metric_key,
                "pattern_label": pattern_label,
                "pattern_index": pattern_index,
                "columns_json": json.dumps(columns, ensure_ascii=False),
                "data_source_type": source_type,
                "row_ref": str(row.row_idx),
                "section_title": block.section_title,
            }
        )


def _handle_metric_rows(
    block: RawBlock, result: IngestResult, pattern_counter: PatternIndexCounter
) -> None:
    metric_key = strip_table_suffix(block.label)
    category = infer_metric_category(block.label, block.section_title)

    for row in block.rows:
        cells = [c for c in row.non_key_cells() if c is not None]
        if not cells:
            continue
        source_type = _row_source_type_override(cells, result.data_source_type)

        if len(cells) >= 3:
            # ヘッダーの無いフラット行だが3セル以上値がある場合、cells[2:]を無言で
            # 切り捨てず、全列を保持した PatternFact として保存する
            pattern_label = str(_dim_value(cells[0]))
            columns = {f"値{i + 1}": _column_entry(c) for i, c in enumerate(cells)}
            pattern_index = pattern_counter.next(metric_key, pattern_label)
            result.pattern_facts.append(
                {
                    "machine_id": result.machine_id,
                    "metric_key": metric_key,
                    "pattern_label": pattern_label,
                    "pattern_index": pattern_index,
                    "columns_json": json.dumps(columns, ensure_ascii=False),
                    "data_source_type": source_type,
                    "row_ref": str(row.row_idx),
                    "section_title": block.section_title,
                }
            )
            continue

        if len(cells) >= 2:
            dims = {"項目": _dim_value(cells[0])}
            value_cell = cells[1]
        else:
            dims = {}
            value_cell = cells[0]

        if block.section_title:
            # 同名ラベル (例:「入力タイミング」) が複数セクションで再利用されるケースを区別する
            dims["セクション"] = block.section_title

        normalized = normalize_cell(value_cell)
        _append_metric_fact(
            result,
            metric_key=metric_key,
            metric_label_ja=block.label,
            category=category,
            dims=dims,
            normalized=normalized,
            source_type=source_type,
            row_ref=str(row.row_idx),
            section_title=block.section_title,
        )


def _handle_text_block(block: RawBlock, result: IngestResult) -> None:
    paragraphs = []
    for row in block.rows:
        cells = [str(c).strip() for c in row.non_key_cells() if c is not None]
        if cells:
            text, _ = strip_citation(" ".join(cells))
            paragraphs.append(text)
    body = "\n".join(paragraphs)
    if not body:
        return

    category = infer_rag_category(block.label, block.section_title)
    doc_id = make_id(result.machine_id, category, block.label, block.row_ref)
    result.rag_documents.append(
        {
            "doc_id": doc_id,
            "machine_id": result.machine_id,
            "category": category,
            "title": block.label,
            "body_text": body,
            "char_count": len(body),
            "section_title": block.section_title,
            "row_ref": block.row_ref,
            "source_url": result.source_url,
            "source_label": result.source_label,
            "data_source_type": result.data_source_type,
        }
    )


def _handle_mixed_block(
    block: RawBlock, result: IngestResult, pattern_counter: PatternIndexCounter
) -> None:
    from pachislot_ai.ingestion.classifier import classify_row_value

    text_rows: list = []
    for row in block.rows:
        row_kind = classify_row_value(row.non_key_cells())
        if row_kind == "text":
            text_rows.append(row)
        elif row_kind == "structured":
            single_block = RawBlock(
                label=block.label, rows=[row], section_title=block.section_title
            )
            _handle_metric_rows(single_block, result, pattern_counter)
        else:
            _handle_unclassified(block, "ambiguous row within mixed block", result, rows=[row])

    if text_rows:
        text_block = RawBlock(label=block.label, rows=text_rows, section_title=block.section_title)
        _handle_text_block(text_block, result)


def _handle_unclassified(
    block: RawBlock, reason: str, result: IngestResult, *, rows: list | None = None
) -> None:
    target_rows = rows if rows is not None else block.rows
    for row in target_rows:
        cells = {
            col: row.non_key_cells()[i]
            for i, col in enumerate(["b", "c", "d", "e", "f", "g"])
            if row.non_key_cells()[i] is not None
        }
        result.unclassified.append(
            {
                "machine_id": result.machine_id,
                "section_title": block.section_title,
                "row_ref": str(row.row_idx),
                "label": block.label,
                "raw_cells_json": json.dumps(cells, ensure_ascii=False, default=str),
                "reason": reason,
            }
        )


# --- finalization ---------------------------------------------------------------------


def _finalize_machine(result: IngestResult, overview_texts: list[str]) -> None:
    m = result.machine
    if overview_texts:
        body = "\n".join(overview_texts)
        result.rag_documents.append(
            {
                "doc_id": make_id(result.machine_id, "overview", "機種概要"),
                "machine_id": result.machine_id,
                "category": "overview",
                "title": "機種概要",
                "body_text": body,
                "char_count": len(body),
                "section_title": None,
                "row_ref": None,
                "source_url": result.source_url,
                "source_label": result.source_label,
                "data_source_type": result.data_source_type,
            }
        )
    m.setdefault("name", None)
    m.setdefault("model_name", None)
    m.setdefault("maker_name", None)
    m.setdefault("release_date", None)
    m.setdefault("release_date_display_raw", None)
    m.setdefault("payout_rate_min", None)
    m.setdefault("payout_rate_max", None)
    m.setdefault("payout_rate_display_raw", None)
    m.setdefault("source_page_last_updated", None)
    m.setdefault("source_page_last_updated_raw", None)


def _run_anomaly_checks(result: IngestResult) -> None:
    anomalies: list[Anomaly] = []

    date_anomaly = check_date_contradiction(
        result.machine.get("release_date"), result.machine.get("source_page_last_updated")
    )
    if date_anomaly:
        anomalies.append(date_anomaly)

    for spec in result.setting_core_specs:
        a = check_payout_rate_range(
            spec["setting"], spec["payout_rate"], row_ref=spec.get("row_ref")
        )
        if a:
            anomalies.append(a)
        a = check_probability_range(
            f"設定{spec['setting']} 初当り",
            spec["hit_rate_probability"],
            row_ref=spec.get("row_ref"),
        )
        if a:
            anomalies.append(a)

    for fact in result.metric_facts:
        if fact["value_type"] == ValueType.PROBABILITY:
            a = check_probability_range(
                fact["metric_key"], fact["value"], row_ref=fact.get("row_ref")
            )
            if a:
                anomalies.append(a)

    anomalies.extend(find_duplicate_metric_facts(result.metric_facts))

    result.anomalies = [
        {"anomaly_type": a.anomaly_type, "description": a.description, "row_ref": a.row_ref}
        for a in anomalies
    ]


def _register_metric_definitions(result: IngestResult) -> None:
    for fact in result.metric_facts:
        key = fact["metric_key"]
        if key in result.metric_definitions:
            continue
        result.metric_definitions[key] = {
            "metric_key": key,
            "category": fact["category"],
            "label_ja": fact["metric_label_ja"],
            "value_type": fact["value_type"],
            "unit": None,
            "description": None,
        }
