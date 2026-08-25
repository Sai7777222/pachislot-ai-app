"""取り込み時の異常値・矛盾・重複検出。

方針: 異常を検出しても値は書き換えない。原文 (display_raw) は常に残し、
`anomaly_records` として別途記録するだけに留める。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class Anomaly:
    anomaly_type: str
    description: str
    row_ref: str | None = None


def check_date_contradiction(
    release_date: date | None, source_last_updated: date | None
) -> Anomaly | None:
    if release_date is None or source_last_updated is None:
        return None
    if source_last_updated < release_date:
        return Anomaly(
            "date_contradiction",
            f"最終更新日 ({source_last_updated}) が導入開始日 ({release_date}) より前になっている。"
            "元データの矛盾の可能性があるため、両方の値をそのまま保持し要確認としてフラグを立てた。",
        )
    return None


def check_payout_rate_range(
    setting: int, payout_rate: float | None, *, row_ref: str | None = None
) -> Anomaly | None:
    if payout_rate is None:
        return None
    if not (0.5 <= payout_rate <= 2.0):
        return Anomaly(
            "out_of_range_payout",
            f"設定{setting}の機械割 {payout_rate} が妥当範囲 [0.5, 2.0] 外。",
            row_ref,
        )
    return None


def check_probability_range(
    label: str, value: float | None, *, row_ref: str | None = None
) -> Anomaly | None:
    if value is None:
        return None
    if not (0.0 <= value <= 1.0):
        return Anomaly(
            "out_of_range_probability", f"{label} の確率値 {value} が [0, 1] の範囲外。", row_ref
        )
    return None


def find_duplicate_metric_facts(facts: list[dict]) -> list[Anomaly]:
    """同一 (metric_key, dimensions_json) の組み合わせが複数回登録されていないか確認する。"""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for f in facts:
        key = (f["metric_key"], f["dimensions_json"])
        grouped.setdefault(key, []).append(f)

    anomalies: list[Anomaly] = []
    for (metric_key, dims_json), group in grouped.items():
        if len(group) <= 1:
            continue
        values = {g["display_raw"] for g in group}
        row_refs = ", ".join(g.get("row_ref") or "?" for g in group)
        if len(values) > 1:
            anomalies.append(
                Anomaly(
                    "duplicate_conflicting_value",
                    f"metric_key={metric_key} dimensions={dims_json} が複数回登録され、"
                    f"値が一致しない: {values} (rows: {row_refs})",
                    row_refs,
                )
            )
        else:
            anomalies.append(
                Anomaly(
                    "duplicate_redundant",
                    f"metric_key={metric_key} dimensions={dims_json} が同じ値で複数回登録されている"
                    f"（元記事内で再掲されている可能性）: rows {row_refs}",
                    row_refs,
                )
            )
    return anomalies
