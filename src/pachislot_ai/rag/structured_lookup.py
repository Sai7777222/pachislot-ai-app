"""structured.db からのキーワードベースの事実検索。

正確な数値 (設定差・機械割・小役確率・天井など) は LLM の内部知識で補完せず、
必ず構造化DBの原文値 (display_raw) を経由してユーザーに提示する方針
(要件「回答方針」) を実現するための最小限のルックアップ層。

高度な自然言語理解は行わず、質問文に含まれるキーワード・数値・既知の
役名/ゾーン名/示唆パターンとの単純な部分一致でファクトを拾う
(「まずはシンプルRAGで構いません」の方針に沿う)。
"""

from __future__ import annotations

import json
import re

from sqlalchemy.orm import Session

from pachislot_ai.data.repositories import machine_repository as mrepo
from pachislot_ai.ingestion.value_parsing import strip_citation

_SETTING_NUMBER_RE = re.compile(r"設定\s*([1-6])")
_PAYOUT_KEYWORDS = ("機械割", "割数", "出玉率")
_HIT_RATE_KEYWORDS = ("初当り", "初当たり", "初当", "当選確率")
_CEILING_KEYWORDS = ("天井",)
_MIN_TERM_LEN = 2


def _extract_setting_numbers(query: str) -> list[int]:
    return sorted({int(m) for m in _SETTING_NUMBER_RE.findall(query)})


def _format_payout(spec) -> str:  # noqa: ANN001
    """機械割の値を % 表記で返す。

    元のExcelセルはパーセント書式だったが openpyxl 上は 0.972 のような小数として
    渡ってくるため、display_raw もその repr ("0.972") になっている。これをそのまま
    LLM に渡すと誤読の元になるため、同じ数値を % 表記に変換するだけで
    (値自体は変えない)、LLMに渡す表記を一本化する。
    """
    raw = spec.payout_rate_display_raw or ""
    if "%" in raw:
        return raw
    if spec.payout_rate is not None:
        return f"{spec.payout_rate * 100:.1f}%"
    return raw


def _is_meaningful_term(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if len(text) < _MIN_TERM_LEN:
        return False
    return not text.isdigit()


class StructuredFinding:
    __slots__ = ("kind", "detail", "source_id")

    def __init__(self, kind: str, detail: str, source_id: int | None) -> None:
        self.kind = kind
        self.detail = detail
        self.source_id = source_id


def find_relevant_structured_facts(
    session: Session, machine_id: str, query: str, *, limit: int = 20
) -> list[StructuredFinding]:
    findings: list[StructuredFinding] = []
    setting_numbers = _extract_setting_numbers(query)
    mentions_payout = any(k in query for k in _PAYOUT_KEYWORDS)
    mentions_hit_rate = any(k in query for k in _HIT_RATE_KEYWORDS)
    mentions_ceiling = any(k in query for k in _CEILING_KEYWORDS)

    # --- 設定別 初当り/機械割 (SettingCoreSpec: ほぼ全機種共通の固定スキーマ) ---
    if setting_numbers or mentions_payout or mentions_hit_rate:
        specs = mrepo.get_setting_core_specs(session, machine_id)
        want_payout = mentions_payout or not (mentions_payout or mentions_hit_rate)
        want_hit = mentions_hit_rate or not (mentions_payout or mentions_hit_rate)
        for spec in specs:
            if setting_numbers and spec.setting not in setting_numbers:
                continue
            if want_hit and spec.hit_rate_display_raw:
                findings.append(
                    StructuredFinding(
                        "setting_core_spec",
                        f"設定{spec.setting}: 初当り確率 {spec.hit_rate_display_raw}",
                        spec.source_id,
                    )
                )
            if want_payout and spec.payout_rate_display_raw:
                findings.append(
                    StructuredFinding(
                        "setting_core_spec",
                        f"設定{spec.setting}: 機械割 {_format_payout(spec)}",
                        spec.source_id,
                    )
                )

    # --- 天井 ---
    if mentions_ceiling:
        ceiling_facts = mrepo.get_metric_facts(session, machine_id, category="天井")
        for fact in ceiling_facts[:limit]:
            dims = json.loads(fact.dimensions_json)
            dims_text = "・".join(f"{k}={v}" for k, v in dims.items())
            findings.append(
                StructuredFinding(
                    "metric_fact",
                    f"[天井/{fact.metric_key}] {dims_text}: {strip_citation(fact.display_raw)[0]}",
                    fact.source_id,
                )
            )

    # --- 汎用: metric_facts の次元値 (役名・パターン名等) が質問文に含まれるか ---
    all_facts = mrepo.get_metric_facts(session, machine_id)
    matched_fact_ids: set[int] = set()
    for fact in all_facts:
        dims = json.loads(fact.dimensions_json)
        if not any(_is_meaningful_term(v) and v in query for v in dims.values()):
            continue
        if fact.id in matched_fact_ids:
            continue
        matched_fact_ids.add(fact.id)
        dims_text = "・".join(f"{k}={v}" for k, v in dims.items())
        findings.append(
            StructuredFinding(
                "metric_fact",
                f"[{fact.metric_key}] {dims_text}: {strip_citation(fact.display_raw)[0]}",
                fact.source_id,
            )
        )
        if len(matched_fact_ids) >= limit:
            break

    # --- 示唆 (Hint): トリガーパターンが質問文に含まれるか ---
    hints = mrepo.get_hints(session, machine_id)
    for hint in hints:
        if not (_is_meaningful_term(hint.trigger_pattern) and hint.trigger_pattern in query):
            continue
        findings.append(
            StructuredFinding(
                "hint",
                f"[示唆/{hint.hint_category}] {hint.trigger_pattern}: "
                f"{strip_citation(hint.raw_text)[0]}",
                hint.source_id,
            )
        )

    # --- ゾーン: zone_key/name/aliases が質問文に含まれるか ---
    zones = mrepo.get_zones(session, machine_id)
    seen_zone_groups: set[str] = set()
    for zone in zones:
        aliases = json.loads(zone.aliases_json) if zone.aliases_json else []
        candidates = [zone.zone_key, zone.name, *aliases]
        if not any(_is_meaningful_term(c) and c in query for c in candidates):
            continue
        if zone.canonical_zone_key in seen_zone_groups:
            continue
        seen_zone_groups.add(zone.canonical_zone_key)
        attrs = json.loads(zone.attributes_json)
        attrs_text = "; ".join(
            f"{k}: {strip_citation(str(v))[0]}" for k, v in attrs.items() if v
        )
        findings.append(
            StructuredFinding("zone", f"[ゾーン/{zone.name}] {attrs_text}", zone.source_id)
        )

    return findings[:limit]
