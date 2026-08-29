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
from pachislot_ai.rag.entity_attribution import extract_query_entities, title_match_score

_SETTING_NUMBER_RE = re.compile(r"設定\s*([1-6])")
_PAYOUT_KEYWORDS = ("機械割", "割数", "出玉率")
_HIT_RATE_KEYWORDS = ("初当り", "初当たり", "初当", "当選確率")
_CEILING_KEYWORDS = ("天井",)
_MIN_TERM_LEN = 2

# Phase4FZ: dimension値(役名・項目名等)をqueryに対してsubstring一致させる際の
# 境界安全性チェック。title_match_score()のASCII単語境界チェックはGG/SGGのような
# 英数字の衝突を防ぐために作られたものだが、「天国」/「天国ロング」のような漢字・
# カタカナの複合語衝突までは防げない(「ロ」は非ASCIIのため既存チェックを素通りする)。
# ここでは逆に、隣接文字が漢字・カタカナ(=単語が継続している可能性が高い)である
# 場合だけを不安全と判定する小さな判定のみを追加する(新しい辞書・巨大regexではない)。
_WORD_CONTINUATION_RE = re.compile(r"[一-鿿ァ-ー]")


_CONTINUATION_PUNCT = {"-"}  # DBの命名規則(AT-F, RT-A等)がハイフンを語の一部として使うため


def _is_word_continuation(ch: str) -> bool:
    return bool(_WORD_CONTINUATION_RE.match(ch)) or ch in _CONTINUATION_PUNCT


def _value_matches_query_with_boundary(value: str, query: str) -> bool:
    """dimension値/zone名がqueryの中に、単語境界的に安全な形で含まれるかを判定する。
    「天国」が「天国ロング」の一部として、あるいは「SGG」が「SGG-EX」の一部として
    誤って一致しないよう、一致位置の前後が漢字/カタカナ/ハイフンで単語が継続して
    いないか(=真に独立した語として現れているか)を確認する。ASCII側は既存の
    title_match_score()と同じ考え方(隣接ASCII英数字があれば継続とみなす)を踏襲する。"""
    idx = query.find(value)
    while idx != -1:
        before_ok = idx == 0 or not (
            _is_word_continuation(query[idx - 1]) or (query[idx - 1].isascii() and query[idx - 1].isalnum())
        )
        after_idx = idx + len(value)
        after_ok = after_idx >= len(query) or not (
            _is_word_continuation(query[after_idx]) or (query[after_idx].isascii() and query[after_idx].isalnum())
        )
        if before_ok and after_ok:
            return True
        idx = query.find(value, idx + 1)
    return False


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

    # --- 汎用: metric_facts の metric_key (概念識別子) が query entity と一致するか ---
    # Phase4FZ: 以前は dims の値(group="天国"/"継続"/"終了" 等、多数の異なる metric_key
    # 間で使い回される汎用stateラベル)を生のqueryに対してsubstring検索していたため、
    # 「天国ロング」が無関係な「天国」関連テーブルを全件引き込む、「AT-Fの…終了後の状態」の
    # 「終了後」が偶然「終了」というgroup値と一致する、といったentityとは無関係な誤爆が
    # 発生していた(phase4fz_root_cause.json参照)。dims値はentityを一意に識別できないが、
    # metric_key(例: 「[GG中] GGストック当選率」「ガイアナビ規定回数振り分け」)はchunkの
    # titleと同じ役割(概念の一意識別子)を果たすため、Phase4FX/FYで既に検証済みの
    # entity抽出・境界安全なtitle_match_score()をそのまま再利用し、metric_key側を照合対象
    # とする(新しい辞書・新しい正規表現は導入しない)。
    # dimension値(役名・項目名等、例: 「ガイアベル」)は、metric_keyには現れないが
    # 正当なentity識別子として機能する場合がある(小役確率テーブルのdims={"項目":"ガイアベル"}等)。
    # これらは境界安全なsubstring一致(_value_matches_query_with_boundary)で個別に許可する。
    # 「天国」のような多数のmetric_keyで使い回される汎用stateラベルは、単独では
    # metric_key一致を伴わないケースが多いため、この境界チェックと合わせて判定する。
    query_entities = extract_query_entities(query)
    all_facts = mrepo.get_metric_facts(session, machine_id)
    matched_fact_ids: set[int] = set()
    for fact in all_facts:
        metric_key_match = any(title_match_score(e, fact.metric_key) > 0 for e in query_entities)
        dims = json.loads(fact.dimensions_json)
        dim_value_match = any(
            _is_meaningful_term(v) and _value_matches_query_with_boundary(v, query)
            for v in dims.values()
        )
        if not (metric_key_match or dim_value_match):
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
    # (Phase4FZ監査でこの経路が既知failureの誤帰属に関与した例は確認されなかったため、
    # 今回はスコープ外として無変更のまま維持する)
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

    # --- ゾーン: zone_key/name/aliases が query に境界安全に一致するか ---
    # Phase4FZ: 以前は生のsubstring testのみで、「GG」が「SGG」の部分文字列として
    # 誤って一致していた(phase4fz_root_cause.jsonで実測確認: 「SGGの仕組みを…」で
    # GG zoneも一致してしまう)。当初title_match_score()の再利用を試みたが、
    # 同関数の「titleがentityより短く、entityに包含される」側の分岐には境界チェックが
    # 無く(chunk titleでは問題化しなかった非対称性)、「SGG-EX」が「SGG」に、
    # 「ガイアステージMAX」が「ガイアステージ」に、「Z-ZONE極」が「Z-ZONE」に、
    # それぞれ誤って一致することが判明した(phase4fz_gt.jsonの GTP-05/06/07 で検出)。
    # そのためchunk側のtitle_match_score()自体は変更せず(Section3の凍結対象)、
    # ここでは_value_matches_query_with_boundary()による対称的な境界チェックを用いる。
    zones = mrepo.get_zones(session, machine_id)
    seen_zone_groups: set[str] = set()
    for zone in zones:
        aliases = json.loads(zone.aliases_json) if zone.aliases_json else []
        candidates = [zone.zone_key, zone.name, *aliases]
        if not any(
            _is_meaningful_term(c) and _value_matches_query_with_boundary(c, query)
            for c in candidates
        ):
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
