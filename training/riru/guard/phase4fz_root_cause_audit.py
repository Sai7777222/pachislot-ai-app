"""Phase4FZ Section4: find_relevant_structured_facts() のroot cause audit。
実DB(read-only)に対し、queryごとにどのdimension値/zone名/aliasesとhint trigger_pattern
がsubstring一致してfindingが生成されたかを機械的に記録する。コードは変更しない。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
REPORTS_DIR = PROJECT_ROOT / "training" / "riru" / "reports"

from pachislot_ai.core.config import Settings  # noqa: E402
from pachislot_ai.data.db import create_structured_engine, open_session  # noqa: E402
from pachislot_ai.data.repositories import machine_repository as mrepo  # noqa: E402
from pachislot_ai.rag.structured_lookup import (  # noqa: E402
    _CEILING_KEYWORDS,
    _HIT_RATE_KEYWORDS,
    _PAYOUT_KEYWORDS,
    _extract_setting_numbers,
    _is_meaningful_term,
    find_relevant_structured_facts,
)

MANDATORY_QUERIES = [
    "天国ロングとは何か説明して",
    "天国について教えて",
    "AT-Fの性能と終了後の状態について教えて",
    "GG継続の条件は？",
]

REAL_ENTITY_QUERIES = [
    "設定6の初当り確率は？",
    "天井は何ゲームですか",
    "ガイアベルとは何か説明して",
    "SGGの仕組みを分かりやすく説明して",
    "GGとSGGの違いを初心者向けに説明して",
    "青7が連続したときのGG当選率は？",
    "Z-ZONEって何？",
    "GG準備中とは何ですか",
    "引き戻しについて教えて",
    "小役確率について教えて",
]


def trace_matches(session, machine_id: str, query: str) -> dict:
    """find_relevant_structured_facts()内部のsubstring判定を再現し、
    どのdimension値/zone名/hint trigger_patternがヒットしたかを個別に記録する
    (コード自体は変更せず、同じ関数・同じヘルパーをread-onlyで呼ぶだけ)。"""
    setting_numbers = _extract_setting_numbers(query)
    mentions_payout = any(k in query for k in _PAYOUT_KEYWORDS)
    mentions_hit_rate = any(k in query for k in _HIT_RATE_KEYWORDS)
    mentions_ceiling = any(k in query for k in _CEILING_KEYWORDS)

    matched_dims = []
    all_facts = mrepo.get_metric_facts(session, machine_id)
    for fact in all_facts:
        dims = json.loads(fact.dimensions_json)
        for k, v in dims.items():
            if _is_meaningful_term(v) and v in query:
                matched_dims.append({
                    "metric_key": fact.metric_key, "dim_key": k, "dim_value": v,
                    "fact_id": fact.id, "display_raw": fact.display_raw,
                })

    matched_zones = []
    zones = mrepo.get_zones(session, machine_id)
    for zone in zones:
        aliases = json.loads(zone.aliases_json) if zone.aliases_json else []
        candidates = [("zone_key", zone.zone_key), ("name", zone.name)] + [("alias", a) for a in aliases]
        for field, c in candidates:
            if _is_meaningful_term(c) and c in query:
                matched_zones.append({"zone_name": zone.name, "matched_field": field, "matched_value": c})

    matched_hints = []
    hints = mrepo.get_hints(session, machine_id)
    for hint in hints:
        if _is_meaningful_term(hint.trigger_pattern) and hint.trigger_pattern in query:
            matched_hints.append({"trigger_pattern": hint.trigger_pattern, "hint_category": hint.hint_category})

    findings = find_relevant_structured_facts(session, machine_id, query)

    return {
        "query": query,
        "setting_numbers": setting_numbers,
        "mentions_payout": mentions_payout,
        "mentions_hit_rate": mentions_hit_rate,
        "mentions_ceiling": mentions_ceiling,
        "matched_metric_fact_dims": matched_dims,
        "matched_zones": matched_zones,
        "matched_hints": matched_hints,
        "n_findings_total": len(findings),
        "findings_detail": [f.detail for f in findings],
    }


def classify_case(trace: dict) -> str:
    """S1-S6 CASE分類(機械的な一次分類、詳細はレポートで人間が確認)。"""
    if not trace["matched_metric_fact_dims"] and not trace["matched_zones"] and not trace["matched_hints"] \
       and not trace["setting_numbers"] and not trace["mentions_payout"] and not trace["mentions_hit_rate"] \
       and not trace["mentions_ceiling"]:
        return "NO_MATCH"
    # S1: 部分文字列一致で、matchしたvalueがqueryの一部でしかない(query全体と一致しない)
    reasons = []
    for m in trace["matched_metric_fact_dims"]:
        if m["dim_value"] != trace["query"].rstrip("とは何か説明してついて教えく。？?").strip():
            reasons.append("S1_substring")
    for m in trace["matched_zones"]:
        reasons.append("S1_substring")
    for m in trace["matched_hints"]:
        reasons.append("S1_substring")
    if reasons:
        return "S1_substring_matching"
    return "S6_mixed_or_unclear"


def main():
    settings = Settings()
    engine = create_structured_engine(settings.structured_db_path)
    with open_session(engine) as session:
        from sqlalchemy import select
        from pachislot_ai.data.models.structured import Machine
        machine_id = session.scalars(select(Machine.machine_id)).first()
        print(f"machine_id={machine_id}")

        results = {"mandatory": [], "real_entity": []}
        for q in MANDATORY_QUERIES:
            trace = trace_matches(session, machine_id, q)
            trace["case"] = classify_case(trace)
            results["mandatory"].append(trace)
            print(f"[mandatory] {q!r} -> n_findings={trace['n_findings_total']} case={trace['case']}")
            for m in trace["matched_metric_fact_dims"]:
                print(f"    metric_fact dim match: key={m['dim_key']} value={m['dim_value']!r} metric_key={m['metric_key']!r}")
            for m in trace["matched_zones"]:
                print(f"    zone match: {m}")
            for m in trace["matched_hints"]:
                print(f"    hint match: {m}")

        for q in REAL_ENTITY_QUERIES:
            trace = trace_matches(session, machine_id, q)
            trace["case"] = classify_case(trace)
            results["real_entity"].append(trace)
            print(f"[real] {q!r} -> n_findings={trace['n_findings_total']} case={trace['case']}")

    out_path = REPORTS_DIR / "phase4fz_root_cause.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
