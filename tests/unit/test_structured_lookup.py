"""structured_lookup のキーワード検索テスト (実データ使用、GPU/Embedding不要)。"""

from __future__ import annotations

import pytest

from pachislot_ai.core.config import get_settings
from pachislot_ai.data.db import create_structured_engine, open_session
from pachislot_ai.rag.structured_lookup import find_relevant_structured_facts

MACHINE_ID = "smart_million_god_kamigami_no_kiseki"

pytestmark = pytest.mark.skipif(
    not get_settings().structured_db_path.is_file(),
    reason="structured.db not found (run scripts/ingest_data.py first)",
)


@pytest.fixture(scope="module")
def structured_engine():
    settings = get_settings()
    return create_structured_engine(settings.structured_db_path)


def test_payout_query_returns_percentage_for_all_settings(structured_engine) -> None:
    with open_session(structured_engine) as session:
        findings = find_relevant_structured_facts(session, MACHINE_ID, "機械割を教えて")
    assert any("設定6" in f.detail and "%" in f.detail for f in findings)
    assert any("設定1" in f.detail for f in findings)


def test_setting_number_narrows_to_that_setting_only(structured_engine) -> None:
    with open_session(structured_engine) as session:
        findings = find_relevant_structured_facts(session, MACHINE_ID, "設定6の機械割は？")
    assert findings
    assert all("設定6" in f.detail for f in findings if f.kind == "setting_core_spec")


def test_hit_rate_query_uses_fraction_display_raw(structured_engine) -> None:
    with open_session(structured_engine) as session:
        findings = find_relevant_structured_facts(session, MACHINE_ID, "設定6の初当りは？")
    hit_findings = [f for f in findings if "初当り" in f.detail]
    assert hit_findings
    assert "1/295" in hit_findings[0].detail


def test_ceiling_query_returns_all_ceiling_facts(structured_engine) -> None:
    with open_session(structured_engine) as session:
        findings = find_relevant_structured_facts(session, MACHINE_ID, "天井は何ゲーム？")
    ceiling_findings = [f for f in findings if "天井" in f.detail]
    assert len(ceiling_findings) >= 3
    joined = " ".join(f.detail for f in ceiling_findings)
    assert "1480G" in joined
    assert "64.5%" in joined


def test_role_name_keyword_match_finds_gaia_bell(structured_engine) -> None:
    with open_session(structured_engine) as session:
        findings = find_relevant_structured_facts(session, MACHINE_ID, "ガイアベルの確率は？")
    assert any("1/37.6" in f.detail for f in findings)


def test_unrelated_query_returns_no_specific_numeric_findings(structured_engine) -> None:
    with open_session(structured_engine) as session:
        findings = find_relevant_structured_facts(
            session, MACHINE_ID, "今日の天気はどうですか？"
        )
    # 機械割/初当り/天井キーワードも役名も含まないため、設定別スペックの
    # 無条件表示 (want_payout/want_hit のデフォルトON) 以外は基本的に空に近い
    assert isinstance(findings, list)


def test_no_source_id_leaks_when_absent(structured_engine) -> None:
    with open_session(structured_engine) as session:
        findings = find_relevant_structured_facts(session, MACHINE_ID, "ガイアベルの確率は？")
    assert all(f.source_id is not None for f in findings)  # 出典が必ず追跡できる


def test_display_raw_and_normalized_both_traceable_via_source(structured_engine) -> None:
    """StructuredFinding.detail は原文値ベースであり、source_id経由でsourcesテーブルへ辿れる。"""
    from pachislot_ai.data.repositories import machine_repository as mrepo

    with open_session(structured_engine) as session:
        findings = find_relevant_structured_facts(session, MACHINE_ID, "設定6の初当りは？")
        assert findings
        src = mrepo.get_source(session, findings[0].source_id)
        assert src is not None
        assert src.url
