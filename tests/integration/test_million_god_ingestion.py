"""実データ (スマスロ ミリオンゴッド-神々の軌跡-_解析.xlsx) を使った取り込み〜DB確認の統合テスト。

Phase 2 要件の「テスト」節に対応。GPU/LLMは不要なので `llm` マーカーは付けない
(デフォルトの `pytest` 実行で毎回検証される)。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pachislot_ai.data.db import create_rag_engine, create_structured_engine, open_session
from pachislot_ai.data.enums import DataSourceType
from pachislot_ai.data.repositories import machine_repository as mrepo
from pachislot_ai.data.repositories import rag_repository as rrepo
from pachislot_ai.ingestion.persist import persist_result
from pachislot_ai.ingestion.pipeline import ingest_excel

EXCEL_PATH = Path(
    r"D:\AI\data\raw\reference\スマスロ ミリオンゴッド-神々の軌跡-_解析.xlsx"
)
MACHINE_ID = "smart_million_god_kamigami_no_kiseki"

pytestmark = pytest.mark.skipif(
    not EXCEL_PATH.is_file(), reason=f"reference excel not found: {EXCEL_PATH}"
)


@pytest.fixture(scope="module")
def ingest_result():
    return ingest_excel(
        EXCEL_PATH,
        machine_id=MACHINE_ID,
        source_url=EXCEL_PATH.resolve().as_uri(),
        data_source_type=DataSourceType.UNKNOWN,
    )


@pytest.fixture(scope="module")
def db_sessions(tmp_path_factory, ingest_result):
    db_dir = tmp_path_factory.mktemp("phase2_db")
    structured_engine = create_structured_engine(db_dir / "structured.db")
    rag_engine = create_rag_engine(db_dir / "rag_store.db")
    persist_result(ingest_result, structured_engine, rag_engine)
    return structured_engine, rag_engine


# --- 元Excelを一切変更していないことの前提確認 ------------------------------------


def test_source_excel_untouched_marker() -> None:
    # 元ファイルへの書き込みは一切行わない (読み取り専用オープンのみ)
    assert EXCEL_PATH.is_file()


# --- IngestResult (DB書き込み前の変換結果) の確認 -----------------------------------


def test_machine_name_extracted(ingest_result) -> None:
    assert ingest_result.machine["name"]
    assert "ミリオンゴッド" in ingest_result.machine["name"]


def test_maker_extracted(ingest_result) -> None:
    assert ingest_result.machine["maker_name"]
    assert "ミズホ" in ingest_result.machine["maker_name"]


def test_setting_core_specs_all_six_settings(ingest_result) -> None:
    settings = {s["setting"] for s in ingest_result.setting_core_specs}
    assert settings == {1, 2, 3, 4, 5, 6}


def test_unclassified_items_are_flagged_not_silently_dropped(ingest_result) -> None:
    # 判断できない行が0件である必要はないが、あれば理由付きで記録されていること
    for item in ingest_result.unclassified:
        assert item["reason"]
        assert item["raw_cells_json"]


def test_date_contradiction_is_detected(ingest_result) -> None:
    types = {a["anomaly_type"] for a in ingest_result.anomalies}
    assert "date_contradiction" in types


# --- 構造化DB (SQLite) への永続化後の確認 --------------------------------------


def test_machine_retrievable_from_db(db_sessions) -> None:
    structured_engine, _ = db_sessions
    with open_session(structured_engine) as session:
        machine = mrepo.get_machine(session, MACHINE_ID)
        assert machine is not None
        assert "ミリオンゴッド" in machine.name


def test_maker_retrievable_from_db(db_sessions) -> None:
    structured_engine, _ = db_sessions
    with open_session(structured_engine) as session:
        machine = mrepo.get_machine(session, MACHINE_ID)
        maker = mrepo.get_maker_by_id(session, machine.maker_id)
        assert maker is not None
        assert "ミズホ" in maker.name


def test_setting6_hit_rate_and_payout(db_sessions) -> None:
    structured_engine, _ = db_sessions
    with open_session(structured_engine) as session:
        spec = mrepo.get_setting_core_spec(session, MACHINE_ID, 6)
        assert spec is not None
        # 原文値
        assert spec.hit_rate_display_raw == "1/533" or spec.hit_rate_display_raw == "1/295"
        # 設定6は 1/295 (最良設定)
        assert spec.hit_rate_display_raw == "1/295"
        assert spec.hit_rate_odds_denominator == pytest.approx(295.0)
        assert spec.hit_rate_probability == pytest.approx(1 / 295)
        assert spec.payout_rate == pytest.approx(1.146)


def test_ceiling_facts_retrievable(db_sessions) -> None:
    structured_engine, _ = db_sessions
    with open_session(structured_engine) as session:
        facts = mrepo.get_metric_facts(session, MACHINE_ID, category="天井")
        assert len(facts) > 0
        by_games = {f.dimensions_json: f for f in facts}
        assert any("1480G" in k for k in by_games)


def test_small_win_probability_facts_retrievable(db_sessions) -> None:
    structured_engine, _ = db_sessions
    with open_session(structured_engine) as session:
        facts = mrepo.get_metric_facts(session, MACHINE_ID, category="小役確率")
        assert len(facts) > 0
        gaia_bell = next(
            (f for f in facts if "ガイアベル" in f.dimensions_json and f.metric_key == "小役確率"),
            None,
        )
        assert gaia_bell is not None
        assert gaia_bell.display_raw == "1/37.6"
        assert gaia_bell.value == pytest.approx(1 / 37.6)


def test_setting_hints_retrievable(db_sessions) -> None:
    structured_engine, _ = db_sessions
    with open_session(structured_engine) as session:
        hints = mrepo.get_hints(session, MACHINE_ID, hint_category="設定示唆")
        assert len(hints) >= 5
        rainbow = next((h for h in hints if h.trigger_pattern == "虹"), None)
        assert rainbow is not None
        assert rainbow.raw_text == "設定6濃厚"
        assert "6" in rainbow.implied_settings_json


def test_rag_documents_retrievable_by_machine_id(db_sessions) -> None:
    _, rag_engine = db_sessions
    with open_session(rag_engine) as session:
        docs = rrepo.get_documents_by_machine(session, MACHINE_ID)
        assert len(docs) > 10
        # 「概要」を含むラベルは複数 (機種概要, 表モード概要 等) が overview カテゴリになりうるため
        # doc_id (機種概要専用) で一意に取得する
        overview_doc_id = f"{MACHINE_ID}::overview::機種概要"
        overview = rrepo.get_document(session, overview_doc_id)
        assert overview is not None
        assert "ミリオンゴッド" in overview.body_text


def test_provenance_is_traceable_per_record(db_sessions) -> None:
    structured_engine, rag_engine = db_sessions
    with open_session(structured_engine) as session:
        machine = mrepo.get_machine(session, MACHINE_ID)
        source = mrepo.get_source(session, machine.source_id)
        assert source is not None
        assert source.url.startswith("file:///")
        assert source.data_source_type == DataSourceType.UNKNOWN

        facts = mrepo.get_metric_facts(session, MACHINE_ID, category="天井")
        assert all(f.source_id == source.id for f in facts)
        assert all(f.retrieved_at is not None for f in facts)
        assert all(f.review_status == "unverified" for f in facts)

        hints = mrepo.get_hints(session, MACHINE_ID)
        assert all(h.source_id == source.id for h in hints)

    with open_session(rag_engine) as session:
        docs = rrepo.get_documents_by_machine(session, MACHINE_ID)
        assert all(d.source_url == source.url for d in docs)
        assert all(d.retrieved_at is not None for d in docs)


def test_raw_and_normalized_values_both_available(db_sessions) -> None:
    structured_engine, _ = db_sessions
    with open_session(structured_engine) as session:
        spec = mrepo.get_setting_core_spec(session, MACHINE_ID, 6)
        # 原文
        assert spec.hit_rate_display_raw == "1/295"
        # 正規化値
        assert spec.hit_rate_probability is not None
        assert abs(spec.hit_rate_probability - 1 / 295) < 1e-9

        ceiling = mrepo.get_metric_facts(session, MACHINE_ID, category="天井")
        raw_only_ok = [f for f in ceiling if f.display_raw and "%" in f.display_raw]
        assert raw_only_ok
        for f in raw_only_ok:
            if f.parse_status == "ok":
                assert f.value is not None
