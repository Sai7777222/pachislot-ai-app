"""Phase 2.5: 実データ (ミリオンゴッド) での品質改善確認。

Phase 2 完了時点の品質確認で見つかった問題が、Phase 2.5 のパーサ修正で
どう変化したかを実データで検証する。GPU/LLM は不要。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import select

from pachislot_ai.data.db import create_rag_engine, create_structured_engine, open_session
from pachislot_ai.data.enums import DataSourceType
from pachislot_ai.data.models.structured import PatternFact
from pachislot_ai.data.repositories import machine_repository as mrepo
from pachislot_ai.ingestion.persist import persist_result
from pachislot_ai.ingestion.pipeline import ingest_excel

EXCEL_PATH = Path(
    r"D:\AI\data\raw\reference\スマスロ ミリオンゴッド-神々の軌跡-_解析.xlsx"
)
MACHINE_ID = "smart_million_god_kamigami_no_kiseki_phase25test"

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
    db_dir = tmp_path_factory.mktemp("phase25_db")
    structured_engine = create_structured_engine(db_dir / "structured.db")
    rag_engine = create_rag_engine(db_dir / "rag_store.db")
    persist_result(ingest_result, structured_engine, rag_engine)
    return structured_engine, rag_engine


def test_source_excel_untouched() -> None:
    assert EXCEL_PATH.is_file()


def test_no_duplicate_conflicting_value_remains(ingest_result) -> None:
    """Phase 2 品質確認で見つかった24件の duplicate_conflicting_value が解消されていること。"""
    conflicting = [
        a for a in ingest_result.anomalies if a["anomaly_type"] == "duplicate_conflicting_value"
    ]
    assert conflicting == []


def test_parse_failed_count_reduced(ingest_result) -> None:
    """7件あった parse_status=failed が大幅に減っていること (完全ゼロは保証しない)。"""
    assert ingest_result.parse_failed_count <= 2


def test_unclassified_count_unchanged_or_better(ingest_result) -> None:
    """未分類は今回のスコープ外だが、増えていないこと。"""
    assert len(ingest_result.unclassified) <= 5


def test_blackhole_table_all_columns_preserved(db_sessions) -> None:
    """「ブラックホール中のテンパイ数字の法則」(rows 588-594) が
    3G目・示唆列を含めて情報欠落なく保存されていること。
    """
    structured_engine, _ = db_sessions
    with open_session(structured_engine) as session:
        patterns = mrepo.get_pattern_facts(session, MACHINE_ID)
        blackhole = [p for p in patterns if "ブラックホール" in p.metric_key]
        assert len(blackhole) >= 6  # 588,589,590,591,592,593,594 のうちヘッダー行を除く6-7行

        for p in blackhole:
            cols = json.loads(p.columns_json)
            # 修正前は 2G目 のみで 3G目・示唆が失われていた
            assert "3G目" in cols
            assert "示唆" in cols
            assert cols["示唆"]["display_raw"]  # 示唆列の原文が空でない


def test_gzone_scenario_table_all_columns_preserved(db_sessions) -> None:
    """「G-ZONE中の演出シナリオ」(rows 903-918) の1G目〜5G目が
    情報欠落なく保存されていること。
    """
    structured_engine, _ = db_sessions
    with open_session(structured_engine) as session:
        patterns = mrepo.get_pattern_facts(session, MACHINE_ID)
        gzone = [p for p in patterns if "G-ZONE中の演出シナリオ" in p.metric_key]
        assert len(gzone) >= 6

        for p in gzone:
            cols = json.loads(p.columns_json)
            # 修正前は列のうち1つしか保存されていなかった
            assert len(cols) >= 4


def test_gzone_lv1_has_multiple_pattern_indices(db_sessions) -> None:
    """LV1 は複数のOR代替パターンを持つため、pattern_index で区別され、
    上書きされていないこと。
    """
    structured_engine, _ = db_sessions
    gzone_key = "G-ZONE中の演出シナリオ（LV別の出目法則）"
    with open_session(structured_engine) as session:
        patterns = mrepo.get_pattern_facts(session, MACHINE_ID, metric_key=gzone_key)
        lv1 = [p for p in patterns if p.pattern_label == "LV1"]
        assert len(lv1) >= 2
        assert len({p.pattern_index for p in lv1}) == len(lv1)  # すべて異なる index


def test_paired_columns_correct_pairing(db_sessions) -> None:
    """「通常告知時(G-ZONE抜け後)の前兆ゲーム数振り分け」で
    2G→0.02%、17G→11.9% のように正しくペアリングされていること
    (修正前は 2G→11.9% のような誤った紐付けが発生していた)。
    """
    structured_engine, _ = db_sessions
    with open_session(structured_engine) as session:
        facts = mrepo.get_metric_facts(session, MACHINE_ID)
        target = [f for f in facts if "通常告知時" in f.metric_key]

        def find(front_value: str):
            for f in target:
                dims = json.loads(f.dimensions_json)
                if dims.get("前兆") == front_value:
                    return f
            return None

        f_2g = find("2G")
        f_17g = find("17G")
        assert f_2g is not None
        assert f_17g is not None
        assert f_2g.display_raw == "0.02% [cite: 369]"
        assert f_17g.display_raw == "11.9% [cite: 369]"
        assert f_2g.value != f_17g.value


def test_compound_value_split_into_separate_facts(db_sessions) -> None:
    """「0.7%・7.5%」が対応する見出しと1対1で分割され、複数factになっていること。"""
    structured_engine, _ = db_sessions
    with open_session(structured_engine) as session:
        facts = mrepo.get_metric_facts(session, MACHINE_ID)
        target = [f for f in facts if f.metric_key == "【通常滞在時】"]
        matched = [
            f
            for f in target
            if json.loads(f.dimensions_json).get("設定") == "4"
            and json.loads(f.dimensions_json).get("group") in ("右上がり黄7", "中段黄7")
        ]
        assert len(matched) == 2
        raws = {f.display_raw for f in matched}
        assert raws == {"15.0%", "3.5%"}


def test_approximate_flag_set_for_yaku_percentages(db_sessions) -> None:
    """「約80%」「約90%」が is_approximate=True で数値化されていること。"""
    structured_engine, _ = db_sessions
    with open_session(structured_engine) as session:
        facts = mrepo.get_metric_facts(session, MACHINE_ID)
        approx = [f for f in facts if f.is_approximate]
        assert len(approx) >= 2
        assert all(f.value is not None for f in approx)


def test_urabmode_bullet_list_is_rag_text_not_metric(ingest_result) -> None:
    """「裏モードのポイント」が metric_fact ではなく RAG 文章として保存されること。"""
    conflicting_keys = {f["metric_key"] for f in ingest_result.metric_facts}
    assert "裏モードのポイント" not in conflicting_keys

    doc_titles = {d["title"] for d in ingest_result.rag_documents}
    assert "裏モードのポイント（テーブル）" in doc_titles


def test_zone_aliases_recorded_without_deleting_data(db_sessions) -> None:
    """GG/ゴッドゲーム(GG)、SGG/スーパーゴッドゲーム(SGG) が
    データを削除・統合せずに canonical_zone_key で束ねられていること。
    """
    structured_engine, _ = db_sessions
    with open_session(structured_engine) as session:
        zones = mrepo.get_zones(session, MACHINE_ID)
        by_key = {z.zone_key: z for z in zones}

        assert "GG" in by_key
        assert "ゴッドゲーム(GG)" in by_key  # 元のゾーンは消えていない
        assert by_key["GG"].canonical_zone_key == by_key["ゴッドゲーム(GG)"].canonical_zone_key

        assert "SGG" in by_key
        assert "スーパーゴッドゲーム (SGG)" in by_key
        assert (
            by_key["SGG"].canonical_zone_key
            == by_key["スーパーゴッドゲーム (SGG)"].canonical_zone_key
        )


def test_all_pattern_facts_have_display_raw_in_columns(db_sessions) -> None:
    """pattern_facts のすべての列に display_raw (原文) が含まれること。"""
    structured_engine, _ = db_sessions
    with open_session(structured_engine) as session:
        patterns = list(
            session.scalars(select(PatternFact).where(PatternFact.machine_id == MACHINE_ID))
        )
        assert len(patterns) > 0
        for p in patterns:
            cols = json.loads(p.columns_json)
            for col_name, entry in cols.items():
                assert "display_raw" in entry, f"{col_name} missing display_raw"
