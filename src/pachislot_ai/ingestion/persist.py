"""IngestResult を構造化DB / RAGストアへ書き込む。

同一 machine_id の再取り込みは「機種単位で洗い替え」する（idempotent）。
将来、値の変更履歴を残したくなった場合はここに履歴テーブル書き込みを追加できる。
"""

from __future__ import annotations

from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from pachislot_ai.data.models.rag import RagDocument
from pachislot_ai.data.models.structured import (
    AnomalyRecord,
    Hint,
    Machine,
    Maker,
    MetricDefinition,
    MetricFact,
    PatternFact,
    SettingCoreSpec,
    Source,
    UnclassifiedItem,
    Zone,
)
from pachislot_ai.ingestion.pipeline import IngestResult


def _get_or_create_source(
    session: Session, url: str, label: str | None, data_source_type: str
) -> Source:
    existing = session.scalar(select(Source).where(Source.url == url))
    if existing:
        return existing
    source = Source(url=url, label=label, data_source_type=data_source_type)
    session.add(source)
    session.flush()
    return source


def _get_or_create_maker(session: Session, name: str | None) -> Maker | None:
    if not name:
        return None
    existing = session.scalar(select(Maker).where(Maker.maker_id == name))
    if existing:
        return existing
    maker = Maker(maker_id=name, name=name)
    session.add(maker)
    session.flush()
    return maker


def persist_structured(session: Session, result: IngestResult) -> None:
    source = _get_or_create_source(
        session, result.source_url, result.source_label, result.data_source_type
    )
    maker = _get_or_create_maker(session, result.machine.get("maker_name"))

    machine = session.scalar(
        select(Machine).where(Machine.machine_id == result.machine_id)
    )
    m = result.machine
    if machine is None:
        machine = Machine(machine_id=result.machine_id)
        session.add(machine)

    machine.name = m.get("name") or machine.machine_id
    machine.model_name = m.get("model_name")
    machine.maker_id = maker.id if maker else None
    machine.release_date_display_raw = m.get("release_date_display_raw")
    machine.release_date = m.get("release_date")
    machine.payout_rate_display_raw = m.get("payout_rate_display_raw")
    machine.payout_rate_min = m.get("payout_rate_min")
    machine.payout_rate_max = m.get("payout_rate_max")
    machine.source_page_last_updated_raw = m.get("source_page_last_updated_raw")
    machine.source_id = source.id
    session.flush()

    # 機種単位で洗い替え
    session.execute(delete(SettingCoreSpec).where(SettingCoreSpec.machine_id == result.machine_id))
    session.execute(delete(MetricFact).where(MetricFact.machine_id == result.machine_id))
    session.execute(delete(PatternFact).where(PatternFact.machine_id == result.machine_id))
    session.execute(delete(Zone).where(Zone.machine_id == result.machine_id))
    session.execute(delete(Hint).where(Hint.machine_id == result.machine_id))
    session.execute(
        delete(UnclassifiedItem).where(UnclassifiedItem.machine_id == result.machine_id)
    )
    session.execute(delete(AnomalyRecord).where(AnomalyRecord.machine_id == result.machine_id))

    existing_defs = {
        d.metric_key
        for d in session.scalars(select(MetricDefinition))
    }
    for key, definition in result.metric_definitions.items():
        if key in existing_defs:
            continue
        session.add(MetricDefinition(**definition))
        existing_defs.add(key)

    for spec in result.setting_core_specs:
        session.add(SettingCoreSpec(source_id=source.id, **spec))

    for fact in result.metric_facts:
        payload = {k: v for k, v in fact.items() if k not in ("metric_label_ja", "category")}
        session.add(MetricFact(source_id=source.id, **payload))

    for pattern in result.pattern_facts:
        session.add(PatternFact(source_id=source.id, **pattern))

    for zone in result.zones:
        session.add(Zone(source_id=source.id, **zone))

    for hint in result.hints:
        session.add(Hint(source_id=source.id, **hint))

    for item in result.unclassified:
        session.add(UnclassifiedItem(source_id=source.id, **item))

    for anomaly in result.anomalies:
        session.add(AnomalyRecord(machine_id=result.machine_id, **anomaly))


def persist_rag(session: Session, result: IngestResult) -> None:
    session.execute(delete(RagDocument).where(RagDocument.machine_id == result.machine_id))
    for doc in result.rag_documents:
        session.add(RagDocument(**doc))


def persist_result(result: IngestResult, structured_engine: Engine, rag_engine: Engine) -> None:
    from pachislot_ai.data.db import open_session

    with open_session(structured_engine) as session:
        persist_structured(session, result)

    with open_session(rag_engine) as session:
        persist_rag(session, result)
