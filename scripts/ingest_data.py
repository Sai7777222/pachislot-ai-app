"""機種データ取り込み CLI (Phase 2)。

例:
    python scripts/ingest_data.py \
        --excel "D:\\AI\\data\\raw\\reference\\スマスロ ミリオンゴッド-神々の軌跡-_解析.xlsx" \
        --machine-id smart_million_god_kamigami_no_kiseki \
        --data-source-type unknown

このスクリプトは「外部サイトからデータを取得・更新する処理」に相当し、
チャット回答処理からは呼び出されない（要件どおり分離している）。
元Excel/元資料は一切変更しない。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pachislot_ai.core.config import get_settings  # noqa: E402
from pachislot_ai.data.db import create_rag_engine, create_structured_engine  # noqa: E402
from pachislot_ai.data.enums import DataSourceType  # noqa: E402
from pachislot_ai.ingestion.persist import persist_result  # noqa: E402
from pachislot_ai.ingestion.pipeline import ingest_excel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="機種データ (Excel) を取り込む")
    parser.add_argument("--excel", required=True, type=Path, help="入力Excelファイルパス")
    parser.add_argument("--machine-id", required=True, help="機種ID (slug)")
    parser.add_argument(
        "--source-url",
        default=None,
        help="情報元URL。未指定時は Excel ファイルパスを file:// URI として使う",
    )
    parser.add_argument("--source-label", default=None, help="情報元の名称 (例: サイト名)")
    parser.add_argument(
        "--data-source-type",
        default=DataSourceType.UNKNOWN,
        choices=[e.value for e in DataSourceType],
    )
    parser.add_argument(
        "--report-out",
        default=None,
        type=Path,
        help="取り込みレポートJSONの出力先 (未指定時は data/processed/reports/<machine_id>.json)",
    )
    args = parser.parse_args()

    if not args.excel.is_file():
        print(f"ERROR: Excel file not found: {args.excel}")
        return 1

    source_url = args.source_url or args.excel.resolve().as_uri()

    print(f"Ingesting: {args.excel}")
    print(f"machine_id={args.machine_id}")
    print(f"source_url={source_url}")

    result = ingest_excel(
        args.excel,
        machine_id=args.machine_id,
        source_url=source_url,
        data_source_type=args.data_source_type,
        source_label=args.source_label,
    )

    settings = get_settings()
    structured_engine = create_structured_engine(settings.structured_db_path)
    rag_engine = create_rag_engine(settings.rag_db_path)
    persist_result(result, structured_engine, rag_engine)

    report_path = args.report_out or (
        settings.data_dir / "processed" / "reports" / f"{args.machine_id}_ingest_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "summary": result.summary(),
        "anomalies": result.anomalies,
        "unclassified_sample": result.unclassified[:50],
        "unclassified_total": len(result.unclassified),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 60)
    print(json.dumps(result.summary(), ensure_ascii=False, indent=2))
    print("-" * 60)
    print(f"Report written to: {report_path}")
    print(f"Structured DB: {settings.structured_db_path}")
    print(f"RAG DB: {settings.rag_db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
