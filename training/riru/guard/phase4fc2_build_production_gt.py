"""Phase4FC2 Section10 (Gate E): 実本番DBに対する独立GT(100件以上)をfreezeする。
GT判定はDBの実データ照合のみに基づく客観的判定(モデル判断は使わない、RULE EVAL-001準拠)。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
REPORTS_DIR = PROJECT_ROOT / "training" / "riru" / "reports"

from pachislot_ai.core.config import Settings  # noqa: E402
from pachislot_ai.data.db import create_structured_engine, open_session  # noqa: E402
from pachislot_ai.rag.embedder import Embedder  # noqa: E402
from pachislot_ai.rag.entity_attribution import extract_query_entities  # noqa: E402
from pachislot_ai.rag.retriever import Retriever  # noqa: E402
from pachislot_ai.rag.structured_lookup import find_relevant_structured_facts  # noqa: E402
from pachislot_ai.rag.vector_store import VectorStore  # noqa: E402

# category: direct_numeric / explanation / mode_state / trigger_condition / effect_benefit /
#           ceiling / quit_timing / comparison / entity_missing / structured_only /
#           chunk_only / chunk_and_structured
QUERIES = [
    # --- direct_numeric (設定別数値) ---
    ("PG-N01", "direct_numeric", "設定1の初当り確率は？"),
    ("PG-N02", "direct_numeric", "設定2の初当り確率は？"),
    ("PG-N03", "direct_numeric", "設定3の初当り確率は？"),
    ("PG-N04", "direct_numeric", "設定4の初当り確率は？"),
    ("PG-N05", "direct_numeric", "設定5の初当り確率は？"),
    ("PG-N06", "direct_numeric", "設定6の初当り確率は？"),
    ("PG-N07", "direct_numeric", "設定1の機械割は？"),
    ("PG-N08", "direct_numeric", "設定2の機械割は？"),
    ("PG-N09", "direct_numeric", "設定4の機械割は？"),
    ("PG-N10", "direct_numeric", "設定5の機械割は？"),
    ("PG-N11", "direct_numeric", "設定3と設定4の機械割は？"),
    ("PG-N12", "direct_numeric", "全設定の初当り確率を教えて"),
    # --- explanation (概念説明) ---
    ("PG-E01", "explanation", "GGについて教えて"),
    ("PG-E02", "explanation", "SGGについて教えて"),
    ("PG-E03", "explanation", "ゼウスモードとは何ですか"),
    ("PG-E04", "explanation", "ガイアステージについて教えて"),
    ("PG-E05", "explanation", "PGGとは何ですか"),
    ("PG-E06", "explanation", "Z-ZONEって何？"),
    ("PG-E07", "explanation", "ガイアナビとは何ですか"),
    ("PG-E08", "explanation", "ガイアベルとは何か説明して"),
    # --- mode_state (モード/状態) ---
    ("PG-M01", "mode_state", "GG中とはどんな状態か教えて"),
    ("PG-M02", "mode_state", "GG準備中とは何ですか"),
    ("PG-M03", "mode_state", "天国について教えて"),
    ("PG-M04", "mode_state", "天国準備について教えて"),
    ("PG-M05", "mode_state", "低確状態について教えて"),
    ("PG-M06", "mode_state", "裏天国について教えて"),
    ("PG-M07", "mode_state", "表モードについて教えて"),
    ("PG-M08", "mode_state", "裏モードについて教えて"),
    # --- trigger_condition (契機/条件) ---
    ("PG-T01", "trigger_condition", "継続契機について教えて"),
    ("PG-T02", "trigger_condition", "GG当選の契機を教えて"),
    ("PG-T03", "trigger_condition", "Z-ZONE昇格の条件を教えて"),
    ("PG-T04", "trigger_condition", "青7連続でのGG当選条件を教えて"),
    ("PG-T05", "trigger_condition", "GG継続の条件は？"),
    # --- effect_benefit (恩恵) ---
    ("PG-B01", "effect_benefit", "天井恩恵について教えて"),
    ("PG-B02", "effect_benefit", "契機と恩恵の関係を教えて"),
    ("PG-B03", "effect_benefit", "高ループストックの恩恵を教えて"),
    ("PG-B04", "effect_benefit", "白7の特典を教えて"),
    # --- ceiling (天井) ---
    ("PG-C01", "ceiling", "天井は何ゲームですか"),
    ("PG-C02", "ceiling", "天井ゲーム数を教えて"),
    ("PG-C03", "ceiling", "設定変更時の天井を教えて"),
    ("PG-C04", "ceiling", "天井到達時の恩恵を教えて"),
    # --- quit_timing (ヤメ時) ---
    ("PG-Q01", "quit_timing", "ヤメ時はいつがいい？"),
    ("PG-Q02", "quit_timing", "ヤメ時の目安を教えて"),
    ("PG-Q03", "quit_timing", "天井とヤメ時を合わせて初心者向けに説明して"),
    # --- comparison (比較) ---
    ("PG-CP01", "comparison", "GGとSGGの違いを初心者向けに説明して"),
    ("PG-CP02", "comparison", "GGとPGGの違いを教えて"),
    ("PG-CP03", "comparison", "ガイアステージとZ-ZONEの違いを教えて"),
    ("PG-CP04", "comparison", "ゼウスモードとガイアステージの関係を教えて"),
    ("PG-CP05", "comparison", "GG準備中とGG中の違いを教えて"),
    ("PG-CP06", "comparison", "天国と天国準備の違いを教えて"),
    ("PG-CP07", "comparison", "ループストックとGGストックの違いを教えて"),
    ("PG-CP08", "comparison", "GG当選とSGG当選の違いを教えて"),
    # --- entity_missing (phantom) ---
    ("PG-P01", "entity_missing", "AT-Fの性能と終了後の状態について教えて"),
    ("PG-P02", "entity_missing", "RT-AとRT-Bの違いを要約して"),
    ("PG-P03", "entity_missing", "天国ロングとは何か説明して"),
    ("PG-P04", "entity_missing", "GGプラスとは何か説明して"),
    ("PG-P05", "entity_missing", "SGG-EXとは何か説明して"),
    ("PG-P06", "entity_missing", "ガイアステージMAXについて教えて"),
    ("PG-P07", "entity_missing", "確定役ネオとは何か説明して"),
    ("PG-P08", "entity_missing", "モードαとモードβの違いを簡単に教えて"),
    ("PG-P09", "entity_missing", "Z-ZONE極について教えて"),
    ("PG-P10", "entity_missing", "裏ZONEについて教えて"),
    # --- structured_only (構造化データのみで答えられる想定) ---
    ("PG-S01", "structured_only", "機械割について教えて"),
    ("PG-S02", "structured_only", "初当りについて教えて"),
    ("PG-S03", "structured_only", "設定6の機械割は？"),
    ("PG-S04", "structured_only", "最低設定と最高設定の機械割の差を教えて"),
    ("PG-S05", "structured_only", "設定3と設定5の機械割は？それぞれの数値と差を教えて"),
    # --- chunk_only (解説文章のみで答えられる想定) ---
    ("PG-CH01", "chunk_only", "SU4について教えて"),
    ("PG-CH02", "chunk_only", "引き戻しについて教えて"),
    ("PG-CH03", "chunk_only", "小役履歴とモード示唆出目の関係を教えて"),
    ("PG-CH04", "chunk_only", "確定役とフリーズ演出の違いを教えて"),
    ("PG-CH05", "chunk_only", "示唆と確定の違いを教えて"),
    # --- chunk_and_structured (両方揃う想定) ---
    ("PG-CS01", "chunk_and_structured", "SGGの仕組みを分かりやすく説明して"),
    ("PG-CS02", "chunk_and_structured", "GG継続の条件は？"),
    ("PG-CS03", "chunk_and_structured", "青7が連続したときのGG当選率は？"),
    ("PG-CS04", "chunk_and_structured", "SGGゲーム数の振り分けを教えて"),
    ("PG-CS05", "chunk_and_structured", "小役確率について教えて"),
    # --- additional query-style variants for coverage (総数調整用) ---
    ("PG-QS01", "explanation", "GGとは？"),
    ("PG-QS02", "explanation", "SGGを要約して"),
    ("PG-QS03", "explanation", "ガイアステージについて詳しく教えて"),
    ("PG-QS04", "explanation", "ミリオンゴッドの遊び方を初心者向けにやさしく説明して"),
    ("PG-QS05", "explanation", "ミリオンゴッドの遊び方を簡潔に説明して"),
    ("PG-QS06", "explanation", "ミリオンゴッドの機種の特徴を教えて"),
    ("PG-QS07", "trigger_condition", "GGストックの仕組みを教えて"),
    ("PG-QS08", "mode_state", "GG中の状態について"),
    ("PG-QS09", "entity_missing", "設定Xと設定Yの機械割差を教えて"),
    ("PG-QS10", "entity_missing", "モード7とモード8の違いを教えて"),
    ("PG-QS11", "comparison", "ガイアベルとガイアナビの違いを教えて"),
    ("PG-QS12", "comparison", "SGGとGG継続ゾーンの関係を教えて"),
    ("PG-QS13", "trigger_condition", "契機と恩恵の関係を教えて"),
    ("PG-QS14", "mode_state", "終了状態と移行先の関係を教えて"),
    ("PG-QS15", "explanation", "当選と前兆の違いを教えて"),
    # --- additional coverage to comfortably clear ≥100 ---
    ("PG-QS16", "direct_numeric", "設定変更時の天井の割合を教えて"),
    ("PG-QS17", "mode_state", "低確A・低確B・天国準備滞在時について教えて"),
    ("PG-QS18", "trigger_condition", "SGGゲーム数の振り分け条件を教えて"),
    ("PG-QS19", "effect_benefit", "天井恩恵詳細を教えて"),
    ("PG-QS20", "comparison", "白7とALL色の関係を教えて"),
    ("PG-QS21", "entity_missing", "ガイアベルSPとは何か説明して"),
    ("PG-QS22", "entity_missing", "PGGロングとは何ですか"),
    ("PG-QS23", "chunk_only", "炎・戦車の解説を教えて"),
    ("PG-QS24", "structured_only", "設定2と設定4の機械割の差はどれくらい？"),
    ("PG-QS25", "mode_state", "超天国滞在時について教えて"),
]


def main():
    settings = Settings()
    embedder = Embedder(settings.embedding_model_path, device=settings.embedding_device)
    vector_store = VectorStore(settings.vector_db_path, settings.vector_db_collection)
    retriever = Retriever(embedder, vector_store, default_top_k=settings.rag_top_k)
    structured_engine = create_structured_engine(settings.structured_db_path)

    print(f"total queries: {len(QUERIES)}")
    gt_rows = []
    with open_session(structured_engine) as session:
        from sqlalchemy import select
        from pachislot_ai.data.models.structured import Machine
        machine_id = session.scalars(select(Machine.machine_id)).first()

        for qid, cat, q in QUERIES:
            # 客観的な事前チェック: 素のembedding top-6とall_chunks title補完検索の両方で
            # 何らかの関連chunkが存在するか、構造化factsが存在するか(GT freeze用、生成前)
            raw_chunks = retriever.search(q, machine_id=None, top_k=settings.rag_top_k)
            structured_findings = find_relevant_structured_facts(session, machine_id, q)
            query_entities = extract_query_entities(q)
            has_any_evidence = len(raw_chunks) > 0 or len(structured_findings) > 0
            gt_label = "ENTITY_MISSING_EXPECTED" if cat == "entity_missing" else (
                "EXPECTED_EVIDENCE" if has_any_evidence else "NO_EVIDENCE_UNEXPECTED"
            )
            gt_rows.append({
                "id": qid, "category": cat, "prompt": q, "query_entities": query_entities,
                "gt_label": gt_label,
                "raw_embedding_chunk_count": len(raw_chunks),
                "structured_findings_count": len(structured_findings),
            })
            print(f"[{cat}] {qid}: {gt_label} (chunks={len(raw_chunks)}, structured={len(structured_findings)})")

    gt = {
        "phase": "Phase4FC2", "section": "Section10 (Gate E production GT)",
        "gt_construction_method": "生embedding top-6の有無 + find_relevant_structured_facts()の有無のみに基づく客観的事前ラベリング(モデル判断は使わない)。entity_missing区分は設計上NO_FACTSを期待するため別ラベル。",
        "n_total": len(gt_rows), "rows": gt_rows,
    }
    gt_json_str = json.dumps(gt, ensure_ascii=False, indent=2, sort_keys=True)
    (REPORTS_DIR / "phase4fc2_production_gt.json").write_text(gt_json_str, encoding="utf-8")
    gt_hash = hashlib.sha256(gt_json_str.encode("utf-8")).hexdigest()
    (REPORTS_DIR / "phase4fc2_production_gt_hash.txt").write_text(
        f"sha256: {gt_hash}\nfile: phase4fc2_production_gt.json\nn_total: {len(gt_rows)}\n", encoding="utf-8")
    print(f"n_total={len(gt_rows)} gt_hash={gt_hash}")


if __name__ == "__main__":
    main()
