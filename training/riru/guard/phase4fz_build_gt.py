"""Phase4FZ Section5: Structured Fact Ground Truth構築(コード変更前にfreeze)。
GT判定基準はDBの実データ(zone一覧・metric_key一覧の直接照合)のみに基づく、
機械的・客観的な判定であり、モデルの判断は一切用いない(RULE EVAL-001準拠)。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
REPORTS_DIR = PROJECT_ROOT / "training" / "riru" / "reports"

# --- Phantom (実在しないentity, 最低15件、天国ロング・AT-Fを含む必須) ---
PHANTOM_QUERIES = [
    {"id": "GTP-01", "query": "天国ロングとは何か説明して", "phantom_entity": "天国ロング", "note": "既存zoneに天国ロングは存在しない。'天国'はmetric_factsのgroup値(汎用ラベル)としてのみ存在"},
    {"id": "GTP-02", "query": "AT-Fの性能と終了後の状態について教えて", "phantom_entity": "AT-F", "note": "zone/metric_keyにAT-Fは一切存在しない"},
    {"id": "GTP-03", "query": "RT-AとRT-Bの違いを要約して", "phantom_entity": "RT-A/RT-B", "note": "zone/metric_keyに存在しない"},
    {"id": "GTP-04", "query": "GGプラスとは何か説明して", "phantom_entity": "GGプラス", "note": "'GG'を含む複合語だが、GGプラスというzoneは存在しない"},
    {"id": "GTP-05", "query": "SGG-EXとは何か説明して", "phantom_entity": "SGG-EX", "note": "存在しない"},
    {"id": "GTP-06", "query": "ガイアステージMAXについて教えて", "phantom_entity": "ガイアステージMAX", "note": "'ガイアステージ'は実在zoneだが'MAX'付きは存在しない"},
    {"id": "GTP-07", "query": "Z-ZONE極について教えて", "phantom_entity": "Z-ZONE極", "note": "'Z-ZONE'は実在zoneだが'極'付きは存在しない"},
    {"id": "GTP-08", "query": "確定役ネオとは何か説明して", "phantom_entity": "確定役ネオ", "note": "存在しない"},
    {"id": "GTP-09", "query": "天国準備中ロングとは何か説明して", "phantom_entity": "天国準備中ロング", "note": "'天国準備'はgroup値として存在するが'天国準備中ロング'は存在しない"},
    {"id": "GTP-10", "query": "ガイアベルSPとは何か説明して", "phantom_entity": "ガイアベルSP", "note": "'ガイアベル'は実在小役だが'SP'付きは存在しない"},
    {"id": "GTP-11", "query": "モードαとモードβの違いを簡単に教えて", "phantom_entity": "モードα/モードβ", "note": "存在しない"},
    {"id": "GTP-12", "query": "裏ZONEについて教えて", "phantom_entity": "裏ZONE", "note": "存在しない"},
    {"id": "GTP-13", "query": "継続ロングとは何ですか", "phantom_entity": "継続ロング", "note": "'継続'はgroup値として存在するが'継続ロング'は存在しない"},
    {"id": "GTP-14", "query": "終了ショートとは何ですか", "phantom_entity": "終了ショート", "note": "'終了'はgroup値として存在するが'終了ショート'は存在しない"},
    {"id": "GTP-15", "query": "PGGロングとは何ですか", "phantom_entity": "PGGロング", "note": "'PGG'(プレミアムゴッドゲーム)は実在zoneだが'PGGロング'は存在しない"},
]

# --- Real entity (実在entity, 最低20件) ---
REAL_QUERIES = [
    {"id": "GTR-01", "query": "GGについて教えて", "real_entity": "GG", "note": "zone GG"},
    {"id": "GTR-02", "query": "SGGについて教えて", "real_entity": "SGG", "note": "zone SGG"},
    {"id": "GTR-03", "query": "ゼウスモードとは何ですか", "real_entity": "ゼウスモード", "note": "zone ゼウスモード"},
    {"id": "GTR-04", "query": "ガイアステージについて教えて", "real_entity": "ガイアステージ", "note": "zone ガイアステージ"},
    {"id": "GTR-05", "query": "PGGとは何ですか", "real_entity": "PGG", "note": "zone プレミアムゴッドゲーム(PGG)"},
    {"id": "GTR-06", "query": "Z-ZONEって何？", "real_entity": "Z-ZONE", "note": "zone Z-ZONE"},
    {"id": "GTR-07", "query": "設定6の初当り確率は？", "real_entity": "設定6/初当り確率", "note": "setting_core_spec専用分岐"},
    {"id": "GTR-08", "query": "天井は何ゲームですか", "real_entity": "天井", "note": "天井keyword専用分岐"},
    {"id": "GTR-09", "query": "機械割について教えて", "real_entity": "機械割", "note": "payout keyword専用分岐"},
    {"id": "GTR-10", "query": "設定3の機械割は？", "real_entity": "設定3/機械割", "note": "setting_core_spec専用分岐"},
    {"id": "GTR-11", "query": "GGとSGGの違いを初心者向けに説明して", "real_entity": "GG,SGG", "note": "zone GG + zone SGG(2entity)"},
    {"id": "GTR-12", "query": "GGとPGGの違いを教えて", "real_entity": "GG,PGG", "note": "zone GG + zone PGG(2entity)"},
    {"id": "GTR-13", "query": "ガイアステージとZ-ZONEの違いを教えて", "real_entity": "ガイアステージ,Z-ZONE", "note": "zone x2"},
    {"id": "GTR-14", "query": "ゼウスモードとガイアステージの関係を教えて", "real_entity": "ゼウスモード,ガイアステージ", "note": "zone x2"},
    {"id": "GTR-15", "query": "GG準備中とは何ですか", "real_entity": "GG準備中(metric_key)", "note": "metric_key '[GG準備中] GGストック当選率'が実在"},
    {"id": "GTR-16", "query": "GG中の状態について教えて", "real_entity": "GG中(metric_key)", "note": "metric_key '[GG中] GGストック当選率'が実在"},
    {"id": "GTR-17", "query": "ガイアナビの規定回数振り分けを教えて", "real_entity": "ガイアナビ規定回数振り分け(metric_key)", "note": "metric_key実在"},
    {"id": "GTR-18", "query": "ガイアステージ終了抽選について教えて", "real_entity": "ガイアステージ終了抽選(metric_key)", "note": "metric_key実在"},
    {"id": "GTR-19", "query": "青7連続時のGG当選率は？", "real_entity": "青7連続/GG当選率", "note": "hint/metric_fact実在"},
    {"id": "GTR-20", "query": "初当りについて教えて", "real_entity": "初当り", "note": "hit_rate keyword専用分岐"},
]

# --- Close concept (実在するが紛らわしい類似名, 最低5件) ---
CLOSE_CONCEPT_QUERIES = [
    {"id": "GTC-01", "query": "SGGの仕組みを分かりやすく説明して", "target_entity": "SGG", "confusable_with": "GG", "note": "'GG'が'SGG'の部分文字列であるため、GGのzoneデータが誤って混入しないことを確認する"},
    {"id": "GTC-02", "query": "PGGについて教えて", "target_entity": "PGG", "confusable_with": "GG,SGG", "note": "'GG'が'PGG'の部分文字列"},
    {"id": "GTC-03", "query": "GG準備中とGG中の違いを教えて", "target_entity": "GG準備中,GG中", "confusable_with": "互いに", "note": "2つの類似metric_keyを取り違えないことを確認"},
    {"id": "GTC-04", "query": "天国と天国準備の違いを教えて", "target_entity": "天国,天国準備", "confusable_with": "互いに", "note": "2つの類似group値を取り違えないことを確認"},
    {"id": "GTC-05", "query": "ガイアステージとガイアナビの違いを教えて", "target_entity": "ガイアステージ,ガイアナビ", "confusable_with": "互いに", "note": "名前が似た2つの異なる概念"},
]


def gt_label_for_phantom() -> str:
    return "NO_FACTS"


def main():
    gt = {
        "phase": "Phase4FZ", "section": "Section5 (structured fact ground truth, frozen before code change)",
        "gt_construction_method": "全てDBの実データ(zone一覧・metric_key一覧の直接照合)のみに基づく客観的判定。モデルの判断は一切使用していない(RULE EVAL-001準拠)。",
        "phantom": [{**q, "gt_label": "NO_FACTS"} for q in PHANTOM_QUERIES],
        "real": [{**q, "gt_label": "EXPECTED_FACTS"} for q in REAL_QUERIES],
        "close_concept": [{**q, "gt_label": "EXPECTED_FACTS_ENTITY_SPECIFIC"} for q in CLOSE_CONCEPT_QUERIES],
    }
    gt["n_phantom"] = len(gt["phantom"])
    gt["n_real"] = len(gt["real"])
    gt["n_close_concept"] = len(gt["close_concept"])
    gt["n_total"] = gt["n_phantom"] + gt["n_real"] + gt["n_close_concept"]

    gt_path = REPORTS_DIR / "phase4fz_gt.json"
    gt_json_str = json.dumps(gt, ensure_ascii=False, indent=2, sort_keys=True)
    gt_path.write_text(gt_json_str, encoding="utf-8")

    gt_hash = hashlib.sha256(gt_json_str.encode("utf-8")).hexdigest()
    hash_path = REPORTS_DIR / "phase4fz_gt_hash.txt"
    hash_path.write_text(f"sha256: {gt_hash}\nfile: phase4fz_gt.json\nn_total: {gt['n_total']}\n", encoding="utf-8")

    print(f"n_phantom={gt['n_phantom']} n_real={gt['n_real']} n_close_concept={gt['n_close_concept']} n_total={gt['n_total']}")
    print(f"gt_hash={gt_hash}")
    print(f"wrote -> {gt_path}, {hash_path}")


if __name__ == "__main__":
    main()
