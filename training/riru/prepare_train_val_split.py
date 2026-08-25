"""Phase 4C: 823件の候補データセットを train/validation に分割する。

対象: training/riru/processed/riru_qwen_messages_v2_candidate.jsonl (読み取り専用)

【分割方針】
- train : validation = 約90% : 10%
- 固定seedでシャッフル (再現性確保)
- 単純にレコード単位でシャッフルすると、同一user文の
  バリエーション違い (variation_id違い) がtrain/validに分散してしまい、
  「ほぼ同じ入力なのにtrainとvalの両方にある」というデータリークになる。
  これを避けるため、同一user文 (マルチターンの場合は最初のuserターン) を
  グループ化し、グループ単位でtrain/validに振り分ける。

このスクリプトは読み取り専用の候補ファイルを読み、
train/valの2ファイルを新規出力するのみ。学習は行わない。
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent
CANDIDATE_PATH = TRAINING_ROOT / "processed" / "riru_qwen_messages_v2_candidate.jsonl"
OUT_DIR = TRAINING_ROOT / "processed"
REPORTS_DIR = TRAINING_ROOT / "reports"

SPLIT_SEED = 42
VAL_RATIO = 0.10


def load_candidate(path: Path = CANDIDATE_PATH) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def group_key(record: dict) -> str:
    """最初のuserターンの文字列をグループキーとする。

    同じuser文 (または同じ最初のuser発話) を持つレコードは
    train/validどちらか一方にまとめて割り当てることで、
    ほぼ同一の入力がtrainとvalの両方に混入するデータリークを避ける。
    """
    first_user = next((m["content"] for m in record["messages"] if m["role"] == "user"), "")
    return first_user


def split_train_val(
    records: list[dict], val_ratio: float = VAL_RATIO, seed: int = SPLIT_SEED
) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[group_key(r)].append(r)

    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    total = len(records)
    target_val_count = round(total * val_ratio)

    val_records: list[dict] = []
    train_records: list[dict] = []
    val_count = 0
    for gk in group_keys:
        group_records = groups[gk]
        # まだvalidationの目標件数に達していなければ、このグループをvalへ
        if val_count < target_val_count:
            val_records.extend(group_records)
            val_count += len(group_records)
        else:
            train_records.extend(group_records)

    return train_records, val_records


def check_no_leakage(train_records: list[dict], val_records: list[dict]) -> dict:
    train_keys = {group_key(r) for r in train_records}
    val_keys = {group_key(r) for r in val_records}
    overlap = train_keys & val_keys
    return {
        "train_unique_user_keys": len(train_keys),
        "val_unique_user_keys": len(val_keys),
        "overlapping_user_keys": len(overlap),
        "overlap_examples": list(overlap)[:10],
    }


def main() -> int:
    records = load_candidate()
    print(f"読み込み件数 (候補823件想定): {len(records)}")

    train_records, val_records = split_train_val(records)
    leakage_check = check_no_leakage(train_records, val_records)

    train_path = OUT_DIR / "riru_train_v1.jsonl"
    val_path = OUT_DIR / "riru_val_v1.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "source": str(CANDIDATE_PATH),
        "seed": SPLIT_SEED,
        "val_ratio_target": VAL_RATIO,
        "total_records": len(records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "actual_val_ratio": round(len(val_records) / len(records), 4),
        "leakage_check": leakage_check,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    (REPORTS_DIR / "train_val_split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"train: {len(train_records)}件 -> {train_path}")
    print(f"val:   {len(val_records)}件 -> {val_path}")
    print(f"leakage check: {leakage_check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
