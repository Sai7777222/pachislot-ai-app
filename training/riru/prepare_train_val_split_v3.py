"""Phase 4H-5: v3候補データセットの統合とtrain/validation分割。

v2候補 (riru_lora_v2_candidate.jsonl, 897件 = 823+Phase4F新規74、読み取り専用) に
Phase 4H-3の新規17件 (riru_phase4h_new_candidate.jsonl、本番RAG形式の
「重要情報省略防止」データ) を統合し、v3候補 (914件) を新規作成する。
v2ファイルは一切変更しない。

その後、v1/v2と同一方針で leak-safe な train/validation 分割を行う。
  - train : validation = 約90% : 10%
  - 固定seed=42でシャッフル
  - 最初のuserターンをグループ化し、グループ単位でtrain/validに振り分ける
  - overlap (train/val間でグループキーが重複するケース) を自動検証する

このスクリプトは読み取り専用の候補ファイルを読み、v3専用のファイルを
新規出力するのみ。v1/v2のtrain/val/candidateファイルは一切変更しない。
学習は行わない。
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent
V2_CANDIDATE_PATH = TRAINING_ROOT / "processed" / "riru_lora_v2_candidate.jsonl"
PHASE4H_NEW_PATH = TRAINING_ROOT / "processed" / "riru_phase4h_new_candidate.jsonl"
OUT_DIR = TRAINING_ROOT / "processed"
REPORTS_DIR = TRAINING_ROOT / "reports"

SPLIT_SEED = 42
VAL_RATIO = 0.10


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_key(record: dict) -> str:
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
    v2_records = load_jsonl(V2_CANDIDATE_PATH)
    new_records = load_jsonl(PHASE4H_NEW_PATH)
    print(f"v2候補 (無変更): {len(v2_records)}件")
    print(f"Phase 4H-3新規: {len(new_records)}件")

    combined = v2_records + new_records
    combined_path = OUT_DIR / "riru_lora_v3_candidate.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"v3統合候補データ出力: {combined_path} ({len(combined)}件)")

    train_records, val_records = split_train_val(combined)
    leakage_check = check_no_leakage(train_records, val_records)

    train_path = OUT_DIR / "riru_train_v3.jsonl"
    val_path = OUT_DIR / "riru_val_v3.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "v2_candidate_count_unchanged": len(v2_records),
        "phase4h_new_count": len(new_records),
        "v3_combined_total": len(combined),
        "seed": SPLIT_SEED,
        "val_ratio_target": VAL_RATIO,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "actual_val_ratio": round(len(val_records) / len(combined), 4),
        "leakage_check": leakage_check,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    (REPORTS_DIR / "train_val_split_report_v3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"train: {len(train_records)}件 -> {train_path}")
    print(f"val:   {len(val_records)}件 -> {val_path}")
    print(f"leakage check: {leakage_check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
