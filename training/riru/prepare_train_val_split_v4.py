"""Phase 4L-2: v4候補データセット(914件)のtrain/validation分割。

対象: training/riru/processed/riru_lora_v4_candidate.jsonl (読み取り専用、
      Phase 4Kで作成・人間レビュー済み。v2の897件のうちH-0/H-3/H-11のみ修正、
      Phase4K新規17件(Type1 11 + Type2 6)を追加)

分割方針はv1/v2/v3と完全に同一。
  - train : validation = 約90% : 10%
  - 固定seed=42でシャッフル
  - 最初のuserターンをグループ化し、グループ単位でtrain/validに振り分ける
  - overlap (train/val間でグループキーが重複するケース) を自動検証する

このスクリプトは読み取り専用の候補ファイルを読み、v4専用のtrain/valファイルを
新規出力するのみ。v4 candidateファイル自体・v1/v2/v3の各ファイルは一切変更しない。
学習は行わない。
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent
CANDIDATE_PATH = TRAINING_ROOT / "processed" / "riru_lora_v4_candidate.jsonl"
OUT_DIR = TRAINING_ROOT / "processed"
REPORTS_DIR = TRAINING_ROOT / "reports"

SPLIT_SEED = 42
VAL_RATIO = 0.10


def load_candidate(path: Path = CANDIDATE_PATH) -> list[dict]:
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


def phase4k_placement_report(train_records: list[dict], val_records: list[dict]) -> dict:
    def collect(records: list[dict]) -> list[str]:
        out = []
        for r in records:
            meta = r["metadata"]
            if meta.get("source") == "phase4k_generated":
                out.append(f"{meta.get('category_code', '?')}-{meta.get('index', '?')}")
        return sorted(out)

    return {
        "type1_type2_in_train": collect(train_records),
        "type1_type2_in_val": collect(val_records),
    }


def main() -> int:
    records = load_candidate()
    print(f"読み込み件数 (v4候補914件想定): {len(records)}")

    train_records, val_records = split_train_val(records)
    leakage_check = check_no_leakage(train_records, val_records)
    placement = phase4k_placement_report(train_records, val_records)

    train_path = OUT_DIR / "riru_train_v4.jsonl"
    val_path = OUT_DIR / "riru_val_v4.jsonl"

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
        "phase4k_placement": placement,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    (REPORTS_DIR / "train_val_split_report_v4.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"train: {len(train_records)}件 -> {train_path}")
    print(f"val:   {len(val_records)}件 -> {val_path}")
    print(f"leakage check: {leakage_check}")
    print(f"Phase4K new-data placement: {placement}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
