"""Phase 4U: ratio-high candidateにpersona identity教師を追加した
ratio_high_identity candidateを構築する。

既存 riru_ratio_high_candidate.jsonl は無改変・読み取り専用で読み込み、
新規identity教師(43件)を追加した別ファイルとして新candidateを作成する。
complex multi-fact教師113件は一切削除しない。

品質検査: placeholder/ChatML/重複/高類似度/実在機種名/Phase4T probe文面との
contamination(naming probe・P04 probe・実Q3・structured17・character39)を確認する。

学習は本ファイルでは行わない。既存ratio-high関連ファイルへの書き込みは一切ない。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = TRAINING_ROOT / "processed"
REPORTS_DIR = TRAINING_ROOT / "reports"
EVAL_DIR = TRAINING_ROOT / "eval"

sys.path.insert(0, str(TRAINING_ROOT))
from phase4u_identity_source_data import IDENTITY_RECORDS, NO_INTRUSION_RECORDS  # noqa: E402

EXISTING_HIGH_CANDIDATE = PROCESSED_DIR / "riru_ratio_high_candidate.jsonl"

PLACEHOLDER_PATTERNS = ["〜〜", "○○", "XXX", "[名前]", "<name>", "placeholder"]
CHATML_PATTERN = re.compile(r"<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]")
REAL_MACHINE_HINTS = ["ミリオンゴッド", "モンキーターン", "北斗", "ジャグラー", "バジリスク"]
WRONG_NAME_ENUMERATION = ["リコ", "リサ", "アリス", "パチ子", "キリコ", "リリ", "リナ"]


def char_bigrams(text: str) -> set[str]:
    t = re.sub(r"\s+", "", text)
    return {t[i : i + 2] for i in range(len(t) - 1)} if len(t) >= 2 else {t}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_existing() -> list[dict]:
    records = []
    with open(EXISTING_HIGH_CANDIDATE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_riru_record(item: dict, source_tag: str, category: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]},
        ],
        "metadata": {"source": source_tag, "category": category, "index": item["id"]},
    }


def load_contamination_sources() -> dict[str, list[str]]:
    """Phase4Tのnaming/P04 probe文面、実Q3、structured17、character39のテキストを
    読み取り専用で集め、新規identity教師との重複コピーがないかを確認する。"""
    sources: dict[str, list[str]] = {}

    sys.path.insert(0, str(EVAL_DIR))
    import phase4t_probes as pt  # noqa: PLC0415

    sources["naming_probes"] = [p["prompt"] for p in pt.NAMING_PROBES]
    sources["p04_probes"] = [p["question"] for p in pt.P04_PROBES]

    rag17q = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    sources["structured_17q"] = [r["question"] for r in rag17q] + [
        r["rag_context_text"] for r in rag17q
    ]

    eval39 = [
        json.loads(line)
        for line in (EVAL_DIR / "riru_eval_set_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    c39_texts = []
    for item in eval39:
        if item["type"] == "single":
            c39_texts.append(item["prompt"])
        else:
            c39_texts.extend(item["turns"])
    sources["character_39"] = c39_texts

    holdout = json.loads(
        (EVAL_DIR / "phase4i_holdout_omission_v2.json").read_text(encoding="utf-8")
    )
    sources["holdout_p01_p10"] = [r["question"] for r in holdout] + [
        r["rag_context_text"] for r in holdout
    ]

    return sources


def quality_check(new_records: list[dict]) -> dict:
    issues: dict[str, list] = {
        "placeholder": [], "chatml": [], "real_machine_name": [],
        "wrong_name_enumeration": [], "exact_duplicate": [], "empty": [],
    }
    seen = {}
    for rec in new_records:
        rid = rec["metadata"]["index"]
        user_text = rec["messages"][0]["content"]
        assistant_text = rec["messages"][1]["content"]
        for p in PLACEHOLDER_PATTERNS:
            if p in assistant_text:
                issues["placeholder"].append({"id": rid, "pattern": p})
        if CHATML_PATTERN.search(assistant_text) or CHATML_PATTERN.search(user_text):
            issues["chatml"].append(rid)
        for name in REAL_MACHINE_HINTS:
            if name in user_text or name in assistant_text:
                issues["real_machine_name"].append({"id": rid, "name": name})
        for wn in WRONG_NAME_ENUMERATION:
            if wn in assistant_text:
                issues["wrong_name_enumeration"].append({"id": rid, "name": wn})
        if not user_text.strip() or not assistant_text.strip():
            issues["empty"].append(rid)
        key = (user_text, assistant_text)
        if key in seen:
            issues["exact_duplicate"].append({"id": rid, "dup_of": seen[key]})
        else:
            seen[key] = rid

    high_sim_pairs = []
    for i in range(len(new_records)):
        bi = char_bigrams(new_records[i]["messages"][1]["content"])
        for j in range(i + 1, len(new_records)):
            sim = jaccard(bi, char_bigrams(new_records[j]["messages"][1]["content"]))
            if sim >= 0.85:
                high_sim_pairs.append(
                    {
                        "a": new_records[i]["metadata"]["index"],
                        "b": new_records[j]["metadata"]["index"],
                        "similarity": round(sim, 3),
                    }
                )

    contamination_sources = load_contamination_sources()
    contamination_hits = []
    for rec in new_records:
        user_text = rec["messages"][0]["content"]
        for src_name, texts in contamination_sources.items():
            for t in texts:
                if t and (t in user_text or user_text in t):
                    contamination_hits.append(
                        {"id": rec["metadata"]["index"], "source": src_name, "matched_text": t[:60]}
                    )

    return {
        "issues": issues,
        "high_similarity_pairs": high_sim_pairs,
        "contamination_hits": contamination_hits,
    }


def group_key(rec: dict) -> str:
    return rec["messages"][0]["content"]


def group_safe_split(records: list[dict], val_ratio: float, seed: int = 42):
    import random as _random

    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(group_key(r), []).append(r)
    keys = list(groups.keys())
    rng = _random.Random(seed)
    rng.shuffle(keys)
    n_val_target = max(1, round(len(records) * val_ratio))
    val_records: list[dict] = []
    train_records: list[dict] = []
    for k in keys:
        if len(val_records) < n_val_target:
            val_records.extend(groups[k])
        else:
            train_records.extend(groups[k])
    return train_records, val_records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    existing = load_existing()
    n_complex_existing = sum(
        1 for r in existing if r["metadata"].get("source") == "phase4s_ratio_high"
    )

    identity_new = [
        to_riru_record(x, "phase4u_identity", "identity_naming") for x in IDENTITY_RECORDS
    ]
    no_intrusion_new = [
        to_riru_record(x, "phase4u_identity", "identity_no_intrusion") for x in NO_INTRUSION_RECORDS
    ]
    new_records = identity_new + no_intrusion_new

    qc = quality_check(new_records)
    n_issues = sum(len(v) if isinstance(v, list) else 0 for v in qc["issues"].values())
    n_issues += len(qc["high_similarity_pairs"]) + len(qc["contamination_hits"])

    (REPORTS_DIR / "phase4u_dataset_quality.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"quality issues total: {n_issues}")
    print(json.dumps({k: len(v) for k, v in qc["issues"].items()}, ensure_ascii=False))
    print(f"high_similarity_pairs: {len(qc['high_similarity_pairs'])}")
    print(f"contamination_hits: {len(qc['contamination_hits'])}")

    if n_issues > 0:
        print("QUALITY ISSUES FOUND -- not proceeding to build candidate.")
        return 1

    combined = existing + new_records
    n_complex_total = n_complex_existing  # 既存complex教師件数は不変(削除していない)
    complex_ratio = round(100 * n_complex_total / len(combined), 2)

    print(f"existing={len(existing)} +identity={len(new_records)} = total={len(combined)}")
    print(f"complex teachers (unchanged): {n_complex_total}, new complex ratio: {complex_ratio}%")

    if complex_ratio < 10.0:
        print(f"STOP: complex ratio {complex_ratio}% < 10% target -- reporting and halting.")
        return 1

    write_jsonl(PROCESSED_DIR / "riru_ratio_high_identity_candidate.jsonl", combined)

    val_ratio = 102 / 1027  # ratio-highのval比率(約9.93%)を踏襲
    train, val = group_safe_split(combined, val_ratio, seed=42)
    train_keys = {group_key(r) for r in train}
    val_keys = {group_key(r) for r in val}
    overlap = len(train_keys & val_keys)

    write_jsonl(PROCESSED_DIR / "riru_ratio_high_identity_train.jsonl", train)
    write_jsonl(PROCESSED_DIR / "riru_ratio_high_identity_val.jsonl", val)

    summary = {
        "existing_count": len(existing),
        "identity_new_count": len(identity_new),
        "no_intrusion_new_count": len(no_intrusion_new),
        "new_total": len(new_records),
        "combined_total": len(combined),
        "complex_teacher_count_unchanged": n_complex_total,
        "complex_ratio_pct": complex_ratio,
        "train_count": len(train),
        "val_count": len(val),
        "train_val_overlap": overlap,
    }
    (REPORTS_DIR / "phase4u_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
