"""Phase 4S: complex multi-fact教師poolから ratio_mid / ratio_high candidateを構築する。

- phase4s_source_data.build_pool()でpoolを生成
- 品質検査 (重複・高類似度・placeholder・実在機種名・禁止表現・fact retention自己検証)
- mid = poolの約半数のsubset (既存914件に対し約5%)
- high = pool全体 (既存914件に対し約10%)
- 既存riru_lora_v4_candidate.jsonl(914件)と結合してcandidateを作成
  (既存ファイルは無改変・読み取り専用)
- group-safe train/val split (seed=42、既存v4と同じ方針: 同一user文字列がtrain/valを跨がない)

学習は本ファイルでは行わない。既存train/val/candidateファイルへの書き込みは一切行わない。
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = TRAINING_ROOT / "processed"
REPORTS_DIR = TRAINING_ROOT / "reports"

sys.path.insert(0, str(TRAINING_ROOT))
from phase4s_source_data import build_pool  # noqa: E402

EXISTING_CANDIDATE_PATH = PROCESSED_DIR / "riru_lora_v4_candidate.jsonl"

FORBIDDEN_PHRASES = [
    "覚えておくといい", "ヤメ時", "やめ時", "おすすめ", "べきです", "べきだ",
    "期待値", "勝率", "約", "倍になる", "差分", "設定推測", "推測できる",
]
REAL_MACHINE_HINTS = ["ミリオンゴッド", "モンキーターン", "北斗", "ジャグラー", "バジリスク"]
EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")
PLACEHOLDER_PATTERN = re.compile(r"[〜ー]{2,}")
CHATML_PATTERN = re.compile(r"<\|im_start\|>|<\|im_end\|>")


def char_bigrams(text: str) -> set[str]:
    t = re.sub(r"\s+", "", text)
    return {t[i : i + 2] for i in range(len(t) - 1)} if len(t) >= 2 else {t}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def normalize_fact(text: str) -> str:
    t = text.strip().replace("％", "%").replace(",", "").replace("ゲーム", "G")
    return re.sub(r"\s+", "", t)


def check_retention(item: dict) -> dict:
    answer = item["assistant"]
    answer_norm = normalize_fact(answer)
    missing = []
    for fact in item["relevant_facts"]:
        if fact in answer or normalize_fact(fact) in answer_norm:
            continue
        missing.append(fact)
    leaked = [f for f in item["irrelevant_facts"] if f in answer]
    return {"missing_relevant": missing, "leaked_irrelevant": leaked}


def quality_check(pool: list[dict]) -> dict:
    issues: dict[str, list] = {
        "missing_relevant_facts": [], "leaked_irrelevant_facts": [], "forbidden_phrase": [],
        "real_machine_name": [], "emoji": [], "placeholder": [], "chatml": [],
        "empty_content": [], "exact_duplicate_answer": [], "exact_duplicate_user": [],
    }
    seen_answers: dict[str, str] = {}
    seen_users: dict[str, str] = {}
    for item in pool:
        rid = item["id"]
        ret = check_retention(item)
        if ret["missing_relevant"]:
            issues["missing_relevant_facts"].append(
                {"id": rid, "missing": ret["missing_relevant"]}
            )
        if ret["leaked_irrelevant"]:
            issues["leaked_irrelevant_facts"].append(
                {"id": rid, "leaked": ret["leaked_irrelevant"]}
            )
        for phrase in FORBIDDEN_PHRASES:
            if phrase in item["assistant"] and phrase != "約":
                issues["forbidden_phrase"].append({"id": rid, "phrase": phrase})
        if "約" in item["assistant"] and re.search(r"約\d", item["assistant"]):
            issues["forbidden_phrase"].append({"id": rid, "phrase": "約+数値(派生計算疑い)"})
        for name in REAL_MACHINE_HINTS:
            if name in item["user"] or name in item["assistant"]:
                issues["real_machine_name"].append({"id": rid, "name": name})
        if EMOJI_PATTERN.search(item["assistant"]):
            issues["emoji"].append(rid)
        if PLACEHOLDER_PATTERN.search(item["assistant"]):
            issues["placeholder"].append(rid)
        if CHATML_PATTERN.search(item["assistant"]) or CHATML_PATTERN.search(item["user"]):
            issues["chatml"].append(rid)
        if not item["user"].strip() or not item["assistant"].strip():
            issues["empty_content"].append(rid)
        if item["assistant"] in seen_answers:
            issues["exact_duplicate_answer"].append(
                {"id": rid, "dup_of": seen_answers[item["assistant"]]}
            )
        else:
            seen_answers[item["assistant"]] = rid
        if item["user"] in seen_users:
            issues["exact_duplicate_user"].append({"id": rid, "dup_of": seen_users[item["user"]]})
        else:
            seen_users[item["user"]] = rid

    # similarity (answer text, char-bigram jaccard, new-vs-new)
    high_sim_pairs = []
    for i in range(len(pool)):
        bi = char_bigrams(pool[i]["assistant"])
        for j in range(i + 1, len(pool)):
            sim = jaccard(bi, char_bigrams(pool[j]["assistant"]))
            if sim >= 0.85:
                high_sim_pairs.append(
                    {"a": pool[i]["id"], "b": pool[j]["id"], "similarity": round(sim, 3)}
                )

    n_total_issues = sum(
        len(v) for k, v in issues.items()
    )
    return {
        "n_pool": len(pool),
        "issues": issues,
        "n_total_issue_entries": n_total_issues,
        "high_similarity_pairs_ge_0.85": high_sim_pairs,
        "n_pairs_ge_0.9": sum(1 for p in high_sim_pairs if p["similarity"] >= 0.9),
        "n_pairs_ge_0.85_lt_0.9": sum(1 for p in high_sim_pairs if 0.85 <= p["similarity"] < 0.9),
    }


def check_against_existing(pool: list[dict], existing_answers: list[str]) -> dict:
    """new vs existing similarity (念のためexisting897/914との高類似も確認)。"""
    existing_bigrams = [char_bigrams(a) for a in existing_answers]
    high_sim = []
    for item in pool:
        bi = char_bigrams(item["assistant"])
        for eb, ea in zip(existing_bigrams, existing_answers, strict=True):
            sim = jaccard(bi, eb)
            if sim >= 0.85:
                high_sim.append(
                    {"new_id": item["id"], "existing_answer": ea, "similarity": round(sim, 3)}
                )
    return {"new_vs_existing_high_similarity": high_sim, "n_pairs": len(high_sim)}


def to_riru_record(item: dict, source_tag: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]},
        ],
        "metadata": {
            "source": source_tag,
            "category": item["category"],
            "category_code": item["category_code"],
            "index": item["id"],
        },
    }


def load_existing() -> list[dict]:
    records = []
    with open(EXISTING_CANDIDATE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_key(rec: dict) -> str:
    """既存v4分割と同じ方針: 最初のuser発話をgroup keyとする。"""
    return rec["messages"][0]["content"]


def group_safe_split(
    records: list[dict], val_ratio: float, seed: int = 42
) -> tuple[list[dict], list[dict]]:
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


def drop_high_similarity(pool: list[dict], threshold: float = 0.85) -> tuple[list[dict], list[str]]:
    """new-vs-new類似度がthreshold以上のペアについて、片方(後方=b)を除外する。
    複数ペアに絡む場合も一度のパスで安全側(除外多め)に倒す。
    """
    dropped: set[str] = set()
    ids = [x["id"] for x in pool]
    bigrams = {x["id"]: char_bigrams(x["assistant"]) for x in pool}
    for i in range(len(pool)):
        if ids[i] in dropped:
            continue
        for j in range(i + 1, len(pool)):
            if ids[j] in dropped:
                continue
            if jaccard(bigrams[ids[i]], bigrams[ids[j]]) >= threshold:
                dropped.add(ids[j])
    kept = [x for x in pool if x["id"] not in dropped]
    return kept, sorted(dropped)


def main() -> int:
    raw_pool = build_pool(n_per_pattern=16, seed=42)
    pool, dropped_ids = drop_high_similarity(raw_pool, threshold=0.85)
    print(f"raw_pool={len(raw_pool)} dropped_for_similarity={len(dropped_ids)} kept={len(pool)}")
    qc = quality_check(pool)
    qc["dropped_for_similarity_ids"] = dropped_ids
    qc["raw_pool_size"] = len(raw_pool)

    existing_records = load_existing()
    existing_answers = [
        r["messages"][-1]["content"]
        for r in existing_records
        if r["messages"][-1]["role"] == "assistant"
    ]
    ext_qc = check_against_existing(pool, existing_answers)

    (REPORTS_DIR / "phase4s_dataset_quality.json").write_text(
        json.dumps({"pool_quality": qc, "vs_existing": ext_qc}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    critical_ok = (
        len(qc["issues"]["missing_relevant_facts"]) == 0
        and len(qc["issues"]["leaked_irrelevant_facts"]) == 0
        and len(qc["issues"]["real_machine_name"]) == 0
        and len(qc["issues"]["exact_duplicate_answer"]) == 0
        and len(qc["issues"]["exact_duplicate_user"]) == 0
        and qc["n_pairs_ge_0.9"] == 0
    )
    print(f"critical_ok: {critical_ok}")
    issue_counts = {k: len(v) if isinstance(v, list) else v for k, v in qc["issues"].items()}
    print(json.dumps(issue_counts, ensure_ascii=False))
    n_sim85 = len(qc["high_similarity_pairs_ge_0.85"])
    print(f"high_sim(>=0.85) pairs: {n_sim85}, >=0.9: {qc['n_pairs_ge_0.9']}")
    print(f"vs_existing high similarity pairs: {ext_qc['n_pairs']}")

    if not critical_ok:
        print("CRITICAL QUALITY ISSUE FOUND -- not proceeding to build candidates.")
        return 1

    # mid = first half of pool (7 per pattern = 56), high = full pool (112)
    mid_pool = []
    by_pattern: dict[str, list[dict]] = {}
    for item in pool:
        by_pattern.setdefault(item["category_code"], []).append(item)
    for _code, items in by_pattern.items():
        mid_pool.extend(items[: len(items) // 2])

    mid_new_records = [to_riru_record(x, "phase4s_ratio_mid") for x in mid_pool]
    high_new_records = [to_riru_record(x, "phase4s_ratio_high") for x in pool]

    mid_candidate = existing_records + mid_new_records
    high_candidate = existing_records + high_new_records

    mid_ratio = len(mid_new_records) / len(mid_candidate)
    high_ratio = len(high_new_records) / len(high_candidate)

    write_jsonl(PROCESSED_DIR / "riru_ratio_mid_candidate.jsonl", mid_candidate)
    write_jsonl(PROCESSED_DIR / "riru_ratio_high_candidate.jsonl", high_candidate)

    val_ratio = 91 / 914  # 既存v4と同じ比率(約9.96%)を踏襲
    mid_train, mid_val = group_safe_split(mid_candidate, val_ratio, seed=42)
    high_train, high_val = group_safe_split(high_candidate, val_ratio, seed=42)

    # overlap確認
    mid_train_keys = {group_key(r) for r in mid_train}
    mid_val_keys = {group_key(r) for r in mid_val}
    high_train_keys = {group_key(r) for r in high_train}
    high_val_keys = {group_key(r) for r in high_val}
    mid_overlap = len(mid_train_keys & mid_val_keys)
    high_overlap = len(high_train_keys & high_val_keys)

    write_jsonl(PROCESSED_DIR / "riru_ratio_mid_train.jsonl", mid_train)
    write_jsonl(PROCESSED_DIR / "riru_ratio_mid_val.jsonl", mid_val)
    write_jsonl(PROCESSED_DIR / "riru_ratio_high_train.jsonl", high_train)
    write_jsonl(PROCESSED_DIR / "riru_ratio_high_val.jsonl", high_val)

    review_samples = {
        "mid_new_records_full": mid_pool,
        "high_new_records_full": pool,
    }
    (REPORTS_DIR / "phase4s_review_samples.json").write_text(
        json.dumps(review_samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "pool_size": len(pool),
        "mid_new_count": len(mid_new_records),
        "high_new_count": len(high_new_records),
        "mid_total_count": len(mid_candidate),
        "high_total_count": len(high_candidate),
        "mid_complex_ratio_pct": round(mid_ratio * 100, 2),
        "high_complex_ratio_pct": round(high_ratio * 100, 2),
        "mid_train_count": len(mid_train),
        "mid_val_count": len(mid_val),
        "high_train_count": len(high_train),
        "high_val_count": len(high_val),
        "mid_train_val_overlap": mid_overlap,
        "high_train_val_overlap": high_overlap,
        "pattern_counts_in_pool": dict(Counter(x["category_code"] for x in pool)),
        "pattern_counts_in_mid": dict(Counter(x["category_code"] for x in mid_pool)),
    }
    (REPORTS_DIR / "phase4s_ratio_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
