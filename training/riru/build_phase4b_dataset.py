"""Phase 4B: 新規リル人格データの組み立て・検証・レポート生成。

`phase4b_source_data.py` に定義された人手作成データ(目標300件)を
messages形式へ組み立て、品質検査・重複検査・事実情報チェックを行う。

このスクリプトは:
  - QLoRA/LoRA学習を一切行わない
  - Qwenモデル・adapter・DB/RAG/Vector DBに一切触れない
  - 既存523件・旧archive・旧原本を一切変更しない
  - 生成した候補データセットは "candidate" として保存するのみで、
    本番学習に使う設定ファイル等は更新しない
"""

from __future__ import annotations

import difflib
import json
import sys
from collections import Counter
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = TRAINING_ROOT / "processed"
REPORTS_DIR = TRAINING_ROOT / "reports"

sys.path.insert(0, str(TRAINING_ROOT))
import convert_dataset as cd  # noqa: E402  (既存523件処理済みのクリーニング/検証ロジックを再利用)
import phase4b_source_data as src  # noqa: E402

# 実在機種名など、パチスロ事実情報として厳密に混入してはいけない固有名詞
# (convert_dataset.MACHINE_SPECIFIC_KEYWORDS は既存プロジェクトの対象機種名を含むため流用)
FORBIDDEN_REAL_MACHINE_TERMS = [
    "ミリオンゴッド", "GOD揃い", "ガイアベル", "ガイアステージ", "Z-ZONE",
    "ゼウスモード", "PGG", "神々の軌跡",
]

CATEGORY_META = {
    "A": ("kimi_usage", src.CATEGORY_A_KIMI, 60),
    "B": ("no_info_available", src.CATEGORY_B_NO_INFO, 70),
    "C": ("faithful_to_given_info", src.CATEGORY_C_FAITHFUL, 50),
    "D": ("correction_handling", src.CATEGORY_D_CORRECTION, 30),
    "F": ("emotion_reaction", src.CATEGORY_F_EMOTION, 20),
    "G": ("length_variation", src.CATEGORY_G_LENGTH, 30),
}


def build_single_turn_record(item: dict, category_code: str, category_name: str, idx: int) -> dict:
    user_content = item["user"].strip()
    assistant_content = item["assistant"].strip()
    metadata = {
        "source": "phase4b_generated",
        "category": category_name,
        "category_code": category_code,
        "index": idx,
    }
    if "length" in item:
        metadata["length_bucket"] = item["length"]
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": metadata,
    }


def build_multiturn_record(item: dict, idx: int) -> dict:
    turns = item["turns"]
    if len(turns) % 2 != 0:
        raise ValueError(f"multiturn record #{idx} has odd number of turns: {len(turns)}")
    messages = []
    for i, text in enumerate(turns):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": text.strip()})
    return {
        "messages": messages,
        "metadata": {
            "source": "phase4b_generated",
            "category": "multiturn",
            "category_code": "E",
            "index": idx,
            "turn_pairs": len(turns) // 2,
        },
    }


def build_all() -> list[dict]:
    records: list[dict] = []
    for code, (name, items, _target) in CATEGORY_META.items():
        for i, item in enumerate(items):
            records.append(build_single_turn_record(item, code, name, i))
    for i, item in enumerate(src.CATEGORY_E_MULTITURN):
        records.append(build_multiturn_record(item, i))
    return records


# ---------------------------------------------------------------------------
# 品質検査
# ---------------------------------------------------------------------------


def all_assistant_texts(record: dict) -> list[str]:
    return [m["content"] for m in record["messages"] if m["role"] == "assistant"]


def all_texts(record: dict) -> list[str]:
    return [m["content"] for m in record["messages"]]


def validate_records(records: list[dict]) -> dict:
    errors = []
    empty_content = 0
    emoji_hits = 0
    kaomoji_hits = 0
    double_excl_hits = 0
    chatml_hits = 0
    real_machine_fact_hits = []

    for i, r in enumerate(records):
        for m in r["messages"]:
            if not m["content"].strip():
                empty_content += 1
                errors.append(f"record {i}: empty content in role={m['role']}")
            if cd.DECORATIVE_SYMBOLS_PATTERN.search(m["content"]):
                emoji_hits += 1
            if "♪" in m["content"]:
                kaomoji_hits += 1
            if cd.REPEATED_EXCLAMATION_PATTERN.search(m["content"]):
                double_excl_hits += 1
            if cd.CHATML_TOKEN_PATTERN.search(m["content"]):
                chatml_hits += 1
        # 実在機種の固有名詞チェック (assistant側のみでなくuser側=参照情報にも注意)
        text_all = " ".join(all_texts(r))
        for term in FORBIDDEN_REAL_MACHINE_TERMS:
            if term in text_all:
                real_machine_fact_hits.append({"record_index": i, "term": term, "text": text_all})

    if len(records) != 300:
        errors.append(f"total_count_mismatch: expected=300 actual={len(records)}")

    return {
        "errors": errors,
        "total": len(records),
        "empty_content": empty_content,
        "decorative_symbol_hits": emoji_hits,
        "kaomoji_note_hits": kaomoji_hits,
        "repeated_exclamation_hits": double_excl_hits,
        "chatml_token_hits": chatml_hits,
        "real_machine_fact_hits": real_machine_fact_hits,
    }


def detect_generic_fact_patterns(records: list[dict]) -> list[dict]:
    """カテゴリC以外で、パチスロ数値らしきパターン(%・1/xxx・xxxG・xxx枚)が
    紛れ込んでいないかを検出する(Cは架空データを意図的に使うため除外)。
    """
    hits = []
    for i, r in enumerate(records):
        if r["metadata"].get("category_code") == "C":
            continue
        text_all = " ".join(all_texts(r))
        found = [pat.pattern for pat in cd.FACT_NUMERIC_PATTERNS if pat.search(text_all)]
        if found:
            hits.append(
                {
                    "record_index": i,
                    "category": r["metadata"].get("category"),
                    "patterns": found,
                    "text": text_all,
                }
            )
    return hits


def find_exact_duplicates(records: list[dict]) -> dict:
    user_first_map: dict[str, list[int]] = {}
    assistant_first_map: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        u = r["messages"][0]["content"]
        a = r["messages"][1]["content"] if len(r["messages"]) > 1 else ""
        user_first_map.setdefault(u, []).append(i)
        assistant_first_map.setdefault(a, []).append(i)
    return {
        "user_exact_dupes": {k: v for k, v in user_first_map.items() if len(v) > 1},
        "assistant_exact_dupes": {k: v for k, v in assistant_first_map.items() if len(v) > 1},
    }


def find_high_similarity(
    records: list[dict], threshold: float = 0.9
) -> list[tuple[int, int, float]]:
    texts = [r["messages"][1]["content"] if len(r["messages"]) > 1 else "" for r in records]
    hits = []
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            if not texts[i] or not texts[j] or texts[i] == texts[j]:
                continue
            ratio = difflib.SequenceMatcher(None, texts[i], texts[j]).ratio()
            if ratio >= threshold:
                hits.append((i, j, round(ratio, 3)))
    return hits


def word_stats(records: list[dict]) -> dict:
    pronoun_counter: Counter[str] = Counter()
    tail_counter: Counter[str] = Counter()
    lengths = []
    TAIL_WORDS = ["だよ", "なんだ", "だね", "だぞ"]
    PRONOUNS = ["私", "リル", "キミ"]
    for r in records:
        for text in all_assistant_texts(r):
            for w in PRONOUNS:
                pronoun_counter[w] += text.count(w)
            for w in TAIL_WORDS:
                tail_counter[w] += text.count(w)
            lengths.append(len(text))
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    length_stats = {}
    if n:
        length_stats = {
            "min": lengths_sorted[0],
            "max": lengths_sorted[-1],
            "avg": round(sum(lengths_sorted) / n, 1),
            "median": lengths_sorted[n // 2],
        }
    return {
        "pronoun_counts": dict(pronoun_counter),
        "tail_word_counts": dict(tail_counter),
        "assistant_length_stats": length_stats,
    }


def same_tail_repetition_within_response(records: list[dict]) -> list[dict]:
    TAIL_PATTERNS = [
        "だよ！", "だよ", "だよ〜", "だよね！", "だよね",
        "なんだ！", "なんだ", "なんだ〜", "なんだよ",
        "だね！", "だね", "だねっ", "だぞ！", "だぞ",
        "かな", "かも", "よ！", "よ〜", "ね！", "ねっ", "っ！",
    ]
    hits = []
    for i, r in enumerate(records):
        for text in all_assistant_texts(r):
            local = {}
            for pat in TAIL_PATTERNS:
                c = text.count(pat)
                if c >= 2:
                    local[pat] = c
            if local:
                hits.append({"record_index": i, "text": text, "tail_counts": local})
    return hits


def category_counts(records: list[dict]) -> dict:
    c: Counter[str] = Counter()
    for r in records:
        c[r["metadata"]["category"]] += 1
    return dict(c)


# ---------------------------------------------------------------------------
# 既存523件との統合統計 (ファイルには保存するが、学習用の最終ファイルとしては使わない)
# ---------------------------------------------------------------------------


def load_existing_523() -> list[dict]:
    path = PROCESSED_DIR / "riru_qwen_messages_v1.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def combined_stats(new_records: list[dict], existing_records: list[dict]) -> dict:
    combined = existing_records + new_records

    def get_assistant_texts(rec: dict) -> list[str]:
        return [m["content"] for m in rec["messages"] if m["role"] == "assistant"]

    pronoun_counter: Counter[str] = Counter()
    tail_counter: Counter[str] = Counter()
    lengths = []
    for r in combined:
        for text in get_assistant_texts(r):
            for w in ["私", "リル", "キミ"]:
                pronoun_counter[w] += text.count(w)
            for w in ["だよ", "なんだ", "だね", "だぞ"]:
                tail_counter[w] += text.count(w)
            lengths.append(len(text))
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    # 重複 (user文面の完全一致) を統合データ全体で確認
    user_map: dict[str, list[int]] = {}
    for i, r in enumerate(combined):
        u = r["messages"][0]["content"]
        user_map.setdefault(u, []).append(i)
    dupe_groups = {k: v for k, v in user_map.items() if len(v) > 1}

    return {
        "existing_count": len(existing_records),
        "new_count": len(new_records),
        "combined_count": len(combined),
        "pronoun_counts": dict(pronoun_counter),
        "pronoun_rate_kimi_pct": round(
            pronoun_counter["キミ"] / max(sum(pronoun_counter.values()), 1) * 100, 2
        ),
        "tail_word_counts": dict(tail_counter),
        "assistant_length_stats": {
            "min": lengths_sorted[0],
            "max": lengths_sorted[-1],
            "avg": round(sum(lengths_sorted) / n, 1),
            "median": lengths_sorted[n // 2],
        }
        if n
        else {},
        "user_text_exact_dupe_groups": len(dupe_groups),
    }


# ---------------------------------------------------------------------------
# 人間確認用サンプル抽出
# ---------------------------------------------------------------------------


def extract_review_samples(records: list[dict], per_category: int = 10) -> dict:
    by_category: dict[str, list[dict]] = {}
    for r in records:
        by_category.setdefault(r["metadata"]["category"], []).append(r)
    samples = {}
    for cat, items in by_category.items():
        step = max(1, len(items) // per_category)
        picked = items[::step][:per_category]
        samples[cat] = picked
    return samples


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    records = build_all()
    print(f"新規レコード数: {len(records)}")

    cat_counts = category_counts(records)
    print("カテゴリ別件数:", cat_counts)

    validation = validate_records(records)
    generic_fact_hits = detect_generic_fact_patterns(records)
    dupes = find_exact_duplicates(records)
    high_sim = find_high_similarity(records)
    stats = word_stats(records)
    tail_repeat_hits = same_tail_repetition_within_response(records)

    # 候補データセットとして保存 (学習には使わない、あくまでcandidate)
    candidate_path = PROCESSED_DIR / "riru_phase4b_new_candidate.jsonl"
    with open(candidate_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"候補データ出力: {candidate_path}")

    # 既存523件とのマージ版 (candidateとしてのみ生成、学習設定は更新しない)
    existing = load_existing_523()
    merged_preview_path = PROCESSED_DIR / "riru_qwen_messages_v2_candidate.jsonl"
    with open(merged_preview_path, "w", encoding="utf-8") as f:
        for r in existing + records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"統合候補データ出力 (学習には未使用): {merged_preview_path}")

    combined = combined_stats(records, existing)

    review_samples = extract_review_samples(records, per_category=10)

    report = {
        "category_target": {name: target for _, (name, _, target) in CATEGORY_META.items()},
        "category_actual_counts": cat_counts,
        "multiturn_count": len(src.CATEGORY_E_MULTITURN),
        "total_new_records": len(records),
        "validation": validation,
        "generic_fact_pattern_hits_outside_category_c": generic_fact_hits,
        "duplicates": {
            "user_exact_dupe_groups": len(dupes["user_exact_dupes"]),
            "assistant_exact_dupe_groups": len(dupes["assistant_exact_dupes"]),
            "high_similarity_pairs_ge_0.9": len(high_sim),
        },
        "word_stats": stats,
        "same_tail_repetition_within_response": {
            "count": len(tail_repeat_hits),
        },
        "combined_with_existing_523": combined,
    }

    (REPORTS_DIR / "phase4b_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "phase4b_duplicates.json").write_text(
        json.dumps(
            {
                "user_exact_dupes": dupes["user_exact_dupes"],
                "assistant_exact_dupes": dupes["assistant_exact_dupes"],
                "high_similarity_pairs": high_sim,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (REPORTS_DIR / "phase4b_tail_repetition.json").write_text(
        json.dumps(tail_repeat_hits, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "phase4b_review_samples.json").write_text(
        json.dumps(review_samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
