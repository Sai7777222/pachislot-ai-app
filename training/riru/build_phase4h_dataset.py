# ruff: noqa: E501
"""Phase 4H-3: 本番RAG形式追加データの組み立て・検証・レポート生成。

`phase4h_source_data.py` の CATEGORY_T (本番RAGコンテキスト形式を模した
「重要情報省略防止」データ) を messages形式へ組み立て、品質検査を行う。

このスクリプトは:
  - QLoRA/LoRA学習を一切行わない
  - Qwenモデル・v1/v2 adapter・v1/v2 checkpoint・DB/RAG/Vector DBに一切触れない
  - v1(823件)・v2(897件)の既存candidateファイルを一切変更しない
  - 生成した候補データセットは新規ファイルとしてのみ保存する
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = TRAINING_ROOT / "processed"
REPORTS_DIR = TRAINING_ROOT / "reports"

sys.path.insert(0, str(TRAINING_ROOT))
import convert_dataset as cd  # noqa: E402
import phase4h_source_data as src  # noqa: E402

FORBIDDEN_REAL_MACHINE_TERMS = [
    "ミリオンゴッド", "GOD揃い", "ガイアベル", "ガイアステージ", "Z-ZONE",
    "ゼウスモード", "PGG", "神々の軌跡",
]

PLACEHOLDER_PATTERN = re.compile(
    r"(〜{3,}|ー{4,}|X{2,}|x{2,}|○{2,}|●{2,}|TBD|TODO|\.{4,}|…{2,}|"
    r"[■□▲△▼▽◆◇]{2,}|�)"
)

# 「不自然な長文化」チェック用の目安上限 (文字数)。これを超える場合は
# 「全部答える」方向へ振れていないか目視確認の対象としてフラグする。
UNNATURAL_LENGTH_THRESHOLD = 140

RAG_HEADER_TEMPLATE = """【対象機種】
{machine_line}
このセクション以下の情報はすべて上記の機種に関するものです。他の機種の名称を補完したり、機種名を推測したりしないでください。

【構造化データ（数値・確率・設定差・天井・示唆など）】
このセクションの数値は必ず原文表記のまま回答に使ってください。計算し直したり丸めたりしないでください。

{structured_lines}

【関連する解説文章】

{prose_sections}"""


def build_rag_context(item: dict) -> str:
    structured_lines = "\n".join(
        f"- [{row['label']}] {row['item']}: {row['value']}" for row in item["structured_rows"]
    )
    prose_blocks = "\n\n".join(
        f"◆ {p['title']}（出典カテゴリ: {p['category']}）\n{p['body']}" for p in item["prose_sections"]
    )
    return RAG_HEADER_TEMPLATE.format(
        machine_line=src.MACHINE_LINE,
        structured_lines=structured_lines,
        prose_sections=prose_blocks,
    )


def build_all() -> list[dict]:
    records = []
    for i, item in enumerate(src.CATEGORY_T_PRODUCTION_RAG_OMISSION):
        rag_context = build_rag_context(item)
        user_content = f"{rag_context}\n\n{item['question']}"
        assistant_content = item["assistant"].strip()
        records.append(
            {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "metadata": {
                    "source": "phase4h_generated",
                    "category": "production_rag_omission_prevention",
                    "category_code": "T",
                    "index": i,
                },
            }
        )
    return records


# ---------------------------------------------------------------------------
# 品質検査
# ---------------------------------------------------------------------------


def all_texts(record: dict) -> list[str]:
    return [m["content"] for m in record["messages"]]


def all_assistant_texts(record: dict) -> list[str]:
    return [m["content"] for m in record["messages"] if m["role"] == "assistant"]


def validate_schema(records: list[dict]) -> list[str]:
    errors = []
    for i, r in enumerate(records):
        msgs = r["messages"]
        if not msgs or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
            errors.append(f"record {i}: must start with user and end with assistant")
        for j, m in enumerate(msgs):
            if not m.get("content", "").strip():
                errors.append(f"record {i} turn {j}: empty content (role={m['role']})")
    return errors


def validate_records(records: list[dict]) -> dict:
    emoji_hits = 0
    kaomoji_hits = 0
    double_excl_hits = 0
    chatml_hits = 0
    placeholder_hits: list[dict] = []
    real_machine_fact_hits: list[dict] = []

    for i, r in enumerate(records):
        for m in r["messages"]:
            if cd.DECORATIVE_SYMBOLS_PATTERN.search(m["content"]):
                emoji_hits += 1
            if "♪" in m["content"]:
                kaomoji_hits += 1
            if cd.REPEATED_EXCLAMATION_PATTERN.search(m["content"]):
                double_excl_hits += 1
            if cd.CHATML_TOKEN_PATTERN.search(m["content"]):
                chatml_hits += 1
            ph = PLACEHOLDER_PATTERN.search(m["content"])
            if ph:
                placeholder_hits.append({"record_index": i, "matched": ph.group()})
        text_all = " ".join(all_texts(r))
        for term in FORBIDDEN_REAL_MACHINE_TERMS + cd.MACHINE_SPECIFIC_KEYWORDS:
            if term in text_all:
                real_machine_fact_hits.append({"record_index": i, "term": term})

    return {
        "total": len(records),
        "decorative_symbol_hits": emoji_hits,
        "kaomoji_note_hits": kaomoji_hits,
        "repeated_exclamation_hits": double_excl_hits,
        "chatml_token_hits": chatml_hits,
        "placeholder_hits": placeholder_hits,
        "real_machine_fact_hits": real_machine_fact_hits,
    }


def validate_relevance_coverage(source_items: list[dict]) -> dict:
    """各例について、relevant=Trueの構造化データの値がassistant回答に含まれ、
    relevant=Falseの値は含まれていないことを確認する
    (「省略しない」と「不要な情報まで答えない」の両方を自動検証する)。
    """
    missing_relevant: list[dict] = []
    leaked_irrelevant: list[dict] = []
    for i, item in enumerate(source_items):
        answer = item["assistant"]
        for row in item["structured_rows"]:
            value_present = row["value"] in answer
            if row["relevant"] and not value_present:
                missing_relevant.append(
                    {"record_index": i, "label": row["label"], "item": row["item"], "value": row["value"]}
                )
            if not row["relevant"] and value_present:
                leaked_irrelevant.append(
                    {"record_index": i, "label": row["label"], "item": row["item"], "value": row["value"]}
                )
    return {
        "missing_relevant_values": missing_relevant,
        "leaked_irrelevant_values": leaked_irrelevant,
        "missing_relevant_count": len(missing_relevant),
        "leaked_irrelevant_count": len(leaked_irrelevant),
    }


def check_unnatural_length(records: list[dict]) -> list[dict]:
    hits = []
    for i, r in enumerate(records):
        for text in all_assistant_texts(r):
            if len(text) > UNNATURAL_LENGTH_THRESHOLD:
                hits.append({"record_index": i, "length": len(text), "text": text})
    return hits


def find_exact_duplicates(records: list[dict]) -> dict:
    user_map: dict[str, list[int]] = {}
    assistant_map: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        u = r["messages"][0]["content"]
        a = r["messages"][1]["content"]
        user_map.setdefault(u, []).append(i)
        assistant_map.setdefault(a, []).append(i)
    return {
        "user_exact_dupes": {k: v for k, v in user_map.items() if len(v) > 1},
        "assistant_exact_dupes": {k: v for k, v in assistant_map.items() if len(v) > 1},
    }


def find_high_similarity_within(records: list[dict], threshold: float = 0.9) -> list[tuple]:
    texts = [r["messages"][1]["content"] for r in records]
    hits = []
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            if texts[i] == texts[j]:
                continue
            ratio = difflib.SequenceMatcher(None, texts[i], texts[j]).ratio()
            if ratio >= threshold:
                hits.append((i, j, round(ratio, 3)))
    return hits


def find_high_similarity_against_existing(
    new_records: list[dict], existing_records: list[dict], threshold: float = 0.9
) -> list[dict]:
    new_texts = [r["messages"][1]["content"] for r in new_records]
    existing_texts = [
        r["messages"][-1]["content"]
        if r["messages"] and r["messages"][-1]["role"] == "assistant"
        else ""
        for r in existing_records
    ]
    hits = []
    for ni, ntext in enumerate(new_texts):
        for ei, etext in enumerate(existing_texts):
            if not etext or ntext == etext:
                continue
            ratio = difflib.SequenceMatcher(None, ntext, etext).ratio()
            if ratio >= threshold:
                hits.append({"new_index": ni, "existing_index": ei, "ratio": round(ratio, 3)})
    return hits


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
                hits.append({"record_index": i, "tail_counts": local})
    return hits


def word_stats(records: list[dict]) -> dict:
    pronoun_counter: Counter[str] = Counter()
    tail_counter: Counter[str] = Counter()
    lengths = []
    for r in records:
        for text in all_assistant_texts(r):
            for w in ["私", "リル", "キミ"]:
                pronoun_counter[w] += text.count(w)
            for w in ["だよ", "なんだ", "だね", "だぞ"]:
                tail_counter[w] += text.count(w)
            lengths.append(len(text))
    n = len(lengths)
    return {
        "pronoun_counts": dict(pronoun_counter),
        "tail_word_counts": dict(tail_counter),
        "length_stats": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "avg": round(sum(lengths) / n, 1) if n else 0,
        },
    }


def load_existing_897() -> list[dict]:
    path = PROCESSED_DIR / "riru_lora_v2_candidate.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    records = build_all()
    print(f"新規レコード数: {len(records)}")

    schema_errors = validate_schema(records)
    validation = validate_records(records)
    relevance_check = validate_relevance_coverage(src.CATEGORY_T_PRODUCTION_RAG_OMISSION)
    length_hits = check_unnatural_length(records)
    dupes = find_exact_duplicates(records)
    high_sim_within = find_high_similarity_within(records)
    tail_repeat_hits = same_tail_repetition_within_response(records)
    stats = word_stats(records)

    existing = load_existing_897()
    high_sim_vs_existing = find_high_similarity_against_existing(records, existing)
    existing_user_texts = {r["messages"][0]["content"] for r in existing}
    new_vs_existing_user_dupes = [
        i for i, r in enumerate(records) if r["messages"][0]["content"] in existing_user_texts
    ]

    new_candidate_path = PROCESSED_DIR / "riru_phase4h_new_candidate.jsonl"
    with open(new_candidate_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"新規データ出力: {new_candidate_path}")

    report = {
        "total_new_records": len(records),
        "schema_validation_errors": schema_errors,
        "validation": validation,
        "relevance_coverage_check": relevance_check,
        "unnatural_length_hits_gt_140chars": length_hits,
        "duplicates": {
            "user_exact_dupe_groups_within_new": len(dupes["user_exact_dupes"]),
            "assistant_exact_dupe_groups_within_new": len(dupes["assistant_exact_dupes"]),
            "new_records_duplicating_existing_897_user_text": new_vs_existing_user_dupes,
            "high_similarity_pairs_within_new_ge_0.9": len(high_sim_within),
            "high_similarity_pairs_new_vs_existing_897_ge_0.9": len(high_sim_vs_existing),
        },
        "same_tail_repetition_within_response_count": len(tail_repeat_hits),
        "word_stats": stats,
    }
    (REPORTS_DIR / "phase4h_dataset_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "phase4h_duplicates.json").write_text(
        json.dumps(
            {
                "user_exact_dupes_within_new": dupes["user_exact_dupes"],
                "assistant_exact_dupes_within_new": dupes["assistant_exact_dupes"],
                "high_similarity_within_new": high_sim_within,
                "high_similarity_vs_existing_897": high_sim_vs_existing,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
