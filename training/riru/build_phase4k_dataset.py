# ruff: noqa: E501
"""Phase 4K: v4改善教師データの組み立て・検証・レポート生成。

v2候補 (riru_lora_v2_candidate.jsonl, 897件、読み取り専用) を読み込み、
H-0/H-3/H-11の3件だけをメモリ上で修正し、Type1(12件)・Type2(6件)の新規データを
追加してv4候補 (riru_lora_v4_candidate.jsonl) を新規作成する。

このスクリプトは:
  - QLoRA/LoRA学習を一切行わない
  - v1/v2/v3 adapter・checkpoint・logsに一切触れない
  - riru_lora_v2_candidate.jsonl 自体は一切書き換えない (読み取りのみ)
  - train/validation分割は行わない (人間レビュー後の次フェーズで実施)
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
V2_CANDIDATE_PATH = PROCESSED_DIR / "riru_lora_v2_candidate.jsonl"

sys.path.insert(0, str(TRAINING_ROOT))
import convert_dataset as cd  # noqa: E402
import phase4k_source_data as src  # noqa: E402

FORBIDDEN_REAL_MACHINE_TERMS = [
    "ミリオンゴッド", "GOD揃い", "ガイアベル", "ガイアステージ", "Z-ZONE",
    "ゼウスモード", "PGG", "神々の軌跡",
]
PLACEHOLDER_PATTERN = re.compile(
    r"(〜{3,}|ー{4,}|X{2,}(?![0-9A-Za-z])|x{2,}|○{2,}|●{2,}|TBD|TODO|\.{4,}|…{2,}|"
    r"[■□▲△▼▽◆◇]{2,}|�)"
)
# hallucination/derived-computation兆候の検出 (禁止事項: 倍率計算・差分計算・
# 勝率推測・設定推測・ヤメ時アドバイス・期待値推測・因果関係の創作)
SUSPICIOUS_PHRASE_PATTERNS = [
    "倍だ", "倍だよ", "倍になる", "ポイント高い", "ポイント上", "だと思う", "かもしれない",
    "した方がいい", "やめ時", "ヤメ時", "おすすめ", "覚えておくと", "参考になりそう",
]
UNNATURAL_LENGTH_THRESHOLD = 160

RAG_HEADER_TEMPLATE = """【対象機種】
{machine_line}
このセクション以下の情報はすべて上記の機種に関するものです。他の機種の名称を補完したり、機種名を推測したりしないでください。

【構造化データ（数値・確率・設定差・天井・示唆など）】
このセクションの数値は必ず原文表記のまま回答に使ってください。計算し直したり丸めたりしないでください。

{structured_lines}

【関連する解説文章】

{prose_sections}"""

# ---------------------------------------------------------------------------
# H-0 / H-3 / H-11 の修正定義 (修正前テキストとの一致検証つき)
# ---------------------------------------------------------------------------
EXISTING_FIXES = [
    {
        "id": "H-0",
        "candidate_index": 823,
        "expected_before_assistant": "天井は999Gで、天井まで行くとAT確定だよ。",
        "after_assistant": "天井は999Gで、天井まで行くとAT確定だよ。ATの純増は5枚/Gなんだ。",
        "reason": "Phase4J監査で『二次的エンティティ(AT)の派生数値を削っている』と判定。天井到達の直接の結果であるAT純増を復元。",
    },
    {
        "id": "H-3",
        "candidate_index": 826,
        "expected_before_assistant": "初当りは1/300で、当たった後はRT10Gが付くよ。",
        "after_assistant": "初当りは1/300で、当たった後はRT10Gが付くよ。RT中はボーナス確率もアップするんだ。",
        "reason": "Phase4J監査で同様のパターンと判定。RTの直接の性質(ボーナス確率アップ)を復元。",
    },
    {
        "id": "H-11",
        "candidate_index": 834,
        "expected_before_assistant": "AT中の純増は4枚/Gで、終了後はRTが付くよ。",
        "after_assistant": "AT中の純増は4枚/Gで、終了後はRTが付くよ。RTは10Gまでなんだ。",
        "reason": "Phase4J監査で同様のパターンと判定。RTの継続期間を復元。",
    },
]


def load_v2_candidate() -> list[dict]:
    records = []
    with open(V2_CANDIDATE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def apply_existing_fixes(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """H-0/H-3/H-11の3件だけをメモリ上で修正する。元リストは変更せず新しいリストを返す。"""
    fixed = [dict(r, messages=[dict(m) for m in r["messages"]]) for r in records]
    diff_report = []
    for fix in EXISTING_FIXES:
        idx = fix["candidate_index"]
        record = fixed[idx]
        assistant_msg = next(m for m in record["messages"] if m["role"] == "assistant")
        before = assistant_msg["content"]
        if before != fix["expected_before_assistant"]:
            raise AssertionError(
                f"{fix['id']} (index={idx}): 想定していた修正前テキストと一致しません。"
                f"expected={fix['expected_before_assistant']!r} actual={before!r}"
            )
        assistant_msg["content"] = fix["after_assistant"]
        diff_report.append(
            {
                "id": fix["id"],
                "candidate_index": idx,
                "user": record["messages"][0]["content"],
                "before_assistant": before,
                "after_assistant": fix["after_assistant"],
                "reason": fix["reason"],
            }
        )
    return fixed, diff_report


# ---------------------------------------------------------------------------
# Type1 組み立て
# ---------------------------------------------------------------------------


def build_rag_context(item: dict) -> str:
    structured_lines = "\n".join(
        f"- [{row['label']}] {row['item']}: {row['value']}" for row in item["structured_rows"]
    )
    prose_blocks = "\n\n".join(
        f"◆ {p['title']}（出典カテゴリ: {p['category']}）\n{p['body']}" for p in item["prose_sections"]
    )
    return RAG_HEADER_TEMPLATE.format(
        machine_line=f"{item['machine']}（パチスロ） 設定判別・天井・ゾーン・解析・打ち方・ヤメ時",
        structured_lines=structured_lines,
        prose_sections=prose_blocks,
    )


def build_type1_records() -> tuple[list[dict], list[dict]]:
    records = []
    structure_reports = []
    for i, item in enumerate(src.CATEGORY_T1_COMPLEX_RAG_STRUCTURE):
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
                    "source": "phase4k_generated",
                    "category": "complex_rag_structure_omission_prevention",
                    "category_code": "T1",
                    "index": i,
                },
            }
        )
        n_structured_relevant = sum(1 for r in item["structured_rows"] if r["relevant"])
        n_structured_irrelevant = sum(1 for r in item["structured_rows"] if not r["relevant"])
        n_prose_relevant = sum(1 for p in item["prose_sections"] if p["relevant"])
        n_prose_irrelevant = sum(1 for p in item["prose_sections"] if not p["relevant"])
        has_compressed_summary = any(p.get("is_compressed_summary") for p in item["prose_sections"])
        # relevant値の網羅チェック (構造化データのrelevant行の値 + prose限定relevant事実)
        relevant_values = [r["value"] for r in item["structured_rows"] if r["relevant"]]
        irrelevant_values = [r["value"] for r in item["structured_rows"] if not r["relevant"]]
        missing = [v for v in relevant_values if v not in assistant_content]
        leaked = [v for v in irrelevant_values if v in assistant_content]
        extra_facts = item.get("extra_relevant_facts_in_prose_not_in_structured", [])
        extra_missing = [f for f in extra_facts if not any(kw in assistant_content for kw in f.split("、"))]
        structure_reports.append(
            {
                "index": i,
                "question": item["question"],
                "structured_rows_total": len(item["structured_rows"]),
                "structured_rows_relevant": n_structured_relevant,
                "structured_rows_irrelevant": n_structured_irrelevant,
                "prose_sections_total": len(item["prose_sections"]),
                "prose_sections_relevant": n_prose_relevant,
                "prose_sections_irrelevant": n_prose_irrelevant,
                "has_compressed_summary_section": has_compressed_summary,
                "relevant_fact_count": n_structured_relevant + len(extra_facts),
                "irrelevant_fact_count": n_structured_irrelevant,
                "missing_relevant_structured_values": missing,
                "leaked_irrelevant_structured_values": leaked,
                "missing_extra_prose_facts": extra_missing,
                "assistant_length": len(assistant_content),
                "relevant_retention_ok": len(missing) == 0 and len(extra_missing) == 0,
                "no_irrelevant_leak": len(leaked) == 0,
            }
        )
    return records, structure_reports


def build_type2_records() -> list[dict]:
    records = []
    for i, item in enumerate(src.CATEGORY_T2_DERIVED_ENTITY_RETENTION):
        records.append(
            {
                "messages": [
                    {"role": "user", "content": item["user"].strip()},
                    {"role": "assistant", "content": item["assistant"].strip()},
                ],
                "metadata": {
                    "source": "phase4k_generated",
                    "category": "derived_entity_retention",
                    "category_code": "T2",
                    "index": i,
                },
            }
        )
    return records


# ---------------------------------------------------------------------------
# 品質検査 (Phase4F/4Hのパターンを踏襲)
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
        expected_role = "user"
        for j, m in enumerate(msgs):
            if m["role"] != expected_role:
                errors.append(f"record {i} turn {j}: expected role={expected_role}")
            expected_role = "assistant" if expected_role == "user" else "user"
            if not m.get("content", "").strip():
                errors.append(f"record {i} turn {j}: empty content")
    return errors


def validate_records(records: list[dict]) -> dict:
    emoji_hits = chatml_hits = excl_hits = kaomoji_hits = 0
    placeholder_hits: list[dict] = []
    real_machine_hits: list[dict] = []
    suspicious_hits: list[dict] = []
    for i, r in enumerate(records):
        for m in r["messages"]:
            if cd.DECORATIVE_SYMBOLS_PATTERN.search(m["content"]):
                emoji_hits += 1
            if "♪" in m["content"]:
                kaomoji_hits += 1
            if cd.REPEATED_EXCLAMATION_PATTERN.search(m["content"]):
                excl_hits += 1
            if cd.CHATML_TOKEN_PATTERN.search(m["content"]):
                chatml_hits += 1
            ph = PLACEHOLDER_PATTERN.search(m["content"])
            if ph:
                placeholder_hits.append({"record_index": i, "matched": ph.group()})
            if m["role"] == "assistant":
                for phrase in SUSPICIOUS_PHRASE_PATTERNS:
                    if phrase in m["content"]:
                        suspicious_hits.append({"record_index": i, "phrase": phrase, "text": m["content"]})
        text_all = " ".join(all_texts(r))
        for term in FORBIDDEN_REAL_MACHINE_TERMS + cd.MACHINE_SPECIFIC_KEYWORDS:
            if term in text_all:
                real_machine_hits.append({"record_index": i, "term": term})
    return {
        "decorative_symbol_hits": emoji_hits,
        "kaomoji_hits": kaomoji_hits,
        "repeated_exclamation_hits": excl_hits,
        "chatml_token_hits": chatml_hits,
        "placeholder_hits": placeholder_hits,
        "real_machine_fact_hits": real_machine_hits,
        "suspicious_hallucination_phrase_hits": suspicious_hits,
    }


def same_tail_repetition_within_response(records: list[dict]) -> list[dict]:
    TAIL_PATTERNS = [
        "だよ！", "だよ", "だよ〜", "だよね", "なんだ！", "なんだ", "なんだよ",
        "だね！", "だね", "だぞ", "っ！", "よ！", "ね！",
    ]
    hits = []
    for i, r in enumerate(records):
        for text in all_assistant_texts(r):
            local = {}
            for pat in TAIL_PATTERNS:
                c = text.count(pat)
                if c >= 3:
                    local[pat] = c
            if local:
                hits.append({"record_index": i, "tail_counts": local})
    return hits


def same_fact_repetition_within_response(records: list[dict]) -> list[dict]:
    """1回答内で同一の10文字以上の部分文字列が3回以上出現する場合を検出。"""
    hits = []
    for i, r in enumerate(records):
        for text in all_assistant_texts(r):
            if len(text) < 30:
                continue
            seen: dict[str, int] = {}
            for length in (10, 15):
                for j in range(0, len(text) - length):
                    chunk = text[j : j + length]
                    seen[chunk] = seen.get(chunk, 0) + 1
                    if seen[chunk] >= 3:
                        hits.append({"record_index": i, "chunk": chunk, "text": text[:120]})
                        break
    return hits


def find_exact_duplicates(records: list[dict]) -> dict:
    user_map: dict[str, list[int]] = {}
    assistant_map: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        u = r["messages"][0]["content"]
        a = r["messages"][-1]["content"]
        user_map.setdefault(u, []).append(i)
        assistant_map.setdefault(a, []).append(i)
    return {
        "user_exact_dupes": {k: v for k, v in user_map.items() if len(v) > 1},
        "assistant_exact_dupes": {k: v for k, v in assistant_map.items() if len(v) > 1},
    }


def find_high_similarity_within(records: list[dict], threshold: float = 0.9) -> list[tuple]:
    texts = [r["messages"][-1]["content"] for r in records]
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
    new_texts = [r["messages"][-1]["content"] for r in new_records]
    existing_texts = [r["messages"][-1]["content"] for r in existing_records]
    hits = []
    for ni, ntext in enumerate(new_texts):
        for ei, etext in enumerate(existing_texts):
            if not etext or ntext == etext:
                continue
            ratio = difflib.SequenceMatcher(None, ntext, etext).ratio()
            if ratio >= threshold:
                hits.append({"new_index": ni, "existing_index": ei, "ratio": round(ratio, 3)})
    return hits


def check_unnatural_length(records: list[dict]) -> list[dict]:
    return [
        {"record_index": i, "length": len(text)}
        for i, r in enumerate(records)
        for text in all_assistant_texts(r)
        if len(text) > UNNATURAL_LENGTH_THRESHOLD
    ]


def length_stats(records: list[dict]) -> dict:
    lengths = sorted(len(t) for r in records for t in all_assistant_texts(r))
    n = len(lengths)
    if n == 0:
        return {}

    def pct(p: float) -> float:
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        return lengths[f] if f == c else lengths[f] + (lengths[c] - lengths[f]) * (k - f)

    return {
        "min": lengths[0],
        "max": lengths[-1],
        "mean": round(sum(lengths) / n, 1),
        "median": round(pct(0.5), 1),
        "p90": round(pct(0.9), 1),
    }


def word_stats(records: list[dict]) -> dict:
    pronoun_counter: Counter[str] = Counter()
    tail_counter: Counter[str] = Counter()
    for r in records:
        for text in all_assistant_texts(r):
            for w in ["私", "リル", "キミ"]:
                pronoun_counter[w] += text.count(w)
            for w in ["だよ", "なんだ", "だね", "だぞ"]:
                tail_counter[w] += text.count(w)
    return {"pronoun_counts": dict(pronoun_counter), "tail_word_counts": dict(tail_counter)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    v2_records = load_v2_candidate()
    assert len(v2_records) == 897, f"unexpected v2 count: {len(v2_records)}"

    fixed_v2_records, fixes_diff = apply_existing_fixes(v2_records)

    type1_records, type1_structure_report = build_type1_records()
    type2_records = build_type2_records()
    new_records = type1_records + type2_records

    print(f"v2 (修正後) : {len(fixed_v2_records)}件")
    print(f"Type1新規    : {len(type1_records)}件")
    print(f"Type2新規    : {len(type2_records)}件")

    v4_records = fixed_v2_records + new_records
    v4_path = PROCESSED_DIR / "riru_lora_v4_candidate.jsonl"
    with open(v4_path, "w", encoding="utf-8") as f:
        for r in v4_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"v4候補データ出力: {v4_path} ({len(v4_records)}件)")

    # --- 品質検査 (新規18件対象) ---
    schema_errors = validate_schema(new_records)
    validation = validate_records(new_records)
    tail_repeat = same_tail_repetition_within_response(new_records)
    fact_repeat = same_fact_repetition_within_response(new_records)
    dupes = find_exact_duplicates(new_records)
    high_sim_within = find_high_similarity_within(new_records)
    high_sim_vs_existing = find_high_similarity_against_existing(new_records, fixed_v2_records)
    unnatural_length = check_unnatural_length(new_records)
    lengths = length_stats(new_records)
    words = word_stats(new_records)

    # 修正した3件も含めた再検査 (修正後テキストが異常を含まないか)
    fix_only_records = [fixed_v2_records[f["candidate_index"]] for f in EXISTING_FIXES]
    fix_validation = validate_records(fix_only_records)
    fix_tail_repeat = same_tail_repetition_within_response(fix_only_records)

    # --- relevant/irrelevant自動検証サマリ (Type1) ---
    relevant_ok = sum(1 for x in type1_structure_report if x["relevant_retention_ok"])
    irrelevant_ok = sum(1 for x in type1_structure_report if x["no_irrelevant_leak"])
    total_missing = sum(len(x["missing_relevant_structured_values"]) for x in type1_structure_report)
    total_missing += sum(len(x["missing_extra_prose_facts"]) for x in type1_structure_report)
    total_leaked = sum(len(x["leaked_irrelevant_structured_values"]) for x in type1_structure_report)

    # 「relevant事実が4個以上あるのに20文字前後で終了」チェック
    human_review_flags = [
        {"index": x["index"], "relevant_fact_count": x["relevant_fact_count"], "assistant_length": x["assistant_length"]}
        for x in type1_structure_report
        if x["relevant_fact_count"] >= 4 and x["assistant_length"] <= 25
    ]

    report = {
        "v4_total_records": len(v4_records),
        "v2_base_records_unchanged_count": len(fixed_v2_records) - len(EXISTING_FIXES),
        "v2_base_records_fixed_count": len(EXISTING_FIXES),
        "type1_new_count": len(type1_records),
        "type2_new_count": len(type2_records),
        "new_total_count": len(new_records),
        "schema_validation_errors": schema_errors,
        "validation_new_records": validation,
        "validation_fixed_records": fix_validation,
        "same_tail_repetition_new_records": len(tail_repeat),
        "same_tail_repetition_fixed_records": len(fix_tail_repeat),
        "same_fact_repetition_within_response": len(fact_repeat),
        "duplicates": {
            "user_exact_dupe_groups": len(dupes["user_exact_dupes"]),
            "assistant_exact_dupe_groups": len(dupes["assistant_exact_dupes"]),
            "high_similarity_within_new_ge_0.9": len(high_sim_within),
            "high_similarity_new_vs_existing897_ge_0.9": len(high_sim_vs_existing),
        },
        "unnatural_length_hits_gt_160chars": len(unnatural_length),
        "length_stats_new_records": lengths,
        "word_stats_new_records": words,
        "type1_relevance_check_summary": {
            "records_with_full_relevant_retention": f"{relevant_ok}/{len(type1_structure_report)}",
            "records_with_no_irrelevant_leak": f"{irrelevant_ok}/{len(type1_structure_report)}",
            "total_missing_relevant_facts": total_missing,
            "total_leaked_irrelevant_facts": total_leaked,
        },
        "human_review_flags_short_answer_many_facts": human_review_flags,
    }

    (REPORTS_DIR / "phase4k_dataset_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "phase4k_type1_structure_report.json").write_text(
        json.dumps(type1_structure_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "phase4k_duplicates.json").write_text(
        json.dumps(
            {
                "user_exact_dupes": dupes["user_exact_dupes"],
                "assistant_exact_dupes": dupes["assistant_exact_dupes"],
                "high_similarity_within_new": high_sim_within,
                "high_similarity_vs_existing897": high_sim_vs_existing,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (REPORTS_DIR / "phase4k_existing_fixes_diff.json").write_text(
        json.dumps(fixes_diff, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 人間レビュー用ファイル ---
    review_samples = []
    for i, item in enumerate(src.CATEGORY_T1_COMPLEX_RAG_STRUCTURE):
        struct_info = type1_structure_report[i]
        review_samples.append(
            {
                "id": f"T1-{i}",
                "category": "complex_rag_structure_omission_prevention",
                "user_full": type1_records[i]["messages"][0]["content"],
                "assistant_full": type1_records[i]["messages"][1]["content"],
                "structured_data": item["structured_rows"],
                "prose_sections": item["prose_sections"],
                "relevant_info_list": [r["value"] for r in item["structured_rows"] if r["relevant"]]
                + item.get("extra_relevant_facts_in_prose_not_in_structured", []),
                "irrelevant_info_list": [r["value"] for r in item["structured_rows"] if not r["relevant"]],
                "why_included": "質問対象と同一の情報ブロック(構造化データの当該見出し、または直接関連する解説文)に属する事実のため。",
                "why_excluded": "質問対象と異なるトピック(別のゾーン/小役/設定判別等)に属する情報のため、質問に無関係と判断し除外。",
                "structural_check": struct_info,
            }
        )
    for i, item in enumerate(src.CATEGORY_T2_DERIVED_ENTITY_RETENTION):
        review_samples.append(
            {
                "id": f"T2-{i}",
                "category": "derived_entity_retention",
                "user_full": type2_records[i]["messages"][0]["content"],
                "assistant_full": type2_records[i]["messages"][1]["content"],
                "why_included": "質問対象から1〜2階層以内の直接の派生エンティティに関する情報のため保持。",
                "why_excluded": "該当なし(このカテゴリは無関係情報を含まないシンプルな参照情報形式のため)。",
            }
        )
    (REPORTS_DIR / "phase4k_review_samples.json").write_text(
        json.dumps(review_samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
