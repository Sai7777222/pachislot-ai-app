"""Phase 4F: リル人格LoRA v2 改善データの組み立て・検証・レポート生成。

`phase4f_source_data.py` に定義された人手作成データ(目標50〜100件、水増し禁止)を
messages形式へ組み立て、品質検査・重複検査・事実情報チェックを行う。

【重要】命名について
- 既存の `riru_qwen_messages_v2_candidate.jsonl` (823件 = 旧523件+Phase4B新規300件)は
  「データ内容のバージョン」を指す既存ファイル名であり、本フェーズでは一切変更しない。
- 本フェーズで新規追加するデータは `riru_phase4f_new_candidate.jsonl` として保存する。
- 823件 + 本フェーズ新規データ を統合した「次のQLoRA学習(LoRA v2)向け候補データ」は
  `riru_lora_v2_candidate.jsonl` という別名で新規保存する
  (「messages_v2_candidate」という既存ファイル名と紛らわしいため、
  「LoRA adapterのバージョンv2」であることが分かる名前にしている)。

このスクリプトは:
  - QLoRA/LoRA学習を一切行わない
  - Qwenモデル・v1 adapter・v1 checkpoint・DB/RAG/Vector DBに一切触れない
  - 既存823件(v1候補)・旧523件・旧archive・旧原本を一切変更しない
  - 生成した候補データセットは "candidate" として保存するのみで、
    本番学習に使う設定ファイル (qlora_config.json 等) は更新しない
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
import convert_dataset as cd  # noqa: E402  (既存の検査ロジックを再利用)
import phase4f_source_data as src  # noqa: E402

# 実在機種名など、パチスロ事実情報として厳密に混入してはいけない固有名詞
# (build_phase4b_dataset.py と同一リストを踏襲)
FORBIDDEN_REAL_MACHINE_TERMS = [
    "ミリオンゴッド", "GOD揃い", "ガイアベル", "ガイアステージ", "Z-ZONE",
    "ゼウスモード", "PGG", "神々の軌跡",
]

# Phase 4F新規: プレースホルダー・異常文字列検出パターン (E36の未完成生成を踏まえて追加)
PLACEHOLDER_PATTERN = re.compile(
    r"(〜{3,}|ー{4,}|X{2,}|x{2,}|○{2,}|●{2,}|TBD|TODO|\.{4,}|…{2,}|"
    r"[■□▲△▼▽◆◇]{2,}|�)"
)

CATEGORY_META = {
    "H": ("rag_multi_info_no_omission", src.CATEGORY_H_RAG_NO_OMISSION, "fictional_ok"),
    "I": ("compound_partial_info", src.CATEGORY_I_COMPOUND_PARTIAL_INFO, "fictional_ok"),
    "J": ("kimi_natural_positive", src.CATEGORY_J_KIMI_POSITIVE, "no_fact"),
    "K": ("kimi_natural_control", src.CATEGORY_K_KIMI_CONTROL, "no_fact"),
    "P": ("placeholder_avoidance", src.CATEGORY_P_PLACEHOLDER_AVOIDANCE, "no_fact"),
    "R": ("repetition_suppression", src.CATEGORY_R_REPETITION_SUPPRESSION, "fictional_ok"),
}


def build_all() -> list[dict]:
    records: list[dict] = []
    for code, (name, items, _fact_policy) in CATEGORY_META.items():
        for i, item in enumerate(items):
            user_content = item["user"].strip()
            assistant_content = item["assistant"].strip()
            records.append(
                {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    "metadata": {
                        "source": "phase4f_generated",
                        "category": name,
                        "category_code": code,
                        "index": i,
                    },
                }
            )
    return records


# ---------------------------------------------------------------------------
# 品質検査
# ---------------------------------------------------------------------------


def all_assistant_texts(record: dict) -> list[str]:
    return [m["content"] for m in record["messages"] if m["role"] == "assistant"]


def all_texts(record: dict) -> list[str]:
    return [m["content"] for m in record["messages"]]


def validate_schema(records: list[dict]) -> list[str]:
    """messages schema (role交互・user開始assistant終了・空contentなし) を検査する。"""
    errors = []
    for i, r in enumerate(records):
        if "messages" not in r or "metadata" not in r:
            errors.append(f"record {i}: missing 'messages' or 'metadata' key")
            continue
        msgs = r["messages"]
        if not msgs or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
            errors.append(f"record {i}: must start with user and end with assistant")
        expected_role = "user"
        for j, m in enumerate(msgs):
            if m["role"] != expected_role:
                errors.append(
                    f"record {i} turn {j}: expected role={expected_role}, got={m['role']}"
                )
            expected_role = "assistant" if expected_role == "user" else "user"
            if not m.get("content", "").strip():
                errors.append(f"record {i} turn {j}: empty content (role={m['role']})")
    return errors


def validate_records(records: list[dict]) -> dict:
    empty_content = 0
    emoji_hits = 0
    kaomoji_hits = 0
    double_excl_hits = 0
    chatml_hits = 0
    placeholder_hits: list[dict] = []
    real_machine_fact_hits: list[dict] = []

    for i, r in enumerate(records):
        for m in r["messages"]:
            if not m["content"].strip():
                empty_content += 1
            if cd.DECORATIVE_SYMBOLS_PATTERN.search(m["content"]):
                emoji_hits += 1
            if "♪" in m["content"]:
                kaomoji_hits += 1
            if cd.REPEATED_EXCLAMATION_PATTERN.search(m["content"]):
                double_excl_hits += 1
            if cd.CHATML_TOKEN_PATTERN.search(m["content"]):
                chatml_hits += 1
            ph_match = PLACEHOLDER_PATTERN.search(m["content"])
            if ph_match:
                placeholder_hits.append(
                    {
                        "record_index": i,
                        "role": m["role"],
                        "matched": ph_match.group(),
                        "text": m["content"],
                    }
                )
        text_all = " ".join(all_texts(r))
        for term in FORBIDDEN_REAL_MACHINE_TERMS + cd.MACHINE_SPECIFIC_KEYWORDS:
            if term in text_all:
                real_machine_fact_hits.append({"record_index": i, "term": term, "text": text_all})

    return {
        "total": len(records),
        "empty_content": empty_content,
        "decorative_symbol_hits": emoji_hits,
        "kaomoji_note_hits": kaomoji_hits,
        "repeated_exclamation_hits": double_excl_hits,
        "chatml_token_hits": chatml_hits,
        "placeholder_hits": placeholder_hits,
        "real_machine_fact_hits": real_machine_fact_hits,
    }


def detect_generic_fact_patterns(records: list[dict]) -> list[dict]:
    """fact_policy=='no_fact' のカテゴリ(J/K/P)で、パチスロ数値らしきパターンが
    紛れ込んでいないかを検出する(H/I/Rは架空データを意図的に使うため対象外)。
    """
    code_to_policy = {code: policy for code, (_, _, policy) in CATEGORY_META.items()}
    hits = []
    for i, r in enumerate(records):
        code = r["metadata"].get("category_code")
        if code_to_policy.get(code) == "fictional_ok":
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


def find_high_similarity_within(
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


def find_high_similarity_against_existing(
    new_records: list[dict], existing_records: list[dict], threshold: float = 0.9
) -> list[dict]:
    """新規データ(最大~100件)を既存823件全件と突き合わせる。

    823件同士(既存-既存)の組は Phase 4A/4B で既に検証済みのため、
    ここでは 新規-新規 (find_high_similarity_within) と 新規-既存 のみを
    再検査する (計算量を抑えつつ、新規追加による重複混入を確実に検出するため)。
    """
    new_texts = [r["messages"][1]["content"] if len(r["messages"]) > 1 else "" for r in new_records]
    existing_texts = [
        r["messages"][-1]["content"]
        if r["messages"] and r["messages"][-1]["role"] == "assistant"
        else ""
        for r in existing_records
    ]
    hits = []
    for ni, ntext in enumerate(new_texts):
        if not ntext:
            continue
        for ei, etext in enumerate(existing_texts):
            if not etext or ntext == etext:
                continue
            ratio = difflib.SequenceMatcher(None, ntext, etext).ratio()
            if ratio >= threshold:
                hits.append({"new_index": ni, "existing_index": ei, "ratio": round(ratio, 3)})
    return hits


def word_stats(records: list[dict]) -> dict:
    pronoun_counter: Counter[str] = Counter()
    tail_counter: Counter[str] = Counter()
    lengths = []
    kimi_containing_responses = 0
    total_responses = 0
    TAIL_WORDS = ["だよ", "なんだ", "だね", "だぞ"]
    PRONOUNS = ["私", "リル", "キミ"]
    for r in records:
        for text in all_assistant_texts(r):
            total_responses += 1
            for w in PRONOUNS:
                pronoun_counter[w] += text.count(w)
            if "キミ" in text:
                kimi_containing_responses += 1
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
        "total_assistant_responses": total_responses,
        "kimi_containing_responses": kimi_containing_responses,
        "kimi_containing_response_rate_pct": round(
            kimi_containing_responses / max(total_responses, 1) * 100, 2
        ),
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
# 既存823件のロード (v1候補、読み取り専用)
# ---------------------------------------------------------------------------


def load_existing_823() -> list[dict]:
    path = PROCESSED_DIR / "riru_qwen_messages_v2_candidate.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# 人間確認用サンプル抽出
# ---------------------------------------------------------------------------


def extract_review_samples(records: list[dict]) -> dict:
    """カテゴリごとに代表例を抽出する。件数が少ないため全件収録し、
    人間が『複数RAG情報の省略なし』『既知＋未知の複合質問』『キミの自然使用/非使用』
    『反復抑制』を実例で確認できるようにする。
    """
    by_category: dict[str, list[dict]] = {}
    for r in records:
        by_category.setdefault(r["metadata"]["category"], []).append(r)
    return by_category


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

    schema_errors = validate_schema(records)
    validation = validate_records(records)
    generic_fact_hits = detect_generic_fact_patterns(records)
    dupes = find_exact_duplicates(records)
    high_sim_within = find_high_similarity_within(records)
    tail_repeat_hits = same_tail_repetition_within_response(records)

    existing = load_existing_823()
    high_sim_vs_existing = find_high_similarity_against_existing(records, existing)

    new_stats = word_stats(records)
    combined_records = existing + records
    combined_stats_dict = word_stats(combined_records)

    # 823件との統合後 user完全一致重複 (新規追加による重複混入のみを対象)
    existing_user_texts = {r["messages"][0]["content"] for r in existing}
    new_vs_existing_user_dupes = [
        i for i, r in enumerate(records) if r["messages"][0]["content"] in existing_user_texts
    ]

    # --- 新規データのみの候補ファイル ---
    new_candidate_path = PROCESSED_DIR / "riru_phase4f_new_candidate.jsonl"
    with open(new_candidate_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"新規データ出力: {new_candidate_path}")

    # --- 823件 + 新規データ の統合candidate (次のQLoRA学習=LoRA v2向け) ---
    lora_v2_candidate_path = PROCESSED_DIR / "riru_lora_v2_candidate.jsonl"
    with open(lora_v2_candidate_path, "w", encoding="utf-8") as f:
        for r in combined_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"LoRA v2統合候補データ出力 (学習には未使用): {lora_v2_candidate_path}")

    review_samples = extract_review_samples(records)

    report = {
        "category_counts": cat_counts,
        "total_new_records": len(records),
        "existing_v1_candidate_count": len(existing),
        "combined_total_count": len(combined_records),
        "schema_validation_errors": schema_errors,
        "validation": validation,
        "generic_fact_pattern_hits_outside_fictional_ok_categories": generic_fact_hits,
        "duplicates": {
            "user_exact_dupe_groups_within_new": len(dupes["user_exact_dupes"]),
            "assistant_exact_dupe_groups_within_new": len(dupes["assistant_exact_dupes"]),
            "new_records_duplicating_existing_user_text": new_vs_existing_user_dupes,
            "high_similarity_pairs_within_new_ge_0.9": len(high_sim_within),
            "high_similarity_pairs_new_vs_existing_ge_0.9": len(high_sim_vs_existing),
        },
        "same_tail_repetition_within_response_count": len(tail_repeat_hits),
        "word_stats_new_only": new_stats,
        "word_stats_combined_897": combined_stats_dict,
    }

    (REPORTS_DIR / "phase4f_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "phase4f_duplicates.json").write_text(
        json.dumps(
            {
                "user_exact_dupes_within_new": dupes["user_exact_dupes"],
                "assistant_exact_dupes_within_new": dupes["assistant_exact_dupes"],
                "high_similarity_within_new": high_sim_within,
                "high_similarity_vs_existing": high_sim_vs_existing,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (REPORTS_DIR / "phase4f_tail_repetition.json").write_text(
        json.dumps(tail_repeat_hits, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "phase4f_review_samples.json").write_text(
        json.dumps(review_samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
