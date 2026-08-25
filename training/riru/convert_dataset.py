"""旧リル(calm2-7b用)LoRAデータをQwen2.5-14B-Instruct向け messages形式へ変換 (Phase 4A)。

【目的】
`D:\\AI\\archive\\riru_ai_legacy_2025\\datasets\\` の7ファイル・523件を、
Qwen向けの `{"messages": [...], "metadata": {...}}` 形式へ機械的に変換し、
安全なクリーニング（♪除去・連続感嘆符の正規化など）のみを行う。

【厳守事項】
- 参照元 (archive) は一切変更しない。読み込みのみ。
- 文章の「意味」を書き換えるような加工は行わない。
  語尾（だよ/なんだ/だね等）は削除・置換しない。
  「同一応答内での過剰反復」のうち、機械的に安全に判定できるもの
  （直接連続する完全重複のみ）だけを対象とし、それ以外は検出してレポートするに留める。
- 二人称「キミ」の追加は一切行わない。
- パチスロ事実情報（数値等）は削除せず、検出のみ行いレポートに記録する。
- 重複は削除せず、検出してレポートするに留める。
- system prompt本文は messages に含めない（LoRAは人格のみを学習する設計のため）。
- ChatMLの `<|im_start|>` 等はデータ本文へ挿入しない。万一元データに混入していた場合は除去する。

このモジュールは python として直接実行可能 (`python convert_dataset.py`) で、
関数群は `tests/unit/test_riru_dataset_conversion.py` からも import して単体テストする。
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------------

TRAINING_ROOT = Path(__file__).resolve().parent  # training/riru/
ARCHIVE_SOURCE_DIR = Path(r"D:\AI\archive\riru_ai_legacy_2025\datasets")  # 読み取り専用
LOCAL_SOURCE_DIR = TRAINING_ROOT / "source"
PROCESSED_DIR = TRAINING_ROOT / "processed"
REPORTS_DIR = TRAINING_ROOT / "reports"

SOURCE_FILES = [
    "riru_character_personality.jsonl",
    "riru_conversations.jsonl",
    "riru_corrections.jsonl",
    "riru_emotions.jsonl",
    "riru_expressive_reactions.jsonl",
    "riru_intro_greetings.jsonl",
    "riru_misc.jsonl",
]

# ---------------------------------------------------------------------------
# クリーニング用パターン
# ---------------------------------------------------------------------------

# A. 装飾記号（♪等）。絵文字・顔文字はもともとsystem_prompt.py上のルールで禁止されている。
DECORATIVE_SYMBOLS_PATTERN = re.compile(
    r"[♪☆★✨💫💕😢😭😊😃🤣😡😎🤔🐱🌟🌀💥❤️💔😆😅😳🙄🤯🥺]"
)

# B. 連続感嘆符 (！！、!!、！!、!！ など2文字以上の連続) を単一の「！」へ正規化
REPEATED_EXCLAMATION_PATTERN = re.compile(r"[！!]{2,}")

# C. 直接連続する完全重複の語尾（間に他の文字が無い機械的な二重化のみを対象、
#    文をまたぐ「同じ語尾が2文で使われている」ようなケースは意味を変える恐れがあるため対象外）
_TAIL_TOKEN = (
    r"(?:だよ[〜ー]?[！!]?|だよね[！!]?|なんだ[〜ー]?[！!]?|なんだよ[〜ー]?[！!]?"
    r"|だね[っ]?[！!]?|だぞ[！!]?|っ[！!])"
)
BACK_TO_BACK_DUP_TAIL_PATTERN = re.compile(r"(" + _TAIL_TOKEN + r")\1+")

# E. ChatML等の特殊トークン文字列
CHATML_TOKEN_PATTERN = re.compile(r"<\|[a-zA-Z_][a-zA-Z0-9_]*\|>")

# F. 文字化け検出: Unicode置換文字、または制御文字（改行/タブ以外）
MOJIBAKE_REPLACEMENT_CHAR = "\ufffd"
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# パチスロ事実情報キーワード（監査で使ったものに、本フェーズで追加指定されたものを統合）
FACT_KEYWORDS = [
    "機械割", "初当り", "初当たり", "天井", "小役確率", "設定差",
    "設定1", "設定2", "設定3", "設定4", "設定5", "設定6",
    "純増", "ループストック", "払い出し",
]
# 数値パターン: %、1/xxx、xxxG、xxx枚、xxx倍
FACT_NUMERIC_PATTERNS = [
    re.compile(r"\d+(\.\d+)?\s*%"),
    re.compile(r"1\s*/\s*\d+"),
    re.compile(r"\d+\s*G(?![a-zA-Z])"),
    re.compile(r"\d+\s*枚"),
    re.compile(r"\d+(\.\d+)?\s*倍"),
]
# 特定機種固有の仕様（本プロジェクトの対象機種名・ゾーン名など）
MACHINE_SPECIFIC_KEYWORDS = [
    "ミリオンゴッド", "GOD揃い", "ガイアベル", "ガイアステージ", "Z-ZONE",
    "GG", "SGG", "ゼウスモード", "PGG",
]


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------


@dataclass
class CleaningChange:
    field: str  # "output" 固定 (今回はoutputのみクリーニング対象)
    reason: str  # 変更理由コード
    before: str
    after: str


@dataclass
class ConvertedRecord:
    messages: list[dict]
    metadata: dict
    changes: list[CleaningChange] = field(default_factory=list)
    fact_flags: list[str] = field(default_factory=list)
    mojibake_flags: list[str] = field(default_factory=list)
    chatml_flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# クリーニング処理
# ---------------------------------------------------------------------------


def clean_output(text: str) -> tuple[str, list[CleaningChange]]:
    """outputフィールドを安全にクリーニングする。

    行う変更 (A〜E相当):
      A. 装飾記号(♪等)を除去
      B. 連続感嘆符(！！、!!等)を単一の「！」へ正規化
      C. 直接連続する完全重複の語尾のみを1回分に圧縮 (機械的に安全な場合のみ)
      D. 前後の空白除去
      E. ChatML特殊トークン文字列があれば除去

    「語尾そのもの」の削除・言い換えは一切行わない。
    文をまたぐ語尾の重複（意味変更のリスクがあるもの）は変更せず、
    呼び出し側で "flag-only" として別途検出する。
    """
    changes: list[CleaningChange] = []
    current = text

    # D. まず前後空白を除去 (差分比較のベースラインを揃える)
    stripped = current.strip()
    if stripped != current:
        changes.append(CleaningChange("output", "whitespace_trim", current, stripped))
        current = stripped

    # E. ChatML特殊トークンの除去
    if CHATML_TOKEN_PATTERN.search(current):
        removed = CHATML_TOKEN_PATTERN.sub("", current).strip()
        changes.append(CleaningChange("output", "chatml_token_removed", current, removed))
        current = removed

    # A. 装飾記号(♪等)の除去
    if DECORATIVE_SYMBOLS_PATTERN.search(current):
        removed = DECORATIVE_SYMBOLS_PATTERN.sub("", current)
        # 記号除去で生じた不要な連続空白を1つに整理 (意味は変えない、見た目の整形のみ)
        removed = re.sub(r"[ \u3000]{2,}", " ", removed).strip()
        changes.append(CleaningChange("output", "decorative_symbol_removed", current, removed))
        current = removed

    # B. 連続感嘆符の正規化
    if REPEATED_EXCLAMATION_PATTERN.search(current):
        normalized = REPEATED_EXCLAMATION_PATTERN.sub("！", current)
        changes.append(
            CleaningChange("output", "repeated_exclamation_normalized", current, normalized)
        )
        current = normalized

    # C. 直接連続する完全重複の語尾を1回分に圧縮 (機械的に安全なケースのみ)
    if BACK_TO_BACK_DUP_TAIL_PATTERN.search(current):
        collapsed = BACK_TO_BACK_DUP_TAIL_PATTERN.sub(r"\1", current)
        changes.append(CleaningChange("output", "back_to_back_tail_collapsed", current, collapsed))
        current = collapsed

    # 最終トリム (記号除去等で末尾に空白が残るケースの後始末)
    final = current.strip()
    if final != current:
        changes.append(CleaningChange("output", "whitespace_trim", current, final))
        current = final

    return current, changes


def detect_mojibake(text: str) -> list[str]:
    flags = []
    if MOJIBAKE_REPLACEMENT_CHAR in text:
        flags.append("unicode_replacement_char")
    if CONTROL_CHAR_PATTERN.search(text):
        flags.append("control_char_present")
    # 明らかに不自然なUnicodeカテゴリ(サロゲート等)の混入チェック
    for ch in text:
        if unicodedata.category(ch) == "Cs":  # surrogate
            flags.append("surrogate_char_present")
            break
    return flags


def scan_fact_info(text: str) -> list[str]:
    """パチスロ事実情報（数値・機種固有仕様）の混入を検出する。削除はしない。"""
    hits: list[str] = []
    for kw in FACT_KEYWORDS:
        if kw in text:
            hits.append(f"keyword:{kw}")
    for kw in MACHINE_SPECIFIC_KEYWORDS:
        if kw in text:
            hits.append(f"machine_keyword:{kw}")
    for pat in FACT_NUMERIC_PATTERNS:
        if pat.search(text):
            hits.append(f"numeric_pattern:{pat.pattern}")
    return hits


def detect_chatml_leftover(text: str) -> list[str]:
    return CHATML_TOKEN_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# 読み込み・変換
# ---------------------------------------------------------------------------


def load_legacy_records(base_dir: Path = ARCHIVE_SOURCE_DIR) -> list[dict]:
    """archive配下の7ファイルを読み取り専用で読み込む。書き込みは一切行わない。"""
    records = []
    for fn in SOURCE_FILES:
        path = base_dir / fn
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                d["_source_file"] = fn
                d["_source_lineno"] = lineno
                records.append(d)
    return records


def convert_record(record: dict, legacy_index: int) -> ConvertedRecord:
    instruction = record.get("instruction", "")
    raw_output = record.get("output", "")

    cleaned_output, changes = clean_output(raw_output)

    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": cleaned_output},
    ]
    metadata = {
        "source_file": record["_source_file"],
        "source_lineno": record["_source_lineno"],
        "legacy_index": legacy_index,
        "variation_type": record.get("variation_type"),
        "variation_id": record.get("variation_id", record.get("variation")),
        "legacy": True,
        "legacy_raw_instruction": instruction,
        "legacy_raw_output": raw_output,
    }

    conv = ConvertedRecord(messages=messages, metadata=metadata, changes=changes)
    conv.fact_flags = scan_fact_info(instruction) + scan_fact_info(cleaned_output)
    conv.mojibake_flags = detect_mojibake(instruction) + detect_mojibake(cleaned_output)
    conv.chatml_flags = detect_chatml_leftover(instruction) + detect_chatml_leftover(raw_output)
    return conv


def convert_all(records: list[dict]) -> list[ConvertedRecord]:
    return [convert_record(r, i) for i, r in enumerate(records)]


# ---------------------------------------------------------------------------
# 重複検査 (検出のみ、削除はしない)
# ---------------------------------------------------------------------------


def find_duplicates(records: list[dict], similarity_threshold: float = 0.9) -> dict:
    instr_map: dict[str, list[int]] = {}
    output_map: dict[str, list[int]] = {}
    pair_map: dict[tuple[str, str], list[int]] = {}

    for i, r in enumerate(records):
        instr = r.get("instruction", "")
        out = r.get("output", "")
        instr_map.setdefault(instr, []).append(i)
        output_map.setdefault(out, []).append(i)
        pair_map.setdefault((instr, out), []).append(i)

    instruction_exact_dupes = {k: v for k, v in instr_map.items() if len(v) > 1}
    output_exact_dupes = {k: v for k, v in output_map.items() if len(v) > 1}
    pair_exact_dupes = {k: v for k, v in pair_map.items() if len(v) > 1}

    # 高類似度 (output同士、O(n^2)だが523件×523件は許容範囲)
    high_similarity_pairs = []
    outputs = [r.get("output", "") for r in records]
    n = len(outputs)
    for i in range(n):
        for j in range(i + 1, n):
            if outputs[i] == outputs[j]:
                continue  # 完全一致は上のoutput_exact_dupesで既に捕捉済み
            ratio = difflib.SequenceMatcher(None, outputs[i], outputs[j]).ratio()
            if ratio >= similarity_threshold:
                high_similarity_pairs.append((i, j, round(ratio, 3)))

    return {
        "instruction_exact_dupes": instruction_exact_dupes,
        "output_exact_dupes": output_exact_dupes,
        "pair_exact_dupes": pair_exact_dupes,
        "high_similarity_pairs": high_similarity_pairs,
    }


# ---------------------------------------------------------------------------
# 品質検証
# ---------------------------------------------------------------------------


def validate_converted(converted: list[ConvertedRecord], expected_count: int) -> dict:
    errors = []
    if len(converted) != expected_count:
        errors.append(f"count_mismatch: expected={expected_count} actual={len(converted)}")

    empty_user = 0
    empty_assistant = 0
    missing_roles = 0
    chatml_contamination = 0
    decorative_symbol_remaining = 0
    repeated_exclaim_remaining = 0
    untraceable = 0

    output_lengths = []
    tail_counter: Counter[str] = Counter()
    pronoun_counter: Counter[str] = Counter()

    TAIL_WORDS = ["だよ", "なんだ", "だね", "だぞ"]
    PRONOUNS = ["私", "リル", "キミ"]

    for c in converted:
        roles = [m["role"] for m in c.messages]
        if roles != ["user", "assistant"]:
            missing_roles += 1
        user_content = c.messages[0]["content"]
        asst_content = c.messages[1]["content"]
        if not user_content.strip():
            empty_user += 1
        if not asst_content.strip():
            empty_assistant += 1
        if DECORATIVE_SYMBOLS_PATTERN.search(asst_content):
            decorative_symbol_remaining += 1
        if REPEATED_EXCLAMATION_PATTERN.search(asst_content):
            repeated_exclaim_remaining += 1
        if CHATML_TOKEN_PATTERN.search(asst_content) or CHATML_TOKEN_PATTERN.search(user_content):
            chatml_contamination += 1
        if c.metadata.get("source_file") is None or c.metadata.get("source_lineno") is None:
            untraceable += 1

        output_lengths.append(len(asst_content))
        for tw in TAIL_WORDS:
            tail_counter[tw] += asst_content.count(tw)
        for pw in PRONOUNS:
            pronoun_counter[pw] += asst_content.count(pw)

    output_lengths_sorted = sorted(output_lengths)
    n = len(output_lengths_sorted)
    length_stats = {}
    if n:
        length_stats = {
            "min": output_lengths_sorted[0],
            "max": output_lengths_sorted[-1],
            "avg": round(sum(output_lengths_sorted) / n, 1),
            "median": output_lengths_sorted[n // 2],
        }

    return {
        "errors": errors,
        "total": len(converted),
        "empty_user": empty_user,
        "empty_assistant": empty_assistant,
        "missing_roles": missing_roles,
        "chatml_contamination": chatml_contamination,
        "decorative_symbol_remaining": decorative_symbol_remaining,
        "repeated_exclaim_remaining": repeated_exclaim_remaining,
        "untraceable": untraceable,
        "output_length_stats": length_stats,
        "tail_word_counts": dict(tail_counter),
        "pronoun_counts": dict(pronoun_counter),
    }


# ---------------------------------------------------------------------------
# メイン処理 (ファイル出力)
# ---------------------------------------------------------------------------


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    # 0. archiveの7ファイルを読み取り専用スナップショットとしてsource/へコピー
    #    (archive自体には一切書き込まない。コピー元→新規領域への複製のみ)
    for fn in SOURCE_FILES:
        shutil.copy2(ARCHIVE_SOURCE_DIR / fn, LOCAL_SOURCE_DIR / fn)

    # 1. 読み込み (source/のローカルスナップショットから読む。archiveは読み取り専用のまま)
    records = load_legacy_records(LOCAL_SOURCE_DIR)
    print(f"読み込み件数: {len(records)}")

    # 2. 変換 + クリーニング
    converted = convert_all(records)

    # 3. per-file + combined 出力
    by_file: dict[str, list[ConvertedRecord]] = {}
    for r, c in zip(records, converted, strict=True):
        by_file.setdefault(r["_source_file"], []).append(c)

    for fn, items in by_file.items():
        out_path = PROCESSED_DIR / fn.replace(".jsonl", "_qwen_messages.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for c in items:
                row = {"messages": c.messages, "metadata": c.metadata}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    combined_path = PROCESSED_DIR / "riru_qwen_messages_v1.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for c in converted:
            row = {"messages": c.messages, "metadata": c.metadata}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"変換後ファイル出力: {combined_path}")

    # 4. 重複検査
    dupes = find_duplicates(records)

    # 5. 品質検証
    validation = validate_converted(converted, expected_count=len(records))

    # 6. クリーニング差分レポート
    diff_records = []
    for r, c in zip(records, converted, strict=True):
        if c.changes:
            diff_records.append(
                {
                    "source_file": r["_source_file"],
                    "source_lineno": r["_source_lineno"],
                    "variation_type": r.get("variation_type"),
                    "variation_id": r.get("variation_id", r.get("variation")),
                    "changes": [
                        {"reason": ch.reason, "before": ch.before, "after": ch.after}
                        for ch in c.changes
                    ],
                }
            )

    # 7. パチスロ事実情報フラグ・mojibake・chatml のレコード一覧
    fact_flag_records = [
        {
            "source_file": r["_source_file"],
            "source_lineno": r["_source_lineno"],
            "instruction": r.get("instruction"),
            "output": c.messages[1]["content"],
            "flags": c.fact_flags,
        }
        for r, c in zip(records, converted, strict=True)
        if c.fact_flags
    ]
    mojibake_flag_records = [
        {
            "source_file": r["_source_file"],
            "source_lineno": r["_source_lineno"],
            "flags": c.mojibake_flags,
        }
        for r, c in zip(records, converted, strict=True)
        if c.mojibake_flags
    ]
    chatml_flag_records = [
        {
            "source_file": r["_source_file"],
            "source_lineno": r["_source_lineno"],
            "flags": c.chatml_flags,
        }
        for r, c in zip(records, converted, strict=True)
        if c.chatml_flags
    ]

    # 8. 「フラグは立てたが自動修正しなかった」同一語尾の文またぎ重複(参考情報)
    TAIL_PATTERNS_FOR_REPORT = [
        "だよ！！", "だよ！", "だよ", "だよ〜", "だよね！", "だよね",
        "なんだ！！", "なんだ！", "なんだ", "なんだ〜", "なんだよ", "なんだよ〜",
        "だね！", "だね", "だねっ", "だぞ！", "だぞ",
        "かな", "かも", "よ！", "よ〜", "ね！", "ねっ", "っ！",
    ]
    unresolved_repetition = []
    for r, c in zip(records, converted, strict=True):
        out = c.messages[1]["content"]
        local: dict[str, int] = {}
        for pat in TAIL_PATTERNS_FOR_REPORT:
            cnt = out.count(pat)
            if cnt >= 2:
                local[pat] = cnt
        if local:
            unresolved_repetition.append(
                {
                    "source_file": r["_source_file"],
                    "source_lineno": r["_source_lineno"],
                    "output": out,
                    "tail_counts": local,
                    "note": (
                        "文をまたぐ自然な語尾反復のため自動修正せず(意味変更リスク回避)。"
                        "人間の確認用に記録。"
                    ),
                }
            )

    report = {
        "summary": {
            "legacy_total": len(records),
            "converted_total": len(converted),
            "changed_records": len(diff_records),
            "unchanged_records": len(converted) - len(diff_records),
            "fact_flag_records": len(fact_flag_records),
            "mojibake_flag_records": len(mojibake_flag_records),
            "chatml_flag_records": len(chatml_flag_records),
            "unresolved_tail_repetition_records": len(unresolved_repetition),
            "instruction_exact_dupe_groups": len(dupes["instruction_exact_dupes"]),
            "output_exact_dupe_groups": len(dupes["output_exact_dupes"]),
            "pair_exact_dupe_groups": len(dupes["pair_exact_dupes"]),
            "high_similarity_pairs": len(dupes["high_similarity_pairs"]),
        },
        "validation": validation,
        "change_reason_counts": dict(
            Counter(ch["reason"] for d in diff_records for ch in d["changes"])
        ),
    }

    (REPORTS_DIR / "conversion_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "cleaning_diff.json").write_text(
        json.dumps(diff_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "fact_info_flags.json").write_text(
        json.dumps(fact_flag_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "mojibake_flags.json").write_text(
        json.dumps(mojibake_flag_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "chatml_flags.json").write_text(
        json.dumps(chatml_flag_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "unresolved_tail_repetition.json").write_text(
        json.dumps(unresolved_repetition, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def _dupe_json_safe(d: dict) -> dict:
        return {
            "instruction_exact_dupes": {k: v for k, v in d["instruction_exact_dupes"].items()},
            "output_exact_dupes": {k: v for k, v in d["output_exact_dupes"].items()},
            "pair_exact_dupes": {f"{k[0]} ||| {k[1]}": v for k, v in d["pair_exact_dupes"].items()},
            "high_similarity_pairs": d["high_similarity_pairs"],
        }

    (REPORTS_DIR / "duplicates.json").write_text(
        json.dumps(_dupe_json_safe(dupes), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
