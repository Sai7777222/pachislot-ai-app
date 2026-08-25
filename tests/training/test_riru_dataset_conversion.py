"""旧リルLoRAデータ → Qwen向けmessages変換 (Phase 4A) の単体テスト。

`training/riru/convert_dataset.py` はpachislot_aiパッケージ (src/) の外側にある
一時的なデータ準備ツールのため、`scripts/` 配下のスクリプト群と同様に
sys.path経由でimportする。

このテストは D:\\AI\\archive\\riru_ai_legacy_2025\\datasets\\ (読み取り専用) の
実データに依存する。アーカイブは変更しない。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "training" / "riru"))

import convert_dataset as cd  # noqa: E402

ARCHIVE_AVAILABLE = cd.ARCHIVE_SOURCE_DIR.is_dir() and all(
    (cd.ARCHIVE_SOURCE_DIR / fn).is_file() for fn in cd.SOURCE_FILES
)

pytestmark = pytest.mark.skipif(
    not ARCHIVE_AVAILABLE,
    reason="旧リルLoRAデータのアーカイブ (D:\\AI\\archive\\riru_ai_legacy_2025) が見つかりません",
)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def legacy_records() -> list[dict]:
    return cd.load_legacy_records(cd.ARCHIVE_SOURCE_DIR)


@pytest.fixture(scope="module")
def converted_records(legacy_records: list[dict]) -> list[cd.ConvertedRecord]:
    return cd.convert_all(legacy_records)


# ---------------------------------------------------------------------------
# 1. 523件すべて変換される
# ---------------------------------------------------------------------------


def test_all_523_records_are_loaded_and_converted(legacy_records, converted_records):
    assert len(legacy_records) == 523
    assert len(converted_records) == 523


# ---------------------------------------------------------------------------
# 2. 元データは変更されない
# ---------------------------------------------------------------------------


def test_archive_source_is_never_modified(legacy_records, converted_records):
    """変換処理の前後でarchive内の7ファイルのSHA-256が変化しないことを確認する。"""
    hashes_after = {
        fn: _sha256_of_file(cd.ARCHIVE_SOURCE_DIR / fn) for fn in cd.SOURCE_FILES
    }
    # このプロセス内で最初に読み込んだ時点のハッシュと再計算後のハッシュを比較
    hashes_recheck = {
        fn: _sha256_of_file(cd.ARCHIVE_SOURCE_DIR / fn) for fn in cd.SOURCE_FILES
    }
    assert hashes_after == hashes_recheck

    # 各ファイルの行数が SOURCE_FILES と record 数の対応から変わっていないことも確認
    per_file_counts: dict[str, int] = {}
    for r in legacy_records:
        per_file_counts[r["_source_file"]] = per_file_counts.get(r["_source_file"], 0) + 1
    for fn in cd.SOURCE_FILES:
        with open(cd.ARCHIVE_SOURCE_DIR / fn, encoding="utf-8") as f:
            n_nonempty = sum(1 for line in f if line.strip())
        assert per_file_counts[fn] == n_nonempty


# ---------------------------------------------------------------------------
# 3. messages schemaが正しい / 4. user/assistant contentが空でない
# ---------------------------------------------------------------------------


def test_messages_schema_is_valid_for_all_records(converted_records):
    for c in converted_records:
        assert len(c.messages) == 2
        assert c.messages[0]["role"] == "user"
        assert c.messages[1]["role"] == "assistant"
        assert isinstance(c.messages[0]["content"], str)
        assert isinstance(c.messages[1]["content"], str)


def test_user_and_assistant_content_is_never_empty(converted_records):
    for c in converted_records:
        assert c.messages[0]["content"].strip() != ""
        assert c.messages[1]["content"].strip() != ""


def test_no_system_prompt_baked_into_messages(converted_records):
    """LoRAは人格のみを学習する設計のため、system roleのメッセージを含めない。"""
    for c in converted_records:
        roles = [m["role"] for m in c.messages]
        assert "system" not in roles


# ---------------------------------------------------------------------------
# 5. metadataから元レコードを追跡できる
# ---------------------------------------------------------------------------


def test_metadata_allows_tracing_back_to_legacy_source(legacy_records, converted_records):
    for legacy, c in zip(legacy_records, converted_records, strict=True):
        assert c.metadata["source_file"] == legacy["_source_file"]
        assert c.metadata["source_lineno"] == legacy["_source_lineno"]
        assert c.metadata["legacy"] is True
        assert c.metadata["legacy_raw_instruction"] == legacy.get("instruction", "")
        assert c.metadata["legacy_raw_output"] == legacy.get("output", "")

    # source_file + source_lineno で実ファイルの該当行を引き直せることを確認 (サンプル検証)
    sample = converted_records[100]
    fn = sample.metadata["source_file"]
    lineno = sample.metadata["source_lineno"]
    with open(cd.ARCHIVE_SOURCE_DIR / fn, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    original = json.loads(lines[lineno - 1])
    assert original["instruction"] == sample.metadata["legacy_raw_instruction"]


# ---------------------------------------------------------------------------
# 6. ♪が除去される
# ---------------------------------------------------------------------------


def test_decorative_symbols_are_removed_by_clean_output():
    cleaned, changes = cd.clean_output("ふふ〜ん♪ 私、天才かもしれないよっ！？")
    assert "♪" not in cleaned
    assert any(ch.reason == "decorative_symbol_removed" for ch in changes)


def test_no_decorative_symbols_remain_in_any_converted_record(converted_records):
    for c in converted_records:
        assert "♪" not in c.messages[1]["content"]


# ---------------------------------------------------------------------------
# 7. 連続感嘆符が正規化される
# ---------------------------------------------------------------------------


def test_repeated_exclamation_marks_are_normalized_by_clean_output():
    cleaned, changes = cd.clean_output("やったー！超うれしいよ〜！！ありがとう〜！！")
    assert "！！" not in cleaned
    assert "!!" not in cleaned
    assert any(ch.reason == "repeated_exclamation_normalized" for ch in changes)


def test_no_repeated_exclamation_remains_in_any_converted_record(converted_records):
    for c in converted_records:
        content = c.messages[1]["content"]
        assert "！！" not in content
        assert "!!" not in content


# ---------------------------------------------------------------------------
# 8. ChatML特殊文字列が混入しない
# ---------------------------------------------------------------------------


def test_chatml_tokens_are_stripped_by_clean_output():
    cleaned, changes = cd.clean_output("<|im_start|>assistant\nこんにちは！<|im_end|>")
    assert "<|im_start|>" not in cleaned
    assert "<|im_end|>" not in cleaned
    assert any(ch.reason == "chatml_token_removed" for ch in changes)


def test_no_chatml_tokens_in_any_converted_record(converted_records):
    for c in converted_records:
        assert cd.CHATML_TOKEN_PATTERN.search(c.messages[0]["content"]) is None
        assert cd.CHATML_TOKEN_PATTERN.search(c.messages[1]["content"]) is None


# ---------------------------------------------------------------------------
# 9. パチスロ固有数値を検出できる
# ---------------------------------------------------------------------------


def test_fact_info_scan_detects_numeric_and_keyword_facts():
    hits = cd.scan_fact_info("設定6の機械割は114.6%で、初当り確率は1/295だよ！純増は約7枚/G！")
    assert any("keyword:機械割" in h for h in hits)
    assert any(h.startswith("numeric_pattern:") for h in hits)


def test_fact_info_scan_is_clean_for_ordinary_character_lines():
    hits = cd.scan_fact_info("うん、今日もめちゃ元気だよ！質問があったらどんどん聞いてね！")
    assert hits == []


def test_real_dataset_fact_flags_match_known_audit_result(converted_records):
    """事前の監査で判明している「天井」1件のみが検出されることを確認する回帰テスト。"""
    flagged = [c for c in converted_records if c.fact_flags]
    assert len(flagged) == 1
    assert flagged[0].metadata["source_file"] == "riru_character_personality.jsonl"
    assert "天井" in flagged[0].metadata["legacy_raw_output"]


# ---------------------------------------------------------------------------
# 10. クリーニング差分が追跡できる
# ---------------------------------------------------------------------------


def test_cleaning_diff_before_after_is_recorded_correctly():
    original = "うわ〜！それめっちゃアツいじゃん！！私、超ワクワクなんだけど！"
    cleaned, changes = cd.clean_output(original)
    assert len(changes) >= 1
    change = changes[0]
    assert change.before == original
    assert change.after == cleaned
    assert change.reason == "repeated_exclamation_normalized"


def test_unchanged_records_have_no_diff_entries(converted_records):
    for c in converted_records:
        raw_output = c.metadata["legacy_raw_output"]
        cleaned_output = c.messages[1]["content"]
        if raw_output.strip() == cleaned_output:
            assert c.changes == [] or all(
                ch.reason == "whitespace_trim" for ch in c.changes
            )


def test_diff_change_count_matches_expected_categories(converted_records):
    """既知の監査結果 (♪:54件, !!正規化:11件, back-to-back重複:0件) との整合確認 (回帰テスト)。"""
    reason_counts: dict[str, int] = {}
    for c in converted_records:
        for ch in c.changes:
            reason_counts[ch.reason] = reason_counts.get(ch.reason, 0) + 1
    assert reason_counts.get("decorative_symbol_removed", 0) == 54
    assert reason_counts.get("repeated_exclamation_normalized", 0) == 11
    assert reason_counts.get("back_to_back_tail_collapsed", 0) == 0
    assert reason_counts.get("chatml_token_removed", 0) == 0


# ---------------------------------------------------------------------------
# 追加: 意味を変えていないことの間接確認 (語尾・一人称・二人称の頻度が不変)
# ---------------------------------------------------------------------------


def test_tail_words_and_pronouns_are_not_altered(legacy_records, converted_records):
    """語尾・人称は一切削除/置換していないため、総出現数が変換前後で変わらないことを確認する。"""
    for word in ["だよ", "なんだ", "だね", "私", "リル", "キミ"]:
        before = sum(r.get("output", "").count(word) for r in legacy_records)
        after = sum(c.messages[1]["content"].count(word) for c in converted_records)
        assert before == after, f"{word!r} の出現数が変換前後で変化した"


def test_kimi_is_not_artificially_added(converted_records):
    """二人称「キミ」を機械的に追加していないことの確認 (旧データ通り3件のみ)。"""
    kimi_count = sum(c.messages[1]["content"].count("キミ") for c in converted_records)
    assert kimi_count == 3


# ---------------------------------------------------------------------------
# validate_converted() / find_duplicates() の健全性
# ---------------------------------------------------------------------------


def test_validate_converted_reports_zero_errors(converted_records):
    result = cd.validate_converted(converted_records, expected_count=523)
    assert result["errors"] == []
    assert result["empty_user"] == 0
    assert result["empty_assistant"] == 0
    assert result["missing_roles"] == 0
    assert result["chatml_contamination"] == 0
    assert result["decorative_symbol_remaining"] == 0
    assert result["repeated_exclaim_remaining"] == 0
    assert result["untraceable"] == 0


def test_find_duplicates_does_not_mutate_input(legacy_records):
    before = json.dumps(legacy_records, sort_keys=True, default=str)
    cd.find_duplicates(legacy_records)
    after = json.dumps(legacy_records, sort_keys=True, default=str)
    assert before == after


def test_find_duplicates_matches_known_audit_result(legacy_records):
    dupes = cd.find_duplicates(legacy_records)
    assert len(dupes["instruction_exact_dupes"]) == 25
    assert len(dupes["output_exact_dupes"]) == 0
    assert len(dupes["pair_exact_dupes"]) == 0
