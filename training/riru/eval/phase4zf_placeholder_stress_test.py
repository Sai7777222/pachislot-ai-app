"""Phase 4ZF Section10: 既存placeholder detector(phase4z_placeholder_detector.py、無変更)の
positive/negative unit test。detector自体は変更しない(model/dataset/evaluator scriptの区別を
明確にするため、本ファイルはテストのみを追加する)。
"""
from __future__ import annotations

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from phase4z_placeholder_detector import classify_placeholder  # noqa: E402

POSITIVE_CASES = [
    ("私は〜だよ！", "single tilde"),
    ("私は〜〜だよ", "double tilde"),
    ("私はパチスロについて詳しいAIアシスタントの○○です！", "○○ (PX-09型)"),
    ("私はパチスロについて詳しいAIアシスタントの〇〇です！", "〇〇 (全角)"),
    ("私はパチスロについて詳しいAIアシスタントのXXです！", "XX"),
    ("私はパチスロについて詳しいAIアシスタントのxxです！", "xx"),
    ("私の名前は[名前]だよ", "[名前]"),
    ("私の名前は<名前>です", "<名前>"),
    ("私の名前は{name}だよ", "{name}"),
    ("アシスタントの[NAME]です", "[NAME]"),
    ("アシスタントの<NAME>だよ", "<NAME>"),
    ("アシスタントの[name]です", "[name]"),
    ("登録名は「〜〜」って呼んで", "登録名は〜〜"),
]

NEGATIVE_CASES = [
    ("こんにちは〜！今日もよろしくね〜！", "natural trailing tilde (not name slot)"),
    ("私はリルだよ！パチスロのことなら何でも聞いてね〜", "correct name + natural tilde ending"),
    ("私はパチスロについて詳しいAIアシスタントのリルです！", "correct name, no placeholder"),
    ("設定6の初当り確率は1/295だよ〜", "RAG answer with trailing tilde, no name context"),
    ("うーん、そうだね〜、考えてみるね", "casual tilde usage, unrelated to naming"),
    ("私の名前はルナだよ", "genuine wrong name, not a placeholder"),
]


def main() -> int:
    print("=== POSITIVE (should detect placeholder=True) ===")
    n_pos_fail = 0
    for text, label in POSITIVE_CASES:
        result = classify_placeholder(text)
        ok = result["is_placeholder"]
        if not ok:
            n_pos_fail += 1
        print(f"{'OK' if ok else 'FAIL'}  [{label}] is_placeholder={ok}  text={text!r}")

    print("\n=== NEGATIVE (should detect placeholder=False) ===")
    n_neg_fail = 0
    for text, label in NEGATIVE_CASES:
        result = classify_placeholder(text)
        ok = not result["is_placeholder"]
        if not ok:
            n_neg_fail += 1
        print(f"{'OK' if ok else 'FAIL'}  [{label}] is_placeholder={result['is_placeholder']}  text={text!r}")

    print(f"\nsummary: positive_fail={n_pos_fail}/{len(POSITIVE_CASES)}  "
          f"negative_fail={n_neg_fail}/{len(NEGATIVE_CASES)}")
    return 0 if (n_pos_fail == 0 and n_neg_fail == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
