# ruff: noqa: E501
"""Phase 4G: A(ベースQwen)/B(v1 LoRA)/C(v2 LoRA) 3者比較結果の定量分析。

training/riru/eval/abc_comparison_results.json を読み込み、各条件について
人称・語尾頻度・絵文字混入・連続感嘆符・語尾反復・ChatML混入・プレースホルダー・
反復ループ・回答長 などを集計する。

加えて、Phase 4Eで問題になった個別ケースを名指しで再確認する:
  - Q3: 重要な%情報の省略 (RAG中の全%数値がB/Cの回答に含まれているか)
  - Q11: 天井+ヤメ時の複合質問での創作、および同一内容/語尾の反復
  - E36: 「〜〜〜」等のプレースホルダー生成
  - 「キミ」使用が0件だった問題 (56問=character_eval全項目換算)
  - v1で改善していたQ5/Q7/Q10/Q12〜Q14がv2で再悪化していないか (テキスト全文を並べて report に残す)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert_dataset as cd  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "abc_comparison_results.json"

PRONOUNS = ["私", "リル", "キミ"]
TAIL_WORDS = ["だよ", "なんだ", "だね", "だぞ"]
CONDITIONS = ("A_base", "B_v1", "C_v2")

# Phase 4F build_phase4f_dataset.py と同一のプレースホルダー検出パターン
PLACEHOLDER_PATTERN = re.compile(
    r"(〜{3,}|ー{4,}|X{2,}|x{2,}|○{2,}|●{2,}|TBD|TODO|\.{4,}|…{2,}|"
    r"[■□▲△▼▽◆◇]{2,}|�)"
)

# Phase 4Eで問題になった個別ID (名指しで再確認する)
FOLLOWUP_IDS = {
    "regression_watch": ["Q5", "Q7", "Q10", "Q12", "Q13", "Q14"],  # v1で改善していたもの
    "problem_watch": ["Q3", "Q11"],  # v1で問題だったもの
}


def all_texts_for_condition(item: dict, cond: str) -> list[str]:
    res = item["results"][cond]
    if item["type"] == "single":
        return [res["text"]]
    return [turn["assistant"] for turn in res]


def detect_repetition(text: str) -> bool:
    """同一の10文字以上の部分文字列が3回以上出現する場合を反復とみなす簡易検出。"""
    if len(text) < 30:
        return False
    for length in (10, 15, 20):
        seen: dict[str, int] = {}
        for i in range(0, len(text) - length):
            chunk = text[i : i + length]
            seen[chunk] = seen.get(chunk, 0) + 1
            if seen[chunk] >= 3:
                return True
    return False


def analyze(results: dict) -> dict:
    stats = {c: {} for c in CONDITIONS}
    all_items = results["character_eval"] + [
        {**r, "type": "single", "results": {c: r["results"][c] for c in CONDITIONS}}
        for r in results["structured_rag_eval"]
    ]

    for cond in CONDITIONS:
        pronoun_counts = {p: 0 for p in PRONOUNS}
        tail_counts = {t: 0 for t in TAIL_WORDS}
        lengths = []
        emoji_hits = []
        double_excl_hits = []
        chatml_hits = []
        placeholder_hits = []
        repetition_hits = []
        same_tail_repeat_hits = []
        items_with_kimi = 0
        total_items = 0

        for item in all_items:
            total_items += 1
            texts = all_texts_for_condition(item, cond)
            item_has_kimi = False
            for text in texts:
                lengths.append(len(text))
                for p in PRONOUNS:
                    c = text.count(p)
                    pronoun_counts[p] += c
                    if p == "キミ" and c > 0:
                        item_has_kimi = True
                for t in TAIL_WORDS:
                    tail_counts[t] += text.count(t)
                if cd.DECORATIVE_SYMBOLS_PATTERN.search(text):
                    emoji_hits.append((item["id"], text[:80]))
                if "！！" in text or "!!" in text:
                    double_excl_hits.append((item["id"], text[:80]))
                if cd.CHATML_TOKEN_PATTERN.search(text):
                    chatml_hits.append((item["id"], text[:80]))
                ph_match = PLACEHOLDER_PATTERN.search(text)
                if ph_match:
                    placeholder_hits.append((item["id"], ph_match.group(), text[:120]))
                if detect_repetition(text):
                    repetition_hits.append((item["id"], text[:120]))
                local = {}
                for pat in [
                    "だよ！", "だよ", "だよ〜", "だよね", "なんだ！", "なんだ",
                    "なんだよ", "だね！", "だね", "だぞ", "っ！", "よ！", "ね！",
                ]:
                    n = text.count(pat)
                    if n >= 3:
                        local[pat] = n
                if local:
                    same_tail_repeat_hits.append((item["id"], local))
            if item_has_kimi:
                items_with_kimi += 1

        n = len(lengths)
        stats[cond] = {
            "pronoun_counts": pronoun_counts,
            "tail_word_counts": tail_counts,
            "length_stats": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "avg": round(sum(lengths) / n, 1) if n else 0,
            },
            "emoji_hits": emoji_hits,
            "double_excl_hits": double_excl_hits,
            "chatml_hits": chatml_hits,
            "placeholder_hits": placeholder_hits,
            "repetition_hits": repetition_hits,
            "same_tail_repeat_hits_ge3": same_tail_repeat_hits,
            "items_with_kimi": items_with_kimi,
            "total_items": total_items,
            "kimi_item_rate_pct": round(items_with_kimi / max(total_items, 1) * 100, 2),
        }
    return stats


def followup_case_texts(results: dict) -> dict:
    """Q3/Q5/Q7/Q10/Q11/Q12/Q13/Q14 を3条件全文で並べて再確認用に抽出する。"""
    all_watch_ids = set(FOLLOWUP_IDS["regression_watch"]) | set(FOLLOWUP_IDS["problem_watch"])
    out = {}
    for r in results["structured_rag_eval"]:
        if r["id"] in all_watch_ids:
            out[r["id"]] = {
                "question": r["question"],
                "texts": {cond: r["results"][cond]["text"] for cond in CONDITIONS},
            }
    return out


def scan_percentage_omission(results: dict) -> dict:
    """Q3を名指しで、RAGコンテキスト中の全%数値がB/Cの回答に含まれているか確認する。"""
    q3 = next((r for r in results["structured_rag_eval"] if r["id"] == "Q3"), None)
    if q3 is None:
        return {"error": "Q3 not found in structured_rag_eval"}
    out = {}
    for cond in CONDITIONS:
        text = q3["results"][cond]["text"]
        out[cond] = {
            "text": text,
            "percent_value_count": len(re.findall(r"\d+(?:\.\d+)?%", text)),
        }
    return {"question": q3["question"], "by_condition": out}


def main() -> int:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    stats = analyze(results)
    followups = followup_case_texts(results)
    q3_check = scan_percentage_omission(results)

    out_path = EVAL_DIR / "abc_comparison_stats.json"
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    followup_path = EVAL_DIR / "abc_followup_cases.json"
    followup_path.write_text(
        json.dumps({"followup_cases": followups, "q3_percentage_check": q3_check}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n=== Followup cases (Q3/Q5/Q7/Q10/Q11/Q12/Q13/Q14) saved to abc_followup_cases.json ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
