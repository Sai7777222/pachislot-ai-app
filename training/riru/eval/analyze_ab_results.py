# ruff: noqa: E501
"""Phase 4E: A/B比較結果の定量分析。

training/riru/eval/ab_comparison_results.json を読み込み、
A(ベースQwen) / B(Qwen+リルLoRA) それぞれについて、
人称・語尾頻度・絵文字混入・連続感嘆符・語尾反復・ChatML混入・反復ループ・
回答長 などを集計する。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert_dataset as cd  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "ab_comparison_results.json"

PRONOUNS = ["私", "リル", "キミ"]
TAIL_WORDS = ["だよ", "なんだ", "だね", "だぞ"]


def all_texts_for_condition(item: dict, cond: str) -> list[str]:
    """1項目・1条件から生成テキストのリストを取り出す (単発 or マルチターン全部)。"""
    res = item["results"][cond]
    if item["type"] == "single":
        return [res["text"]]
    # multiturn: list of {"user":..., "assistant":..., "meta":...}
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
    stats = {"A_base": {}, "B_lora": {}}
    for cond in ("A_base", "B_lora"):
        pronoun_counts = {p: 0 for p in PRONOUNS}
        tail_counts = {t: 0 for t in TAIL_WORDS}
        lengths = []
        emoji_hits = []
        double_excl_hits = []
        chatml_hits = []
        repetition_hits = []
        same_tail_repeat_hits = []
        items_with_kimi = 0
        total_items = 0

        for item in results["character_eval"] + [
            {**r, "type": "single", "results": {c: r["results"][c] for c in ("A_base", "B_lora")}}
            for r in results["structured_rag_eval"]
        ]:
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
            "repetition_hits": repetition_hits,
            "same_tail_repeat_hits_ge3": same_tail_repeat_hits,
            "items_with_kimi": items_with_kimi,
            "total_items": total_items,
        }
    return stats


def main() -> int:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    stats = analyze(results)
    out_path = EVAL_DIR / "ab_comparison_stats.json"
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
