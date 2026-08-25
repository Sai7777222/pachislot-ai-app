# ruff: noqa: E501
"""Phase 4H-7: A(base)/B(v1)/C(v2)/D(v3) 4者比較結果の定量分析。

training/riru/eval/abcd_comparison_results.json を読み込み、各条件について
人称・語尾頻度・絵文字混入・連続感嘆符・語尾反復・ChatML混入・プレースホルダー・
反復ループ・回答長 などを集計する (character_eval + structured_rag_eval)。

加えて:
  - kimi_eval_v2 (12問): 「キミ」自然使用テスト(7問)・非使用対照テスト(5問)の
    条件別使用率を集計する。
  - Q3 (既存構造化RAG) + omission_eval_v2 (5問、学習データとは異なる架空値・
    トピックの held-out テスト) について、質問対象に直接関連する重要事実
    (key_facts) の網羅率と、無関係な情報 (irrelevant_markers) の混入率を
    条件別に定量集計する。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert_dataset as cd  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "abcd_comparison_results.json"

PRONOUNS = ["私", "リル", "キミ"]
TAIL_WORDS = ["だよ", "なんだ", "だね", "だぞ"]
CONDITIONS = ("A_base", "B_v1", "C_v2", "D_v3")

PLACEHOLDER_PATTERN = re.compile(
    r"(〜{3,}|ー{4,}|X{2,}|x{2,}|○{2,}|●{2,}|TBD|TODO|\.{4,}|…{2,}|"
    r"[■□▲△▼▽◆◇]{2,}|�)"
)

# Q3型「重要情報網羅率／不要情報混入率」の定量評価用グラウンドトゥルース。
# key_facts: 質問に直接関連し、省略すべきでない事実 (RAGコンテキスト中の値)。
# irrelevant_markers: 同じRAGコンテキスト中にあるが、質問には無関係な情報
# (回答に含まれていたら「不要情報の混入」とみなす)。
OMISSION_GROUND_TRUTH = {
    "Q3": {
        "key_facts": ["510G", "1000G", "1480G", "15.2%", "20.3%", "64.5%"],
        "irrelevant_markers": ["裏天国", "天空の扉", "雷雨", "0テンパイ"],
    },
    "O01": {
        "key_facts": ["1/420", "1/260"],
        "irrelevant_markers": ["ゾーンδ", "1/70"],
    },
    "O02": {
        "key_facts": ["0テンパイ", "ゾーンε", "偶数テンパイ"],
        "irrelevant_markers": ["引き戻し", "30%"],
    },
    "O03": {
        "key_facts": ["30%", "40%"],
        "irrelevant_markers": ["0テンパイ", "ゾーンε"],
    },
    "O04": {
        "key_facts": ["AT確定", "ゾーンζ", "250枚"],
        "irrelevant_markers": ["前兆出現率", "設定判別"],
    },
    "O05": {
        "key_facts": ["3枚/G", "6枚/G"],
        "irrelevant_markers": ["ゾーンζ", "15G"],
    },
}


def all_texts_for_condition(item: dict, cond: str) -> list[str]:
    res = item["results"][cond]
    if item["type"] == "single":
        return [res["text"]]
    return [turn["assistant"] for turn in res]


def detect_repetition(text: str) -> bool:
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


def analyze_general(results: dict) -> dict:
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
        }
    return stats


def analyze_kimi_v2(results: dict) -> dict:
    items = results["kimi_eval_v2"]
    positive_items = [i for i in items if i["category"].startswith("kimi_positive")]
    control_items = [i for i in items if i["category"].startswith("kimi_control")]

    out = {}
    for cond in CONDITIONS:
        pos_hits = sum(1 for i in positive_items if "キミ" in i["results"][cond]["text"])
        ctrl_hits = sum(1 for i in control_items if "キミ" in i["results"][cond]["text"])
        out[cond] = {
            "positive_context_items": len(positive_items),
            "positive_context_kimi_used": pos_hits,
            "positive_context_kimi_rate_pct": round(pos_hits / max(len(positive_items), 1) * 100, 1),
            "control_context_items": len(control_items),
            "control_context_kimi_used": ctrl_hits,
            "control_context_kimi_leak_rate_pct": round(ctrl_hits / max(len(control_items), 1) * 100, 1),
        }
    return out


def analyze_omission(results: dict) -> dict:
    """Q3 + omission_eval_v2(O01-O05) について、条件別の重要情報網羅率／
    不要情報混入率を算出する。
    """
    all_omission_items = [r for r in results["structured_rag_eval"] if r["id"] == "Q3"]
    all_omission_items += results["omission_eval_v2"]

    per_item = {}
    condition_totals = {
        c: {"key_fact_hits": 0, "key_fact_total": 0, "irrelevant_hits": 0, "irrelevant_total": 0}
        for c in CONDITIONS
    }

    for item in all_omission_items:
        qid = item["id"]
        gt = OMISSION_GROUND_TRUTH.get(qid)
        if gt is None:
            continue
        per_item[qid] = {"question": item["question"]}
        for cond in CONDITIONS:
            text = item["results"][cond]["text"]
            key_hits = [k for k in gt["key_facts"] if k in text]
            irr_hits = [k for k in gt["irrelevant_markers"] if k in text]
            per_item[qid][cond] = {
                "text": text,
                "key_facts_found": key_hits,
                "key_facts_missing": [k for k in gt["key_facts"] if k not in text],
                "key_fact_coverage_pct": round(len(key_hits) / len(gt["key_facts"]) * 100, 1),
                "irrelevant_markers_leaked": irr_hits,
            }
            condition_totals[cond]["key_fact_hits"] += len(key_hits)
            condition_totals[cond]["key_fact_total"] += len(gt["key_facts"])
            condition_totals[cond]["irrelevant_hits"] += len(irr_hits)
            condition_totals[cond]["irrelevant_total"] += len(gt["irrelevant_markers"])

    summary = {}
    for cond in CONDITIONS:
        t = condition_totals[cond]
        summary[cond] = {
            "key_fact_coverage_rate_pct": round(t["key_fact_hits"] / max(t["key_fact_total"], 1) * 100, 1),
            "irrelevant_leak_rate_pct": round(t["irrelevant_hits"] / max(t["irrelevant_total"], 1) * 100, 1),
        }

    return {"per_item": per_item, "condition_summary": summary}


def main() -> int:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    general_stats = analyze_general(results)
    kimi_stats = analyze_kimi_v2(results)
    omission_stats = analyze_omission(results)

    (EVAL_DIR / "abcd_comparison_stats.json").write_text(
        json.dumps(general_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (EVAL_DIR / "abcd_kimi_v2_stats.json").write_text(
        json.dumps(kimi_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (EVAL_DIR / "abcd_omission_stats.json").write_text(
        json.dumps(omission_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== general stats ===")
    print(json.dumps(general_stats, ensure_ascii=False, indent=2))
    print("=== kimi v2 stats ===")
    print(json.dumps(kimi_stats, ensure_ascii=False, indent=2))
    print("=== omission condition summary ===")
    print(json.dumps(omission_stats["condition_summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
