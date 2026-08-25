# ruff: noqa: E501
"""Phase 4L: comprehensive_results.json の定量・定性分析。

- Q3 5-seed分析 (relevant recall, %出現状況)
- Phase4I held-out (Q3+P01-P10) 分析
- 既存17問の一般統計 + hallucination兆候検出
- キャラクター39問の一般統計
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert_dataset as cd  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "phase4l_comprehensive_results.json"

CONDITIONS = ("A_base", "B_v2", "C_v4", "D_v3")

PLACEHOLDER_PATTERN = re.compile(
    r"(〜{3,}|ー{4,}|X{2,}(?![0-9A-Za-z])|x{2,}|○{2,}|●{2,}|TBD|TODO|\.{4,}|…{2,}|"
    r"[■□▲△▼▽◆◇]{2,}|�)"
)
# 「私は〜〜だよ」のような、主語の直後に説明内容が無く記号だけで終わる
# 未完成文パターン (E36で v3/v4 に再発を確認したため追加検出)。
INCOMPLETE_PREDICATE_PATTERN = re.compile(r"(私は|僕は|リルは)[〜ー]{1,3}(だよ|なんだ|だね)")
SUSPICIOUS_PHRASE_PATTERNS = [
    "倍だ", "倍だよ", "倍になる", "ポイント高い", "ポイント上", "ポイントだ",
    "だと思う", "かもしれない", "した方がいい", "やめ時", "ヤメ時", "おすすめ",
    "覚えておくと", "参考になりそう", "見極める", "検討しましょう",
]

Q3_GROUND_TRUTH = {
    "key_facts": ["510G", "1000G", "1480G", "15.2%", "20.3%", "64.5%"],
    "irrelevant_markers": ["裏天国", "天空の扉", "雷雨", "0テンパイ"],
}


def load_holdout_ground_truth() -> dict:
    """Phase 4Iのheld-outファイル(P01〜P10)自身が持つkey_facts/irrelevant_markersを
    読み取り専用で使用する (ファイル自体は変更しない)。"""
    path = EVAL_DIR / "phase4i_holdout_omission_v2.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["id"]: {"key_facts": item["key_facts"], "irrelevant_markers": item["irrelevant_markers"]}
        for item in items
    }


HOLDOUT_GROUND_TRUTH = load_holdout_ground_truth()


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


def analyze_q3_multiseed(results: dict) -> dict:
    data = results["q3_multiseed"]
    out = {}
    for cond in CONDITIONS:
        per_seed = {}
        recalls = []
        pct_appear_count = 0
        gamecount_appear_count = 0
        for seed, gen in data[cond].items():
            text = gen["text"]
            found = [k for k in Q3_GROUND_TRUTH["key_facts"] if k in text]
            recall = round(len(found) / len(Q3_GROUND_TRUTH["key_facts"]) * 100, 1)
            recalls.append(recall)
            has_pct = any(p in text for p in ("15.2%", "20.3%", "64.5%"))
            has_gamecount = any(g in text for g in ("510G", "1000G", "1480G"))
            if has_pct:
                pct_appear_count += 1
            if has_gamecount:
                gamecount_appear_count += 1
            per_seed[seed] = {
                "text": text,
                "key_facts_found": found,
                "recall_pct": recall,
                "has_any_percentage": has_pct,
                "has_any_gamecount": has_gamecount,
            }
        out[cond] = {
            "per_seed": per_seed,
            "avg_recall_pct": round(sum(recalls) / len(recalls), 1),
            "min_recall_pct": min(recalls),
            "max_recall_pct": max(recalls),
            "seeds_with_any_percentage": f"{pct_appear_count}/{len(data[cond])}",
            "seeds_with_any_gamecount": f"{gamecount_appear_count}/{len(data[cond])}",
        }
    return out


def analyze_holdout_11(results: dict) -> dict:
    data = results["holdout_11"]
    out = {}
    condition_totals = {c: {"recall_sum": 0.0, "n": 0, "irr_hits": 0, "irr_total": 0} for c in CONDITIONS}
    for qid, item in data.items():
        gt = Q3_GROUND_TRUTH if qid == "Q3" else HOLDOUT_GROUND_TRUTH.get(qid)
        if gt is None:
            continue
        out[qid] = {"question": item["question"], "by_condition": {}}
        for cond in CONDITIONS:
            text = item["conditions"][cond]["text"]
            found = [k for k in gt["key_facts"] if k in text]
            irr_found = [k for k in gt["irrelevant_markers"] if k in text]
            recall = round(len(found) / len(gt["key_facts"]) * 100, 1)
            out[qid]["by_condition"][cond] = {
                "text": text,
                "key_facts_found": found,
                "key_facts_missing": [k for k in gt["key_facts"] if k not in found],
                "recall_pct": recall,
                "irrelevant_leaked": irr_found,
            }
            condition_totals[cond]["recall_sum"] += recall
            condition_totals[cond]["n"] += 1
            condition_totals[cond]["irr_hits"] += len(irr_found)
            condition_totals[cond]["irr_total"] += len(gt["irrelevant_markers"])
    summary = {
        cond: {
            "avg_recall_pct": round(t["recall_sum"] / max(t["n"], 1), 1),
            "irrelevant_leak_rate_pct": round(t["irr_hits"] / max(t["irr_total"], 1) * 100, 1),
        }
        for cond, t in condition_totals.items()
    }
    return {"per_question": out, "condition_summary": summary}


def analyze_17q(results: dict) -> dict:
    data = results["structured_17q"]
    stats = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        lengths = []
        repetition_hits = []
        placeholder_hits = []
        suspicious_hits = []
        for qid, item in data.items():
            text = item["conditions"][cond]["text"]
            lengths.append(len(text))
            if detect_repetition(text):
                repetition_hits.append(qid)
            ph = PLACEHOLDER_PATTERN.search(text)
            if ph:
                placeholder_hits.append({"qid": qid, "matched": ph.group()})
            for phrase in SUSPICIOUS_PHRASE_PATTERNS:
                if phrase in text:
                    suspicious_hits.append({"qid": qid, "phrase": phrase, "text": text[:150]})
        n = len(lengths)
        stats[cond] = {
            "length_stats": {
                "min": min(lengths), "max": max(lengths),
                "mean": round(sum(lengths) / n, 1),
            },
            "repetition_hits": repetition_hits,
            "placeholder_hits": placeholder_hits,
            "suspicious_hallucination_phrase_hits": suspicious_hits,
        }
    return stats


def analyze_character_39(results: dict) -> dict:
    data = results["character_39"]
    stats = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        pronoun_counts = {"私": 0, "リル": 0, "キミ": 0}
        tail_counts = {"だよ": 0, "なんだ": 0, "だね": 0, "だぞ": 0}
        lengths = []
        emoji_hits = 0
        excl_hits = 0
        chatml_hits = 0
        placeholder_hits = []
        incomplete_predicate_hits = []
        repetition_hits = []
        same_tail_hits = []
        for eid, item in data.items():
            cond_res = item["conditions"][cond]
            texts = [cond_res["text"]] if item["type"] == "single" else [t["assistant"] for t in cond_res]
            for text in texts:
                lengths.append(len(text))
                for p in pronoun_counts:
                    pronoun_counts[p] += text.count(p)
                for t in tail_counts:
                    tail_counts[t] += text.count(t)
                if cd.DECORATIVE_SYMBOLS_PATTERN.search(text) or "♪" in text:
                    emoji_hits += 1
                if cd.REPEATED_EXCLAMATION_PATTERN.search(text):
                    excl_hits += 1
                if cd.CHATML_TOKEN_PATTERN.search(text):
                    chatml_hits += 1
                ph = PLACEHOLDER_PATTERN.search(text)
                if ph:
                    placeholder_hits.append({"eid": eid, "matched": ph.group(), "text": text[:80]})
                ip = INCOMPLETE_PREDICATE_PATTERN.search(text)
                if ip:
                    incomplete_predicate_hits.append({"eid": eid, "matched": ip.group(), "text": text[:80]})
                if detect_repetition(text):
                    repetition_hits.append(eid)
                local = {}
                for pat in ["だよ", "なんだ", "だね", "だぞ", "っ！", "よ！"]:
                    c = text.count(pat)
                    if c >= 3:
                        local[pat] = c
                if local:
                    same_tail_hits.append({"eid": eid, "counts": local})
        n = len(lengths)
        stats[cond] = {
            "pronoun_counts": pronoun_counts,
            "tail_word_counts": tail_counts,
            "length_stats": {"min": min(lengths), "max": max(lengths), "mean": round(sum(lengths) / n, 1)},
            "emoji_hits": emoji_hits,
            "double_excl_hits": excl_hits,
            "chatml_hits": chatml_hits,
            "placeholder_hits": placeholder_hits,
            "incomplete_predicate_hits": incomplete_predicate_hits,
            "repetition_hits": repetition_hits,
            "same_tail_repeat_hits_ge3": same_tail_hits,
            "total_items": len(data),
        }
    return stats


def main() -> int:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    q3_analysis = analyze_q3_multiseed(results)
    holdout_analysis = analyze_holdout_11(results)
    q17_analysis = analyze_17q(results)
    char_analysis = analyze_character_39(results)

    (EVAL_DIR / "phase4l_q3_multiseed_analysis.json").write_text(
        json.dumps(q3_analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (EVAL_DIR / "phase4l_holdout11_analysis.json").write_text(
        json.dumps(holdout_analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (EVAL_DIR / "phase4l_17q_analysis.json").write_text(
        json.dumps(q17_analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (EVAL_DIR / "phase4l_character39_analysis.json").write_text(
        json.dumps(char_analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "q3_recall_summary": {c: {"avg": q3_analysis[c]["avg_recall_pct"], "min": q3_analysis[c]["min_recall_pct"], "max": q3_analysis[c]["max_recall_pct"], "pct_seeds": q3_analysis[c]["seeds_with_any_percentage"]} for c in CONDITIONS},
        "holdout11_summary": holdout_analysis["condition_summary"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
