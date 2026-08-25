# ruff: noqa: E501
"""Phase 4I-6/7: kimi_temperature実験結果の定量分析。

- temperature別「キミ」使用率 (肯定文脈7問 x 3 seed)
- 対照問題 (温度0.7のみ) での「キミ」漏出率
- prompt指示あり/なしの比較
- 語尾崩れ・反復・回答長の簡易チェック
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert_dataset as cd  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "phase4i_kimi_temperature_results.json"

TEMPERATURES = ("0.3", "0.5", "0.7", "0.9")


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


def analyze_positive(results: dict) -> dict:
    out = {}
    for prompt_name, items in results["positive"].items():
        out[prompt_name] = {}
        for temp in TEMPERATURES:
            total = 0
            kimi_used = 0
            mechanical_flags = 0
            repetition_flags = 0
            emoji_flags = 0
            excl_flags = 0
            chatml_flags = 0
            lengths = []
            for _item_id, item_data in items.items():
                gens = item_data["by_temperature"].get(temp, [])
                for g in gens:
                    total += 1
                    lengths.append(len(g["text"]))
                    if g["kimi_count"] > 0:
                        kimi_used += 1
                    if g.get("mechanical_start_flag"):
                        mechanical_flags += 1
                    if detect_repetition(g["text"]):
                        repetition_flags += 1
                    if cd.DECORATIVE_SYMBOLS_PATTERN.search(g["text"]) or "♪" in g["text"]:
                        emoji_flags += 1
                    if cd.REPEATED_EXCLAMATION_PATTERN.search(g["text"]):
                        excl_flags += 1
                    if cd.CHATML_TOKEN_PATTERN.search(g["text"]):
                        chatml_flags += 1
            n = len(lengths)
            out[prompt_name][temp] = {
                "total_generations": total,
                "kimi_used_count": kimi_used,
                "kimi_used_rate_pct": round(kimi_used / max(total, 1) * 100, 1),
                "mechanical_kimi_start_count": mechanical_flags,
                "repetition_count": repetition_flags,
                "emoji_hits": emoji_flags,
                "double_excl_hits": excl_flags,
                "chatml_hits": chatml_flags,
                "avg_length": round(sum(lengths) / n, 1) if n else 0,
            }
    return out


def analyze_control(results: dict) -> dict:
    out = {}
    for prompt_name, items in results["control"].items():
        total = 0
        kimi_leaked = 0
        repetition_flags = 0
        lengths = []
        for _item_id, item_data in items.items():
            for g in item_data["generations"]:
                total += 1
                lengths.append(len(g["text"]))
                if g["kimi_count"] > 0:
                    kimi_leaked += 1
                if detect_repetition(g["text"]):
                    repetition_flags += 1
        n = len(lengths)
        out[prompt_name] = {
            "total_generations": total,
            "kimi_leaked_count": kimi_leaked,
            "kimi_leak_rate_pct": round(kimi_leaked / max(total, 1) * 100, 1),
            "repetition_count": repetition_flags,
            "avg_length": round(sum(lengths) / n, 1) if n else 0,
        }
    return out


def collect_all_kimi_texts(results: dict) -> list[dict]:
    """「キミ」が実際に出現した生成をすべて抽出する (自然さの目視確認用)。"""
    hits = []
    for prompt_name, items in results["positive"].items():
        for item_id, item_data in items.items():
            for temp, gens in item_data["by_temperature"].items():
                for g in gens:
                    if g["kimi_count"] > 0:
                        hits.append(
                            {
                                "prompt": prompt_name,
                                "item_id": item_id,
                                "temperature": temp,
                                "seed": g["seed"],
                                "text": g["text"],
                            }
                        )
    for prompt_name, items in results["control"].items():
        for item_id, item_data in items.items():
            for g in item_data["generations"]:
                if g["kimi_count"] > 0:
                    hits.append(
                        {
                            "prompt": prompt_name,
                            "item_id": item_id,
                            "temperature": "0.7 (control)",
                            "seed": g["seed"],
                            "text": g["text"],
                        }
                    )
    return hits


def main() -> int:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    positive_stats = analyze_positive(results)
    control_stats = analyze_control(results)
    all_kimi_texts = collect_all_kimi_texts(results)

    out = {
        "positive_by_temperature": positive_stats,
        "control_leak_at_temp0.7": control_stats,
        "all_generations_containing_kimi": all_kimi_texts,
        "note": "「キミ」を含む生成が0件の場合、all_generations_containing_kimiは空配列になる。",
    }
    (EVAL_DIR / "phase4i_kimi_temperature_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved")
    print(json.dumps({"positive_by_temperature": positive_stats, "control_leak_at_temp0.7": control_stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
