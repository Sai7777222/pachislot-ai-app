# ruff: noqa: E501
"""Phase 4J: v2学習データ897件 (riru_lora_v2_candidate.jsonl) の監査スクリプト。

QLoRA/LoRA学習は行わない。データファイルの読み取りと分析のみ。
riru_lora_v2_candidate.jsonl / riru_train_v2.jsonl / riru_val_v2.jsonl 等の
既存ファイルは一切変更しない。

出力: training/riru/reports/phase4j_dataset_audit.json (全量集計)
      training/riru/reports/phase4j_h_category_review.json (Hカテゴリ16件個別評価)
      training/riru/reports/phase4j_q3_similar_examples.json (Q3類似度ランキング)

【自動推定の限界について】
本スクリプトの「事実数」「情報保持率」は、数値・候補パターン(正規表現)に基づく
機械的な推定である。意味的に等価だが表記が異なる場合 (例:「およそ」と「約」)は
過小評価されうる。個別の重要判定 (H・held-out類似度・良い/悪い教師例) は
このスクリプトの後、目視で改めて判断する。
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = TRAINING_ROOT / "processed"
REPORTS_DIR = TRAINING_ROOT / "reports"
CANDIDATE_PATH = PROCESSED_DIR / "riru_lora_v2_candidate.jsonl"

# ---------------------------------------------------------------------------
# 事実(数値・候補)抽出パターン (自動推定用)
# ---------------------------------------------------------------------------
NUMERIC_TOKEN_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*%"          # パーセント
    r"|1\s*/\s*\d+(?:\.\d+)?"     # 分数 (1/295等)
    r"|\d+\s*G(?![a-zA-Z])"       # ゲーム数
    r"|\d+\s*枚"                  # 枚数
    r"|\d+(?:\.\d+)?\s*倍"        # 倍率
    r"|\d+\s*種類"                # 種類数
    r"|\d+\s*回(?!転)"            # 回数 (「回転」は除く誤検知回避)
)

# 短文化を示唆する表現 (完全一致・部分一致の両方を検索)
SHORTENING_PHRASES = [
    "短く答える", "簡潔に", "一言で", "ざっくり", "要するに", "まとめると",
    "端的に", "長く説明しない", "必要最低限", "詳細を省く", "手短に",
    "簡単に言うと", "かいつまんで", "要点だけ",
]

# 列挙・比較を要求するキーワード (userテキスト側)
ENUMERATION_KEYWORDS = ["それぞれ", "全部", "詳しく", "一覧", "比較", "違い"]


def load_records(path: Path = CANDIDATE_PATH) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def origin_of(record: dict) -> str:
    meta = record["metadata"]
    src = meta.get("source")
    if src == "phase4b_generated":
        return f"phase4b:{meta.get('category', '?')}"
    if src == "phase4f_generated":
        return f"phase4f:{meta.get('category', '?')}"
    if meta.get("legacy"):
        return "legacy_523"
    return "unknown"


def origin_group(record: dict) -> str:
    meta = record["metadata"]
    if meta.get("legacy"):
        return "legacy_523"
    if meta.get("source") == "phase4b_generated":
        return "phase4b_300"
    if meta.get("source") == "phase4f_generated":
        return "phase4f_74"
    return "unknown"


def all_user_text(record: dict) -> str:
    return "\n".join(m["content"] for m in record["messages"] if m["role"] == "user")


def all_assistant_text(record: dict) -> str:
    return "\n".join(m["content"] for m in record["messages"] if m["role"] == "assistant")


def assistant_turns(record: dict) -> list[str]:
    return [m["content"] for m in record["messages"] if m["role"] == "assistant"]


# ---------------------------------------------------------------------------
# 4J-2: 回答長分布
# ---------------------------------------------------------------------------
LENGTH_BUCKETS = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 120), (121, 10**9)]


def length_bucket_label(n: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi if hi < 10**9 else '121+'}"
    return "unknown"


def percentile(sorted_vals: list[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def analyze_length_distribution(records: list[dict]) -> dict:
    lengths = []
    for r in records:
        for text in assistant_turns(r):
            lengths.append(len(text))
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    bucket_counts: Counter[str] = Counter()
    for length in lengths:
        bucket_counts[length_bucket_label(length)] += 1

    overall = {
        "assistant_response_total": n,
        "min": lengths_sorted[0] if n else 0,
        "max": lengths_sorted[-1] if n else 0,
        "mean": round(statistics.mean(lengths_sorted), 1) if n else 0,
        "median": round(statistics.median(lengths_sorted), 1) if n else 0,
        "p25": round(percentile(lengths_sorted, 0.25), 1),
        "p50": round(percentile(lengths_sorted, 0.50), 1),
        "p75": round(percentile(lengths_sorted, 0.75), 1),
        "p90": round(percentile(lengths_sorted, 0.90), 1),
        "p95": round(percentile(lengths_sorted, 0.95), 1),
        "bucket_counts": dict(bucket_counts),
        "bucket_pct": {k: round(v / n * 100, 1) for k, v in bucket_counts.items()} if n else {},
    }

    # カテゴリ別 (record単位の代表長=assistant全結合の長さ)
    by_category: dict[str, list[int]] = defaultdict(list)
    by_origin: dict[str, list[int]] = defaultdict(list)
    for r in records:
        rec_len = len(all_assistant_text(r))
        cat = r["metadata"].get("category", "legacy_or_unknown")
        by_category[cat].append(rec_len)
        by_origin[origin_group(r)].append(rec_len)

    def summarize(bucket: dict[str, list[int]]) -> dict:
        out = {}
        for k, vals in bucket.items():
            out[k] = {
                "count": len(vals),
                "mean": round(statistics.mean(vals), 1) if vals else 0,
                "median": round(statistics.median(vals), 1) if vals else 0,
            }
        return out

    return {
        "overall": overall,
        "by_category": summarize(by_category),
        "by_origin_group": summarize(by_origin),
    }


# ---------------------------------------------------------------------------
# 4J-3/4J-5: 情報量圧縮監査 (数値ベース、自動推定)
# ---------------------------------------------------------------------------


REFERENCE_INFO_PREFIX_PATTERN = re.compile(r"参照情報[：:]\s*(.+?)(?:\n|$)")


def extract_facts(text: str) -> list[str]:
    """「参照情報：A、B、C」形式の場合は「、」区切りの各節を1事実として扱い、
    節内に数値があればその数値を事実として使う (paraphrase後も一致しやすいため)。
    数値が無い節 (例:「天井到達でAT確定」) は節そのものを定性的事実として扱う
    (この場合、assistant側が言い換えていると一致しないため過小評価されうる。
    レポートではこの限界を明記する)。
    「参照情報：」形式でない場合は、テキスト全体から数値パターンのみを拾う。
    """
    ref_match = REFERENCE_INFO_PREFIX_PATTERN.search(text)
    if ref_match:
        clauses = [c.strip() for c in re.split("[、,]", ref_match.group(1)) if c.strip()]
        facts = []
        for clause in clauses:
            nums = NUMERIC_TOKEN_PATTERN.findall(clause)
            if nums:
                facts.extend(nums)
            else:
                facts.append(clause)
        return facts
    return NUMERIC_TOKEN_PATTERN.findall(text)


def classify_compression(input_facts: list[str], output_facts: list[str], retention: float) -> str:
    """A〜Gの粗い自動分類 (最終判断は人間レビュー推奨、ここではヒューリスティック)。"""
    n_in = len(input_facts)
    if n_in == 0:
        return "N/A(数値なし)"
    if retention >= 90:
        return "A(ほぼ全部保持,自動推定)"
    if retention >= 60:
        return "B(軽微な省略,自動推定)"
    # 両端(最初と最後)だけ残しているかを確認
    in_set_ordered = list(dict.fromkeys(input_facts))  # 順序保持・重複除去
    out_set = set(output_facts)
    if n_in >= 3 and len(out_set) <= 2:
        endpoints = {in_set_ordered[0], in_set_ordered[-1]}
        if out_set and out_set.issubset(endpoints):
            return "C(代表値/両端のみ,自動推定)"
        return "D(1〜2個のみ保持,自動推定)"
    if retention == 0:
        return "E/F(数値ゼロ,要約化の疑い,自動推定→要目視)"
    return "D(一部のみ保持,自動推定)"


def analyze_information_compression(records: list[dict]) -> dict:
    per_record = []
    for i, r in enumerate(records):
        user_text = all_user_text(r)
        assistant_text = all_assistant_text(r)
        input_facts = extract_facts(user_text)
        if len(input_facts) < 2:
            continue
        input_set_ordered = list(dict.fromkeys(input_facts))
        matched = [f for f in input_set_ordered if f in assistant_text]
        retention = round(len(matched) / len(input_set_ordered) * 100, 1)
        per_record.append(
            {
                "index": i,
                "category": r["metadata"].get("category", "legacy_or_unknown"),
                "origin": origin_group(r),
                "input_fact_count": len(input_set_ordered),
                "output_fact_count": len(matched),
                "input_facts": input_set_ordered,
                "output_facts_matched": matched,
                "retention_ratio_pct": retention,
                "auto_classification": classify_compression(input_set_ordered, matched, retention),
                "user_text": user_text,
                "assistant_text": assistant_text,
            }
        )

    class_counts = Counter(x["auto_classification"] for x in per_record)

    # 数値数別の平均保持率 (4J-5)
    by_count_bucket: dict[str, list[float]] = defaultdict(list)
    for x in per_record:
        n = x["input_fact_count"]
        if n == 2:
            key = "2個"
        elif n == 3:
            key = "3個"
        elif n == 4:
            key = "4個"
        else:
            key = "5個以上"
        by_count_bucket[key].append(x["retention_ratio_pct"])

    retention_by_count = {
        k: {
            "n_records": len(v),
            "avg_retention_pct": round(statistics.mean(v), 1) if v else 0,
        }
        for k, v in by_count_bucket.items()
    }

    return {
        "note": "input_fact_count>=2 (userテキストに数値的事実が2個以上出現) のレコードのみを対象とする自動推定。",
        "total_multi_fact_records": len(per_record),
        "auto_classification_counts": dict(class_counts),
        "retention_by_input_count": retention_by_count,
        "per_record": per_record,
    }


# ---------------------------------------------------------------------------
# 4J-6: 列挙・比較データの監査
# ---------------------------------------------------------------------------


def analyze_enumeration_records(records: list[dict]) -> dict:
    hits = []
    for i, r in enumerate(records):
        user_text = all_user_text(r)
        assistant_text = all_assistant_text(r)
        matched_keywords = [kw for kw in ENUMERATION_KEYWORDS if kw in user_text]
        if not matched_keywords:
            continue
        input_facts = list(dict.fromkeys(extract_facts(user_text)))
        matched = [f for f in input_facts if f in assistant_text]
        hits.append(
            {
                "index": i,
                "category": r["metadata"].get("category", "legacy_or_unknown"),
                "keywords_matched": matched_keywords,
                "input_fact_count": len(input_facts),
                "output_fact_count_matched": len(matched),
                "fully_answered": (len(input_facts) == 0) or (len(matched) == len(input_facts)),
                "user_text": user_text,
                "assistant_text": assistant_text,
            }
        )
    partial_only = [h for h in hits if h["input_fact_count"] > 0 and not h["fully_answered"]]
    return {
        "total_enumeration_keyword_records": len(hits),
        "partial_answer_despite_enumeration_keyword": len(partial_only),
        "partial_examples": partial_only,
    }


# ---------------------------------------------------------------------------
# 4J-7: 短文化を教えている表現の探索
# ---------------------------------------------------------------------------


def analyze_shortening_language(records: list[dict]) -> dict:
    direct_hits = []
    for i, r in enumerate(records):
        full_text = all_user_text(r) + "\n" + all_assistant_text(r)
        matched = [p for p in SHORTENING_PHRASES if p in full_text]
        if matched:
            direct_hits.append({"index": i, "matched_phrases": matched})

    # 「広い質問 + 極端に短い回答」の抽出 (質問が10文字以上 かつ 回答が15文字以下)
    broad_q_short_a = []
    for i, r in enumerate(records):
        for m_i in range(len(r["messages"]) - 1):
            if r["messages"][m_i]["role"] == "user" and r["messages"][m_i + 1]["role"] == "assistant":
                u = r["messages"][m_i]["content"]
                a = r["messages"][m_i + 1]["content"]
                if len(u) >= 10 and len(a) <= 15:
                    broad_q_short_a.append(
                        {"index": i, "category": r["metadata"].get("category", "legacy_or_unknown"), "user": u, "assistant": a}
                    )

    return {
        "explicit_shortening_instruction_hits": len(direct_hits),
        "explicit_shortening_instruction_examples": direct_hits[:20],
        "broad_question_extremely_short_answer_count": len(broad_q_short_a),
        "broad_question_extremely_short_answer_examples": broad_q_short_a[:30],
    }


# ---------------------------------------------------------------------------
# 4J-12: 回答長と情報保持率の相関
# ---------------------------------------------------------------------------


def analyze_length_vs_retention(compression_result: dict) -> dict:
    buckets = {"1-20": [], "21-40": [], "41-60": [], "61+": []}
    for x in compression_result["per_record"]:
        alen = len(x["assistant_text"])
        if alen <= 20:
            key = "1-20"
        elif alen <= 40:
            key = "21-40"
        elif alen <= 60:
            key = "41-60"
        else:
            key = "61+"
        buckets[key].append(x["retention_ratio_pct"])
    return {
        k: {
            "n_records": len(v),
            "avg_retention_pct": round(statistics.mean(v), 1) if v else None,
        }
        for k, v in buckets.items()
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    records = load_records()
    assert len(records) == 897, f"unexpected record count: {len(records)}"

    length_dist = analyze_length_distribution(records)
    compression = analyze_information_compression(records)
    enumeration = analyze_enumeration_records(records)
    shortening = analyze_shortening_language(records)
    length_vs_retention = analyze_length_vs_retention(compression)

    # per_recordはサイズが大きいので、集計本体には要約のみ残し、詳細は別ファイルに分離
    compression_summary = {k: v for k, v in compression.items() if k != "per_record"}

    report = {
        "total_records": len(records),
        "length_distribution": length_dist,
        "information_compression_summary": compression_summary,
        "enumeration_records": {k: v for k, v in enumeration.items() if k != "partial_examples"},
        "enumeration_partial_examples": enumeration["partial_examples"],
        "shortening_language": shortening,
        "length_vs_retention": length_vs_retention,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "phase4j_dataset_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "phase4j_compression_detail.json").write_text(
        json.dumps(compression["per_record"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Saved phase4j_dataset_audit.json and phase4j_compression_detail.json")
    print(f"total_records={len(records)}")
    print(f"multi_fact_records={compression['total_multi_fact_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
