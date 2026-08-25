"""Phase 4R: 教師データ Fact Retention / Information Selection 監査。

riru_train_v4.jsonl(823件) / riru_val_v4.jsonl(91件) / riru_lora_v4_candidate.jsonl(914件)
の各レコードについて、user発話に埋め込まれたcontext(RAG的構造化データ・参照情報・解説文)
からdeterministicにfactを抽出し、質問との関連性(relevance)を判定した上で、
assistant教師回答がどれだけそのfactを保持しているか(retention)を定量化する。

学習・adapter変更・データ変更は一切行わない。分析専用の読み取りのみ。

=== データ形式 (実データから確認済み) ===
各レコード: {"messages": [{"role":"user","content":...}, {"role":"assistant","content":...}, ...],
             "metadata": {"source": ..., "category": ..., "category_code": ..., ...}}
RAG的contextはuser contentの中に直接埋め込まれている(system roleは存在しない)。
2つの主要形式:
  (1) 簡易形式: "参照情報：A、B、C\n<question>"
  (2) Phase4K構造化形式: "【対象機種】...\n【構造化データ】\n- [label] key: value\n...
       【関連する解説文章】\n◆ 見出し（出典カテゴリ: xxx）\n本文\n...\n<question>"
questionはuser contentの最終行として抽出する(全レコードで概ね一貫していることを確認済み)。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = TRAINING_ROOT / "processed"
REPORTS_DIR = TRAINING_ROOT / "reports"

TRAIN_PATH = PROCESSED_DIR / "riru_train_v4.jsonl"
VAL_PATH = PROCESSED_DIR / "riru_val_v4.jsonl"
CANDIDATE_PATH = PROCESSED_DIR / "riru_lora_v4_candidate.jsonl"

# --- deterministic fact patterns ---
PCT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*[%％]")
GAME_PATTERN = re.compile(r"\d+(?:,\d{3})*\s*G(?![a-zA-Z])")
MAI_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*枚(?:/G)?")
KAI_PATTERN = re.compile(r"\d+\s*回(?!転)")
BAI_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*倍")
SHURUI_PATTERN = re.compile(r"\d+\s*(?:種類|段階|パターン)")
FRACTION_PATTERN = re.compile(r"1\s*/\s*\d+(?:\.\d+)?")
RANGE_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*[〜~]\s*\d+(?:\.\d+)?")

ALL_NUMERIC_PATTERN = re.compile(
    "|".join(
        p.pattern
        for p in (
            PCT_PATTERN, RANGE_PATTERN, FRACTION_PATTERN, MAI_PATTERN,
            BAI_PATTERN, SHURUI_PATTERN, GAME_PATTERN, KAI_PATTERN,
        )
    )
)

STRUCTURED_ROW_PATTERN = re.compile(r"^-\s*\[([^\]]+)\]\s*(.+)$", re.MULTILINE)
PROSE_SECTION_PATTERN = re.compile(
    r"◆\s*([^（\n]+)（出典カテゴリ:\s*([^）]+)）\n(.+?)(?=\n◆|\Z)", re.DOTALL
)
REFERENCE_INFO_PREFIX_PATTERN = re.compile(r"参照情報[：:]\s*(.+)")

# domain topic words used for rule-based (Level1) relevance matching
TOPIC_WORDS = [
    "天井", "ボーナス", "ART", "AT", "確率", "ゾーン", "モード", "小役", "設定",
    "前兆", "投資", "回収", "出目", "示唆", "継続", "上乗せ", "ループストック",
    "純増", "機械割", "初当り", "抽選", "振り分け", "解析", "打ち方", "ヤメ時",
    "設定判別", "特化", "演出",
]

EXCEPTION_WORDS = ["ただし", "除く", "場合のみ", "例外", "但し"]
CONDITION_WORDS = ["場合", "時", "成立時", "到達時"]

COMPRESSION_PHRASES = [
    "は3種類あって", "は3種類あり", "抽選で決定する", "抽選で決まる", "主に",
    "代表的には", "など", "詳しくは", "基本的には", "詳細は省略", "細かくは",
]


def normalize_fact(text: str) -> str:
    """15.2% / 15.2％ / 510G / 510ゲーム / 1,000G / 1000G 等を同一視するための正規化。"""
    t = text.strip()
    t = t.replace("％", "%")
    t = t.replace(",", "")
    t = t.replace("ゲーム", "G")
    t = re.sub(r"\s+", "", t)
    return t


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec["_lineno"] = lineno
                records.append(rec)
    return records


def split_question_context(user_text: str) -> tuple[str, str]:
    """user contentを (context, question) に分割する。
    最終非空行を質問として扱う(実データ観察より確認済みの一貫パターン)。
    """
    lines = [line for line in user_text.split("\n")]
    non_empty_idx = [i for i, line in enumerate(lines) if line.strip()]
    if not non_empty_idx:
        return "", user_text.strip()
    last_idx = non_empty_idx[-1]
    question = lines[last_idx].strip()
    context = "\n".join(lines[:last_idx]).strip()
    return context, question


def topic_words_in(text: str) -> set[str]:
    return {w for w in TOPIC_WORDS if w in text}


def classify_relevance(label_or_clause: str, question: str) -> tuple[str, str]:
    """Level1(ルールベース) + Level2(トピック語の緩やか一致によるsemantic近似)。
    戻り値: (relevance, confidence)
    relevance in {relevant, irrelevant}, confidence in {high, medium, low}
    """
    label_topics = topic_words_in(label_or_clause)
    question_topics = topic_words_in(question)
    if label_topics and question_topics:
        if label_topics & question_topics:
            return "relevant", "high"
        return "irrelevant", "high"
    # Level2: どちらかにtopic語が無い曖昧ケース -> ゆるい部分文字列一致
    if label_topics:
        # ラベルにtopic語はあるが質問にtopic語が明示的に無い(口語的質問等)
        # -> ラベルのtopic語が質問文字列に部分一致するか(語幹一致)を再確認
        for w in label_topics:
            if w in question:
                return "relevant", "medium"
        return "irrelevant", "medium"
    # ラベル側にtopic語が無い(定性的な短いラベル等) -> 質問が単一トピック質問なら関連とみなす
    if len(question_topics) <= 1:
        return "relevant", "low"
    return "irrelevant", "low"


def extract_context_facts(context: str, question: str) -> list[dict]:
    facts = []

    structured_rows = list(STRUCTURED_ROW_PATTERN.finditer(context))
    for m in structured_rows:
        label, rest = m.group(1), m.group(2).strip()
        relevance, confidence = classify_relevance(label, question)
        if "：" in rest or ":" in rest:
            sep = "：" if "：" in rest else ":"
            key, _, value = rest.partition(sep)
            key, value = key.strip(), value.strip()
            is_numeric_pair = bool(ALL_NUMERIC_PATTERN.search(key)) or bool(
                ALL_NUMERIC_PATTERN.search(value)
            )
            facts.append(
                {
                    "type": "mapping" if is_numeric_pair else "categorical",
                    "label": label,
                    "key": key,
                    "value": value,
                    "raw": rest,
                    "normalized": normalize_fact(value) if value else normalize_fact(key),
                    "relevance": relevance,
                    "confidence": confidence,
                    "is_percentage": bool(PCT_PATTERN.search(value)),
                    "is_exception": any(w in rest for w in EXCEPTION_WORDS),
                    "is_condition": any(w in label or w in key for w in CONDITION_WORDS),
                }
            )
        else:
            facts.append(
                {
                    "type": "categorical",
                    "label": label,
                    "key": None,
                    "value": rest,
                    "raw": rest,
                    "normalized": normalize_fact(rest),
                    "relevance": relevance,
                    "confidence": confidence,
                    "is_percentage": bool(PCT_PATTERN.search(rest)),
                    "is_exception": any(w in rest for w in EXCEPTION_WORDS),
                    "is_condition": any(w in label for w in CONDITION_WORDS),
                }
            )

    # prose sections: embedded numeric facts (semantic-adjacent, still deterministic regex based)
    for m in PROSE_SECTION_PATTERN.finditer(context):
        heading, _src_cat, body = m.group(1), m.group(2), m.group(3)
        nums = ALL_NUMERIC_PATTERN.findall(body)
        relevance, confidence = classify_relevance(heading, question)
        for n in nums:
            facts.append(
                {
                    "type": "numeric_in_prose",
                    "label": heading.strip(),
                    "key": None,
                    "value": n,
                    "raw": n,
                    "normalized": normalize_fact(n),
                    "relevance": relevance,
                    "confidence": confidence,
                    "is_percentage": bool(PCT_PATTERN.fullmatch(n.strip())),
                    "is_exception": any(w in body for w in EXCEPTION_WORDS),
                    "is_condition": any(w in body for w in CONDITION_WORDS),
                }
            )

    # simple "参照情報：" format (no structured rows present)
    if not structured_rows and not list(PROSE_SECTION_PATTERN.finditer(context)):
        ref_match = REFERENCE_INFO_PREFIX_PATTERN.search(context)
        if ref_match:
            clauses = [c.strip() for c in re.split("[、,]", ref_match.group(1)) if c.strip()]
            for clause in clauses:
                nums = ALL_NUMERIC_PATTERN.findall(clause)
                relevance, confidence = classify_relevance(clause, question)
                if nums:
                    for n in nums:
                        facts.append(
                            {
                                "type": "numeric",
                                "label": clause,
                                "key": None,
                                "value": n,
                                "raw": clause,
                                "normalized": normalize_fact(n),
                                "relevance": relevance,
                                "confidence": confidence,
                                "is_percentage": bool(PCT_PATTERN.search(n)),
                                "is_exception": any(w in clause for w in EXCEPTION_WORDS),
                                "is_condition": any(w in clause for w in CONDITION_WORDS),
                            }
                        )
                else:
                    facts.append(
                        {
                            "type": "categorical",
                            "label": clause,
                            "key": None,
                            "value": clause,
                            "raw": clause,
                            "normalized": normalize_fact(clause),
                            "relevance": relevance,
                            "confidence": confidence,
                            "is_percentage": False,
                            "is_exception": any(w in clause for w in EXCEPTION_WORDS),
                            "is_condition": any(w in clause for w in CONDITION_WORDS),
                        }
                    )
        elif not context.strip():
            pass  # persona系: contextなし
        else:
            # フォールバック: 未知形式、全体から数値のみ拾う(低confidence)
            for n in ALL_NUMERIC_PATTERN.findall(context):
                facts.append(
                    {
                        "type": "numeric",
                        "label": "(unparsed_context)",
                        "key": None,
                        "value": n,
                        "raw": n,
                        "normalized": normalize_fact(n),
                        "relevance": "relevant",
                        "confidence": "low",
                        "is_percentage": bool(PCT_PATTERN.search(n)),
                        "is_exception": False,
                        "is_condition": False,
                    }
                )

    return facts


def check_retention(fact: dict, answer: str, answer_norm: str) -> dict:
    exact = fact["raw"] in answer or (fact["value"] and fact["value"] in answer)
    norm_val = fact["normalized"]
    normalized = bool(norm_val) and norm_val in answer_norm
    return {"exact_retained": exact, "normalized_retained": exact or normalized}


def source_group(meta: dict) -> str:
    src = meta.get("source")
    if src:
        return src
    if meta.get("legacy"):
        return "legacy"
    return "unknown"


def analyze_record(rec: dict, split: str) -> dict:
    messages = rec["messages"]
    meta = rec["metadata"]
    # 最初のuser/最後のassistant (multi-turnはPhase4Rでは最終ターンを主対象とする)
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
    user_text = user_msgs[-1] if user_msgs else ""
    answer = assistant_msgs[-1] if assistant_msgs else ""
    answer_norm = normalize_fact(answer)

    context, question = split_question_context(user_text)
    facts = extract_context_facts(context, question)

    for f in facts:
        f.update(check_retention(f, answer, answer_norm))

    relevant_facts = [f for f in facts if f["relevance"] == "relevant"]
    relevant_high = [f for f in relevant_facts if f["confidence"] == "high"]

    def retention_stats(fset: list[dict], key: str) -> dict | None:
        if not fset:
            return None
        retained = sum(1 for f in fset if f[key])
        return {
            "relevant_fact_count": len(fset),
            "retained_fact_count": retained,
            "omitted_fact_count": len(fset) - retained,
            "retention_rate": round(retained / len(fset) * 100, 1),
        }

    pct_facts = [f for f in relevant_facts if f["is_percentage"]]
    mapping_facts = [f for f in relevant_facts if f["type"] == "mapping"]
    numeric_facts = [
        f for f in relevant_facts if f["type"] in ("numeric", "numeric_in_prose", "mapping")
    ]
    condition_facts = [f for f in relevant_facts if f["is_condition"]]
    exception_facts = [f for f in relevant_facts if f["is_exception"]]

    return {
        "id": f"{split}#{rec['_lineno']}",
        "split": split,
        "source": source_group(meta),
        "category": meta.get("category"),
        "category_code": meta.get("category_code"),
        "question": question,
        "context": context,
        "answer": answer,
        "context_length": len(context),
        "answer_length": len(answer),
        "compression_ratio": round(len(answer) / len(context), 3) if context else None,
        "facts": facts,
        "overall_exact": retention_stats(relevant_facts, "exact_retained"),
        "overall_normalized": retention_stats(relevant_facts, "normalized_retained"),
        "high_confidence_normalized": retention_stats(relevant_high, "normalized_retained"),
        "numeric_normalized": retention_stats(numeric_facts, "normalized_retained"),
        "percentage_normalized": retention_stats(pct_facts, "normalized_retained"),
        "mapping_normalized": retention_stats(mapping_facts, "normalized_retained"),
        "condition_normalized": retention_stats(condition_facts, "normalized_retained"),
        "exception_normalized": retention_stats(exception_facts, "normalized_retained"),
    }


def aggregate(results: list[dict], key: str) -> dict:
    vals = [r[key]["retention_rate"] for r in results if r[key] is not None]
    n_with = len(vals)
    n_na = len(results) - n_with
    if not vals:
        return {"n_records_with_facts": 0, "n_records_na": n_na, "mean": None, "median": None}
    vals_sorted = sorted(vals)
    mid = len(vals_sorted) // 2
    median = (
        vals_sorted[mid]
        if len(vals_sorted) % 2 == 1
        else (vals_sorted[mid - 1] + vals_sorted[mid]) / 2
    )
    return {
        "n_records_with_facts": n_with,
        "n_records_na": n_na,
        "mean": round(sum(vals) / len(vals), 1),
        "median": round(median, 1),
        "min": min(vals),
        "max": max(vals),
    }


def main() -> int:
    train_recs = load_records(TRAIN_PATH)
    val_recs = load_records(VAL_PATH)
    cand_recs = load_records(CANDIDATE_PATH)

    train_results = [analyze_record(r, "train") for r in train_recs]
    val_results = [analyze_record(r, "val") for r in val_recs]
    all_results = train_results + val_results

    report = {
        "n_train": len(train_results),
        "n_val": len(val_results),
        "n_total": len(all_results),
        "n_candidate_raw": len(cand_recs),
        "aggregate": {
            "overall_exact": {
                "train": aggregate(train_results, "overall_exact"),
                "val": aggregate(val_results, "overall_exact"),
                "total": aggregate(all_results, "overall_exact"),
            },
            "overall_normalized": {
                "train": aggregate(train_results, "overall_normalized"),
                "val": aggregate(val_results, "overall_normalized"),
                "total": aggregate(all_results, "overall_normalized"),
            },
            "high_confidence_normalized": {
                "train": aggregate(train_results, "high_confidence_normalized"),
                "val": aggregate(val_results, "high_confidence_normalized"),
                "total": aggregate(all_results, "high_confidence_normalized"),
            },
            "numeric_normalized": {"total": aggregate(all_results, "numeric_normalized")},
            "percentage_normalized": {"total": aggregate(all_results, "percentage_normalized")},
            "mapping_normalized": {"total": aggregate(all_results, "mapping_normalized")},
            "condition_normalized": {"total": aggregate(all_results, "condition_normalized")},
            "exception_normalized": {"total": aggregate(all_results, "exception_normalized")},
        },
        "records_with_zero_relevant_facts": sum(
            1 for r in all_results if r["overall_normalized"] is None
        ),
    }

    # source-group breakdown (legacy vs phase4b/4f/4k)
    by_source = defaultdict(list)
    for r in all_results:
        by_source[r["source"]].append(r)
    report["by_source"] = {
        src: aggregate(recs, "overall_normalized") for src, recs in by_source.items()
    }
    report["by_source_percentage"] = {
        src: aggregate(recs, "percentage_normalized") for src, recs in by_source.items()
    }

    # category_code breakdown
    by_catcode = defaultdict(list)
    for r in all_results:
        by_catcode[r["category_code"] or "N/A"].append(r)
    report["by_category_code"] = {
        c: aggregate(recs, "overall_normalized") for c, recs in by_catcode.items()
    }

    out_path = REPORTS_DIR / "phase4r_fact_retention_results.json"
    slim_results = []
    for r in all_results:
        slim = {k: v for k, v in r.items() if k not in ("context",)}
        slim_results.append(slim)
    out_path.write_text(
        json.dumps({"report": report, "records": slim_results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved -> {out_path}")
    print(json.dumps(report["aggregate"]["overall_normalized"]["total"], ensure_ascii=False))
    print(json.dumps(report["aggregate"]["percentage_normalized"]["total"], ensure_ascii=False))
    print(json.dumps(report["aggregate"]["mapping_normalized"]["total"], ensure_ascii=False))

    # also dump full records (with context) separately for downstream scripts
    full_path = REPORTS_DIR / "_phase4r_full_records.json"
    full_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Full records (with context) -> {full_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
