"""Phase 4R: 教師データの「圧縮パターン」探索 + Q3出力との近傍探索 + persona/RAG比較。

- 教師回答中の圧縮を誘発しうる定型表現の頻度
- v4のQ3代表回答「天井ゲーム数は3種類あって、抽選で決定するよ。」に対する、
  教師914件中の文字列近傍(TF-IDF風の軽量Jaccard類似度、外部ライブラリ不使用)
- persona系 (fact無し) subset と RAG/QA系 (fact有り) subsetの回答長比較
学習・データ変更は一切行わない。既存RAG/Vector DBへはアクセスしない。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = TRAINING_ROOT / "processed"
REPORTS_DIR = TRAINING_ROOT / "reports"

COMPRESSION_PHRASES = [
    "は3種類あって", "は3種類あり", "抽選で決定する", "抽選で決まる", "主に",
    "代表的には", "など", "詳しくは", "基本的には", "だよ。", "なんだ。", "だね。",
]

Q3_REPRESENTATIVE_ANSWER = "天井ゲーム数は3種類あって、抽選で決定するよ。"


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def char_bigrams(text: str) -> set[str]:
    text = re.sub(r"\s+", "", text)
    return {text[i : i + 2] for i in range(len(text) - 1)} if len(text) >= 2 else {text}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> int:
    train = load_records(PROCESSED_DIR / "riru_train_v4.jsonl")
    val = load_records(PROCESSED_DIR / "riru_val_v4.jsonl")
    all_recs = train + val

    answers = []
    for r in all_recs:
        assistant_msgs = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
        if assistant_msgs:
            answers.append({"answer": assistant_msgs[-1], "meta": r["metadata"]})

    # --- compression phrase frequency ---
    phrase_counts = {}
    for phrase in COMPRESSION_PHRASES:
        n = sum(1 for a in answers if phrase in a["answer"])
        phrase_counts[phrase] = {"count": n, "pct": round(100 * n / len(answers), 1)}

    # --- structural pattern: 「Xは3種類あって/あり、抽選で決定する」型 (列挙せず上位概念のみ) ---
    STRUCTURAL_PATTERN = re.compile(r"は\d+種類(あって|あり)[、。]\s*(抽選で決(定する|まる))")
    structural_matches = [a for a in answers if STRUCTURAL_PATTERN.search(a["answer"])]

    # --- nearest teacher examples to Q3's representative short answer (char-bigram Jaccard) ---
    q3_bigrams = char_bigrams(Q3_REPRESENTATIVE_ANSWER)
    scored = []
    for a in answers:
        sim = jaccard(q3_bigrams, char_bigrams(a["answer"]))
        scored.append((sim, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    top20 = scored[:20]

    # --- persona (no relevant facts detected) vs RAG/QA (has facts) length comparison ---
    # NOTE: uses the same relevance/fact logic as phase4r_fact_retention_audit.py, re-derived
    # lightly here via source/category heuristic (fact-bearing categories only) to avoid
    # re-importing the full audit module.
    fact_bearing_categories = {
        "faithful_to_given_info", "rag_multi_info_no_omission", "compound_partial_info",
        "complex_rag_structure_omission_prevention", "derived_entity_retention",
        "repetition_suppression",
    }
    persona_like = [
        a for a in answers if a["meta"].get("category") not in fact_bearing_categories
    ]
    rag_like = [a for a in answers if a["meta"].get("category") in fact_bearing_categories]

    def length_stats(items):
        lens = [len(x["answer"]) for x in items]
        return {
            "n": len(items),
            "mean_length": round(sum(lens) / len(lens), 1) if lens else None,
            "pct_of_total": round(100 * len(items) / len(answers), 1),
        }

    persona_vs_rag = {
        "persona_like_non_fact_categories": length_stats(persona_like),
        "rag_like_fact_categories": length_stats(rag_like),
        "note": (
            "category名ベースの粗い分類(fact-bearingカテゴリ以外は全てpersona_likeとして扱う)。"
            "実際にはpersona_like側にも一部説明的な回答が含まれうるため参考値。"
        ),
    }

    report = {
        "n_total_answers": len(answers),
        "compression_phrase_frequency": phrase_counts,
        "structural_short_pattern_count": len(structural_matches),
        "structural_short_pattern_pct": round(100 * len(structural_matches) / len(answers), 2),
        "structural_short_pattern_examples": [
            {"answer": a["answer"], "category": a["meta"].get("category")}
            for a in structural_matches[:10]
        ],
        "q3_nearest_teacher_examples_top20": [
            {
                "similarity": round(sim, 3),
                "answer": a["answer"],
                "category": a["meta"].get("category"),
                "source": a["meta"].get("source"),
            }
            for sim, a in top20
        ],
        "persona_vs_rag_length_comparison": persona_vs_rag,
    }

    out_path = REPORTS_DIR / "phase4r_teacher_pattern_analysis.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(json.dumps(phrase_counts, ensure_ascii=False))
    print(
        "structural_short_pattern_count:",
        len(structural_matches),
        f"({report['structural_short_pattern_pct']}%)",
    )
    print(json.dumps(persona_vs_rag, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
