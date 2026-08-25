"""Phase 4R: fact density別分析 + Q3-like subset分析。

phase4r_fact_retention_audit.py が出力した _phase4r_full_records.json を読み込み、
- relevant_fact_countのbucket別retention
- 「Q3-like」(relevant facts>=5, 複数数値, percentageあり, mappingあり)のsubset分析
- retention worst/good caseランキング
- Phase4K追加17件 vs 既存897件の比較
- T1-0個別監査
を行う。読み取り専用。学習・データ変更は一切行わない。
"""

from __future__ import annotations

import json
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = TRAINING_ROOT / "reports"

BUCKETS = [(1, 1), (2, 2), (3, 4), (5, 7), (8, 10_000)]
BUCKET_LABELS = ["1fact", "2facts", "3-4facts", "5-7facts", "8+facts"]


def bucket_for(n: int) -> str:
    for (lo, hi), label in zip(BUCKETS, BUCKET_LABELS, strict=True):
        if lo <= n <= hi:
            return label
    return "unknown"


def mean(vals: list[float]) -> float | None:
    return round(sum(vals) / len(vals), 1) if vals else None


def main() -> int:
    recs = json.loads((REPORTS_DIR / "_phase4r_full_records.json").read_text(encoding="utf-8"))
    with_facts = [r for r in recs if r["overall_normalized"] is not None]

    # --- fact density bucket analysis ---
    density: dict[str, list[dict]] = {label: [] for label in BUCKET_LABELS}
    for r in with_facts:
        n = r["overall_normalized"]["relevant_fact_count"]
        density[bucket_for(n)].append(r)
    density_report = {}
    for label in BUCKET_LABELS:
        group = density[label]
        rates = [g["overall_normalized"]["retention_rate"] for g in group]
        density_report[label] = {
            "n_records": len(group),
            "mean_retention": mean(rates),
            "min_retention": min(rates) if rates else None,
            "max_retention": max(rates) if rates else None,
        }

    # --- Q3-like subset: relevant facts>=5, has percentage, has mapping ---
    q3_like = []
    for r in with_facts:
        n_rel = r["overall_normalized"]["relevant_fact_count"]
        has_pct = r["percentage_normalized"] is not None
        has_mapping = r["mapping_normalized"] is not None
        if n_rel >= 5 and has_pct and has_mapping:
            q3_like.append(r)

    q3_like_report = {
        "n_records": len(q3_like),
        "mean_overall_retention": mean(
            [r["overall_normalized"]["retention_rate"] for r in q3_like]
        ),
        "mean_percentage_retention": mean(
            [r["percentage_normalized"]["retention_rate"] for r in q3_like]
        ),
        "mean_mapping_retention": mean(
            [r["mapping_normalized"]["retention_rate"] for r in q3_like]
        ),
        "mean_answer_length": mean([r["answer_length"] for r in q3_like]),
        "ids": [r["id"] for r in q3_like],
    }

    # --- length vs retention scatter data (correlation, no external libs) ---
    def pearson(xs: list[float], ys: list[float]) -> float | None:
        n = len(xs)
        if n < 2:
            return None
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx == 0 or vy == 0:
            return None
        return round(cov / (vx**0.5 * vy**0.5), 3)

    lens = [r["answer_length"] for r in with_facts]
    retentions = [r["overall_normalized"]["retention_rate"] for r in with_facts]
    comp_ratios = [r["compression_ratio"] for r in with_facts if r["compression_ratio"] is not None]
    comp_retentions = [
        r["overall_normalized"]["retention_rate"]
        for r in with_facts
        if r["compression_ratio"] is not None
    ]
    correlation_report = {
        "answer_length_vs_retention_pearson_r": pearson(lens, retentions),
        "compression_ratio_vs_retention_pearson_r": pearson(comp_ratios, comp_retentions),
        "n": len(with_facts),
    }

    # --- worst / good cases ---
    def sortkey_overall(r):
        return r["overall_normalized"]["retention_rate"]

    worst_overall = sorted(with_facts, key=sortkey_overall)[:30]
    pct_recs = [r for r in with_facts if r["percentage_normalized"] is not None]
    worst_pct = sorted(pct_recs, key=lambda r: r["percentage_normalized"]["retention_rate"])[:30]
    map_recs = [r for r in with_facts if r["mapping_normalized"] is not None]
    worst_map = sorted(map_recs, key=lambda r: r["mapping_normalized"]["retention_rate"])[:30]
    high_density = [r for r in with_facts if r["overall_normalized"]["relevant_fact_count"] >= 5]
    worst_density = sorted(high_density, key=sortkey_overall)[:30]

    good_cases = [
        r
        for r in with_facts
        if r["overall_normalized"]["relevant_fact_count"] >= 5
        and r["overall_normalized"]["retention_rate"] >= 90
    ][:20]

    # --- legacy/existing vs Phase4K 17 comparison ---
    existing = [r for r in with_facts if r["source"] != "phase4k_generated"]
    phase4k = [r for r in with_facts if r["source"] == "phase4k_generated"]
    legacy_vs_4k = {
        "existing_897_with_facts": {
            "n": len(existing),
            "mean_overall": mean([r["overall_normalized"]["retention_rate"] for r in existing]),
            "mean_pct": mean(
                [
                    r["percentage_normalized"]["retention_rate"]
                    for r in existing
                    if r["percentage_normalized"]
                ]
            ),
            "mean_mapping": mean(
                [
                    r["mapping_normalized"]["retention_rate"]
                    for r in existing
                    if r["mapping_normalized"]
                ]
            ),
            "mean_answer_length": mean([r["answer_length"] for r in existing]),
        },
        "phase4k_17_with_facts": {
            "n": len(phase4k),
            "mean_overall": mean([r["overall_normalized"]["retention_rate"] for r in phase4k]),
            "mean_pct": mean(
                [
                    r["percentage_normalized"]["retention_rate"]
                    for r in phase4k
                    if r["percentage_normalized"]
                ]
            ),
            "mean_mapping": mean(
                [
                    r["mapping_normalized"]["retention_rate"]
                    for r in phase4k
                    if r["mapping_normalized"]
                ]
            ),
            "mean_answer_length": mean([r["answer_length"] for r in phase4k]),
        },
    }

    # --- T1-0 individual audit ---
    t1_records = [r for r in recs if r["category_code"] == "T1"]
    t1_0_candidates = [r for r in t1_records if "天井" in (r["question"] or "")]
    t1_0 = t1_0_candidates[0] if t1_0_candidates else (t1_records[0] if t1_records else None)

    out = {
        "fact_density_analysis": density_report,
        "q3_like_analysis": q3_like_report,
        "correlations": correlation_report,
        "legacy_vs_phase4k": legacy_vs_4k,
        "t1_records_found": len(t1_records),
    }
    (REPORTS_DIR / "phase4r_fact_density_analysis.json").write_text(
        json.dumps(density_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "phase4r_q3like_analysis.json").write_text(
        json.dumps(
            {
                "q3_like_analysis": q3_like_report,
                "correlations": correlation_report,
                "legacy_vs_phase4k": legacy_vs_4k,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    (REPORTS_DIR / "phase4r_density_qlike_combined.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- human-readable case dumps ---
    def fmt_case(r: dict) -> list[str]:
        stats = r["overall_normalized"]
        lines = [
            f"ID={r['id']} source={r['source']} category={r['category']}({r['category_code']})",
            f"question: {r['question']}",
            f"relevant_fact_count={stats['relevant_fact_count']} "
            f"retained={stats['retained_fact_count']} "
            f"omitted={stats['omitted_fact_count']} "
            f"retention={stats['retention_rate']}%",
        ]
        pct = r["percentage_normalized"]
        mapf = r["mapping_normalized"]
        if pct:
            lines.append(f"  percentage_retention={pct['retention_rate']}%")
        if mapf:
            lines.append(f"  mapping_retention={mapf['retention_rate']}%")
        rel_facts = [f for f in r["facts"] if f["relevance"] == "relevant"]
        omitted_facts = [f["raw"] for f in rel_facts if not f["normalized_retained"]]
        retained_facts = [f["raw"] for f in rel_facts if f["normalized_retained"]]
        lines.append(f"  retained_facts: {retained_facts}")
        lines.append(f"  omitted_facts: {omitted_facts}")
        lines.append(f"answer: {r['answer']}")
        lines.append("")
        return lines

    worst_lines = ["=== A. Overall retention worst 30 ==="]
    for r in worst_overall:
        worst_lines += fmt_case(r)
    worst_lines.append("=== B. Percentage retention worst 30 ===")
    for r in worst_pct:
        worst_lines += fmt_case(r)
    worst_lines.append("=== C. Mapping retention worst 30 ===")
    for r in worst_map:
        worst_lines += fmt_case(r)
    worst_lines.append("=== D. High fact density (>=5) worst 30 ===")
    for r in worst_density:
        worst_lines += fmt_case(r)
    (REPORTS_DIR / "_phase4r_worst_cases_utf8.txt").write_text(
        "\n".join(worst_lines), encoding="utf-8"
    )

    good_lines = ["=== Good cases: relevant facts>=5 AND retention>=90% ==="]
    for r in good_cases:
        good_lines += fmt_case(r)
    if t1_0:
        good_lines.append("=== T1-0 individual audit ===")
        good_lines += fmt_case(t1_0)
    (REPORTS_DIR / "_phase4r_good_cases_utf8.txt").write_text(
        "\n".join(good_lines), encoding="utf-8"
    )

    print("density:", json.dumps(density_report, ensure_ascii=False))
    print("q3_like:", json.dumps(q3_like_report, ensure_ascii=False))
    print("correlations:", json.dumps(correlation_report, ensure_ascii=False))
    print("legacy_vs_4k:", json.dumps(legacy_vs_4k, ensure_ascii=False))
    print(f"good_cases n={len(good_cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
