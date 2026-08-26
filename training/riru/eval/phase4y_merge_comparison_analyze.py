"""Phase 4Y Section13: A_lora_final vs B_merged_hf の同等性比較。

同一prompt・同一seedでの生成テキストを、exact match / normalized match /
required-fact recall / identity matchの観点で比較する。重大な意味差があれば
目視確認対象として出力する(このスクリプト自体は最終判定を行わない)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EVAL_DIR.parents[0] / "reports"

sys.path.insert(0, str(EVAL_DIR))

REQUIRED_FACTS = {
    "Q3": ["510G", "1000G", "1480G", "15.2%", "20.3%", "64.5%"],
    "P01": ["450G", "750G", "1300G", "18%", "27%", "55%"],
    "P02": ["1/450", "1/410", "1/370", "1/320", "1/280"],
    "PT-01_scope": None,  # PT-01のrequired_factsをphase4t_probesから取得
    "Q9": ["1/533", "1/295", "97.2%", "114.6%"],
    "Q11": ["510G", "1000G", "1480G", "15.2%", "20.3%", "64.5%"],
    "V1-A_broad": None,  # phase4v_probesから取得
    "LC-01_longcontext": None,  # phase4w_probesから取得
}


def normalize(t: str) -> str:
    return t.replace("％", "%").replace(",", "").replace("ゲーム", "G")


def recall_pct(text: str, required: list[str]) -> float:
    if not required:
        return None
    text_n = normalize(text)
    found = [f for f in required if f in text or normalize(f) in text_n]
    return round(len(found) / len(required) * 100, 1)


def resolve_required_facts() -> dict:
    facts = dict(REQUIRED_FACTS)
    from phase4t_probes import P04_PROBES

    pt01 = next(p for p in P04_PROBES if p["id"] == "PT-01")
    facts["PT-01_scope"] = pt01["required_facts"]

    from phase4v_probes import PROBES as BROAD_PROBES

    v1a = next(p for p in BROAD_PROBES if p["id"] == "V1-A")
    facts["V1-A_broad"] = v1a["required_facts"]

    from phase4w_probes import LONGCONTEXT_PROBES

    lc01 = next(p for p in LONGCONTEXT_PROBES if p["id"] == "LC-01")
    facts["LC-01_longcontext"] = lc01["required_facts"]
    return facts


def main() -> int:
    a = json.loads((EVAL_DIR / "phase4y_a_lora_final_results.json").read_text(encoding="utf-8"))
    b = json.loads((EVAL_DIR / "phase4y_b_merged_hf_results.json").read_text(encoding="utf-8"))
    required_facts = resolve_required_facts()

    per_probe = {}
    n_pairs = 0
    n_exact = 0
    n_normalized = 0
    divergences = []

    for pid in a:
        a_rec, b_rec = a[pid], b.get(pid, {})
        a_texts = {"greedy": a_rec.get("greedy"), **a_rec.get("sampled", {})}
        b_texts = {"greedy": b_rec.get("greedy"), **b_rec.get("sampled", {})}
        req = required_facts.get(pid)

        probe_pairs = []
        for label in a_texts:
            if label not in b_texts:
                continue
            at, bt = a_texts[label], b_texts[label]
            n_pairs += 1
            exact = at == bt
            norm_match = normalize(at) == normalize(bt)
            if exact:
                n_exact += 1
            if norm_match:
                n_normalized += 1
            a_recall = recall_pct(at, req) if req else None
            b_recall = recall_pct(bt, req) if req else None
            entry = {
                "label": label, "exact_match": exact, "normalized_match": norm_match,
                "a_text": at, "b_text": bt, "a_recall": a_recall, "b_recall": b_recall,
            }
            probe_pairs.append(entry)
            if not norm_match:
                divergences.append({"probe": pid, **entry})
        per_probe[pid] = probe_pairs

    summary = {
        "n_pairs": n_pairs,
        "exact_match_count": n_exact,
        "exact_match_rate_pct": round(100 * n_exact / n_pairs, 1) if n_pairs else None,
        "normalized_match_count": n_normalized,
        "normalized_match_rate_pct": round(100 * n_normalized / n_pairs, 1) if n_pairs else None,
        "divergence_count": len(divergences),
    }

    out = {"summary": summary, "per_probe": per_probe, "divergences": divergences}
    out_path = REPORTS_DIR / "phase4y_merge_gguf_comparison.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    review_lines = ["=== A_lora_final vs B_merged_hf divergences (normalized mismatch) ==="]
    for d in divergences:
        review_lines.append(f"[{d['probe']}/{d['label']}]")
        review_lines.append(f"  A: {d['a_text']}")
        review_lines.append(f"  B: {d['b_text']}")
        if d["a_recall"] is not None:
            review_lines.append(f"  recall: A={d['a_recall']}% B={d['b_recall']}%")
    review_path = REPORTS_DIR / "_phase4y_merge_divergence_review_utf8.txt"
    review_path.write_text("\n".join(review_lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved -> {out_path}")
    print(f"Saved -> {review_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
