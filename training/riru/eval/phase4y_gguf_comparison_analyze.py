"""Phase 4Y-R: A_lora_final / B_merged_hf / C_gguf_bf16 / D_gguf_q8_0 / E_gguf_q5_k_m
の5条件比較。同一12代表probe・同一seedでのexact/normalized match・
required-fact recall・wrong-name・placeholderを比較する。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EVAL_DIR.parents[0] / "reports"

sys.path.insert(0, str(EVAL_DIR))
from phase4x_naming_reclassify import classify as classify_naming  # noqa: E402
from phase4x_placeholder_detector import classify_placeholder  # noqa: E402

REQUIRED_FACTS = {
    "Q3": ["510G", "1000G", "1480G", "15.2%", "20.3%", "64.5%"],
    "P01": ["450G", "750G", "1300G", "18%", "27%", "55%"],
    "P02": ["1/450", "1/410", "1/370", "1/320", "1/280"],
    "Q9": ["1/533", "1/295", "97.2%", "114.6%"],
    "Q11": ["510G", "1000G", "1480G", "15.2%", "20.3%", "64.5%"],
}

CONDITIONS = [
    ("A_lora_final", "phase4y_a_lora_final_results.json"),
    ("B_merged_hf", "phase4y_b_merged_hf_results.json"),
    ("C_gguf_bf16", "phase4y_c_gguf_bf16_results.json"),
    ("D_gguf_q8_0", "phase4y_d_gguf_q8_0_results.json"),
    ("E_gguf_q5_k_m", "phase4y_e_gguf_q5_k_m_results.json"),
]


def normalize(t: str) -> str:
    return t.replace("％", "%").replace(",", "").replace("ゲーム", "G")


def recall_pct(text: str, required: list[str] | None) -> float | None:
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


def load_all():
    data = {}
    for label, fname in CONDITIONS:
        path = EVAL_DIR / fname
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw.pop("_meta", None)
        data[label] = raw
    return data


def main() -> int:
    data = load_all()
    required_facts = resolve_required_facts()
    labels = [c[0] for c in CONDITIONS]

    per_probe = {}
    identity_flags = []
    fact_table = {}

    probe_ids = list(data["A_lora_final"].keys())
    for pid in probe_ids:
        req = required_facts.get(pid)
        per_probe[pid] = {}
        recalls_by_label = {}
        for label in labels:
            rec = data[label].get(pid, {})
            texts = {"greedy": rec.get("greedy"), **rec.get("sampled", {})}
            recalls = [recall_pct(t, req) for t in texts.values() if t]
            mean_recall = (
                round(sum(r for r in recalls if r is not None) / len(recalls), 1)
                if req and recalls else None
            )
            recalls_by_label[label] = mean_recall

            # identity checks on naming-relevant probes
            if pid in ("E02", "E36", "NW-01_naming"):
                for seed_label, t in texts.items():
                    if not t:
                        continue
                    naming = classify_naming(t)
                    ph = classify_placeholder(t)
                    if naming["category"] == "A" or ph["is_placeholder"]:
                        identity_flags.append({
                            "probe": pid, "condition": label, "seed": seed_label,
                            "text": t, "naming_category": naming["category"],
                            "placeholder": ph["is_placeholder"],
                        })
        fact_table[pid] = recalls_by_label

    # pairwise exact/normalized match vs A_lora_final (the reference)
    pairwise = {}
    for label in labels[1:]:
        n_pairs = n_exact = n_norm = 0
        divergences = []
        for pid in probe_ids:
            a_rec = data["A_lora_final"].get(pid, {})
            b_rec = data[label].get(pid, {})
            a_texts = {"greedy": a_rec.get("greedy"), **a_rec.get("sampled", {})}
            b_texts = {"greedy": b_rec.get("greedy"), **b_rec.get("sampled", {})}
            for seed_label in a_texts:
                if seed_label not in b_texts:
                    continue
                at, bt = a_texts[seed_label], b_texts[seed_label]
                if not at or not bt:
                    continue
                n_pairs += 1
                exact = at == bt
                norm_match = normalize(at) == normalize(bt)
                if exact:
                    n_exact += 1
                if norm_match:
                    n_norm += 1
                if not norm_match:
                    divergences.append({
                        "probe": pid, "seed": seed_label, "a_text": at, "b_text": bt,
                    })
        pairwise[label] = {
            "n_pairs": n_pairs,
            "exact_match_rate_pct": round(100 * n_exact / n_pairs, 1) if n_pairs else None,
            "normalized_match_rate_pct": round(100 * n_norm / n_pairs, 1) if n_pairs else None,
            "divergence_count": len(divergences),
            "divergences": divergences,
        }

    out = {
        "conditions": labels,
        "fact_recall_by_probe_and_condition": fact_table,
        "pairwise_vs_lora_final": pairwise,
        "identity_flags": identity_flags,
    }
    out_path = REPORTS_DIR / "phase4y_gguf_gate_analysis.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    review_lines = ["=== identity flags (wrong-name A or placeholder) across all conditions ==="]
    for f in identity_flags:
        review_lines.append(
            f"[{f['condition']}/{f['probe']}/{f['seed']}] cat={f['naming_category']} "
            f"placeholder={f['placeholder']}"
        )
        review_lines.append(f"  {f['text']}")
    review_lines.append("")
    for label in labels[1:]:
        review_lines.append(f"=== divergences: A_lora_final vs {label} ===")
        for d in pairwise[label]["divergences"]:
            review_lines.append(f"[{d['probe']}/{d['seed']}]")
            review_lines.append(f"  A: {d['a_text']}")
            review_lines.append(f"  {label}: {d['b_text']}")
        review_lines.append("")

    review_path = REPORTS_DIR / "_phase4y_gguf_divergence_review_utf8.txt"
    review_path.write_text("\n".join(review_lines), encoding="utf-8")

    exact_rates = {k: v["exact_match_rate_pct"] for k, v in pairwise.items()}
    print(json.dumps(exact_rates, ensure_ascii=False))
    print(f"identity_flags: {len(identity_flags)}")
    print(f"Saved -> {out_path}")
    print(f"Saved -> {review_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
