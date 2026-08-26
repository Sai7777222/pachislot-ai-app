"""Phase 4Z: B_merged_hf vs C_gguf_bf16 大規模identity診断の集約分析。

probe family別のgenuine_wrong_name/correct_name/placeholder/hedge/
generic_role/no_name率、paired比較(WIN/TIE/LOSS/CRITICAL_LOSS)、
反復架空名の検索、Scope required_fact_recallの比較を行う。
このスクリプト自体はCASE判定を行わない(目視確認・最終判定は別途)。
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EVAL_DIR.parents[0] / "reports"

sys.path.insert(0, str(EVAL_DIR))
from phase4t_probes import P04_PROBES  # noqa: E402
from phase4z_naming_classify import classify_naming  # noqa: E402

FAMILY_MAP = {
    "set_a_naming_stress": "phase4w_naming_stress",
    "set_b_heldout_naming": "phase4x_heldout_naming",
}

SAFE_CATS = {"D", "E"}  # generic_role, correct_name = "B側でsafe"とみなす基準
UNSAFE_CATS = {"A", "C"}  # genuine wrong-name, placeholder = "C側でunsafe"

KNOWN_FICTIONAL_NAMES = ["ルリ", "ルリコ", "リリ", "ルナ", "リコ", "パチ子", "キリエ", "あいこ"]
KATAKANA_NAME_PATTERN = re.compile(r"[ァ-ヴー]{2,6}")


def normalize(t: str) -> str:
    return t.replace("％", "%").replace(",", "").replace("ゲーム", "G")


def recall_pct(text: str, required: list[str]) -> float:
    if not required:
        return None
    text_n = normalize(text)
    found = [f for f in required if f in text or normalize(f) in text_n]
    return round(len(found) / len(required) * 100, 1)


def all_labeled_texts(rec: dict) -> dict:
    out = {}
    if "greedy" in rec:
        out["greedy"] = rec["greedy"]
    out.update(rec.get("sampled", {}))
    return out


def classify_block(block: dict) -> dict:
    """probe_id -> {seed_label: classification}"""
    out = {}
    for pid, rec in block.items():
        out[pid] = {}
        for label, text in all_labeled_texts(rec).items():
            if text is None:
                continue
            out[pid][label] = classify_naming(text, is_naming_context=True)
    return out


def rate_summary(classified_flat: list[dict]) -> dict:
    counts = Counter(c["category"] for c in classified_flat)
    total = len(classified_flat)
    if total == 0:
        return {"total": 0}
    return {
        "total": total,
        "counts": dict(counts),
        "genuine_wrong_name_rate_pct": round(100 * counts.get("A", 0) / total, 2),
        "hedge_rate_pct": round(100 * counts.get("B", 0) / total, 2),
        "placeholder_rate_pct": round(100 * counts.get("C", 0) / total, 2),
        "generic_role_rate_pct": round(100 * counts.get("D", 0) / total, 2),
        "correct_name_rate_pct": round(100 * counts.get("E", 0) / total, 2),
        "no_name_rate_pct": round(100 * counts.get("G", 0) / total, 2),
    }


def flatten(classified: dict) -> list[dict]:
    out = []
    for pid, seeds in classified.items():
        for label, c in seeds.items():
            out.append({"probe": pid, "seed": label, **c})
    return out


def main() -> int:
    b = json.loads((EVAL_DIR / "phase4z_identity_results_hf.json").read_text(encoding="utf-8"))
    c = json.loads((EVAL_DIR / "phase4z_identity_results_gguf.json").read_text(encoding="utf-8"))

    identity_blocks = ["set_a_naming_stress", "set_b_heldout_naming", "set_c_e36", "set_d_e02"]

    b_classified = {blk: classify_block(b[blk]) for blk in identity_blocks}
    c_classified = {blk: classify_block(c[blk]) for blk in identity_blocks}

    # family-level rates, splitting E36/E02 original vs paraphrase
    def split_family(block_name: str, classified: dict) -> dict:
        if block_name == "set_c_e36":
            orig = {"E36_ORIGINAL": classified.get("E36_ORIGINAL", {})}
            para = {k: v for k, v in classified.items() if k != "E36_ORIGINAL"}
            return {"e36_original": flatten(orig), "e36_paraphrase": flatten(para)}
        if block_name == "set_d_e02":
            orig = {"E02_ORIGINAL": classified.get("E02_ORIGINAL", {})}
            para = {k: v for k, v in classified.items() if k != "E02_ORIGINAL"}
            return {"e02_original": flatten(orig), "e02_paraphrase": flatten(para)}
        return {block_name: flatten(classified)}

    rates = {"B_merged_hf": {}, "C_gguf_bf16": {}}
    b_all_flat = []
    c_all_flat = []
    for blk in identity_blocks:
        b_split = split_family(blk, b_classified[blk])
        c_split = split_family(blk, c_classified[blk])
        for k, v in b_split.items():
            rates["B_merged_hf"][k] = rate_summary(v)
            b_all_flat.extend(v)
        for k, v in c_split.items():
            rates["C_gguf_bf16"][k] = rate_summary(v)
            c_all_flat.extend(v)
    rates["B_merged_hf"]["overall"] = rate_summary(b_all_flat)
    rates["C_gguf_bf16"]["overall"] = rate_summary(c_all_flat)

    # paired comparison (same probe + seed) across all 4 identity blocks
    win = tie = loss = critical_loss = 0
    critical_loss_detail = []
    loss_detail = []
    for blk in identity_blocks:
        for pid in b_classified[blk]:
            if pid not in c_classified[blk]:
                continue
            b_seeds = b_classified[blk][pid]
            c_seeds = c_classified[blk][pid]
            for label in b_seeds:
                if label not in c_seeds:
                    continue
                b_cat = b_seeds[label]["category"]
                c_cat = c_seeds[label]["category"]
                b_score = {"E": 2, "D": 1, "B": 1, "G": 0, "A": -2, "C": -2}.get(b_cat, 0)
                c_score = {"E": 2, "D": 1, "B": 1, "G": 0, "A": -2, "C": -2}.get(c_cat, 0)
                if c_score > b_score:
                    win += 1
                elif c_score < b_score:
                    loss += 1
                    b_text = b[blk][pid].get("greedy") if label == "greedy" else \
                        b[blk][pid].get("sampled", {}).get(label)
                    c_text = c[blk][pid].get("greedy") if label == "greedy" else \
                        c[blk][pid].get("sampled", {}).get(label)
                    entry = {"block": blk, "probe": pid, "seed": label,
                             "b_cat": b_cat, "c_cat": c_cat,
                             "b_text": b_text, "c_text": c_text}
                    loss_detail.append(entry)
                    if b_cat in SAFE_CATS and c_cat in UNSAFE_CATS:
                        critical_loss += 1
                        critical_loss_detail.append(entry)
                else:
                    tie += 1
    n_pairs = win + tie + loss
    paired = {
        "n_pairs": n_pairs, "win": win, "tie": tie, "loss": loss,
        "critical_loss": critical_loss,
        "critical_loss_rate_pct": round(100 * critical_loss / n_pairs, 3) if n_pairs else None,
    }

    # repeated fictional name search (C side, all identity blocks)
    name_hits = []
    for blk in identity_blocks:
        for pid, rec in c[blk].items():
            for label, text in all_labeled_texts(rec).items():
                if text is None:
                    continue
                for name in KNOWN_FICTIONAL_NAMES:
                    if name in text:
                        name_hits.append({"block": blk, "probe": pid, "seed": label,
                                           "name": name, "text": text})
    name_freq = Counter(h["name"] for h in name_hits)

    # scope comparison
    scope_required = {p["id"]: p["required_facts"] for p in P04_PROBES}
    scope_compare = {}
    for pid in scope_required:
        req = scope_required[pid]
        b_texts = all_labeled_texts(b["scope"][pid])
        c_texts = all_labeled_texts(c["scope"][pid])
        b_recalls = [recall_pct(t, req) for t in b_texts.values()]
        c_recalls = [recall_pct(t, req) for t in c_texts.values()]
        scope_compare[pid] = {
            "b_mean_recall_pct": round(sum(b_recalls) / len(b_recalls), 1),
            "c_mean_recall_pct": round(sum(c_recalls) / len(c_recalls), 1),
        }
    scope_overall_b = round(
        sum(v["b_mean_recall_pct"] for v in scope_compare.values()) / len(scope_compare), 1
    )
    scope_overall_c = round(
        sum(v["c_mean_recall_pct"] for v in scope_compare.values()) / len(scope_compare), 1
    )

    out = {
        "rates_by_family": rates,
        "paired_comparison": paired,
        "fictional_name_search": {
            "frequency": dict(name_freq),
            "hits": name_hits,
        },
        "scope_comparison": {
            "per_probe": scope_compare,
            "overall_b_mean_recall_pct": scope_overall_b,
            "overall_c_mean_recall_pct": scope_overall_c,
        },
    }
    out_path = REPORTS_DIR / "phase4z_identity_analysis.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    paired_out_path = REPORTS_DIR / "phase4z_paired_analysis.json"
    paired_out_path.write_text(
        json.dumps({"summary": paired, "critical_loss_detail": critical_loss_detail,
                    "loss_detail": loss_detail}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    scope_out_path = REPORTS_DIR / "phase4z_scope_analysis.json"
    scope_out_path.write_text(json.dumps(out["scope_comparison"], ensure_ascii=False, indent=2),
                               encoding="utf-8")

    review_lines = ["=== CRITICAL LOSS cases (B safe -> C wrong-name/placeholder) ==="]
    for item in critical_loss_detail:
        review_lines.append(
            f"[{item['block']}/{item['probe']}/{item['seed']}] B={item['b_cat']} C={item['c_cat']}"
        )
        review_lines.append(f"  B: {item['b_text']}")
        review_lines.append(f"  C: {item['c_text']}")
    review_lines.append("")
    review_lines.append("=== fictional name hits (C side) ===")
    for h in name_hits:
        review_lines.append(f"[{h['block']}/{h['probe']}/{h['seed']}] name={h['name']}")
        review_lines.append(f"  {h['text']}")

    review_path = REPORTS_DIR / "_phase4z_review_required_utf8.txt"
    review_path.write_text("\n".join(review_lines), encoding="utf-8")

    print("overall B:", rates["B_merged_hf"]["overall"])
    print("overall C:", rates["C_gguf_bf16"]["overall"])
    print("paired:", paired)
    print("fictional name freq:", dict(name_freq))
    print("scope overall B/C:", scope_overall_b, scope_overall_c)
    print(f"Saved -> {out_path}")
    print(f"Saved -> {paired_out_path}")
    print(f"Saved -> {scope_out_path}")
    print(f"Saved -> {review_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
