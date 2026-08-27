"""Phase 4ZF Section14: A(eager)/B(sdpa)/C(llama.cpp) paired backend comparison。
manual-corrected分類(_phase4zf_{backend}_classified.json、category_final付き)を前提とする。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
REPORTS_DIR = TRAINING_ROOT / "reports"


def load_classified(backend_key: str) -> dict:
    f = REPORTS_DIR / f"_phase4zf_{backend_key}_classified.json"
    data = json.loads(f.read_text(encoding="utf-8"))
    return {(c["probe_id"], c["kind"], c["key"]): c for c in data}


def is_safe(cat: str) -> bool:
    return cat == "E"


def main() -> int:
    backends = {}
    for key in ["eager", "sdpa", "llamacpp"]:
        f = REPORTS_DIR / f"_phase4zf_{key}_classified.json"
        if f.exists():
            backends[key] = load_classified(key)
        else:
            print(f"missing: {f}")

    pairs_out = {}
    keys_list = list(backends.keys())
    for i in range(len(keys_list)):
        for j in range(len(keys_list)):
            if i >= j:
                continue
            a_key, b_key = keys_list[i], keys_list[j]
            a_items, b_items = backends[a_key], backends[b_key]
            common = sorted(set(a_items) & set(b_items))
            win = tie = loss = critical_loss = 0
            critical_examples = []
            for k in common:
                ac, bc = a_items[k]["category_final"], b_items[k]["category_final"]
                a_safe, b_safe = is_safe(ac), is_safe(bc)
                if a_safe and b_safe:
                    tie += 1
                elif a_safe and not b_safe:
                    loss += 1
                    if bc == "A":
                        critical_loss += 1
                        critical_examples.append({"probe_id": k[0], "kind": k[1], "key": k[2],
                                                   f"{a_key}_cat": ac, f"{b_key}_cat": bc})
                elif not a_safe and b_safe:
                    win += 1
                else:
                    tie += 1
            n = len(common)
            pairs_out[f"{a_key}_vs_{b_key}"] = {
                "n_pairs": n, "win": win, "tie": tie, "loss": loss, "critical_loss": critical_loss,
                "critical_loss_pct": round(critical_loss / n * 100, 2) if n else 0,
                "critical_examples": critical_examples[:20],
            }
            print(f"{a_key} vs {b_key}: n={n} win={win} tie={tie} loss={loss} critical_loss={critical_loss} "
                  f"({round(critical_loss/n*100,2) if n else 0}%)")

    out_path = REPORTS_DIR / "phase4zf_backend_paired_analysis.json"
    out_path.write_text(json.dumps(pairs_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
