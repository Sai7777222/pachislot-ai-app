"""Phase4ZM Section17-18: OOD/RAG regressionの再評価。Phase4ZGモデル・
regeneration constraint文言・seedを一切変更していないため、既存の
raw/finalテキストへ新validatorを適用するだけでpipeline全体の挙動を
正確に再現できる(GPU再実行不要)。"""
from __future__ import annotations
import json
import sys
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GUARD_DIR))
REPORTS_DIR = GUARD_DIR.parent / "reports"

from identity_validator import validate_identity  # noqa: E402

_FALLBACK = "私はリルだよ！"


def load(name):
    return json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))


def recheck_single_turn(raw, prompt, old_final, old_modified):
    v1 = validate_identity(raw, prompt)
    if v1.safe:
        return raw, "pass", v1.__dict__, None
    # old run's `final` (when modified=True) is the already-generated regenerated text.
    candidate_regen = old_final if old_modified else None
    if candidate_regen is not None:
        v2 = validate_identity(candidate_regen, prompt)
        if v2.safe:
            return candidate_regen, "regenerated_pass", v1.__dict__, v2.__dict__
        return _FALLBACK, "fallback", v1.__dict__, v2.__dict__
    return _FALLBACK, "regeneration_needed_not_run", v1.__dict__, None


def recheck_dataset(filename, out_name, label):
    data = load(filename)
    rows = []
    n_modified = 0
    n_flagged = 0
    for pid, v in data.items():
        final, stage, v1, v2 = recheck_single_turn(v["raw"], v.get("prompt") or v.get("question", ""),
                                                     v["final"], v["modified"])
        modified = final != v["raw"]
        if modified:
            n_modified += 1
        if not v1["safe"]:
            n_flagged += 1
        rows.append({"id": pid, "category": v.get("category") or v.get("set"), "raw": v["raw"],
                     "new_final": final, "new_stage": stage, "modified_by_new_guard": modified,
                     "old_stage": v["stage"], "old_modified": v["modified"]})
    out = {
        "purpose": f"{label}の縮小後guardによる再評価。",
        "n": len(rows), "flagged_at_first_call": n_flagged, "modified_final_count": n_modified,
        "rows": rows,
    }
    out_path = REPORTS_DIR / out_name
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{label}: n={len(rows)} flagged={n_flagged} modified={n_modified} -> {out_path}")


def main():
    recheck_dataset("phase4zl_ood_regression_results.json", "phase4zm_ood_recheck.json", "OOD regression (Section17)")
    recheck_dataset("phase4zl_rag_regression_results.json", "phase4zm_rag_recheck.json", "RAG regression (Section18)")


if __name__ == "__main__":
    main()
