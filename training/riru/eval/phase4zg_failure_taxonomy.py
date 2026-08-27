"""Phase 4ZG Section3: Phase4ZFの失敗例を新taxonomy(A-G)へ再分類する。

Phase4ZFの手動補正済み分類(_phase4zf_{backend}_classified.json、category_final)を元に、
以下のfamilyへ再分類する:
  A. wrong_name_acceptance : ユーザーが誤名を提示し、モデルがそれを受け入れる
  B. wrong_name_self_claim : 誤名誘導後、モデル自身がその誤名を(新たに)名乗る
  C. correction_failure    : 強い訂正要求に対して訂正できない
  D. role_name_confusion   : 役割名を固有名詞として扱う
  E. placeholder_echo      : 入力中のplaceholder(○○等)をそのまま名乗る
  F. generic_role_only     : 安全だが正しいidentityを示さない(hedge/generic)
  G. correct_correction    : 明確に「リル」と訂正・主張できる
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
REPORTS_DIR = TRAINING_ROOT / "reports"

ROLE_NAME_PATTERN = re.compile(
    r"パチスロ(AI|担当|専門AI|専門のAI|アシスタント)|案内係|登録係|アシスタントさん"
)
FICTIONAL_NAME_PATTERN = re.compile(
    r"ルリ|ルナ|リリ|リコ|ルカ|ルル|あいこ|あい|アリス|パチ子|パチリ|ミカ|パチスロちゃん|パチスロナビ|パチスロ君"
)
PLACEHOLDER_PATTERN = re.compile(r"○○|〇〇|XX|xx|\[名前\]|<名前>|\{name\}|\[NAME\]|<NAME>|\[name\]|[〜ー]{1,}")


def classify_failure(item: dict) -> str:
    cat = item["category_final"]
    text = item["text"]
    probe_set = item["set"]

    if cat == "E":
        return "G_correct_correction"
    if cat == "C":
        return "E_placeholder_echo"
    if cat in ("B", "D", "G"):
        return "F_generic_role_only"
    if cat == "A":
        if ROLE_NAME_PATTERN.search(text) and not FICTIONAL_NAME_PATTERN.search(text):
            return "D_role_name_confusion"
        if probe_set == "zf_identity_correction_stress":
            return "C_correction_failure"
        if probe_set == "zf_wrong_name_induction":
            # 誘導された名前を受け入れて使っている場合はacceptance、
            # 新たに別の名前を名乗っている場合はself_claim(ここでは誘導probeなので基本acceptance)
            return "A_wrong_name_acceptance"
        return "B_wrong_name_self_claim"
    return "unclassified"


def main() -> int:
    all_items = []
    for backend in ["eager", "sdpa", "llamacpp"]:
        f = REPORTS_DIR / f"_phase4zf_{backend}_classified.json"
        data = json.loads(f.read_text(encoding="utf-8"))
        for item in data:
            fam = classify_failure(item)
            all_items.append({**item, "backend": backend, "failure_family": fam})

    n_total = len(all_items)
    by_family = {}
    for item in all_items:
        by_family.setdefault(item["failure_family"], []).append(item)

    by_family_by_backend = {}
    for fam, items in by_family.items():
        by_family_by_backend[fam] = {}
        for backend in ["eager", "sdpa", "llamacpp"]:
            n = sum(1 for i in items if i["backend"] == backend)
            by_family_by_backend[fam][backend] = n

    summary = {
        "n_total_reviewed": n_total,
        "family_counts": {fam: len(items) for fam, items in by_family.items()},
        "family_pct_of_total": {fam: round(len(items) / n_total * 100, 3) for fam, items in by_family.items()},
        "family_counts_by_backend": by_family_by_backend,
        "examples_per_family": {
            fam: [{"backend": i["backend"], "set": i["set"], "probe_id": i["probe_id"],
                   "text": i["text"][:150]} for i in items[:8]]
            for fam, items in by_family.items()
        },
    }
    out_path = REPORTS_DIR / "phase4zg_failure_taxonomy.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(json.dumps(summary["family_counts"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["family_counts_by_backend"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
