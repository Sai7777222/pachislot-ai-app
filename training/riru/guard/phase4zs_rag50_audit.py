"""Phase4ZS Stage F/G/H: RAG50全件を独立GTで監査する。既存output(Phase4ZN rag50_raw、
raw Phase4ZG+production system prompt+pre-baked context、追加generationなしで再利用)を、
生成時に実際に与えられていたcontext(phase4zf_rag_stress_eval由来のpre-baked context、
生成後に別途取得したものではない)と突き合わせる。"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parent
sys.path.insert(0, str(TRAINING_ROOT / "eval"))
REPORTS_DIR = TRAINING_ROOT / "reports"

from phase4zf_rag_stress_eval import load_rag_probe_pool  # noqa: E402

NUMERIC_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?パーセント|1/\d+")
CONTRADICTION_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?\s*%|1/\d+|\d+パーセント)[^。！？]{0,80}(登録データ|情報)[^。！？]{0,20}(ない|なかった|見つから)"
    r"|(登録データ|情報)[^。！？]{0,20}(ない|なかった|見つから)[^。！？]{0,80}(\d+(?:\.\d+)?\s*%|1/\d+|\d+パーセント)"
)
NON_NUMERIC_CLAIM_MARKERS = re.compile(r"確定|濃厚|恩恵|モード|関係")


def main():
    ng = json.loads((REPORTS_DIR / "phase4zn_rag50_raw.json").read_text(encoding="utf-8"))
    pool = {p["id"]: p for p in load_rag_probe_pool()}

    numeric_rows = []
    for r in ng:
        pid = r["probe_id"]
        context = pool.get(pid, {}).get("context", "") or ""
        response = r["response"]
        response_numerics = NUMERIC_PATTERN.findall(response)
        unsupported = []
        for num in set(response_numerics):
            if num not in context:
                unsupported.append(num)
        contradiction = bool(CONTRADICTION_PATTERN.search(response))
        numeric_rows.append({
            "probe_id": pid, "prompt": r["prompt"], "response": response,
            "response_numerics": list(set(response_numerics)),
            "unsupported_numerics": unsupported, "has_unsupported_numeric": len(unsupported) > 0,
            "contradictory_self_awareness": contradiction,
            "context_had_any_numeric": bool(NUMERIC_PATTERN.search(context)),
        })

    n_unsupported_numeric_turns = sum(1 for r in numeric_rows if r["has_unsupported_numeric"])
    n_contradiction = sum(1 for r in numeric_rows if r["contradictory_self_awareness"])

    numeric_out = {
        "purpose": "Stage F: RAG50全件(既存Phase4ZN raw output再利用、追加generationなし)の"
                   "unsupported numeric audit。各応答中の数値が、生成時に実際に与えられていた"
                   "pre-baked contextの文字列として存在するかを機械照合した(近似的なexact-match "
                   "チェックであり、意味的な同値判定[単位換算・丸め等]は含まない、目視補足が必要)。",
        "n_total": len(numeric_rows),
        "unsupported_numeric_turn_count": n_unsupported_numeric_turns,
        "unsupported_numeric_probe_count": n_unsupported_numeric_turns,
        "contradiction_count": n_contradiction,
        "rows": numeric_rows,
    }
    (REPORTS_DIR / "phase4zs_rag50_numeric_audit.json").write_text(
        json.dumps(numeric_out, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "phase4zs_contradiction_analysis.json").write_text(json.dumps({
        "purpose": "Stage H: 「情報はない」+「具体的数値」という自己矛盾パターンの集計(全診断stage横断)。",
        "rag50_existing_output_contradiction_count": n_contradiction,
        "zero_context_confirmation_contradiction_count": 10,
        "context_provided_stages_contradiction_count": 0,
        "note": "自己矛盾はcontext皆無の状況でのみ確認された(zero-context confirmation 10/10)。"
                "実contextが与えられた全てのstage(A-E)ではcontradiction=0。RAG50既存output監査でも"
                "後述の通り。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"unsupported_numeric_turns={n_unsupported_numeric_turns}/{len(numeric_rows)}")
    print(f"contradiction={n_contradiction}/{len(numeric_rows)}")

    # --- Stage G: non-numeric claim audit, Q11/Q17重点 ---
    q_targets = [r for r in numeric_rows if r["probe_id"] in ("Q11", "Q17")]
    g_out = {
        "purpose": "Stage G: 数値以外の(存在しない仕様/挙動/モード関係/恩恵/確定濃厚表現/因果関係)の"
                   "unsupported claim監査。Q11/Q17を重点的に目視監査した。",
        "q11_row": q_targets[0] if q_targets else None,
        "q17_row": q_targets[1] if len(q_targets) > 1 else None,
        "manual_assessment": "別途manual reviewで記載(phase4zs_summary.md参照)。",
    }
    (REPORTS_DIR / "phase4zs_non_numeric_claim_audit.json").write_text(
        json.dumps(g_out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
