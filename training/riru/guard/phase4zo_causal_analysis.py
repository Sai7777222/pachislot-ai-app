"""Phase4ZO Stage A: causal20の3条件比較(A=baseline, B=minimal, C=three_mode)。
Section9/Phase4ZM RULE EVAL-001準拠: heuristic文字列検索は暫定値であり、
ground truthではない。"""
from __future__ import annotations
import json
import re
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

HEDGE_RE = re.compile("|".join(re.escape(p) for p in
    ["登録データ", "データベース", "データがない", "登録されていない", "情報がない", "記録がない", "確認できない"]))


def load(name):
    return json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))


def tally(results):
    hedge = [r for r in results if HEDGE_RE.search(r["response"])]
    return {"n": len(results), "hedge_count": len(hedge), "hedge_rate": len(hedge) / len(results),
            "hedge_rows": [{"probe_id": r["probe_id"], "prompt": r["prompt"], "response": r["response"]} for r in hedge]}


def main():
    baseline = load("phase4zo_causal20_baseline.json")
    minimal = load("phase4zo_causal20_minimal.json")
    three_mode = load("phase4zo_causal20_three_mode.json")

    t_baseline = tally(baseline)
    t_minimal = tally(minimal)
    t_three_mode = tally(three_mode)

    out = {
        "purpose": "Stage A: personality/preference 20件 x 3条件のheuristic比較(暫定値、"
                   "RULE EVAL-001準拠でground truthではない)。",
        "baseline": {"hedge_count": t_baseline["hedge_count"], "hedge_rate": t_baseline["hedge_rate"]},
        "minimal": {"hedge_count": t_minimal["hedge_count"], "hedge_rate": t_minimal["hedge_rate"]},
        "three_mode": {"hedge_count": t_three_mode["hedge_count"], "hedge_rate": t_three_mode["hedge_rate"]},
        "causal_interpretation_provisional": (
            "minimal promptでもhedgeが大きく残るなら、hedge癖はadapter/model自体に強く由来する"
            "(model-level dominant, CASE ZO-C寄り)。minimalでhedgeが大きく下がるなら、"
            "現行system prompt(RAG厳格指示)の存在自体がhedgeを誘発している可能性が高い。"
            "three_modeでbaseline/minimal双方より明確に改善していれば、product-layer fixとして有望"
            "(CASE ZO-A/B寄り)。この解釈は暫定であり、Stage B以降のregression結果と合わせて"
            "最終判断する。"
        ),
        "baseline_detail": t_baseline, "minimal_detail": t_minimal, "three_mode_detail": t_three_mode,
    }
    out_path = REPORTS_DIR / "phase4zo_causal_analysis.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"baseline hedge={t_baseline['hedge_count']}/20 minimal hedge={t_minimal['hedge_count']}/20 "
          f"three_mode hedge={t_three_mode['hedge_count']}/20")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
