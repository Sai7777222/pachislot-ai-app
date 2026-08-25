# ruff: noqa: E501
"""Phase 4M-5/6/8/9/10: fresh-process結果の統合分析。

各条件 (base/v2/v3/v4) のphase4m_fresh_<condition>.jsonを読み込み、
- active adapter診断
- Q3 sampled/greedy全文比較
- E36/Q11 fresh-process結果比較
- logits差分 (v2-v3, v2-v4, v3-v4など)
- v4のadapter ON/OFF比較
をまとめる。
"""

from __future__ import annotations

import json
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
CONDITIONS = ("base", "v2", "v3", "v4")


def load_all() -> dict:
    return {c: json.loads((REPORTS_DIR / f"phase4m_fresh_{c}.json").read_text(encoding="utf-8")) for c in CONDITIONS}


def load_full_logits(condition: str) -> list[float]:
    path = REPORTS_DIR / f"phase4m_fresh_{condition}_full_logits.json"
    return json.loads(path.read_text(encoding="utf-8"))


def logits_diff(a: list[float], b: list[float]) -> dict:
    max_abs = 0.0
    sum_abs = 0.0
    for x, y in zip(a, b, strict=True):
        d = abs(x - y)
        sum_abs += d
        if d > max_abs:
            max_abs = d
    return {"max_abs_diff": round(max_abs, 6), "mean_abs_diff": round(sum_abs / len(a), 8)}


def main() -> int:
    data = load_all()

    q3_texts_sampled = {c: data[c]["q3_sampled"]["text"] for c in CONDITIONS}
    q3_texts_greedy = {c: data[c]["q3_greedy"]["text"] for c in CONDITIONS}
    e36_texts = {c: data[c]["e36_sampled"]["text"] for c in CONDITIONS}
    q11_texts = {c: data[c]["q11_sampled"]["text"] for c in CONDITIONS}

    identity_check_sampled = {
        "v2_v3_identical": q3_texts_sampled["v2"] == q3_texts_sampled["v3"],
        "v2_v4_identical": q3_texts_sampled["v2"] == q3_texts_sampled["v4"],
        "v3_v4_identical": q3_texts_sampled["v3"] == q3_texts_sampled["v4"],
        "base_v2_identical": q3_texts_sampled["base"] == q3_texts_sampled["v2"],
    }
    identity_check_greedy = {
        "v2_v3_identical": q3_texts_greedy["v2"] == q3_texts_greedy["v3"],
        "v2_v4_identical": q3_texts_greedy["v2"] == q3_texts_greedy["v4"],
        "v3_v4_identical": q3_texts_greedy["v3"] == q3_texts_greedy["v4"],
    }

    # logits diffs
    full_logits = {c: load_full_logits(c) for c in CONDITIONS}
    pairs = [("base", "v2"), ("v2", "v3"), ("v2", "v4"), ("v3", "v4"), ("base", "v4")]
    logits_diffs = {f"{a}_vs_{b}": logits_diff(full_logits[a], full_logits[b]) for a, b in pairs}

    # diagnostics
    diagnostics_summary = {c: data[c]["diagnostics_after_load"] for c in CONDITIONS}

    # adapter ON/OFF (v4)
    onoff = data["v4"].get("adapter_on_off_check")
    onoff_summary = None
    if onoff:
        v4_on_text = data["v4"]["q3_sampled"]["text"]
        v4_off_text = onoff["q3_sampled_with_adapter_off"]["text"]
        off_logits = load_full_logits("v4_adapter_off")
        onoff_summary = {
            "text_identical_on_vs_off": v4_on_text == v4_off_text,
            "diagnostics_with_adapter_disabled": onoff["diagnostics_with_adapter_disabled"],
            "q3_text_on": v4_on_text,
            "q3_text_off": v4_off_text,
            "logits_diff_on_vs_off": logits_diff(full_logits["v4"], off_logits),
        }

    summary = {
        "diagnostics_summary": diagnostics_summary,
        "q3_sampled_texts": q3_texts_sampled,
        "q3_greedy_texts": q3_texts_greedy,
        "q3_sampled_identity_check": identity_check_sampled,
        "q3_greedy_identity_check": identity_check_greedy,
        "e36_texts": e36_texts,
        "q11_texts": q11_texts,
        "logits_diffs": logits_diffs,
        "adapter_on_off_check_v4": onoff_summary,
    }

    (REPORTS_DIR / "phase4m_fresh_process_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in ("q3_sampled_texts", "q3_greedy_texts", "e36_texts", "q11_texts", "adapter_on_off_check_v4")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
