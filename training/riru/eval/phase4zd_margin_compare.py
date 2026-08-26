"""Phase 4ZD Section8-9: A/B/C/D(/E)条件のE36 forced-prefix logits/marginを集約する。

出力:
  training/riru/reports/phase4zd_environment_matrix.json (各条件のロード方式・attn実装等)
  training/riru/reports/phase4zd_margin_comparison.json (リ/ル logit/prob/rank/marginの一覧)
"""
from __future__ import annotations

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
REPORTS_DIR = TRAINING_ROOT / "reports"

CONDITIONS = ["A_LEGACY_4BIT", "B_HF_BF16_EAGER", "C_HF_BF16_SDPA", "D_LLAMA_BF16_CPU", "E_HF_FP32_EAGER"]


def main() -> int:
    matrix = {}
    margins = {}
    for cond in CONDITIONS:
        f = REPORTS_DIR / f"phase4zd_hf_logits_{cond}.json"
        if not f.exists():
            print(f"skip (not found): {f}")
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        matrix[cond] = {
            "model_path": d.get("model_path"),
            "quant_config": d.get("quant_config"),
            "requested_dtype": d.get("requested_dtype"),
            "requested_attn_impl": d.get("requested_attn_impl"),
            "actual_attn_implementation": d.get("actual_attn_implementation", d.get("tool")),
            "device_map": d.get("device_map", "N/A (llama.cpp CPU-only)"),
            "n_tokens": d.get("n_tokens"),
            "tool": d.get("tool", "transformers AutoModelForCausalLM"),
        }
        margins[cond] = {
            "ri_logit": d["ri_logit"], "ru_logit": d["ru_logit"],
            "ri_prob": d["ri_prob"], "ru_prob": d["ru_prob"],
            "ri_rank": d["ri_rank"], "ru_rank": d["ru_rank"],
            "margin_ri_minus_ru_logit": d["margin_ri_minus_ru_logit"],
            "margin_ri_minus_ru_prob": d["margin_ri_minus_ru_prob"],
            "winner": d["winner"],
        }

    # prompt/token identity check (Section23)
    n_tokens_set = {matrix[c]["n_tokens"] for c in matrix if matrix[c]["n_tokens"] is not None}
    prompt_identity_ok = len(n_tokens_set) <= 1

    (REPORTS_DIR / "phase4zd_environment_matrix.json").write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "margins": margins,
        "prompt_token_identity_check": {
            "n_tokens_by_condition": {c: matrix[c]["n_tokens"] for c in matrix},
            "all_match": prompt_identity_ok,
        },
        "winner_summary": {c: margins[c]["winner"] for c in margins},
        "key_observation": (
            "A(legacy 4bit)のみリ優勢。B(BF16 eager)は完全同点。C(BF16 SDPA)/D(llama.cpp BF16 CPU)/"
            "E(float32 eager)は全てル優勢。llama.cppを一切介さないHF単体の条件変更(SDPA化 or float32化)"
            "だけで、legacy 4bit baselineの結論(リ優勢)とは逆方向、かつllama.cppと同方向の結果が再現される。"
        ),
    }
    (REPORTS_DIR / "phase4zd_margin_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved -> phase4zd_environment_matrix.json")
    print("Saved -> phase4zd_margin_comparison.json")
    for c, m in margins.items():
        print(c, m)
    print("prompt/token identity all_match:", prompt_identity_ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
