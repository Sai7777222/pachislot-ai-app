"""Phase 4ZD Section19-20: HF eager vs HF SDPA のlayer-wise hidden state比較。

llama.cppを一切使わず、Phase4ZCで既に取得済みの
phase4zc_hf_hidden_states_eager_bf16.safetensors / phase4zc_hf_hidden_states_sdpa_bf16.safetensors
(同一merged HF BF16 weight、同一E36 forced-prefixプロンプト、attn_implementationのみ相違)を再利用し、
attention backendの変更だけでPhase4ZCと同種の分散型driftが生じるかを検証する。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
REPORTS_DIR = TRAINING_ROOT / "reports"

EAGER_FILE = REPORTS_DIR / "phase4zc_hf_hidden_states_eager_bf16.safetensors"
SDPA_FILE = REPORTS_DIR / "phase4zc_hf_hidden_states_sdpa_bf16.safetensors"


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff ** 2)))
    na, nb_ = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    l2_rel = float(np.linalg.norm(diff) / na) if na else float("nan")
    cos = float(np.dot(a, b) / (na * nb_)) if na > 0 and nb_ > 0 else float("nan")
    return {"max_abs_diff": max_abs, "mean_abs_diff": mean_abs, "rms_diff": rms,
            "l2_relative_error": l2_rel, "cosine_similarity": cos}


def main() -> int:
    eager = {}
    with safe_open(str(EAGER_FILE), framework="pt") as f:
        for k in f.keys():
            eager[k] = f.get_tensor(k).float().numpy().astype(np.float64)
    sdpa = {}
    with safe_open(str(SDPA_FILE), framework="pt") as f:
        for k in f.keys():
            sdpa[k] = f.get_tensor(k).float().numpy().astype(np.float64)

    layer_rows = []
    for il in range(48):
        key = f"layer_{il:02d}_output"
        m = metrics(eager[key], sdpa[key])
        m["layer"] = il
        layer_rows.append(m)

    norm_metrics = metrics(eager["post_final_norm"], sdpa["post_final_norm"])
    logits_metrics = metrics(eager["logits_full"], sdpa["logits_full"])

    RI_ID, RU_ID = 36723, 32610
    logits_token_compare = {
        "ri_eager_logit": float(eager["logits_full"][RI_ID]), "ri_sdpa_logit": float(sdpa["logits_full"][RI_ID]),
        "ru_eager_logit": float(eager["logits_full"][RU_ID]), "ru_sdpa_logit": float(sdpa["logits_full"][RU_ID]),
        "margin_eager": float(eager["logits_full"][RI_ID] - eager["logits_full"][RU_ID]),
        "margin_sdpa": float(sdpa["logits_full"][RI_ID] - sdpa["logits_full"][RU_ID]),
    }

    max_abs_series = [r["max_abs_diff"] for r in layer_rows]
    first_nonzero = next((r["layer"] for r in layer_rows if r["max_abs_diff"] > 1e-6), None)
    first_10x_jump = None
    for i in range(1, len(layer_rows)):
        prev = max_abs_series[i - 1] if max_abs_series[i - 1] > 1e-9 else 1e-9
        if max_abs_series[i] > prev * 10:
            first_10x_jump = layer_rows[i]["layer"]
            break
    first_cosine_drop = next((r["layer"] for r in layer_rows if r["cosine_similarity"] < 0.9999), None)

    result = {
        "purpose": "llama.cppを使わないHF eager vs HF SDPA(同一BF16 weight)のlayer-wise比較。Phase4ZCのHF-vs-llama.cpp比較と同種のパターンが再現するかを確認する。",
        "layer_rows": layer_rows,
        "post_final_norm_metrics": norm_metrics,
        "logits_metrics": logits_metrics,
        "logits_token_compare": logits_token_compare,
        "first_divergence": {
            "first_nonzero_diff_layer": first_nonzero,
            "first_10x_jump_layer": first_10x_jump,
            "first_cosine_drop_below_0.9999_layer": first_cosine_drop,
        },
        "comparison_to_phase4zc_hf_vs_llamacpp": {
            "phase4zc_first_nonzero_layer": 0,
            "phase4zc_first_10x_jump_layer": None,
            "phase4zc_first_cosine_drop_layer": 27,
            "note": "Phase4ZCではHF(eager)とllama.cpp(CPU GGUF)の比較で同様の値を得た。本比較(HF eager vs HF SDPA、llama.cpp不使用)で同種のパターンが出れば、drift要因がllama.cpp固有ではなくattention実装差自体にあることの追加証拠となる。"
        },
    }
    out_path = REPORTS_DIR / "phase4zd_hf_backend_hidden_diff.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print("first_nonzero:", first_nonzero, "first_10x_jump:", first_10x_jump, "first_cosine_drop:", first_cosine_drop)
    print("logits_token_compare:", logits_token_compare)
    for r in layer_rows:
        print(r["layer"], f"max_abs={r['max_abs_diff']:.6e}", f"cos={r['cosine_similarity']:.8f}",
              f"l2_rel={r['l2_relative_error']:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
