"""Phase 4ZC Section13-15: HF(PyTorch, full BF16, eager)とllama.cpp(BF16 GGUF, CPU,
llama-debug経由の公式cb_evalコールバックによるtensor dump)のlayer-wise hidden state比較。

各layer(0-47)の最終トークン位置の出力ベクトル(l_out-<il> / layer_<il>_output)、
post-final-norm(result_norm / post_final_norm)、logits(result_output / logits_full)について
max_abs_diff / mean_abs_diff / rms_diff / cosine_similarity / L2_relative_error を計算し、
"first divergence layer"(first-nonzero, first-10x-jump, first-cosine-drop)を特定する。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from safetensors import safe_open

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = EVAL_DIR.parents[0]
REPORTS_DIR = TRAINING_ROOT / "reports"
DUMP_DIR = REPORTS_DIR / "_phase4zc_llamacpp_dump"

HF_SAFETENSORS = REPORTS_DIR / "phase4zc_hf_hidden_states_eager_bf16.safetensors"


def read_manifest(dump_dir: Path) -> dict:
    """manifest.txtは2パス分(warmup + real)追記されているため、各tensor名の
    「最後に出現したエントリ」(=実データ、後述の理由によりファイル内容も最後の書き込みで上書き済み)
    を採用する。"""
    entries = {}
    for line in (dump_dir / "manifest.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        name = parts[0]
        ne = [int(x) for x in parts[2].split("=")[1].split(",")]
        entries[name] = {"ne": ne, "file": parts[4].split("=")[1]}
    return entries


def load_llamacpp_last_token_vec(dump_dir: Path, manifest: dict, tensor_name: str) -> np.ndarray:
    info = manifest[tensor_name]
    ne = info["ne"]  # [n_embd_or_vocab, n_tokens_or_1, 1, 1]
    raw = (dump_dir / info["file"]).read_bytes()
    arr = np.frombuffer(raw, dtype=np.float32)
    n0, n1 = ne[0], ne[1]
    assert arr.size == n0 * n1, f"{tensor_name}: size mismatch {arr.size} vs {n0 * n1}"
    arr = arr.reshape(n1, n0)  # row-major: [token, dim] since nb[0]=4 (dim fastest)
    return arr[-1, :].astype(np.float64)  # last token position


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff ** 2)))
    denom = float(np.linalg.norm(a)) or 1.0
    l2_rel = float(np.linalg.norm(diff) / denom)
    na = np.linalg.norm(a)
    nb_ = np.linalg.norm(b)
    cos = float(np.dot(a, b) / (na * nb_)) if na > 0 and nb_ > 0 else float("nan")
    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "rms_diff": rms,
        "l2_relative_error": l2_rel,
        "cosine_similarity": cos,
    }


def main() -> int:
    manifest = read_manifest(DUMP_DIR)

    hf_tensors = {}
    with safe_open(str(HF_SAFETENSORS), framework="pt") as f:
        for key in f.keys():
            hf_tensors[key] = f.get_tensor(key).float().numpy().astype(np.float64)

    layer_rows = []
    for il in range(48):
        hf_vec = hf_tensors[f"layer_{il:02d}_output"]
        cpp_vec = load_llamacpp_last_token_vec(DUMP_DIR, manifest, f"l_out-{il}")
        m = metrics(hf_vec, cpp_vec)
        m["layer"] = il
        m["hf_vec_norm"] = float(np.linalg.norm(hf_vec))
        m["cpp_vec_norm"] = float(np.linalg.norm(cpp_vec))
        layer_rows.append(m)

    # post-final-norm and logits
    hf_norm = hf_tensors["post_final_norm"]
    cpp_norm = load_llamacpp_last_token_vec(DUMP_DIR, manifest, "result_norm")
    norm_metrics = metrics(hf_norm, cpp_norm)

    hf_logits = hf_tensors["logits_full"]
    cpp_logits = load_llamacpp_last_token_vec(DUMP_DIR, manifest, "result_output")
    logits_metrics = metrics(hf_logits, cpp_logits)

    RI_ID, RU_ID = 36723, 32610
    logits_token_compare = {
        "ri_hf_logit": float(hf_logits[RI_ID]), "ri_cpp_logit": float(cpp_logits[RI_ID]),
        "ru_hf_logit": float(hf_logits[RU_ID]), "ru_cpp_logit": float(cpp_logits[RU_ID]),
        "ri_minus_ru_hf": float(hf_logits[RI_ID] - hf_logits[RU_ID]),
        "ri_minus_ru_cpp": float(cpp_logits[RI_ID] - cpp_logits[RU_ID]),
    }

    # first divergence layer detection
    max_abs_series = [r["max_abs_diff"] for r in layer_rows]
    first_nonzero = next((r["layer"] for r in layer_rows if r["max_abs_diff"] > 1e-6), None)
    first_10x_jump = None
    for i in range(1, len(layer_rows)):
        prev = max_abs_series[i - 1] if max_abs_series[i - 1] > 1e-9 else 1e-9
        if max_abs_series[i] > prev * 10:
            first_10x_jump = layer_rows[i]["layer"]
            break
    first_cosine_drop = next(
        (r["layer"] for r in layer_rows if r["cosine_similarity"] < 0.9999), None
    )

    result = {
        "layer_rows": layer_rows,
        "post_final_norm_metrics": norm_metrics,
        "logits_metrics": logits_metrics,
        "logits_token_compare": logits_token_compare,
        "first_divergence": {
            "first_nonzero_diff_layer": first_nonzero,
            "first_10x_jump_layer": first_10x_jump,
            "first_cosine_drop_below_0.9999_layer": first_cosine_drop,
        },
    }
    out_path = REPORTS_DIR / "phase4zc_layerwise_hidden_diff.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print("first_nonzero:", first_nonzero, "first_10x_jump:", first_10x_jump,
          "first_cosine_drop:", first_cosine_drop)
    print("logits_token_compare:", logits_token_compare)
    for r in layer_rows:
        print(r["layer"], f"max_abs={r['max_abs_diff']:.6e}", f"cos={r['cosine_similarity']:.8f}",
              f"l2_rel={r['l2_relative_error']:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
