"""Phase 4ZB Section7-11: merged HF (safetensors) と BF16 GGUF の全tensorを
数値的に比較する。読み取り専用。既存資産(merged HF/BF16 GGUF)は一切変更しない。

公式gguf-py(llama.cppに同梱、新規インストールなし)のGGUFReader/dequantize/
TensorNameMapを用いて、推測ではなく公式のtensor名対応関係を使用する。

メモリ節約のため、tensorを1つずつストリーム処理する(全tensorを同時に
メモリへ展開しない)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"D:\AI\tools\llama.cpp\gguf-py")
import gguf  # noqa: E402
from safetensors import safe_open  # noqa: E402

TRAINING_ROOT = Path(__file__).resolve().parents[1]
HF_SAFETENSORS_PATH = TRAINING_ROOT / "merged" / "riru-qwen-final-hf" / "model.safetensors"
GGUF_PATH = TRAINING_ROOT / "gguf" / "riru-qwen-final-bf16.gguf"
REPORTS_DIR = TRAINING_ROOT / "reports"

N_LAYERS = 48  # Qwen2.5-14B

RI_TOKEN_ID = 36723  # 「リ」(tokenizerで直接確認、推測ではない)
RU_TOKEN_ID = 32610  # 「ル」(tokenizerで直接確認、推測ではない)


def hf_tensor_to_f32(t: torch.Tensor) -> np.ndarray:
    """torch bfloat16/他dtype tensorをlosslessにfloat32 numpy配列へ変換する。
    torch自体のbfloat16->float32アップキャストは定義上exactである。"""
    return t.float().numpy()


def classify_tensor_type(gguf_name: str) -> str:
    if gguf_name in ("token_embd.weight",):
        return "token_embeddings"
    if gguf_name in ("output.weight",):
        return "output_head"
    if gguf_name in ("output_norm.weight",):
        return "final_norm"
    if ".attn_q." in gguf_name:
        return "attention_q"
    if ".attn_k." in gguf_name:
        return "attention_k"
    if ".attn_v." in gguf_name:
        return "attention_v"
    if ".attn_output." in gguf_name:
        return "attention_output"
    if ".attn_norm." in gguf_name:
        return "attn_norm"
    if ".ffn_gate." in gguf_name:
        return "mlp_gate"
    if ".ffn_up." in gguf_name:
        return "mlp_up"
    if ".ffn_down." in gguf_name:
        return "mlp_down"
    if ".ffn_norm." in gguf_name:
        return "ffn_norm"
    return "other"


def extract_layer(gguf_name: str) -> int | None:
    if gguf_name.startswith("blk."):
        try:
            return int(gguf_name.split(".")[1])
        except (IndexError, ValueError):
            return None
    return None


def main() -> int:
    tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN2, N_LAYERS)

    reader = gguf.GGUFReader(str(GGUF_PATH), mode="r")
    gguf_tensors_by_name = {t.name: t for t in reader.tensors}

    hf_file = safe_open(str(HF_SAFETENSORS_PATH), framework="pt")
    hf_keys = list(hf_file.keys())

    # --- Section7: tensor inventory ---
    hf_inventory = []
    for k in hf_keys:
        slc = hf_file.get_slice(k)
        hf_inventory.append(
            {"name": k, "shape": list(slc.get_shape()), "dtype": str(slc.get_dtype())}
        )

    gguf_inventory = []
    for t in reader.tensors:
        gguf_inventory.append({
            "name": t.name, "shape": [int(x) for x in t.shape][::-1],
            "tensor_type": int(t.tensor_type), "n_elements": int(t.n_elements),
            "n_bytes": int(t.n_bytes),
        })

    # --- name mapping ---
    mapping_results = []
    missing_in_gguf = []
    matched_pairs = []
    for k in hf_keys:
        gguf_name = tensor_map.get_name(key=k, try_suffixes=(".weight", ".bias"))
        if gguf_name is None:
            missing_in_gguf.append(k)
            continue
        if gguf_name not in gguf_tensors_by_name:
            missing_in_gguf.append(k)
            mapping_results.append({"hf_name": k, "gguf_name_expected": gguf_name, "found": False})
            continue
        mapping_results.append({"hf_name": k, "gguf_name_expected": gguf_name, "found": True})
        matched_pairs.append((k, gguf_name))

    mapped_gguf_names = {gn for _, gn in matched_pairs}
    unexpected_gguf_tensors = [n for n in gguf_tensors_by_name if n not in mapped_gguf_names]

    inventory_out = {
        "hf_tensor_count": len(hf_keys),
        "gguf_tensor_count": len(gguf_tensors_by_name),
        "matched_pairs_count": len(matched_pairs),
        "missing_in_gguf_count": len(missing_in_gguf),
        "missing_in_gguf": missing_in_gguf,
        "unexpected_gguf_tensors_count": len(unexpected_gguf_tensors),
        "unexpected_gguf_tensors": unexpected_gguf_tensors,
        "hf_inventory_sample": hf_inventory[:5],
        "gguf_inventory_sample": gguf_inventory[:5],
    }
    (REPORTS_DIR / "phase4zb_tensor_inventory.json").write_text(
        json.dumps(inventory_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"HF tensors: {len(hf_keys)}, GGUF tensors: {len(gguf_tensors_by_name)}, "
          f"matched: {len(matched_pairs)}, missing_in_gguf: {len(missing_in_gguf)}, "
          f"unexpected: {len(unexpected_gguf_tensors)}")

    if missing_in_gguf or unexpected_gguf_tensors:
        print("STOP: tensor mapping incomplete, see phase4zb_tensor_inventory.json")
        return 1

    # --- Section8-10: weight diff, streamed per-tensor ---
    per_tensor_results = []
    global_max_abs_diff = 0.0
    global_sum_abs_diff = 0.0
    global_sum_sq_diff = 0.0
    global_n_elements = 0
    global_n_differing = 0
    global_exact_identical_count = 0
    global_non_identical_count = 0

    for hf_name, gguf_name in matched_pairs:
        hf_tensor = hf_file.get_tensor(hf_name)  # torch tensor, native bfloat16
        gguf_t = gguf_tensors_by_name[gguf_name]

        hf_f32 = hf_tensor_to_f32(hf_tensor).reshape(-1)

        gguf_f32 = gguf.dequantize(gguf_t.data, gguf_t.tensor_type).astype(np.float32).reshape(-1)

        shape_match = hf_f32.shape == gguf_f32.shape
        if not shape_match:
            per_tensor_results.append({
                "hf_name": hf_name, "gguf_name": gguf_name, "shape_match": False,
                "hf_n": int(hf_f32.shape[0]), "gguf_n": int(gguf_f32.shape[0]),
            })
            continue

        diff = np.abs(hf_f32.astype(np.float64) - gguf_f32.astype(np.float64))
        max_abs_diff = float(diff.max()) if diff.size else 0.0
        mean_abs_diff = float(diff.mean()) if diff.size else 0.0
        rms = float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0
        n_differing = int(np.count_nonzero(diff > 0))
        exact_equal = n_differing == 0
        rel_err = float(diff.max() / (np.abs(hf_f32).max() + 1e-12)) if diff.size else 0.0

        per_tensor_results.append({
            "hf_name": hf_name, "gguf_name": gguf_name, "shape_match": True,
            "n_elements": int(hf_f32.size), "exact_equal": exact_equal,
            "max_abs_diff": max_abs_diff, "mean_abs_diff": mean_abs_diff, "rms_diff": rms,
            "differing_element_count": n_differing,
            "differing_element_ratio": round(n_differing / hf_f32.size, 6) if hf_f32.size else 0.0,
            "relative_error": rel_err,
            "tensor_type": classify_tensor_type(gguf_name),
            "layer": extract_layer(gguf_name),
        })

        global_max_abs_diff = max(global_max_abs_diff, max_abs_diff)
        global_sum_abs_diff += float(diff.sum())
        global_sum_sq_diff += float((diff ** 2).sum())
        global_n_elements += hf_f32.size
        global_n_differing += n_differing
        if exact_equal:
            global_exact_identical_count += 1
        else:
            global_non_identical_count += 1

        del hf_f32, gguf_f32, diff

    summary = {
        "exact_identical_tensor_count": global_exact_identical_count,
        "non_identical_tensor_count": global_non_identical_count,
        "global_max_abs_diff": global_max_abs_diff,
        "global_mean_abs_diff": (
            global_sum_abs_diff / global_n_elements if global_n_elements else None
        ),
        "global_rms_diff": (
            (global_sum_sq_diff / global_n_elements) ** 0.5 if global_n_elements else None
        ),
        "global_differing_element_ratio": (
            round(global_n_differing / global_n_elements, 6) if global_n_elements else None
        ),
        "total_elements_compared": global_n_elements,
    }

    by_tensor_type: dict[str, list] = {}
    by_layer: dict[str, list] = {}
    for r in per_tensor_results:
        if not r.get("shape_match", True):
            continue
        by_tensor_type.setdefault(r["tensor_type"], []).append(r)
        if r["layer"] is not None:
            by_layer.setdefault(str(r["layer"]), []).append(r)

    def agg(records):
        if not records:
            return None
        return {
            "count": len(records),
            "max_abs_diff": max(x["max_abs_diff"] for x in records),
            "mean_of_mean_abs_diff": sum(x["mean_abs_diff"] for x in records) / len(records),
            "exact_identical_count": sum(1 for x in records if x["exact_equal"]),
        }

    by_tensor_type_summary = {k: agg(v) for k, v in by_tensor_type.items()}
    by_layer_summary = {k: agg(v) for k, v in sorted(by_layer.items(), key=lambda x: int(x[0]))}

    # --- Section11: output embedding for リ/ル tokens ---
    output_weight_pair = next(
        (p for p in matched_pairs if p[1] == "output.weight"), None
    )
    token_embd_pair = next(
        (p for p in matched_pairs if p[1] == "token_embd.weight"), None
    )

    def token_row_compare(hf_name, gguf_name, token_id):
        hf_tensor = hf_file.get_tensor(hf_name)
        hf_row = hf_tensor_to_f32(hf_tensor[token_id])
        gguf_t = gguf_tensors_by_name[gguf_name]
        gguf_full = gguf.dequantize(gguf_t.data, gguf_t.tensor_type).astype(np.float32)
        gguf_row = gguf_full[token_id]
        diff = np.abs(hf_row.astype(np.float64) - gguf_row.astype(np.float64))
        cos_sim = float(
            np.dot(hf_row, gguf_row) / (np.linalg.norm(hf_row) * np.linalg.norm(gguf_row) + 1e-12)
        )
        return {
            "token_id": token_id,
            "exact_equal": bool(np.count_nonzero(diff) == 0),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "cosine_similarity": cos_sim,
        }

    token_analysis = {}
    if output_weight_pair:
        token_analysis["output_weight"] = {
            "ri_token_36723": token_row_compare(*output_weight_pair, RI_TOKEN_ID),
            "ru_token_32610": token_row_compare(*output_weight_pair, RU_TOKEN_ID),
        }
    if token_embd_pair:
        token_analysis["token_embd"] = {
            "ri_token_36723": token_row_compare(*token_embd_pair, RI_TOKEN_ID),
            "ru_token_32610": token_row_compare(*token_embd_pair, RU_TOKEN_ID),
        }

    out = {
        "summary": summary,
        "by_tensor_type": by_tensor_type_summary,
        "by_layer": by_layer_summary,
        "token_analysis_ri_ru": token_analysis,
        "shape_mismatches": [r for r in per_tensor_results if not r.get("shape_match", True)],
    }
    (REPORTS_DIR / "phase4zb_weight_diff_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(token_analysis, ensure_ascii=False, indent=2))
    print("Saved -> phase4zb_weight_diff_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
