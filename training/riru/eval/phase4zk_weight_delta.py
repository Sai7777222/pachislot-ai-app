"""Phase 4ZK Diagnostic C: Phase4ZG adapterとPhase4ZJ adapterのLoRA weight直接比較。

重要な方法論上の注意: LoRAのB@A分解には回転の自由度がある(B→BR, A→R^-1A
としてもB@Aは不変)。全candidateはbase modelから毎回新規に学習しており、
ZG/ZJは独立した学習トラジェクトリを持つため、生のlora_A/lora_Bテンソルを
直接比較しても意味がない(たとえ実質的に同じ機能的更新であっても、無関係な
回転によって見かけ上コサイン類似度が低くなる)。

したがって本スクリプトは、各layer×moduleについて実効的な重み更新
ΔW = B @ A を計算し、ΔW_ZG と ΔW_ZJ を比較する(これは回転不変)。
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = TRAINING_ROOT / "reports"

ZG_PATH = TRAINING_ROOT / "lora-riru-qwen-phase4zg-identity-hardened" / "adapter_model.safetensors"
ZJ_PATH = TRAINING_ROOT / "lora-riru-qwen-phase4zj-instruction-override-hardened" / "adapter_model.safetensors"

KEY_PATTERN = re.compile(r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.lora_(A|B)\.weight")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_delta_w_map(state_dict: dict) -> dict[tuple[int, str], torch.Tensor]:
    """各(layer, module)についてΔW = B @ A (out_dim x in_dim)を計算する。"""
    pairs: dict[tuple[int, str], dict[str, torch.Tensor]] = defaultdict(dict)
    for key, tensor in state_dict.items():
        m = KEY_PATTERN.search(key)
        if not m:
            continue
        layer = int(m.group(1))
        module = m.group(2)
        ab = m.group(3)
        pairs[(layer, module)][ab] = tensor

    delta_w = {}
    for (layer, module), d in pairs.items():
        if "A" not in d or "B" not in d:
            continue
        A = d["A"].to(DEVICE).float()
        B = d["B"].to(DEVICE).float()
        delta_w[(layer, module)] = (B @ A).cpu()
    return delta_w


def main() -> int:
    zg_sd = load_file(str(ZG_PATH))
    zj_sd = load_file(str(ZJ_PATH))

    dw_zg = build_delta_w_map(zg_sd)
    dw_zj = build_delta_w_map(zj_sd)

    common_keys = sorted(set(dw_zg.keys()) & set(dw_zj.keys()))
    print(f"ΔW pairs: ZG={len(dw_zg)}, ZJ={len(dw_zj)}, common={len(common_keys)}")

    per_layer_module = []
    module_stats = defaultdict(list)

    for layer, module in common_keys:
        a = dw_zg[(layer, module)]
        b = dw_zj[(layer, module)]
        delta = b - a
        a_norm = float(torch.norm(a))
        b_norm = float(torch.norm(b))
        delta_norm = float(torch.norm(delta))
        max_abs = float(delta.abs().max())
        mean_abs = float(delta.abs().mean())
        rms = float(torch.sqrt((delta ** 2).mean()))
        cos_sim = float(torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)))
        relative_norm_delta = delta_norm / a_norm if a_norm > 1e-12 else float("nan")

        stat = {"layer": layer, "module": module,
                "zg_deltaW_norm": round(a_norm, 4), "zj_deltaW_norm": round(b_norm, 4),
                "delta_norm": round(delta_norm, 4), "relative_norm_delta": round(relative_norm_delta, 6),
                "cosine_similarity": round(cos_sim, 6),
                "max_abs_delta": round(max_abs, 6), "mean_abs_delta": round(mean_abs, 8), "rms_delta": round(rms, 8)}
        per_layer_module.append(stat)
        module_stats[module].append(stat)

    module_summary = {}
    for module, stats in module_stats.items():
        module_summary[module] = {
            "n_layers": len(stats),
            "mean_zg_deltaW_norm": round(sum(s["zg_deltaW_norm"] for s in stats) / len(stats), 4),
            "mean_zj_deltaW_norm": round(sum(s["zj_deltaW_norm"] for s in stats) / len(stats), 4),
            "mean_relative_norm_delta": round(sum(s["relative_norm_delta"] for s in stats) / len(stats), 6),
            "max_relative_norm_delta": round(max(s["relative_norm_delta"] for s in stats), 6),
            "mean_cosine_similarity": round(sum(s["cosine_similarity"] for s in stats) / len(stats), 6),
            "min_cosine_similarity": round(min(s["cosine_similarity"] for s in stats), 6),
            "max_cosine_similarity": round(max(s["cosine_similarity"] for s in stats), 6),
        }

    # layer-wise trend (early/mid/late layers)
    n_layers = max(k[0] for k in common_keys) + 1
    layer_bucket_stats = {"early(0-15)": [], "mid(16-31)": [], "late(32-47)": []}
    for s in per_layer_module:
        if s["layer"] < 16:
            layer_bucket_stats["early(0-15)"].append(s["cosine_similarity"])
        elif s["layer"] < 32:
            layer_bucket_stats["mid(16-31)"].append(s["cosine_similarity"])
        else:
            layer_bucket_stats["late(32-47)"].append(s["cosine_similarity"])
    layer_trend = {k: round(sum(v) / len(v), 6) if v else None for k, v in layer_bucket_stats.items()}

    out = {
        "zg_path": str(ZG_PATH), "zj_path": str(ZJ_PATH),
        "method": "effective ΔW=B@A per (layer,module), rotation-invariant comparison",
        "n_layers_detected": n_layers,
        "module_summary": module_summary,
        "layer_trend_mean_cosine_similarity": layer_trend,
        "per_layer_module": per_layer_module,
    }
    out_path = REPORTS_DIR / "phase4zk_adapter_weight_delta.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(json.dumps(module_summary, indent=2))
    print("layer trend (mean cosine similarity):", layer_trend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
