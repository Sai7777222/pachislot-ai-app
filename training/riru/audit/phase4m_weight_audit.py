"""Phase 4M-1/4M-2: adapter実体・内部重みの監査。

v1/v2/v3/v4のadapter_model.safetensorsを直接読み取り (フルモデルロード不要)、
ファイル識別情報とLoRAテンソルの統計量・SHA-256を比較する。
QLoRA/LoRA学習は行わない。adapterファイルは一切変更しない (読み取りのみ)。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from safetensors import safe_open

TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = TRAINING_ROOT / "reports"

VERSIONS = ["v1", "v2", "v3", "v4"]
SAMPLE_LAYERS = [0, 5, 10, 20, 30]  # Qwen2.5-14Bの層インデックスから抜粋
SAMPLE_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tensor_stats(path: Path, tensor_names: list[str]) -> dict:
    out = {}
    with safe_open(str(path), framework="pt") as f:
        available = set(f.keys())
        for name in tensor_names:
            if name not in available:
                out[name] = {"error": "tensor not found"}
                continue
            t = f.get_tensor(name)
            t_bytes = t.numpy().tobytes()
            out[name] = {
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "mean": float(t.float().mean()),
                "std": float(t.float().std()),
                "norm": float(t.float().norm()),
                "sha256": hashlib.sha256(t_bytes).hexdigest(),
            }
    return out


def build_candidate_tensor_names(all_keys: list[str]) -> list[str]:
    names = []
    for key in all_keys:
        if any(f".{layer}." in key for layer in [f"layers.{i}" for i in SAMPLE_LAYERS]) and any(
            m in key for m in SAMPLE_MODULES
        ):
            names.append(key)
    return sorted(names)


def main() -> int:
    file_identity = {}
    tensor_stats = {}
    all_keys_by_version = {}

    for v in VERSIONS:
        model_path = TRAINING_ROOT / f"lora-riru-qwen-{v}" / "adapter_model.safetensors"
        config_path = TRAINING_ROOT / f"lora-riru-qwen-{v}" / "adapter_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        stat = model_path.stat()
        file_identity[v] = {
            "path": str(model_path),
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256_model": sha256_file(model_path),
            "sha256_config": sha256_file(config_path),
            "lora_r": config.get("r"),
            "lora_alpha": config.get("lora_alpha"),
            "lora_dropout": config.get("lora_dropout"),
            "target_modules": sorted(config.get("target_modules", [])),
            "base_model_name_or_path": config.get("base_model_name_or_path"),
        }
        with safe_open(str(model_path), framework="pt") as f:
            all_keys_by_version[v] = sorted(f.keys())

    # 全safetensorsファイルのSHA-256が異なることを確認 (重大異常チェック)
    model_shas = {v: file_identity[v]["sha256_model"] for v in VERSIONS}
    sha_values = list(model_shas.values())
    all_distinct = len(set(sha_values)) == len(sha_values)

    # サンプルテンソル名を決定 (v2のキー一覧から、v1/v2/v3/v4で共通する層をサンプリング)
    common_keys = set(all_keys_by_version["v2"])
    for v in VERSIONS:
        common_keys &= set(all_keys_by_version[v])
    sample_names = build_candidate_tensor_names(sorted(common_keys))

    for v in VERSIONS:
        model_path = TRAINING_ROOT / f"lora-riru-qwen-{v}" / "adapter_model.safetensors"
        tensor_stats[v] = load_tensor_stats(model_path, sample_names)

    # v2/v3/v4間でのテンソル一致有無を判定
    tensor_diff_summary = {}
    for name in sample_names:
        shas = {v: tensor_stats[v].get(name, {}).get("sha256") for v in ("v2", "v3", "v4")}
        tensor_diff_summary[name] = {
            "sha256_by_version": shas,
            "v2_v3_identical": shas["v2"] == shas["v3"],
            "v2_v4_identical": shas["v2"] == shas["v4"],
            "v3_v4_identical": shas["v3"] == shas["v4"],
        }

    any_identical_pair = any(
        d["v2_v3_identical"] or d["v2_v4_identical"] or d["v3_v4_identical"]
        for d in tensor_diff_summary.values()
    )

    report = {
        "file_identity": file_identity,
        "adapter_model_sha256_all_distinct": all_distinct,
        "sample_tensor_names_checked": sample_names,
        "sample_tensor_count": len(sample_names),
        "tensor_diff_summary": tensor_diff_summary,
        "any_identical_tensor_pair_found": any_identical_pair,
        "tensor_stats_full": tensor_stats,
        "total_tensor_count_per_version": {v: len(all_keys_by_version[v]) for v in VERSIONS},
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "phase4m_weight_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"adapter_model_sha256_all_distinct: {all_distinct}")
    print(f"sample_tensor_count: {len(sample_names)}")
    print(f"any_identical_tensor_pair_found (should be False): {any_identical_pair}")
    for v in VERSIONS:
        sha_short = file_identity[v]["sha256_model"][:16]
        size = file_identity[v]["size_bytes"]
        print(f"{v}: sha256={sha_short}... size={size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
