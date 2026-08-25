"""Phase 4O-17: v5-qkv adapter tensor一覧確認。

q_proj/k_proj/v_projのLoRA tensorが存在し、o_proj LoRA tensorが0件であることを
safetensorsファイルを直接開いて (フルモデルロードなしで) 確認する。
"""

from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open

TRAINING_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = TRAINING_ROOT / "lora-riru-qwen-v5-qkv" / "adapter_model.safetensors"
REPORTS_DIR = TRAINING_ROOT / "reports"


def main() -> int:
    with safe_open(str(ADAPTER_PATH), framework="pt") as f:
        names = list(f.keys())

    module_counts = {"q_proj": 0, "k_proj": 0, "v_proj": 0, "o_proj": 0}
    for n in names:
        for m in module_counts:
            if f".{m}." in n:
                module_counts[m] += 1

    report = {
        "adapter_path": str(ADAPTER_PATH),
        "total_tensors": len(names),
        "module_tensor_counts": module_counts,
        "o_proj_tensor_count_is_zero": module_counts["o_proj"] == 0,
        "q_k_v_all_present": all(module_counts[m] > 0 for m in ("q_proj", "k_proj", "v_proj")),
        "sample_tensor_names": names[:10],
    }
    out_path = REPORTS_DIR / "phase4o_tensor_check.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (report["o_proj_tensor_count_is_zero"] and report["q_k_v_all_present"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
