"""Phase 4Q-6: o8/o4 adapter tensor実体監査。

adapter_config.jsonのconfigだけでなく、adapter_model.safetensorsを直接開いて
q_proj/k_proj/v_proj/o_prójそれぞれのtensor数・lora_A/lora_B shapeを実測する。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from safetensors import safe_open

TRAINING_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = TRAINING_ROOT / "reports"

MODULE_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj")
LAYER_PAT = re.compile(r"\.layers\.(\d+)\.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=["o8", "o4"])
    parser.add_argument("--expected-o-rank", type=int, required=True)
    args = parser.parse_args()

    adapter_dir = TRAINING_ROOT / f"lora-riru-qwen-{args.candidate}"
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"

    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))

    module_info = {
        mt: {"tensor_count": 0, "lora_A_shapes": set(), "lora_B_shapes": set()}
        for mt in MODULE_TYPES
    }
    with safe_open(str(safetensors_path), framework="pt") as f:
        names = list(f.keys())
        for n in names:
            for mt in MODULE_TYPES:
                if f".{mt}." in n:
                    module_info[mt]["tensor_count"] += 1
                    shape = tuple(f.get_slice(n).get_shape())
                    if "lora_A" in n:
                        module_info[mt]["lora_A_shapes"].add(shape)
                    elif "lora_B" in n:
                        module_info[mt]["lora_B_shapes"].add(shape)

    for mt in MODULE_TYPES:
        module_info[mt]["lora_A_shapes"] = sorted(module_info[mt]["lora_A_shapes"])
        module_info[mt]["lora_B_shapes"] = sorted(module_info[mt]["lora_B_shapes"])

    o_shapes = module_info["o_proj"]["lora_A_shapes"]
    o_rank_actual = o_shapes[0][0] if o_shapes else None
    qkv_rank_ok = all(
        module_info[mt]["lora_A_shapes"] and module_info[mt]["lora_A_shapes"][0][0] == 16
        for mt in ("q_proj", "k_proj", "v_proj")
    )
    o_rank_ok = o_rank_actual == args.expected_o_rank

    report = {
        "candidate": args.candidate,
        "adapter_config_target_modules": adapter_config.get("target_modules"),
        "adapter_config_rank_pattern": adapter_config.get("rank_pattern"),
        "adapter_config_alpha_pattern": adapter_config.get("alpha_pattern"),
        "total_tensors": len(names),
        "module_info": module_info,
        "o_proj_rank_actual": o_rank_actual,
        "o_proj_rank_expected": args.expected_o_rank,
        "o_proj_rank_matches": o_rank_ok,
        "qkv_rank_16_confirmed": qkv_rank_ok,
        "verification_passed": o_rank_ok and qkv_rank_ok,
    }
    out_path = REPORTS_DIR / f"phase4q_tensor_check_{args.candidate}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    brief = {k: v for k, v in report.items() if k != "module_info"}
    print(json.dumps(brief, ensure_ascii=False, default=str))
    print(f"Saved -> {out_path}")
    return 0 if report["verification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
