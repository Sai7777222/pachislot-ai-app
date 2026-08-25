"""Phase 4Q-3: PEFT rank_pattern/alpha_pattern の実装確認 (学習前の必須検証)。

実際にQwen2.5-14B-InstructへLoraConfig(rank_pattern={"o_proj": R}, ...)を適用し、
q_proj/k_proj/v_projがr=16のまま、o_projだけが指定rankになっていることを
lora_A/lora_B の実tensor形状から実測する。推測でconfigを書いて学習開始しない、
という指示に従い、学習ジョブ開始前に必ずこのスクリプトで検証する。

学習は一切行わない。モデルはロードするがadapterは保存しない。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL_PATH = r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"

TEST_CASES = [
    {"name": "o8", "o_rank": 8, "o_alpha": 8},
    {"name": "o4", "o_rank": 4, "o_alpha": 4},
]


def inspect_layers(model, o_rank_expected: int, o_alpha_expected: int) -> dict:
    sample = {"q_proj": None, "k_proj": None, "v_proj": None, "o_proj": None}
    for name, module in model.named_modules():
        for mt in sample:
            if name.endswith(mt) and sample[mt] is None and hasattr(module, "lora_A"):
                adapter_name = next(iter(module.lora_A.keys()))
                lora_A = module.lora_A[adapter_name].weight
                lora_B = module.lora_B[adapter_name].weight
                sample[mt] = {
                    "module_name": name,
                    "lora_A_shape": list(lora_A.shape),
                    "lora_B_shape": list(lora_B.shape),
                    "r": module.r[adapter_name],
                    "lora_alpha": module.lora_alpha[adapter_name],
                    "scaling": module.scaling[adapter_name],
                }
    ok = (
        sample["q_proj"]["r"] == 16
        and sample["k_proj"]["r"] == 16
        and sample["v_proj"]["r"] == 16
        and sample["o_proj"]["r"] == o_rank_expected
        and sample["o_proj"]["lora_alpha"] == o_alpha_expected
    )
    return {"sample_layers": sample, "verification_passed": ok}


def main() -> int:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    report = {}
    for case in TEST_CASES:
        print(f"=== verifying rank_pattern for case={case['name']} ===")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, quantization_config=quant_config, device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.08,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            rank_pattern={"o_proj": case["o_rank"]},
            alpha_pattern={"o_proj": case["o_alpha"]},
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        result = inspect_layers(model, case["o_rank"], case["o_alpha"])
        report[case["name"]] = result
        brief = {k: v for k, v in result.items() if k != "sample_layers"}
        print(json.dumps(brief, ensure_ascii=False))
        print(json.dumps(result["sample_layers"], ensure_ascii=False, indent=2))
        del model
        torch.cuda.empty_cache()

    all_ok = all(report[c["name"]]["verification_passed"] for c in TEST_CASES)
    report["all_verification_passed"] = all_ok
    out_path = REPORTS_DIR / "phase4q_peft_pattern_verify.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print(f"all_verification_passed: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
