"""Phase 4U-8: ratio_high_identity 学習前検証。異常があれば学習せず停止する。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parents[1]

EXPECTED_SYSTEM_PROMPT_MD5 = "f3ea72a9ea9a400fcfae0018896350b8"
EXPECTED_HIGH_CANDIDATE_SHA256 = (
    "368b8aab7e5aeada1f8ff2ddff05b1d234d04a45b470fc57f15ef161e6c61c3a"
)
EXPECTED_ADAPTER_SHA256 = {
    "v1": "9037adaa1f85299a803294fa4a88c32fc1de82c499c9cb320e2c6d3d55d2a6c1",
    "v2": "ecbfef5ff208fbff08c56553f53d6f83a925e346bb30a867a9a389bd16b7be8d",
    "v3": "7d348705de552bc4c2a33a4546845dc09809385664b042193731e26337c055aa",
    "v4": "b5f1646cf823e4b382cdac91ab973e9859cf60aebce665ba8cc7e2240d6b5bec",
    "v5-qkv": "358b0610d55496252324d92c006d409d31ee4d22032e9d8b5d856bc7f4d97774",
    "o8": "9177ddea8d302b43279b135f465e38e9b3106e81c93f4cd49ed4190622df854d",
    "o4": "daa3efcde43d7fd1189c9fbe8d0d4bc090be25548c3e18a813beb86a306f18f8",
    "ratio-mid": "578b465b5c92863726759b5f41c560aa94551bff1b538a0acf67b27b5820eb39",
    "ratio-high": "b0c3e65764dec4a9c840aacdad6a7bbc27bc0ff1442165c4d9eac87684de2568",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    return sum(1 for line in open(path, encoding="utf-8") if line.strip())


def main() -> int:
    ok = True
    report: dict = {}

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
        report["cuda_available"] = cuda_ok
        report["gpu_name"] = gpu_name
        if not cuda_ok or (gpu_name and "5090" not in gpu_name):
            ok = False
    except ImportError as exc:
        report["torch_import_error"] = str(exc)
        ok = False

    train_path = TRAINING_ROOT / "processed" / "riru_ratio_high_identity_train.jsonl"
    val_path = TRAINING_ROOT / "processed" / "riru_ratio_high_identity_val.jsonl"
    cand_path = TRAINING_ROOT / "processed" / "riru_ratio_high_identity_candidate.jsonl"
    high_cand_path = TRAINING_ROOT / "processed" / "riru_ratio_high_candidate.jsonl"

    report["train_count"] = count_lines(train_path)
    report["val_count"] = count_lines(val_path)
    report["candidate_count"] = count_lines(cand_path)
    if report["candidate_count"] != 1070:
        ok = False

    report["ratio_high_candidate_sha256"] = sha256_file(high_cand_path)
    report["ratio_high_candidate_unchanged"] = (
        report["ratio_high_candidate_sha256"] == EXPECTED_HIGH_CANDIDATE_SHA256
    )
    if not report["ratio_high_candidate_unchanged"]:
        ok = False

    sp_md5 = md5_file(PROJECT_ROOT / "config" / "prompts" / "system.jinja2")
    report["system_prompt_md5"] = sp_md5
    report["system_prompt_unchanged"] = sp_md5 == EXPECTED_SYSTEM_PROMPT_MD5
    if not report["system_prompt_unchanged"]:
        ok = False

    adapter_check = {}
    for name, expected in EXPECTED_ADAPTER_SHA256.items():
        p = TRAINING_ROOT / f"lora-riru-qwen-{name}" / "adapter_model.safetensors"
        actual = sha256_file(p) if p.is_file() else None
        adapter_check[name] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            ok = False
    report["adapter_sha256_check"] = adapter_check

    # dataset quality re-verify (0 issues expected, from build_phase4u_dataset.py's last run)
    quality_path = TRAINING_ROOT / "reports" / "phase4u_dataset_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    n_issues = sum(len(v) if isinstance(v, list) else 0 for v in quality["issues"].values())
    n_issues += len(quality["high_similarity_pairs"]) + len(quality["contamination_hits"])
    report["dataset_quality_issue_count"] = n_issues
    if n_issues != 0:
        ok = False

    summary_path = TRAINING_ROOT / "reports" / "phase4u_dataset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report["complex_ratio_pct"] = summary["complex_ratio_pct"]
    report["complex_ratio_ok"] = summary["complex_ratio_pct"] >= 10.0
    report["train_val_overlap"] = summary["train_val_overlap"]
    report["train_val_overlap_ok"] = summary["train_val_overlap"] == 0
    if not (report["complex_ratio_ok"] and report["train_val_overlap_ok"]):
        ok = False

    # config diff check: ratio_high_identity vs ratio_high (lora設定が一致することを確認)
    high_cfg_path = TRAINING_ROOT / "configs" / "qlora_config_ratio_high.json"
    high_cfg = json.loads(high_cfg_path.read_text(encoding="utf-8"))
    identity_cfg = json.loads(
        (TRAINING_ROOT / "configs" / "qlora_config_ratio_high_identity.json").read_text(
            encoding="utf-8"
        )
    )
    lora_match = (
        high_cfg["lora"]["target_modules"] == identity_cfg["lora"]["target_modules"]
        and high_cfg["lora"]["r"] == identity_cfg["lora"]["r"]
        and high_cfg["lora"]["lora_alpha"] == identity_cfg["lora"]["lora_alpha"]
        and high_cfg["lora"]["lora_dropout"] == identity_cfg["lora"]["lora_dropout"]
        and "rank_pattern" not in identity_cfg["lora"]
        and "alpha_pattern" not in identity_cfg["lora"]
    )
    report["lora_config_matches_ratio_high"] = lora_match
    if not lora_match:
        ok = False
    for key in (
        "num_train_epochs", "learning_rate", "optim", "max_seq_length", "seed",
        "per_device_train_batch_size", "gradient_accumulation_steps",
    ):
        if high_cfg["training"][key] != identity_cfg["training"][key]:
            ok = False
            report.setdefault("training_param_mismatches", []).append(key)

    report["overall_status"] = "READY" if ok else "TRAINING ABORT"
    out_path = TRAINING_ROOT / "reports" / "phase4u_pretrain_checks.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overall_status": report["overall_status"]}, ensure_ascii=False))
    print(f"Full report -> {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
