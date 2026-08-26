"""Phase 4X-17: ratio_high_identity_stable 学習前preflight。
1項目でも重大FAILなら学習せず停止する(exit code != 0)。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parents[1]

EXPECTED_SYSTEM_PROMPT_MD5 = "f3ea72a9ea9a400fcfae0018896350b8"
EXPECTED_IDENTITY_CANDIDATE_SHA256 = (
    "2329d3c0513cb5a91cc4512957b68baa144747aecf172441de4fb87e40e8b7e7"
)
EXPECTED_ADAPTER_SHA256 = {
    "v4": "b5f1646cf823e4b382cdac91ab973e9859cf60aebce665ba8cc7e2240d6b5bec",
    "ratio-high": "b0c3e65764dec4a9c840aacdad6a7bbc27bc0ff1442165c4d9eac87684de2568",
    "ratio-high-identity": "ab4f55b8f948b50a70d14cb99758bc2165f575c1693d5ebd2a57834e5b4f9886",
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
        if not cuda_ok:
            ok = False
    except ImportError as exc:
        report["torch_import_error"] = str(exc)
        ok = False

    train_path = TRAINING_ROOT / "processed" / "riru_ratio_high_identity_stable_train.jsonl"
    val_path = TRAINING_ROOT / "processed" / "riru_ratio_high_identity_stable_val.jsonl"
    cand_path = TRAINING_ROOT / "processed" / "riru_ratio_high_identity_stable_candidate.jsonl"
    identity_cand_path = TRAINING_ROOT / "processed" / "riru_ratio_high_identity_candidate.jsonl"

    # dataset parse OK
    try:
        report["train_count"] = count_lines(train_path)
        report["val_count"] = count_lines(val_path)
        report["candidate_count"] = count_lines(cand_path)
        for p in (train_path, val_path, cand_path):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        json.loads(line)
        report["dataset_parse_ok"] = True
    except (OSError, json.JSONDecodeError) as exc:
        report["dataset_parse_ok"] = False
        report["dataset_parse_error"] = str(exc)
        ok = False

    if report.get("candidate_count") != 1095:
        ok = False

    report["ratio_high_identity_candidate_sha256"] = sha256_file(identity_cand_path)
    report["ratio_high_identity_candidate_unchanged"] = (
        report["ratio_high_identity_candidate_sha256"] == EXPECTED_IDENTITY_CANDIDATE_SHA256
    )
    if not report["ratio_high_identity_candidate_unchanged"]:
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

    # dataset quality re-verify (0 issues expected)
    quality_path = TRAINING_ROOT / "reports" / "phase4x_dataset_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    n_issues = sum(len(v) if isinstance(v, list) else 0 for v in quality["issues"].values())
    n_issues += len(quality["high_similarity_pairs"]) + len(quality["contamination_hits"])
    report["dataset_quality_issue_count"] = n_issues
    if n_issues != 0:
        ok = False

    summary_path = TRAINING_ROOT / "reports" / "phase4x_dataset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report["complex_ratio_pct"] = summary["complex_ratio_pct"]
    report["complex_ratio_ok"] = summary["complex_ratio_pct"] >= 10.0
    report["train_val_overlap"] = summary["train_val_overlap"]
    report["train_val_overlap_ok"] = summary["train_val_overlap"] == 0
    if not (report["complex_ratio_ok"] and report["train_val_overlap_ok"]):
        ok = False

    # config diff check: stable vs ratio_high_identity (LoRA構造が完全一致することを確認)
    identity_cfg_path = TRAINING_ROOT / "configs" / "qlora_config_ratio_high_identity.json"
    identity_cfg = json.loads(identity_cfg_path.read_text(encoding="utf-8"))
    stable_cfg_path = TRAINING_ROOT / "configs" / "qlora_config_ratio_high_identity_stable.json"
    stable_cfg = json.loads(stable_cfg_path.read_text(encoding="utf-8"))
    lora_match = (
        identity_cfg["lora"]["target_modules"] == stable_cfg["lora"]["target_modules"]
        and identity_cfg["lora"]["r"] == stable_cfg["lora"]["r"]
        and identity_cfg["lora"]["lora_alpha"] == stable_cfg["lora"]["lora_alpha"]
        and identity_cfg["lora"]["lora_dropout"] == stable_cfg["lora"]["lora_dropout"]
        and "rank_pattern" not in stable_cfg["lora"]
        and "alpha_pattern" not in stable_cfg["lora"]
    )
    report["lora_config_matches_ratio_high_identity"] = lora_match
    if not lora_match:
        ok = False
    for key in (
        "num_train_epochs", "learning_rate", "optim", "max_seq_length", "seed",
        "per_device_train_batch_size", "gradient_accumulation_steps",
    ):
        if identity_cfg["training"][key] != stable_cfg["training"][key]:
            ok = False
            report.setdefault("training_param_mismatches", []).append(key)

    # output directoryが既存adapterを指していないことを確認
    output_dir = stable_cfg["output"]["adapter_dir"]
    report["output_dir"] = output_dir
    existing_adapter_dirs = {
        "training/riru/lora-riru-qwen-v4",
        "training/riru/lora-riru-qwen-ratio-high",
        "training/riru/lora-riru-qwen-ratio-high-identity",
    }
    report["output_dir_collision"] = output_dir in existing_adapter_dirs
    if report["output_dir_collision"]:
        ok = False
    output_path = PROJECT_ROOT / output_dir
    report["output_dir_already_exists"] = output_path.exists()
    if report["output_dir_already_exists"]:
        ok = False

    report["overall_status"] = "READY" if ok else "TRAINING ABORT"
    out_path = TRAINING_ROOT / "reports" / "phase4x_pretrain_checks.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overall_status": report["overall_status"]}, ensure_ascii=False))
    print(f"Full report -> {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
