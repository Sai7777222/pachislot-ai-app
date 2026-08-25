"""Phase 4Q-4: o8/o4 学習前検証 (preflight)。

異常があれば学習せず停止する (exit code != 0)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parents[1]

EXPECTED_TRAIN_SHA256 = "d331fef4296c67a2bca396cc974ab6ba970f7b945530f55583b8f67333090314"
EXPECTED_VAL_SHA256 = "3df5c2a8d598046319574cafd7dc516fa0d2585b3a6e9bf53db14271c3b613eb"
EXPECTED_CANDIDATE_SHA256 = "341c44d06243f9d0492737ad244790ae77270ed44a2e46af205bdd16ace4a8be"
EXPECTED_SYSTEM_PROMPT_MD5 = "f3ea72a9ea9a400fcfae0018896350b8"

EXPECTED_ADAPTER_SHA256 = {
    "v1": "9037adaa1f85299a803294fa4a88c32fc1de82c499c9cb320e2c6d3d55d2a6c1",
    "v2": "ecbfef5ff208fbff08c56553f53d6f83a925e346bb30a867a9a389bd16b7be8d",
    "v3": "7d348705de552bc4c2a33a4546845dc09809385664b042193731e26337c055aa",
    "v4": "b5f1646cf823e4b382cdac91ab973e9859cf60aebce665ba8cc7e2240d6b5bec",
    "v5-qkv": "358b0610d55496252324d92c006d409d31ee4d22032e9d8b5d856bc7f4d97774",
}

ALLOWED_DIFF_PATHS = {
    "._comment",
    ".base_model.note",
    ".lora.target_modules",
    ".lora.rank_pattern",
    ".lora.alpha_pattern",
    ".lora.mlp_modules_note",
    ".loss.description",
    ".data.format",
    ".output.adapter_dir",
    ".output.checkpoint_dir",
    ".output.log_dir",
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


def compare(prefix, a, b, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a.keys()) | set(b.keys())
        for k in sorted(keys):
            compare(f"{prefix}.{k}", a.get(k), b.get(k), diffs)
    elif a != b:
        diffs.append({"path": prefix, "v4": a, "candidate": b})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=["o8", "o4"])
    parser.add_argument("--expected-o-rank", type=int, required=True)
    args = parser.parse_args()

    ok = True
    report: dict = {"candidate": args.candidate}

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

    train_path = TRAINING_ROOT / "processed" / "riru_train_v4.jsonl"
    val_path = TRAINING_ROOT / "processed" / "riru_val_v4.jsonl"
    candidate_path = TRAINING_ROOT / "processed" / "riru_lora_v4_candidate.jsonl"

    n_train = count_lines(train_path)
    n_val = count_lines(val_path)
    report["train_count"] = n_train
    report["val_count"] = n_val
    if n_train != 823 or n_val != 91:
        ok = False

    train_sha = sha256_file(train_path)
    val_sha = sha256_file(val_path)
    cand_sha = sha256_file(candidate_path)
    report["train_sha256_match"] = train_sha == EXPECTED_TRAIN_SHA256
    report["val_sha256_match"] = val_sha == EXPECTED_VAL_SHA256
    report["candidate_sha256_match"] = cand_sha == EXPECTED_CANDIDATE_SHA256
    if not (
        report["train_sha256_match"]
        and report["val_sha256_match"]
        and report["candidate_sha256_match"]
    ):
        ok = False

    sp_path = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
    report["system_prompt_md5_match"] = md5_file(sp_path) == EXPECTED_SYSTEM_PROMPT_MD5
    if not report["system_prompt_md5_match"]:
        ok = False

    adapter_check = {}
    for name, expected in EXPECTED_ADAPTER_SHA256.items():
        p = TRAINING_ROOT / f"lora-riru-qwen-{name}" / "adapter_model.safetensors"
        actual = sha256_file(p) if p.is_file() else None
        adapter_check[name] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            ok = False
    report["adapter_sha256_check"] = adapter_check

    v4_config = json.loads(
        (TRAINING_ROOT / "configs" / "qlora_config_v4.json").read_text(encoding="utf-8")
    )
    cand_config_path = TRAINING_ROOT / "configs" / f"qlora_config_{args.candidate}.json"
    cand_config = json.loads(cand_config_path.read_text(encoding="utf-8"))
    diffs = []
    compare("", v4_config, cand_config, diffs)
    report["config_diffs"] = diffs
    unexpected = [d for d in diffs if d["path"] not in ALLOWED_DIFF_PATHS]
    report["unexpected_config_diffs"] = unexpected
    if unexpected:
        ok = False

    report["rank_pattern"] = cand_config["lora"].get("rank_pattern")
    report["alpha_pattern"] = cand_config["lora"].get("alpha_pattern")
    rank_ok = cand_config["lora"].get("rank_pattern", {}).get("o_proj") == args.expected_o_rank
    alpha_ok = cand_config["lora"].get("alpha_pattern", {}).get("o_proj") == args.expected_o_rank
    report["o_rank_alpha_as_expected"] = rank_ok and alpha_ok
    if not report["o_rank_alpha_as_expected"]:
        ok = False
    report["qkv_r_16_alpha_16"] = (
        cand_config["lora"]["r"] == 16 and cand_config["lora"]["lora_alpha"] == 16
    )
    if not report["qkv_r_16_alpha_16"]:
        ok = False

    for key in (
        "num_train_epochs", "learning_rate", "optim", "max_seq_length", "seed",
        "per_device_train_batch_size", "gradient_accumulation_steps",
    ):
        if v4_config["training"][key] != cand_config["training"][key]:
            ok = False
            report.setdefault("training_param_mismatches", []).append(key)
    if v4_config["lora"]["lora_dropout"] != cand_config["lora"]["lora_dropout"]:
        ok = False
        report.setdefault("training_param_mismatches", []).append("lora_dropout")

    report["overall_status"] = "READY" if ok else "TRAINING ABORT"
    out_path = TRAINING_ROOT / "reports" / f"phase4q_pretrain_checks_{args.candidate}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overall_status": report["overall_status"]}, ensure_ascii=False))
    print(f"Full report -> {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
