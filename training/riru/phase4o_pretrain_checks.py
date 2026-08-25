"""Phase 4O-4: v5-qkv 学習前検証。

異常があれば学習せず停止する (exit code != 0)。

確認項目:
  - CUDA available / GPU名 (RTX5090であること)
  - train件数823 / validation件数91
  - train/val SHA-256がPhase4L時点の記録値と一致
  - system prompt (system.jinja2) 無変更 (MD5)
  - base modelパスがv4と同一
  - v4 config との差分が target_modules のみ
  - v1〜v4 adapter SHA-256 不変
"""

from __future__ import annotations

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

    # --- CUDA / GPU ---
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
        report["cuda_available"] = cuda_ok
        report["gpu_name"] = gpu_name
        if not cuda_ok:
            ok = False
        if gpu_name is not None and "5090" not in gpu_name:
            report["gpu_name_warning"] = f"RTX5090が期待されるが検出されたのは: {gpu_name}"
            ok = False
    except ImportError as exc:
        report["torch_import_error"] = str(exc)
        ok = False

    # --- train/val record counts ---
    train_path = TRAINING_ROOT / "processed" / "riru_train_v4.jsonl"
    val_path = TRAINING_ROOT / "processed" / "riru_val_v4.jsonl"
    candidate_path = TRAINING_ROOT / "processed" / "riru_lora_v4_candidate.jsonl"

    n_train = count_lines(train_path)
    n_val = count_lines(val_path)
    report["train_count"] = n_train
    report["val_count"] = n_val
    if n_train != 823:
        ok = False
        report["train_count_error"] = f"expected 823, got {n_train}"
    if n_val != 91:
        ok = False
        report["val_count_error"] = f"expected 91, got {n_val}"

    # --- SHA-256 checks (unchanged since Phase 4L/4K) ---
    train_sha = sha256_file(train_path)
    val_sha = sha256_file(val_path)
    candidate_sha = sha256_file(candidate_path)
    report["train_sha256"] = train_sha
    report["val_sha256"] = val_sha
    report["candidate_sha256"] = candidate_sha
    report["train_sha256_matches_phase4l"] = train_sha == EXPECTED_TRAIN_SHA256
    report["val_sha256_matches_phase4l"] = val_sha == EXPECTED_VAL_SHA256
    report["candidate_sha256_matches_phase4k"] = candidate_sha == EXPECTED_CANDIDATE_SHA256
    if not (
        report["train_sha256_matches_phase4l"]
        and report["val_sha256_matches_phase4l"]
        and report["candidate_sha256_matches_phase4k"]
    ):
        ok = False

    # --- system prompt unchanged ---
    system_prompt_path = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
    sp_md5 = md5_file(system_prompt_path)
    report["system_prompt_md5"] = sp_md5
    report["system_prompt_unchanged"] = sp_md5 == EXPECTED_SYSTEM_PROMPT_MD5
    if not report["system_prompt_unchanged"]:
        ok = False

    # --- adapter SHA-256 unchanged (v1-v4) ---
    adapter_check = {}
    for name, expected in EXPECTED_ADAPTER_SHA256.items():
        p = TRAINING_ROOT / f"lora-riru-qwen-{name}" / "adapter_model.safetensors"
        actual = sha256_file(p)
        adapter_check[name] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            ok = False
    report["adapter_sha256_check"] = adapter_check

    # --- config diff: v4 vs v5-qkv (only target_modules should differ) ---
    v4_config_path = TRAINING_ROOT / "configs" / "qlora_config_v4.json"
    v4_config = json.loads(v4_config_path.read_text(encoding="utf-8"))
    v5_config = json.loads(
        (TRAINING_ROOT / "configs" / "qlora_config_v5_qkv.json").read_text(encoding="utf-8")
    )
    diffs = []

    def compare(prefix, a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            keys = set(a.keys()) | set(b.keys())
            for k in sorted(keys):
                compare(f"{prefix}.{k}", a.get(k), b.get(k))
        elif a != b:
            diffs.append({"path": prefix, "v4": a, "v5_qkv": b})

    compare("", v4_config, v5_config)
    report["config_diffs"] = diffs

    allowed_diff_paths = {
        "._comment",
        ".base_model.note",
        ".lora.target_modules",
        ".lora.mlp_modules_note",
        ".loss.description",
        ".data.format",
        ".output.adapter_dir",
        ".output.checkpoint_dir",
        ".output.log_dir",
    }
    unexpected_diffs = [d for d in diffs if d["path"] not in allowed_diff_paths]
    report["unexpected_config_diffs"] = unexpected_diffs
    if unexpected_diffs:
        ok = False
    report["target_modules_v4"] = v4_config["lora"]["target_modules"]
    report["target_modules_v5_qkv"] = v5_config["lora"]["target_modules"]
    report["target_modules_diff_is_o_proj_only"] = (
        set(v4_config["lora"]["target_modules"]) - set(v5_config["lora"]["target_modules"])
    ) == {"o_proj"}
    if not report["target_modules_diff_is_o_proj_only"]:
        ok = False

    # --- base model path identical ---
    report["base_model_path_v4"] = v4_config["base_model"]["local_path"]
    report["base_model_path_v5_qkv"] = v5_config["base_model"]["local_path"]
    report["base_model_path_identical"] = (
        v4_config["base_model"]["local_path"] == v5_config["base_model"]["local_path"]
    )
    if not report["base_model_path_identical"]:
        ok = False
    base_model_dir = Path(v5_config["base_model"]["local_path"])
    report["base_model_dir_exists"] = base_model_dir.is_dir()
    if not report["base_model_dir_exists"]:
        ok = False

    report["overall_status"] = "READY" if ok else "NOT_READY"
    out_path = TRAINING_ROOT / "reports" / "phase4o_pretrain_checks.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overall_status": report["overall_status"]}, ensure_ascii=False))
    print(f"Full report -> {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
