"""Llama-3.1-Swallow-8B-Instruct-v0.5 (Q4_K_M GGUF) を D:\\AI\\models\\llm\\swallow に取得。

A/B比較 (Phase 3.5) 用。既存のQwenモデルには一切手を触れない。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

REPO_ID = "okamototk/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf"
FILENAME = "Llama-3.1-Swallow-8B-Instruct-v0.5_Q4_K_M.gguf"
LICENSE_FILES = ["LICENSE", "GEMMA_TERMS_OF_USE.md"]


def main() -> int:
    hf_home = os.getenv("HF_HOME", r"D:\AI\cache\huggingface")
    dest_dir = Path(os.getenv("MODELS_DIR", r"D:\AI\models")) / "llm" / "swallow"
    dest_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = hf_home

    print(f"Downloading {REPO_ID}/{FILENAME}")
    print(f"Destination: {dest_dir}")

    dest = dest_dir / FILENAME
    if not dest.is_file():
        hf_hub_download(
            repo_id=REPO_ID, filename=FILENAME, cache_dir=hf_home, local_dir=str(dest_dir)
        )
    else:
        print(f"Already exists, skipping: {dest}")

    for lf in LICENSE_FILES:
        lf_dest = dest_dir / lf
        if not lf_dest.is_file():
            hf_hub_download(
                repo_id=REPO_ID, filename=lf, cache_dir=hf_home, local_dir=str(dest_dir)
            )
        else:
            print(f"Already exists, skipping: {lf_dest}")

    if not dest.is_file():
        print("ERROR: download failed")
        return 1

    print(f"Saved: {dest} ({dest.stat().st_size / (1024**3):.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
