"""LLM-jp-3-13B-Instruct (Q4_K_M GGUF) を D:\\AI\\models\\llm\\llm-jp-3-13b に取得。

3モデル比較 (Phase 3.6) 用。既存のQwen/Swallowモデルには一切手を触れない。

GGUF配布元: alfredplpl/llm-jp-3-13b-instruct-gguf
- 変換元: 公式 llm-jp/llm-jp-3-13b-instruct と一致（README記載を確認済み）
- 変換内容: 量子化のみ（ファインチューニング・再学習なし）
- ライセンス: Apache License 2.0（公式モデルと整合）
詳細は LICENSE_NOTES.md §2.5 を参照。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

REPO_ID = "alfredplpl/llm-jp-3-13b-instruct-gguf"
FILENAME = "llm-jp-3-13b-instruct-Q4_K_M.gguf"


def main() -> int:
    hf_home = os.getenv("HF_HOME", r"D:\AI\cache\huggingface")
    dest_dir = Path(os.getenv("MODELS_DIR", r"D:\AI\models")) / "llm" / "llm-jp-3-13b"
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

    if not dest.is_file():
        print("ERROR: download failed")
        return 1

    print(f"Saved: {dest} ({dest.stat().st_size / (1024**3):.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
