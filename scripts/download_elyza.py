"""Llama-3-ELYZA-JP-8B (Q4_K_M GGUF, ELYZA公式配布) を D:\\AI\\models\\llm\\elyza に取得。

4モデル比較 (Phase 3.7) 用。既存のQwen/Swallow/LLM-jpモデルには一切手を触れない。

GGUF配布元: elyza/Llama-3-ELYZA-JP-8B-GGUF (ELYZA公式による直接配布、第三者量子化ではない)
- 元モデル: elyza/Llama-3-ELYZA-JP-8B (meta-llama/Meta-Llama-3-8B-Instructの日本語継続事前学習)
- 変換内容: llama.cppによる量子化のみ
- ライセンス: Meta Llama 3 Community License (公式モデルと整合)
詳細は LICENSE_NOTES.md §2.6 を参照。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

REPO_ID = "elyza/Llama-3-ELYZA-JP-8B-GGUF"
FILENAME = "Llama-3-ELYZA-JP-8B-q4_k_m.gguf"
LICENSE_FILES = ["LICENSE", "USE_POLICY.md", "Notice"]


def main() -> int:
    hf_home = os.getenv("HF_HOME", r"D:\AI\cache\huggingface")
    dest_dir = Path(os.getenv("MODELS_DIR", r"D:\AI\models")) / "llm" / "elyza"
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
