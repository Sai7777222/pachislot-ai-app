"""Phase 4D: Qwen2.5-14B-Instruct (HF形式, 学習用) を取得する。

GGUF (推論専用、D:\\AI\\models\\llm\\qwen2.5-14b-instruct-q4_k_m-*.gguf) とは
完全に別ディレクトリで管理し、GGUF側には一切手を加えない。

公式リポジトリ: Qwen/Qwen2.5-14B-Instruct (Apache License 2.0)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen2.5-14B-Instruct"
DEST_DIR = Path(r"D:\AI\models\llm-hf\Qwen2.5-14B-Instruct")

# 学習に必要なファイルのみ取得 (README/画像等の付随物は除外)
ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.txt",
    "tokenizer*",
    "merges.txt",
    "vocab.json",
    "LICENSE",
]


def main() -> int:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    hf_home = os.getenv("HF_HOME", r"D:\AI\cache\huggingface")
    os.environ["HF_HOME"] = hf_home

    print(f"Downloading {REPO_ID} -> {DEST_DIR}")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(DEST_DIR),
        allow_patterns=ALLOW_PATTERNS,
        cache_dir=hf_home,
    )
    print("Download complete.")

    total_bytes = sum(f.stat().st_size for f in DEST_DIR.rglob("*") if f.is_file())
    print(f"Total size: {total_bytes / (1024**3):.2f} GB")
    for f in sorted(DEST_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size / (1024**2):.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
