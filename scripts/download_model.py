"""Qwen2.5-14B-Instruct-GGUF Q4_K_M (3分割) を D:\\AI\\models\\llm に取得.

このモデルは単一ファイルではなく q4_k_m-0000{1,2,3}-of-00003.gguf の
3分割で配布されている。llama.cpp / llama-cpp-python は先頭シャード
(00001-of-00003) を指定すれば残りを自動的に読み込む。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

REPO_ID = "Qwen/Qwen2.5-14B-Instruct-GGUF"
FILENAMES = [
    "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
    "qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf",
    "qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf",
]
# LLM_MODEL_PATH はこの先頭シャードを指す
FIRST_SHARD = FILENAMES[0]


def main() -> int:
    hf_home = os.getenv("HF_HOME", r"D:\AI\cache\huggingface")
    models_llm = Path(os.getenv("MODELS_DIR", r"D:\AI\models")) / "llm"
    models_llm.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = hf_home

    print(f"Downloading {REPO_ID} ({len(FILENAMES)} shards)")
    print(f"Cache: {hf_home}")
    print(f"Destination: {models_llm}")

    for filename in FILENAMES:
        dest = models_llm / filename
        if dest.is_file():
            print(f"Already exists, skipping: {dest} ({dest.stat().st_size / (1024**3):.2f} GB)")
            continue

        print(f"Downloading {filename} ...")
        # local_dir を指定して実体をそのまま D:\AI\models\llm に置く
        # （cache_dir 経由だと Windows でシンボリックリンクが作られ、
        # 後から移動すると壊れたリンクになるため）
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            cache_dir=hf_home,
            local_dir=str(models_llm),
        )

        if not dest.is_file():
            print(f"ERROR: Download failed, file not found: {dest}")
            return 1

        print(f"Saved: {dest} ({dest.stat().st_size / (1024**3):.2f} GB)")

    first_shard_path = models_llm / FIRST_SHARD
    if not first_shard_path.is_file():
        print(f"ERROR: First shard missing after download: {first_shard_path}")
        return 1

    print("-" * 60)
    print("All shards downloaded.")
    print(f"Set LLM_MODEL_PATH to: {first_shard_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
