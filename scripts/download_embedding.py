"""intfloat/multilingual-e5-base を D:\\AI\\models\\embedding に取得."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

REPO_ID = "intfloat/multilingual-e5-base"
LOCAL_DIRNAME = "multilingual-e5-base"


def main() -> int:
    hf_home = os.getenv("HF_HOME", r"D:\AI\cache\huggingface")
    models_dir = Path(os.getenv("MODELS_DIR", r"D:\AI\models"))
    dest = models_dir / "embedding" / LOCAL_DIRNAME
    dest.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = hf_home

    print(f"Downloading {REPO_ID}")
    print(f"Cache: {hf_home}")
    print(f"Destination: {dest}")

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(dest),
        cache_dir=hf_home,
    )

    total_bytes = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"Saved: {dest}")
    print(f"Total size: {total_bytes / (1024**3):.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
