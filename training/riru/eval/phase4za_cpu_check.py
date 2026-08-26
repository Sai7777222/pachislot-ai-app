"""Phase 4ZA Section4: CPU-only(n_gpu_layers=0)が実際にGPU offload=0であることを
verboseログとGPUメモリ計測で確認する。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
GGUF_PATH = str(TRAINING_ROOT / "gguf" / "riru-qwen-final-bf16.gguf")


def gpu_mem_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def main() -> int:
    from llama_cpp import Llama

    before = gpu_mem_used_mib()
    print(f"GPU mem before load: {before} MiB")

    t0 = time.perf_counter()
    llm = Llama(model_path=GGUF_PATH, n_gpu_layers=0, n_ctx=2048, verbose=True)
    load_time = time.perf_counter() - t0
    print(f"n_ctx actually set: {llm.n_ctx()}")

    after = gpu_mem_used_mib()
    print(f"GPU mem after load: {after} MiB")
    delta = after - before

    report = {
        "n_gpu_layers_requested": 0,
        "gpu_mem_before_mib": before,
        "gpu_mem_after_mib": after,
        "gpu_mem_delta_mib": delta,
        "load_time_sec": round(load_time, 2),
        "llama_cpp_python_version": __import__("llama_cpp").__version__,
        "gguf_path": GGUF_PATH,
    }
    out_path = TRAINING_ROOT / "reports" / "_phase4za_cpu_load_check_utf8.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
