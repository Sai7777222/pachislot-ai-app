"""Phase 0: GPU ローカル LLM スモークテスト."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def main() -> int:
    model_path = os.getenv("LLM_MODEL_PATH", "")
    n_gpu_layers = int(os.getenv("LLM_N_GPU_LAYERS", "-1"))
    context_size = int(os.getenv("LLM_CONTEXT_SIZE", "8192"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "128"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    if not model_path:
        print("ERROR: LLM_MODEL_PATH is not set.")
        return 1

    path = Path(model_path)
    if not path.is_file():
        print(f"ERROR: Model file not found: {path}")
        return 1

    try:
        import llama_cpp
    except ImportError:
        print("ERROR: llama_cpp is not installed in the current environment.")
        return 1

    gpu_offload = llama_cpp.llama_supports_gpu_offload()
    print(f"llama_cpp version: {getattr(llama_cpp, '__version__', 'unknown')}")
    print(f"gpu_offload supported: {gpu_offload}")
    print(f"model: {path}")
    print(f"n_gpu_layers: {n_gpu_layers}")
    print("-" * 60)

    start = time.perf_counter()
    llm = llama_cpp.Llama(
        model_path=str(path),
        n_ctx=context_size,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    load_sec = time.perf_counter() - start
    print(f"Model loaded in {load_sec:.2f}s")

    prompt = "こんにちは。パチスロについて簡単に自己紹介してください。"
    print(f"Prompt: {prompt}")
    print("-" * 60)

    infer_start = time.perf_counter()
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    infer_sec = time.perf_counter() - infer_start

    content = output["choices"][0]["message"]["content"]
    usage = output.get("usage", {})

    print("Response:")
    print(content)
    print("-" * 60)
    print(f"Inference time: {infer_sec:.2f}s")
    if usage:
        print(f"Tokens: prompt={usage.get('prompt_tokens')}, completion={usage.get('completion_tokens')}")

    if not gpu_offload:
        print("WARNING: GPU offload is not supported; inference ran on CPU.")
        return 2

    print("SUCCESS: GPU smoke test completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
