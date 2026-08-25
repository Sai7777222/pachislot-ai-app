"""llama-cpp-python (CUDA ビルド) を使ったローカル LLM プロバイダー。

Phase 0 で動作確認済みの GPU (RTX 5090) オフロード環境をそのまま利用する。
`llama_cpp.Llama` はスレッドセーフではなく単一の KV キャッシュを共有するため、
`threading.Lock` で同時実行を直列化する。
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import AsyncIterator
from pathlib import Path

from pachislot_ai.core.exceptions import LLMGenerationError, ModelNotLoadedError
from pachislot_ai.llm.base import ChatCompletionResult, ChatMessage, LLMProvider

logger = logging.getLogger(__name__)

_STREAM_DONE = object()


class LocalLlamaCppProvider(LLMProvider):
    """`D:\\AI\\models\\llm` 配下の GGUF モデルを直接ロードして推論する。"""

    def __init__(
        self,
        model_path: Path,
        *,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        default_max_tokens: int = 512,
        default_temperature: float = 0.7,
        chat_format: str | None = None,
    ) -> None:
        if not model_path.is_file():
            raise ModelNotLoadedError(f"LLM model file not found: {model_path}")

        import llama_cpp  # 遅延インポート: CUDA 未ビルド環境でも他機能をテスト可能にする

        self._model_path = model_path
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature
        self._lock = threading.Lock()

        self.gpu_offload_supported = llama_cpp.llama_supports_gpu_offload()
        self.n_gpu_layers = n_gpu_layers

        # chat_format=None の場合、llama-cpp-python は GGUF に埋め込まれた
        # tokenizer.chat_template (モデル配布元が変換時に含めたテンプレート) を
        # 自動使用する。モデルごとに正しいテンプレートを使うため、Qwen用の
        # 設定をSwallowに流用しない (逆も同様)。
        logger.info(
            "Loading LLM model: %s (n_gpu_layers=%s, chat_format=%s)",
            model_path,
            n_gpu_layers,
            chat_format or "auto (embedded in GGUF)",
        )
        self._llm = llama_cpp.Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_format,
            verbose=False,
        )
        logger.info(
            "LLM model loaded. gpu_offload_supported=%s", self.gpu_offload_supported
        )

    @property
    def model_name(self) -> str:
        return self._model_path.stem

    def _to_llama_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletionResult:
        def _run() -> ChatCompletionResult:
            with self._lock:
                try:
                    output = self._llm.create_chat_completion(
                        messages=self._to_llama_messages(messages),
                        max_tokens=max_tokens or self._default_max_tokens,
                        temperature=(
                            self._default_temperature
                            if temperature is None
                            else temperature
                        ),
                        stream=False,
                    )
                except Exception as exc:  # noqa: BLE001 - llama.cpp 側の例外を包む
                    raise LLMGenerationError(str(exc)) from exc

            choice = output["choices"][0]["message"]
            usage = output.get("usage", {})
            return ChatCompletionResult(
                content=choice.get("content") or "",
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )

        return await asyncio.to_thread(_run)

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        q: queue.Queue = queue.Queue()

        def _produce() -> None:
            with self._lock:
                try:
                    stream = self._llm.create_chat_completion(
                        messages=self._to_llama_messages(messages),
                        max_tokens=max_tokens or self._default_max_tokens,
                        temperature=(
                            self._default_temperature
                            if temperature is None
                            else temperature
                        ),
                        stream=True,
                    )
                    for chunk in stream:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            q.put(content)
                except Exception as exc:  # noqa: BLE001
                    q.put(LLMGenerationError(str(exc)))
                finally:
                    q.put(_STREAM_DONE)

        thread = threading.Thread(target=_produce, daemon=True)
        thread.start()

        while True:
            item = await asyncio.to_thread(q.get)
            if item is _STREAM_DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def health_check(self) -> bool:
        return self._llm is not None
