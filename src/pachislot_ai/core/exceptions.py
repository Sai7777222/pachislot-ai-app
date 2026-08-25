"""アプリケーション共通の例外クラス。"""

from __future__ import annotations


class PachislotAIError(Exception):
    """アプリケーション独自例外の基底クラス。"""


class ModelNotLoadedError(PachislotAIError):
    """LLM モデルがロードされていない状態で推論が呼び出された。"""


class LLMGenerationError(PachislotAIError):
    """LLM 推論中にエラーが発生した。"""
