"""Phase4FC3: production dispatch(会話モード判定)。

RAG context system messageを注入すべきでない会話(雑談・自己紹介等)を
安全側(precision優先)に判別するための軽量モジュール。
"""

from pachislot_ai.dispatch.conservative_dispatch import DispatchResult, dispatch

__all__ = ["DispatchResult", "dispatch"]
