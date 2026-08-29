"""Phase4FM: 決定的な製品モデレーション層。

RAG/学習/モデルアラインメント研究とは独立した、別レイヤーの製品安全機構。
外部モデレーションAPIやLLM分類器は使わない(Section2/Section7)。
"""

from __future__ import annotations

from pachislot_ai.moderation.engine import ModerationEngine, ModerationResult
from pachislot_ai.moderation.policy import (
    MatchForm,
    ModerationPolicy,
    ModerationRule,
    SidePolicy,
    load_policy,
)

__all__ = [
    "ModerationEngine",
    "ModerationResult",
    "ModerationPolicy",
    "ModerationRule",
    "MatchForm",
    "SidePolicy",
    "load_policy",
]
