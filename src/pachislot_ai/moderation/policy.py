"""Phase4FM: モデレーションポリシー設定(config/moderation.yaml)のスキーマ・読み込み。

ポリシーデータはコードにハードコードせず、専用configファイルへ分離する(Section6)。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml


class MatchForm(str, Enum):
    """Section8のmatch mode。"""

    EXACT = "exact"
    TOKEN_BOUNDARY = "token_boundary"
    NORMALIZED_SEQUENCE = "normalized_sequence"


class SidePolicy(str, Enum):
    """入力/出力それぞれ独立した二値ポリシー。Section4のA-D分類は、この2値の
    組み合わせから導かれる(HARD_BLOCK/HARD_BLOCK=A、ALLOW/HARD_BLOCK=B、
    HARD_BLOCK/ALLOW=想定しないが安全側、ALLOW/ALLOW=D)。"""

    HARD_BLOCK = "HARD_BLOCK"
    ALLOW = "ALLOW"


@dataclass(frozen=True, slots=True)
class ModerationRule:
    id: str
    category: str
    match_form: MatchForm
    terms: tuple[str, ...]
    input_policy: SidePolicy
    output_policy: SidePolicy
    enabled: bool
    fallback_id: str

    @property
    def policy_class(self) -> str:
        """Section4のA-D表記に対応するラベル(ログ・テスト用、ユーザーには非公開)。"""
        if self.input_policy is SidePolicy.HARD_BLOCK:
            return "HARD_BLOCK_INPUT"
        if self.output_policy is SidePolicy.HARD_BLOCK:
            return "SUPPRESS_ECHO"
        return "ALLOW_CONTEXTUAL"


@dataclass(frozen=True, slots=True)
class ModerationPolicy:
    rules: tuple[ModerationRule, ...]
    fallbacks: dict[str, str]

    def enabled_rules(self) -> list[ModerationRule]:
        return [r for r in self.rules if r.enabled]


_DEFAULT_FALLBACK_INPUT = "ごめんね、それについては答えられないんだ。"
_DEFAULT_FALLBACK_OUTPUT = "ごめんね、うまく答えられなかったみたい。もう一度違う聞き方をしてもらえる？"


def load_policy(path: Path) -> ModerationPolicy:
    """config/moderation.yaml を読み込み、構造化されたModerationPolicyを返す。
    シークレットは一切扱わない(禁止語ルールのみ)。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fallbacks = dict(raw.get("fallbacks") or {})
    fallbacks.setdefault("default_block", _DEFAULT_FALLBACK_INPUT)
    fallbacks.setdefault("default_output_block", _DEFAULT_FALLBACK_OUTPUT)

    rules: list[ModerationRule] = []
    for row in raw.get("rules") or []:
        rules.append(
            ModerationRule(
                id=str(row["id"]),
                category=str(row.get("category", "uncategorized")),
                match_form=MatchForm(row["match_form"]),
                terms=tuple(str(t) for t in row["terms"]),
                input_policy=SidePolicy(row.get("input_policy", "ALLOW")),
                output_policy=SidePolicy(row.get("output_policy", "ALLOW")),
                enabled=bool(row.get("enabled", True)),
                fallback_id=str(row.get("fallback_id", "default_block")),
            )
        )
    return ModerationPolicy(rules=tuple(rules), fallbacks=fallbacks)
