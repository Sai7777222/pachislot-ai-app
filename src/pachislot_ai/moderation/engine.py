"""Phase4FM: モデレーションエンジン本体。

決定的ルールマッチングのみ(LLM呼び出しなし、外部APIなし、ファジー分類なし)。
入力チェックはdispatch/RAG/生成より前に、出力チェックは生成後・ユーザー表示前に
呼び出される(呼び出し側はChatService、Section9/Section11)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pachislot_ai.moderation.matcher import rule_matches
from pachislot_ai.moderation.policy import ModerationPolicy, ModerationRule, load_policy


@dataclass(frozen=True, slots=True)
class ModerationResult:
    """内部専用の判定結果。rule_id/categoryはログ用であり、ユーザーには一切
    露出しない(Section9/Section13: 内部ルールID・カテゴリ・検出ロジックを
    非公開にする)。"""

    allowed: bool
    rule_id: str | None
    category: str | None
    policy_action: str  # "ALLOWED" | "HARD_BLOCK_INPUT" | "HARD_BLOCK_OUTPUT"
    safe_response: str | None  # allowed=Falseのときのみ、ユーザーに見せてよい代替応答


_ALLOWED_RESULT = ModerationResult(
    allowed=True, rule_id=None, category=None, policy_action="ALLOWED", safe_response=None
)


class ModerationEngine:
    def __init__(self, policy: ModerationPolicy) -> None:
        self._policy = policy
        self._rules = policy.enabled_rules()

    @classmethod
    def from_yaml(cls, path: Path) -> "ModerationEngine":
        return cls(load_policy(path))

    def _first_match(self, text: str, side: str) -> ModerationRule | None:
        for rule in self._rules:
            side_policy = rule.input_policy if side == "input" else rule.output_policy
            if side_policy.value != "HARD_BLOCK":
                continue
            if rule_matches(text, rule):
                return rule
        return None

    def check_input(self, text: str) -> ModerationResult:
        """Section9: HARD_BLOCK_INPUT対象のルールにのみ反応する
        (input_policy=HARD_BLOCKのルールだけがここでブロックしうる。
        SUPPRESS_ECHO対象語=input_policy=ALLOWのルールは、ここでは通過させ、
        Section10の通りoutput側チェックに判断を委ねる)。"""
        if not text:
            return _ALLOWED_RESULT
        rule = self._first_match(text, "input")
        if rule is None:
            return _ALLOWED_RESULT
        fallback = self._policy.fallbacks.get(rule.fallback_id) or self._policy.fallbacks["default_block"]
        return ModerationResult(
            allowed=False, rule_id=rule.id, category=rule.category,
            policy_action="HARD_BLOCK_INPUT", safe_response=fallback,
        )

    def check_output(self, text: str) -> ModerationResult:
        """Section11: output_policy=HARD_BLOCKのルール(HARD_BLOCK_INPUT語・
        SUPPRESS_ECHO語の両方を含む)に反応する。生成後・ユーザー表示前に
        必ず1回呼ばれる(Section12のstreamingバッファリング済み全文に対して)。"""
        if not text:
            return _ALLOWED_RESULT
        rule = self._first_match(text, "output")
        if rule is None:
            return _ALLOWED_RESULT
        fallback = self._policy.fallbacks.get(rule.fallback_id) or self._policy.fallbacks["default_output_block"]
        return ModerationResult(
            allowed=False, rule_id=rule.id, category=rule.category,
            policy_action="HARD_BLOCK_OUTPUT", safe_response=fallback,
        )
