"""Phase4ZT Section2-4: Policy C(UNKNOWN状態の処理方針)3variant。

conservative dispatch(Phase4ZR)は本モジュールから一切importしない/変更しない
(Section1)。このモジュールはdispatchがUNKNOWNと判定した後の後続処理のみを扱う。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# 日本語の一般的な助詞・助動詞・機能語のみ(機種固有名詞辞書ではない、Section4のC3設計注記参照)。
_STOPWORDS = frozenset({
    "は", "が", "を", "に", "で", "と", "の", "か", "も", "よ", "ね", "だ", "です", "ます",
    "て", "た", "る", "し", "な", "い", "う", "そう", "この", "その", "あの", "どう", "何",
    "教えて", "説明して", "違い", "って", "という", "こと", "もの",
})
_TOKEN_RE = re.compile(r"[一-龥ァ-ヶーa-zA-Z0-9]{2,}")


def _content_tokens(text: str) -> set[str]:
    tokens = _TOKEN_RE.findall(text)
    return {t for t in tokens if t not in _STOPWORDS}


@dataclass
class PolicyCDecision:
    variant: str
    retrieval_called: bool
    context_injected: bool
    selected_path: str  # "rag_with_context" or "clarification"
    lexical_overlap_tokens: list[str] | None = None


def decide_c1(query: str, context: str) -> PolicyCDecision:
    return PolicyCDecision(variant="C1", retrieval_called=True, context_injected=True, selected_path="rag_with_context")


def decide_c2(query: str, context: str) -> PolicyCDecision:
    # C2はretrievalすら実行しない設計(Section4: 「UNKNOWNではgenerationせず...」)。
    # ただしこの関数は事前計算済みcontextを受け取る呼び出し規約のため、retrieval_called=Falseと明記する。
    return PolicyCDecision(variant="C2", retrieval_called=False, context_injected=False, selected_path="clarification")


def decide_c3(query: str, context: str, titles: list[str], texts: list[str]) -> PolicyCDecision:
    query_tokens = _content_tokens(query)
    retrieved_text = " ".join(titles) + " " + " ".join(texts)
    overlap = [t for t in query_tokens if t in retrieved_text]
    if overlap:
        return PolicyCDecision(variant="C3", retrieval_called=True, context_injected=True,
                                selected_path="rag_with_context", lexical_overlap_tokens=overlap)
    return PolicyCDecision(variant="C3", retrieval_called=True, context_injected=False,
                            selected_path="clarification", lexical_overlap_tokens=[])


VARIANTS = {"C1": decide_c1, "C2": decide_c2, "C3": decide_c3}
