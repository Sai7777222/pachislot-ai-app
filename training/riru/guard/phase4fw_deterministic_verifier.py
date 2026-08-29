# -*- coding: utf-8 -*-
"""Phase4FW Stage B(entity grounding)/C(evidence binding)/D(deterministic verifier baseline)。
retrieval scoreは一切使用しない。固有名詞dictionaryも使わない(NFKC正規化のみ許可)。"""
from __future__ import annotations
import re
import unicodedata

TOKEN_RE = re.compile(r"[一-龥ァ-ヶーa-zA-Z0-9]{2,}")
NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?パーセント|1/\d+(?:\.\d+)?|\d+G|\d+枚")
SYMBOL_RE = re.compile(r"[×○●?？・]{2,}")


def normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip()


def entity_grounding(subject: str, context: str) -> str:
    """Stage B: subjectがcontext内に存在するか。EXACT/NORMALIZED/NOT_FOUND/AMBIGUOUSを返す。"""
    if not subject:
        return "AMBIGUOUS"
    subj_n = normalize(subject)
    ctx_n = normalize(context)
    if subject in context:
        return "EXACT"
    if subj_n in ctx_n:
        return "NORMALIZED"
    # 「AとB」のような複合主語は分割して再確認
    parts = re.split(r"[とやor・/]", subj_n)
    if len(parts) > 1 and all((p in ctx_n) for p in parts if p):
        return "NORMALIZED"
    return "NOT_FOUND"


def find_evidence_window(subject: str, context: str, window_chars: int = 120) -> list[str]:
    """subjectが出現するcontext内の位置を中心に、前後window_chars文字を評価窓として返す。
    複数箇所に出現する場合は複数窓を返す。"""
    windows = []
    subj_n = normalize(subject) if subject else ""
    ctx_n = normalize(context)
    if not subj_n:
        return windows
    start = 0
    while True:
        idx = ctx_n.find(subj_n, start)
        if idx == -1:
            break
        w_start = max(0, idx - window_chars)
        w_end = min(len(ctx_n), idx + len(subj_n) + window_chars)
        windows.append(ctx_n[w_start:w_end])
        start = idx + len(subj_n)
    return windows


def token_overlap_ratio(claim_text: str, window: str) -> float:
    claim_tokens = set(TOKEN_RE.findall(normalize(claim_text)))
    window_tokens = set(TOKEN_RE.findall(window))
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & window_tokens) / len(claim_tokens)


def evidence_binding(claim_text: str, subject: str, context: str) -> dict:
    """Stage C: claimのsubjectが実際に出現する位置の近傍に、claimの内容(predicate/object)を
    裏付けるトークンが十分含まれているかを判定する。「Xという文字列がcontextのどこかに
    存在する」だけではPASSにしない(=グローバル一致ではなくローカル近傍一致を要求)。"""
    windows = find_evidence_window(subject, context)
    if not windows:
        return {"bound": False, "best_overlap": 0.0, "reason": "subject_not_found_in_context"}
    best = max(token_overlap_ratio(claim_text, w) for w in windows)
    return {"bound": best >= 0.35, "best_overlap": round(best, 3), "reason": "local_window_overlap"}


def deterministic_verify(claim_text: str, subject: str, context: str) -> dict:
    """Stage D: entity grounding + evidence binding + numeric/symbol exact matchを統合した
    deterministic verifier baseline。"""
    grounding = entity_grounding(subject, context)
    ctx_n = normalize(context)
    claim_n = normalize(claim_text)

    # numeric/symbol exact match check(claim中の数値・記号がcontext中にそのまま存在するか)
    claim_numerics = set(NUMERIC_RE.findall(claim_n))
    claim_symbols = set(SYMBOL_RE.findall(claim_n))
    ctx_numerics = set(NUMERIC_RE.findall(ctx_n))
    ctx_symbols = set(SYMBOL_RE.findall(ctx_n))
    unsupported_numerics = claim_numerics - ctx_numerics
    unsupported_symbols = claim_symbols - ctx_symbols

    if grounding == "NOT_FOUND":
        # subjectがcontextに存在しない: claimに実質的な内容があればMISATTRIBUTED、
        # 「見つからない」等の不足申告ならNON_FACTUAL(安全)とみなす。
        hedge_words = ["登録データ", "見つから", "見つかりません", "ありません", "分から", "情報がない"]
        if any(h in claim_text for h in hedge_words):
            return {"status": "SUPPORTED", "grounding": grounding, "evidence": None,
                    "reason": "entity not found, but claim is itself a correct absence-declaration"}
        return {"status": "MISATTRIBUTED", "grounding": grounding, "evidence": None,
                "reason": "subject not found in context; claim asserts content about a non-existent entity"}

    binding = evidence_binding(claim_text, subject, context)
    if unsupported_numerics:
        return {"status": "UNSUPPORTED", "grounding": grounding, "evidence": binding,
                "reason": f"numeric value(s) not found verbatim in context: {sorted(unsupported_numerics)}"}
    if unsupported_symbols:
        return {"status": "UNSUPPORTED", "grounding": grounding, "evidence": binding,
                "reason": f"symbol sequence(s) not found verbatim in context: {sorted(unsupported_symbols)}"}
    if not binding["bound"]:
        return {"status": "MISATTRIBUTED", "grounding": grounding, "evidence": binding,
                "reason": "subject is present in context, but claim content has low overlap with the local evidence window near the subject (possible cross-entity misattribution)"}
    return {"status": "SUPPORTED", "grounding": grounding, "evidence": binding, "reason": "entity grounded and locally bound"}
