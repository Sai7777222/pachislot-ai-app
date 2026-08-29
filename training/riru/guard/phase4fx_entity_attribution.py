# -*- coding: utf-8 -*-
"""Phase4FX: query entity extraction(Q1 deterministic) + evidence attribution + context assembly。
新しいentity dictionaryは作らない。既存chunkのtitle/category metadataのみを構造的に利用する。"""
from __future__ import annotations
import re
import unicodedata

# Q1: deterministic surface extraction。巨大regexは使わない。
# 「AとBの違い」「Aについて」「Aとは」等の助詞パターン + カタカナ/英数字トークンの抽出のみ。
CANDIDATE_TOKEN_RE = re.compile(r"[A-Za-zΑ-Ωα-ω]{1,4}-?[A-Za-zΑ-Ωα-ω0-9]{0,4}|[ァ-ヶー]{2,}|[一-龥]{2,6}")
COMPARISON_SPLIT_RE = re.compile(r"と(?=[^、。]*[のは違い])|、")
PARTICLE_STRIP_RE = re.compile(r"(について|とは|の違い|を|は|が|に関して|に|、).*$")

STOPWORD_TOKENS = {"について", "とは", "教えて", "説明", "違い", "何", "関係", "要約",
                    "初心者", "向け", "簡単", "詳しく", "少し", "登録", "データ"}


def normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip()


def extract_query_entities_q1(query: str) -> list[str]:
    """deterministic surface extraction: 「AとBの違い」型の比較構文をまず検出し、
    A/Bそれぞれの候補エンティティを抽出する。単一entityの場合は主要な名詞句を1つ返す。"""
    q = normalize(query)
    entities: list[str] = []

    # 比較構文: 「XとYの違い」「XとYどちら」等
    m = re.search(r"(.+?)と(.+?)(?:の違い|はどう違う|どちら|どっち|の関係)", q)
    if m:
        for part in (m.group(1), m.group(2)):
            part = part.strip("、 ")
            # 助詞や冒頭の修飾語を取り除く
            part = re.sub(r"^(それとも|あるいは)", "", part)
            if part and part not in STOPWORD_TOKENS:
                entities.append(part)
        if entities:
            return entities

    # 単一entity構文: 「Xについて」「Xとは」「Xを教えて」等
    m = re.search(r"^(.+?)(について|とは|を教えて|とは何か|の仕組み|の関係|に関して)", q)
    if m:
        cand = m.group(1).strip()
        # candが「Aの性能と終了後の状態」のような複合名詞句の場合のみ、明示的な助詞(の/と)の
        # 直前までを主entityとして切り出す。「GGプラス」「ガイアステージMAX」「確定役ネオ」
        # のように助詞を挟まず直接連結された複合語(カタカナ+英数字+漢字が続けて書かれている)は、
        # 分割せずそのまま1つのentityとして扱う(接尾辞を取りこぼすとphantom entityの検出を
        # 損なうため)。
        particle_split = re.search(r"^(.+?)(の|と)(?=[^ー]{2,})", cand)
        if particle_split and len(particle_split.group(1)) >= 1:
            cand = particle_split.group(1)
        if cand and cand not in STOPWORD_TOKENS:
            entities.append(cand)
            return entities

    # フォールバック: カタカナ語・英数字コード・漢字複合語をトークンとして抽出
    tokens = [t for t in CANDIDATE_TOKEN_RE.findall(q) if t not in STOPWORD_TOKENS and len(t) >= 2]
    return tokens[:3] if tokens else [q]


def evidence_attribution(chunks: list[dict]) -> list[dict]:
    """各chunkについて、構造化metadata(title/category)からentity/conceptを推定する。
    優先順位: title(そのまま) > categoryとの組み合わせ。modelには一切事実を作らせない。"""
    out = []
    for c in chunks:
        title = c.get("title", "")
        category = c.get("category", "")
        # titleが数字のみ・記号のみの場合は弱い attribution(親categoryに依存)
        if re.fullmatch(r"[0-9GgなしNone]{1,4}", title.strip()) or title.strip() in ("", "なし"):
            entity = f"{category}(パターン:{title})" if title else category
            confidence_source = "category_only"
        else:
            entity = title
            confidence_source = "title"
        out.append({**c, "attributed_entity": entity, "confidence_source": confidence_source})
    return out


_TRIVIAL_TITLE_RE = re.compile(r"^[0-9GgなしNone]{1,3}$")


def title_match_score(entity: str, title: str) -> float:
    """entity候補とchunk titleの表層一致度を返す(0=不一致, 1=完全一致)。
    NFKC正規化のみ、固有名詞dictionaryは使用しない。「1」「2」「4」等の短い数字/記号のみの
    titleは、部分文字列一致だけでは無関係な語(例: 「SU4」の「4」)と誤って衝突するため、
    完全一致以外は許可しない。"""
    e = normalize(entity)
    t = normalize(title)
    if not e or not t:
        return 0.0
    if e == t:
        return 1.0
    if _TRIVIAL_TITLE_RE.match(t):
        return 0.0
    if len(t) >= 2 and t in e:
        return 0.7
    if len(t) >= 2 and e in t:
        # 単語境界チェック: 「GG」が「SGG」に部分文字列として含まれてしまうような
        # 誤った衝突(entity dictionaryなしで解決可能な一般的な文字列処理)を防ぐため、
        # 一致箇所の直前直後が英数字で連続していないことを確認する。
        def _is_ascii_alnum(ch):
            return ch.isascii() and ch.isalnum()
        idx = t.find(e)
        before_ok = idx == 0 or not _is_ascii_alnum(t[idx - 1])
        after_idx = idx + len(e)
        after_ok = after_idx >= len(t) or not _is_ascii_alnum(t[after_idx])
        if before_ok and after_ok:
            return 0.7
        return 0.0
    return 0.0


def find_title_matched_chunks(entity: str, all_chunks: list[dict], max_results: int = 4) -> list[dict]:
    """entityとtitleが表層一致するchunkを、embedding scoreとは独立に全chunkから検索する。
    これはPhase4FXの中心的な発見(embedding top-kが完全一致titleを見逃すケースがある)への対策。
    titleに一致するchunkがない場合は、本文(text)中にentity文字列が実在するchunkを次点で探す
    (Section8の優先順位: title/heading > source section/content local heading に対応)。
    いずれも固有名詞dictionaryは使用せず、単純な文字列包含判定のみ。"""
    scored = [(title_match_score(entity, c.get("title", "")), c) for c in all_chunks]
    scored = [(s, c) for s, c in scored if s > 0]
    scored.sort(key=lambda x: -x[0])
    if scored:
        return [c for s, c in scored[:max_results]]

    # tier2: 本文一致(titleには出てこないがtextに実在するケース。例: ガイアベル/GGストック)
    e = normalize(entity)
    body_matches = [c for c in all_chunks if e and e in normalize(c.get("text", ""))]
    return body_matches[:max_results]


def bind_entities_to_evidence(query_entities: list[str], embedding_chunks: list[dict],
                                all_chunks: list[dict]) -> dict:
    """Stage B: entity match matrix。query entityごとに、embedding top-k由来のchunkと
    title-match由来のchunkを統合してbindingする。どちらにもbindingされないembedding
    chunkはUNBOUNDとする。"""
    result: dict[str, list[dict]] = {e: [] for e in query_entities}
    bound_chunk_ids: set[str] = set()

    # 長い(=より具体的な)entityから先に処理する。例:「SGG」を先に処理してから「GG」を処理することで、
    # 「SGG」というtitleを持つchunkが「GG」という部分文字列一致で誤って「GG」側にbindingされるのを防ぐ。
    for entity in sorted(query_entities, key=len, reverse=True):
        # (1) title表層一致による補完検索(embedding scoreに依存しない)
        title_matches = find_title_matched_chunks(entity, all_chunks)
        for c in title_matches:
            if c["chunk_id"] not in bound_chunk_ids:
                result[entity].append({**c, "bind_reason": "title_match"})
                bound_chunk_ids.add(c["chunk_id"])
        # (2) embedding top-kのうち、titleにentity文字列が含まれるものを追加でbinding
        for c in embedding_chunks:
            if c["chunk_id"] in bound_chunk_ids:
                continue
            if title_match_score(entity, c.get("title", "")) > 0:
                result[entity].append({**c, "bind_reason": "embedding_title_match"})
                bound_chunk_ids.add(c["chunk_id"])

    unbound = [c for c in embedding_chunks if c["chunk_id"] not in bound_chunk_ids]
    result["UNBOUND"] = [{**c, "bind_reason": "embedding_only_no_title_match"} for c in unbound]
    return result
