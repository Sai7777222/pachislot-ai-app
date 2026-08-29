"""Phase4FX で実現可能性を確認した entity-aware context assembly の production 実装。

Phase4FC〜FW で発見された RAG factual safety 問題(比較・要約・初心者向け説明クエリでの
クロスエンティティ誤帰属、存在しない固有名詞への誤紐付け)に対し、生成前の
retrieval/context assembly 段階で対策する。新しい固有名詞辞書やDB schema変更は
一切行わず、既存チャンクの `title` メタデータのみを構造的に利用する
(Phase4FX Stage A の監査で、DBの67.2%のチャンクは title だけで一意に
entity/concept を特定可能、12.6%は category との組み合わせで復元可能と確認済み)。

このモジュールのロジックは training/riru/guard/phase4fx_entity_attribution.py の
Phase4FX 最終PASS版を、production の `RetrievedChunk` データクラスに合わせて
移植したものであり、新しい設計判断は加えていない(Phase4FY Section3 の方針)。
"""

from __future__ import annotations

import re
import unicodedata

from pachislot_ai.rag.retriever import RetrievedChunk

# --- Query entity extraction (Phase4FX Q1: deterministic surface extraction) ---
# 巨大regexは使わない。「AとBの違い」「Aについて」等の助詞パターン + カタカナ/英数字
# トークンの抽出のみ。固有名詞辞書は一切参照しない。
_CANDIDATE_TOKEN_RE = re.compile(r"[A-Za-zΑ-Ωα-ω]{1,4}-?[A-Za-zΑ-Ωα-ω0-9]{0,4}|[ァ-ヶー]{2,}|[一-龥]{2,6}")

_STOPWORD_TOKENS = {
    "について", "とは", "教えて", "説明", "違い", "何", "関係", "要約",
    "初心者", "向け", "簡単", "簡潔", "詳しく", "少し", "登録", "データ",
}

_TRIVIAL_TITLE_RE = re.compile(r"^[0-9GgなしNone]{1,3}$")

# 単一entityでbindingされたevidenceが0件だったことを示す、model向けの内部シグナル用
# 合成チャンク。ユーザー向け文言としてhard-codeするのではなく、既存のRAG grounding
# prompt(config/prompts/system.jinja2, 変更なし)が「情報が見つからない場合は正直に
# 伝える」という既存ルールに従って処理することを想定した、モデルへの入力データの一部。
_NO_EVIDENCE_TEXT = "この対象についての情報は検索結果の中に見つかりませんでした。"


def normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip()


def extract_query_entities(query: str) -> list[str]:
    """query から対象entityを抽出する(Phase4FX Q1: deterministic surface extraction)。"""
    q = normalize(query)
    entities: list[str] = []

    # 比較構文: 「XとYの違い」「XとYどちら」「XとYの関係」等
    m = re.search(r"(.+?)と(.+?)(?:の違い|はどう違う|どちら|どっち|の関係)", q)
    if m:
        for part in (m.group(1), m.group(2)):
            part = part.strip("、 ")
            part = re.sub(r"^(それとも|あるいは)", "", part)
            if part and part not in _STOPWORD_TOKENS:
                entities.append(part)
        if entities:
            return entities

    # 単一entity構文: 「Xについて」「Xとは」「Xを教えて」等
    m = re.search(r"^(.+?)(について|とは|を教えて|とは何か|の仕組み|の関係|に関して)", q)
    if m:
        cand = m.group(1).strip()
        # 「AT-Fの性能と終了後の状態」のような複合名詞句は、助詞(の/と)の直前までを
        # 主entityとする。「GGプラス」「ガイアステージMAX」のように助詞を挟まない
        # 複合語(カタカナ+英数字+漢字が連続)は、切り詰めずそのまま1entityとして扱う
        # (接尾辞を落とすとphantom entity検出を損なうため)。
        particle_split = re.search(r"^(.+?)(の|と)(?=[^ー]{2,})", cand)
        if particle_split and len(particle_split.group(1)) >= 1:
            cand = particle_split.group(1)
        if cand and cand not in _STOPWORD_TOKENS:
            entities.append(cand)
            return entities

    # フォールバック: カタカナ語・英数字コード・漢字複合語をトークンとして抽出
    # 「初心者向け」のように、隣接する漢字が区切り文字なしでstopwordと連結すると
    # (「初心者」+「向」→「初心者向」)、貪欲な正規表現がstopword自体とは完全一致しない
    # 複合トークンとして抽出してしまうことがある。そのため完全一致だけでなく、
    # stopwordを部分文字列として含むトークンも除外する(「初心者向」は「初心者」を含むため除外)。
    tokens = [
        t for t in _CANDIDATE_TOKEN_RE.findall(q)
        if t not in _STOPWORD_TOKENS and len(t) >= 2
        and not any(sw in t for sw in _STOPWORD_TOKENS if len(sw) >= 2)
    ]
    return tokens[:3] if tokens else [q]


def title_match_score(entity: str, title: str) -> float:
    """entity候補とchunk titleの表層一致度を返す(0=不一致, 0.7=部分一致, 1.0=完全一致)。
    NFKC正規化のみ、固有名詞辞書は使用しない。"""
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
        # 単語境界チェック: 「GG」が「SGG」に部分文字列として含まれる誤衝突を防ぐ
        # (entity辞書なしで実現できる一般的な文字列処理)。
        def _is_ascii_alnum(ch: str) -> bool:
            return ch.isascii() and ch.isalnum()

        idx = t.find(e)
        before_ok = idx == 0 or not _is_ascii_alnum(t[idx - 1])
        after_idx = idx + len(e)
        after_ok = after_idx >= len(t) or not _is_ascii_alnum(t[after_idx])
        if before_ok and after_ok:
            return 0.7
        return 0.0
    return 0.0


def _find_title_matched_chunks(
    entity: str, all_chunks: list[RetrievedChunk], max_results: int = 4
) -> list[RetrievedChunk]:
    """entityとtitleが表層一致するchunkを、embedding scoreとは独立に全チャンクから検索する
    (title supplemental retrieval)。titleに一致するchunkがなければ本文一致を次点で探す。
    retrieval scoreはattributionに一切使用しない。"""
    scored = [(title_match_score(entity, c.title), c) for c in all_chunks]
    scored = [(s, c) for s, c in scored if s > 0]
    scored.sort(key=lambda x: -x[0])
    if scored:
        return [c for _, c in scored[:max_results]]

    e = normalize(entity)
    body_matches = [c for c in all_chunks if e and e in normalize(c.text)]
    return body_matches[:max_results]


def bind_entities_to_evidence(
    query_entities: list[str],
    embedding_chunks: list[RetrievedChunk],
    all_chunks: list[RetrievedChunk],
) -> dict[str, list[RetrievedChunk]]:
    """query entityごとに、embedding検索結果とtitle表層一致検索結果を統合してbindingする。
    どちらにもbindingされないembedding chunkは "UNBOUND" キーに格納する。"""
    result: dict[str, list[RetrievedChunk]] = {e: [] for e in query_entities}
    bound_chunk_ids: set[str] = set()

    # 長い(=より具体的な)entityから先に処理し、「SGG」titleのchunkが「GG」に
    # 部分文字列一致で誤ってbindingされるのを防ぐ。
    for entity in sorted(query_entities, key=len, reverse=True):
        for c in _find_title_matched_chunks(entity, all_chunks):
            if c.chunk_id not in bound_chunk_ids:
                result[entity].append(c)
                bound_chunk_ids.add(c.chunk_id)
        for c in embedding_chunks:
            if c.chunk_id in bound_chunk_ids:
                continue
            if title_match_score(entity, c.title) > 0:
                result[entity].append(c)
                bound_chunk_ids.add(c.chunk_id)

    result["UNBOUND"] = [c for c in embedding_chunks if c.chunk_id not in bound_chunk_ids]
    return result


def _no_evidence_chunk(entity: str) -> RetrievedChunk:
    """evidenceが0件のentityについて、生成前にその旨をmodelへ伝えるための合成チャンク。
    ユーザー向け文言のhard-codeではなく、既存のsystem prompt(変更なし)がこの入力を
    見て「情報が見つからない」という既存ルールに従って応答することを想定している。"""
    return RetrievedChunk(
        chunk_id=f"__no_evidence__::{entity}",
        text=_NO_EVIDENCE_TEXT,
        doc_id="",
        machine_id="",
        category="__no_evidence__",
        title=f"「{entity}」について",
        source_url="",
        source_label=None,
        data_source_type="__no_evidence__",
        score=0.0,
    )


def select_grounded_chunks(
    query: str,
    embedding_chunks: list[RetrievedChunk],
    all_chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Phase4FX A2(query-bound-only)に相当する、production向けのchunk選別。

    1. query entityを抽出する
    2. 各entityにevidenceをbindingする(embedding結果 + title補完検索)
    3. bindingされたchunkのみを最終候補とする(UNBOUNDは除外)
    4. entityが2つ以上あり、一部entityのevidenceが0件の場合は、その旨を伝える
       合成チャンクを1件だけ追加する(単一entityで0件の場合は追加しない — 既存の
       「該当する構造化データ・解説文章は登録されていません」という空contextの
       fallback文言に任せる)
    5. retention比率によるfallbackは行わない。Phase4FXのRAG50固有の◆項目テキスト
       絞り込みには50%保持率fallbackが必要だったが、それは静的contextの文字列フィルタ
       という別のコードパスの話であり、このchunkベースの選別には適用しない
       (Phase4FY統合時に一度誤って適用し、6件中1件だけが真に関連するケースで絞り込みが
       silentlyに無効化されるバグを引き起こしたため、意図的に外してある。詳細は
       関数末尾のコメントを参照)。
    """
    if not embedding_chunks:
        return embedding_chunks

    query_entities = extract_query_entities(query)
    binding = bind_entities_to_evidence(query_entities, embedding_chunks, all_chunks)

    selected: list[RetrievedChunk] = []
    seen_ids: set[str] = set()
    any_entity_has_evidence = False
    missing_entities: list[str] = []

    for entity in query_entities:
        chunks = binding.get(entity, [])
        if chunks:
            any_entity_has_evidence = True
            for c in chunks:
                if c.chunk_id not in seen_ids:
                    selected.append(c)
                    seen_ids.add(c.chunk_id)
        else:
            missing_entities.append(entity)

    if len(query_entities) >= 2 and missing_entities and any_entity_has_evidence:
        for entity in missing_entities:
            selected.append(_no_evidence_chunk(entity))

    # Phase4FXのchunkベースのA2選別(known_failure/phantom_entity/concept_binding)は、
    # retention比率によるfallbackなしで0/22 phantom misbinding・0件のcross-entity
    # misattributionを達成しており、ここでも同じロジックをそのまま踏襲する
    # (_FALLBACK_RETENTION_THRESHOLDはRAG50静的contextの◆項目テキスト絞り込みにのみ
    # 相当する概念であり、chunkベースの選別には適用しない。誤って適用すると、entityで
    # 正しく絞り込めているケース[例: 6件中1件だけが真に関連する場合]まで、絞り込み前の
    # noisy chunk一覧に戻してしまい、Phase4FXが解決したcross-entity誤帰属が再発する)。
    # query entity全てが完全にphantom(evidence 0件)の場合はselectedが空リストのまま
    # 返り、既存の空context fallback文言(build_rag_context側で処理済み)に任せる。
    # これはPhase4FXでRT-A/RT-B・AT-F等の既知failureに対して検証済みの安全な挙動である。
    return selected
