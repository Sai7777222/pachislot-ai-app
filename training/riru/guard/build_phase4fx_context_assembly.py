# -*- coding: utf-8 -*-
"""Phase4FX: 全評価probeについてA0(現行)/A1(entity-grouped)/A2(query-bound-only) contextを
組み立てる。新規retrievalは9probe分のみ(phase4fx_new_probe_contexts.json)、残りは
Phase4FU/4FVの既存embedding結果を再利用する。"""
from __future__ import annotations
import json
import sys
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parent
REPORTS_DIR = GUARD_DIR.parent / "reports"
EVAL_DIR = GUARD_DIR.parent / "eval"
sys.path.insert(0, str(GUARD_DIR))
sys.path.insert(0, str(EVAL_DIR))

from phase4fx_entity_attribution import extract_query_entities_q1, bind_entities_to_evidence  # noqa: E402
from phase4fx_context_assembly import build_context_a0, build_context_a1, build_context_a2  # noqa: E402
from phase4fx_probes import KNOWN_FAILURES, CONCEPT_BINDING_NEW  # noqa: E402


def load_all_chunks():
    raw = json.loads((REPORTS_DIR / "phase4fx_all_chunks_raw.json").read_text(encoding="utf-8"))
    return [{"chunk_id": c["chunk_id"], "text": c["text"], "title": c["metadata"]["title"],
             "category": c["metadata"]["category"]} for c in raw]


def norm_chunks(chunks, title_key="title", text_key="text"):
    return [{"chunk_id": c.get("chunk_id", c.get(title_key, "") + str(id(c))), "text": c[text_key],
             "title": c[title_key], "category": c.get("category", "")} for c in chunks]


def main():
    all_chunks = load_all_chunks()
    new_probe_ctx = {p["id"]: p for p in json.loads((REPORTS_DIR / "phase4fx_new_probe_contexts.json").read_text(encoding="utf-8"))}

    fu_precomputed = {p["id"]: p for p in json.loads((REPORTS_DIR / "phase4fu_precomputed_contexts.json").read_text(encoding="utf-8"))}
    fv_phantom_new = {p["id"]: p for p in json.loads((REPORTS_DIR / "phase4fv_phantom_new_contexts.json").read_text(encoding="utf-8"))}
    fv_concept_new = {p["id"]: p for p in json.loads((REPORTS_DIR / "phase4fv_concept_new_contexts.json").read_text(encoding="utf-8"))}
    fu_stage_f = json.loads((REPORTS_DIR / "phase4fu_stage_f_contexts.json").read_text(encoding="utf-8"))

    import importlib
    pool_mod_spec = None
    sys.path.insert(0, str(EVAL_DIR))
    import phase4zf_rag_stress_eval as pool_mod  # noqa: E402
    rag50_pool = {p["id"]: p for p in pool_mod.load_rag_probe_pool()}
    rag50_baseline = json.loads((REPORTS_DIR / "phase4zn_rag50_raw.json").read_text(encoding="utf-8"))

    probes = []  # list of {id, category, prompt, embedding_chunks(list of dict with chunk_id/title/text)}

    # ---- known_failure (8) ----
    kf_prompt_to_source = {
        "GGとSGGの違いを初心者向けに説明して": ("FU-D01", fu_precomputed),
        "SGGの仕組みを分かりやすく説明して": ("FU-D03", fu_precomputed),
        "ガイアベルとは何か説明して": ("FU-E02", fu_precomputed),
        "ループストックとGGストックの違いを教えて": ("FV-C03", fv_concept_new),
        "AT-Fの性能と終了後の状態について教えて": ("FU-A03", fu_precomputed),
        "RT-AとRT-Bの違いを要約して": ("FU-B05", fu_precomputed),
        "GG当選とSGG当選の違いを教えて": ("FV-C05", fv_concept_new),
    }
    for kf in KNOWN_FAILURES:
        prompt = kf["prompt"]
        if prompt in kf_prompt_to_source:
            src_id, src_dict = kf_prompt_to_source[prompt]
            chunks = norm_chunks(src_dict[src_id]["retrieved_chunks"])
        else:
            chunks = norm_chunks(new_probe_ctx[kf["id"]]["retrieved_chunks"])
        probes.append({"id": kf["id"], "category": "known_failure", "label": kf["label"], "prompt": prompt, "embedding_chunks": chunks})

    # ---- phantom_entity (22, reuse Phase4FV) ----
    phantom_reused_ids = ["FU-A03", "FU-B03", "FU-B04", "FU-B05", "FU-D05", "FU-F03", "FU-F04", "FU-F05"]
    for pid in phantom_reused_ids:
        p = fu_precomputed[pid]
        probes.append({"id": pid, "category": "phantom_entity", "prompt": p["prompt"], "embedding_chunks": norm_chunks(p["retrieved_chunks"])})
    for pid, p in fv_phantom_new.items():
        probes.append({"id": pid, "category": "phantom_entity", "prompt": p["prompt"], "embedding_chunks": norm_chunks(p["retrieved_chunks"])})

    # ---- concept_binding (12 reused + 8 new = 20) ----
    concept_reused_ids = ["FU-D01", "FU-D03", "FU-D05"]
    for pid in concept_reused_ids:
        p = fu_precomputed[pid]
        probes.append({"id": f"CB-{pid}", "category": "concept_binding", "prompt": p["prompt"], "embedding_chunks": norm_chunks(p["retrieved_chunks"])})
    for pid, p in fv_concept_new.items():
        probes.append({"id": pid, "category": "concept_binding", "prompt": p["prompt"], "embedding_chunks": norm_chunks(p["retrieved_chunks"])})
    for cb in CONCEPT_BINDING_NEW:
        p = new_probe_ctx[cb["id"]]
        probes.append({"id": cb["id"], "category": "concept_binding", "prompt": cb["prompt"], "embedding_chunks": norm_chunks(p["retrieved_chunks"])})

    # ---- query_style (5) ----
    d01 = fu_precomputed["FU-D01"]
    probes.append({"id": "QS-D01", "category": "query_style", "prompt": d01["prompt"], "embedding_chunks": norm_chunks(d01["retrieved_chunks"])})
    for sp in fu_stage_f:
        probes.append({"id": f"QS-{sp['id']}", "category": "query_style", "prompt": sp["prompt"], "embedding_chunks": norm_chunks(sp["retrieved_chunks"])})

    # ---- rag50 (8 mandatory + 12 sample = 20) ----
    # RAG50は機種ごとに事前整理されたstatic context(【構造化データ】+【関連する解説文章】)であり、
    # 本フェーズのchunkベースのentity attribution機構とはschemaが異なる。ここでは
    # 【構造化データ】セクションは常に全件保持(Phase4ZO由来のcompleteness regressionへの警戒)しつつ、
    # 【関連する解説文章】の◆項目だけを、query entityが見出し/本文に含まれるものに絞り込む
    # (title-based filteringと同じ考え方を、RAG50独自のセクション構造に適用したもの)。
    mandatory_rag50 = ["P02", "P04", "LC-08", "Q6", "Q11", "Q15", "Q17", "AD-04"]
    all_rag50_ids = [r["probe_id"] for r in rag50_baseline]
    extra_sample = [i for i in all_rag50_ids if i not in mandatory_rag50][::4][:12]
    rag50_ids = mandatory_rag50 + extra_sample
    for pid in rag50_ids:
        p = rag50_pool[pid]
        probes.append({"id": pid, "category": "rag50", "prompt": p["question"], "static_context": p["context"],
                        "embedding_chunks": []})

    (REPORTS_DIR / "phase4fx_probes_consolidated.json").write_text(
        json.dumps(probes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"total probes consolidated: {len(probes)}")
    from collections import Counter
    print(Counter(p["category"] for p in probes))

    # ---- entity attribution + context assembly (A0/A1/A2) ----
    import re as _re
    assembled = []
    for p in probes:
        if p["category"] == "rag50":
            entities = extract_query_entities_q1(p["prompt"])
            ctx = p["static_context"]
            # 【構造化データ】セクションは常に全保持。【関連する解説文章】の◆項目のみを
            # query entityが見出しまたは内容に含まれるものに絞り込む。
            m = _re.search(r"(.*?)(【関連する解説文章】.*)", ctx, _re.DOTALL)
            if m:
                header_part, explain_part = m.group(1), m.group(2)
                items = _re.split(r"(?=◆ )", explain_part)
                kept = [items[0]]  # 【関連する解説文章】の見出し行
                for item in items[1:]:
                    if not entities or any(e in item for e in entities):
                        kept.append(item)
                # フォールバック: entity一致による絞り込みで◆項目の半分以上が失われる場合、
                # 絞り込みによって回答に必要な補強情報が失われるリスクの方が大きいと判断し、
                # 元の【関連する解説文章】全体を保持する(Phase4ZO/4FVで確認済みのcompleteness
                # regressionパターンへの対策)。「構造化データ」が存在しない機種紹介系の
                # probe(Q15/Q17等)では、この閾値判定がより重要になる。
                kept_ratio = (len(kept) - 1) / max(1, len(items) - 1)
                if kept_ratio < 0.5:
                    a2 = ctx
                else:
                    a2 = header_part + "".join(kept)
            else:
                a2 = ctx
            assembled.append({"id": p["id"], "category": p["category"], "prompt": p["prompt"],
                               "query_entities": entities, "A0": ctx, "A1": ctx, "A2": a2})
            continue
        entities = extract_query_entities_q1(p["prompt"])
        binding = bind_entities_to_evidence(entities, p["embedding_chunks"], all_chunks)
        a0 = build_context_a0(p["embedding_chunks"])
        a1 = build_context_a1(entities, binding)
        a2 = build_context_a2(entities, binding)
        assembled.append({"id": p["id"], "category": p["category"], "prompt": p["prompt"],
                           "query_entities": entities, "A0": a0, "A1": a1, "A2": a2,
                           "binding_debug": {e: [c["title"] for c in chunks] for e, chunks in binding.items()}})

    (REPORTS_DIR / "phase4fx_context_assembly.json").write_text(
        json.dumps(assembled, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"assembled contexts saved: {len(assembled)}")


if __name__ == "__main__":
    main()
