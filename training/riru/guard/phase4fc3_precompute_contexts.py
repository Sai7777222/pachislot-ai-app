"""Phase4FC3 Stage B-J: 実本番ChatService.build_rag_context()(dispatch統合後)を
使い、各probeの最終rag_context(またはNone)を事前計算する。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
REPORTS_DIR = PROJECT_ROOT / "training" / "riru" / "reports"

from pachislot_ai.core.config import RAG_CONTEXT_PROMPT_PATH, Settings  # noqa: E402
from pachislot_ai.data.db import create_structured_engine  # noqa: E402
from pachislot_ai.dispatch import dispatch  # noqa: E402
from pachislot_ai.llm.base import ChatMessage  # noqa: E402
from pachislot_ai.rag.embedder import Embedder  # noqa: E402
from pachislot_ai.rag.pipeline import RagPipeline  # noqa: E402
from pachislot_ai.rag.retriever import Retriever  # noqa: E402
from pachislot_ai.rag.vector_store import VectorStore  # noqa: E402
from pachislot_ai.services.chat_service import ChatService  # noqa: E402


class _DummyLLM:
    model_name = "dummy"

    async def chat(self, *a, **k):  # pragma: no cover
        pass

    async def chat_stream(self, *a, **k):  # pragma: no cover
        pass

    async def health_check(self):  # pragma: no cover
        return True


def load_smalltalk_65():
    rows = json.loads((REPORTS_DIR / "phase4zp_smalltalk_recheck_raw.json").read_text(encoding="utf-8"))
    return [{"id": r["probe_id"], "prompt": r["prompt"]} for r in rows]


def load_identity_19():
    picks = json.loads((REPORTS_DIR / "_tmp_identity_picks.json").read_text(encoding="utf-8"))
    out = [{"id": pid, "prompt": prompt} for pid, prompt in picks]
    out.append({"id": "FC2-ID-CANON", "prompt": "君の名前は？"})
    extra = [
        {"id": "FC3-ID-01", "prompt": "名前なんだっけ？"},
        {"id": "FC3-ID-02", "prompt": "自己紹介して"},
        {"id": "FC3-ID-03", "prompt": "リルって呼べばいい？"},
        {"id": "FC3-ID-04", "prompt": "どんな性格？"},
    ]
    return out + extra


def load_ood_15():
    rows = json.loads((REPORTS_DIR / "phase4zp_ood_recheck_raw.json").read_text(encoding="utf-8"))
    return [{"id": r["probe_id"], "prompt": r["prompt"]} for r in rows]


def load_conversational_10():
    sys.path.insert(0, str(PROJECT_ROOT / "training" / "riru" / "eval"))
    from phase4zn_unattended_probes import ALL_PROBES
    return [{"id": p["id"], "prompt": p["prompt"]} for p in ALL_PROBES if p["category"] == "pachislot_conversational"]


def load_evidence_arbitration_30():
    """chunk-only/structured-only/both/neither の4カテゴリを最低30件カバーする。
    Phase4FC2のGate Eで既に分類済みの代表queryを再利用する(新しい研究ではない)。"""
    gt = json.loads((REPORTS_DIR / "phase4fc2_production_gt.json").read_text(encoding="utf-8"))
    id_to_cat = {r["id"]: r["category"] for r in gt["rows"]}
    picks_by_cat = {"structured_only": [], "chunk_only": [], "chunk_and_structured": [], "entity_missing": []}
    for r in gt["rows"]:
        cat = r["category"]
        if cat in picks_by_cat and len(picks_by_cat[cat]) < 8:
            picks_by_cat[cat].append({"id": r["id"], "prompt": r["prompt"], "gt_category": cat})
    mandatory = [
        {"id": "FX-CB06", "prompt": "天井とヤメ時の関係を教えて", "gt_category": "mandatory_arbitration_case"},
        {"id": "PG-CP06", "prompt": "天国と天国準備の違いを教えて", "gt_category": "mandatory_arbitration_case"},
    ]
    out = mandatory
    for cat, items in picks_by_cat.items():
        out.extend(items)
    return out


def load_known_failure_12():
    return [
        {"id": "H-Q6", "prompt": "GGとSGGの違いを初心者向けに説明して"},
        {"id": "H-AT-F", "prompt": "AT-Fの性能と終了後の状態について教えて"},
        {"id": "H-RT-AB", "prompt": "RT-AとRT-Bの違いを要約して"},
        {"id": "H-SGG-JUNBI", "prompt": "SGGの仕組みを分かりやすく説明して"},
        {"id": "H-GG-NAKA", "prompt": "GG中とはどんな状態か教えて"},
        {"id": "H-GAIABELL", "prompt": "ガイアベルとは何か説明して"},
        {"id": "H-SU4", "prompt": "SU4について教えて"},
        {"id": "H-GG-TOSEN", "prompt": "GG当選とSGG当選の違いを教えて"},
        {"id": "H-LOOP-GG", "prompt": "ループストックとGGストックの違いを教えて"},
        {"id": "H-TENKOKU-LONG", "prompt": "天国ロングとは何か説明して"},
        {"id": "H-GG-KEIZOKU", "prompt": "GG継続の条件は？"},
        {"id": "H-AD04", "prompt": "ヤメ時はいつがいい？"},
    ]


def load_rag8_mandatory():
    return [
        {"id": "P02", "prompt": "ボーナス確率について教えて"},
        {"id": "P04", "prompt": "最低設定と最高設定の機械割の差を教えて"},
        {"id": "LC-08", "prompt": "AT-Fの性能と終了後の状態を教えて"},
        {"id": "Q6", "prompt": "GGとSGGの違いを初心者向けに説明して"},
        {"id": "Q11", "prompt": "天井とヤメ時を合わせて初心者向けに説明して"},
        {"id": "Q15", "prompt": "ミリオンゴッドの遊び方を初心者向けにやさしく説明して"},
        {"id": "Q17", "prompt": "ミリオンゴッドの遊び方を少し詳しく説明して"},
        {"id": "AD-04", "prompt": "ヤメ時はいつがいい？"},
    ]


def load_multiturn_transitions():
    return [
        {"id": "FC3-MT-01", "description": "small-talk -> pachislot factual",
         "turns": [{"user": "おはよう！"}, {"user": "天井は何ゲームですか"}]},
        {"id": "FC3-MT-02", "description": "pachislot factual -> small-talk",
         "turns": [{"user": "GGとSGGの違いを初心者向けに説明して"}, {"user": "ありがとう、助かったよ"}]},
        {"id": "FC3-MT-03", "description": "OOD -> pachislot factual",
         "turns": [{"user": "今日の東京の天気を教えて"}, {"user": "天井は何ゲームですか"}]},
        {"id": "FC3-MT-04", "description": "identity -> factual",
         "turns": [{"user": "君の名前は？"}, {"user": "ヤメ時はいつがいい？"}]},
        {"id": "FC3-MT-05", "description": "factual -> identity",
         "turns": [{"user": "天井は何ゲームですか"}, {"user": "君の名前は？"}]},
    ]


def build_context_for(svc: ChatService, query: str) -> dict:
    dispatch_result = dispatch(query)
    ctx = svc.build_rag_context([ChatMessage(role="user", content=query)], None)
    if ctx is None:
        return {
            "mode": dispatch_result.mode, "matched_rule": dispatch_result.matched_rule,
            "retrieval_called": False, "rag_context_injected": False,
            "chunk_evidence": False, "structured_evidence": False, "combined_evidence": False,
            "no_evidence_marker_present": False, "prompt_text": "",
        }
    return {
        "mode": dispatch_result.mode, "matched_rule": dispatch_result.matched_rule,
        "retrieval_called": True, "rag_context_injected": bool(ctx.prompt_text),
        "chunk_evidence": len(ctx.chunk_sources) > 0, "structured_evidence": len(ctx.structured_sources) > 0,
        "combined_evidence": (len(ctx.chunk_sources) > 0 or len(ctx.structured_sources) > 0),
        "is_empty": ctx.is_empty, "prompt_text": ctx.prompt_text,
        "chunk_titles": [c["title"] for c in ctx.chunk_sources],
    }


def main():
    settings = Settings()
    embedder = Embedder(settings.embedding_model_path, device=settings.embedding_device)
    vector_store = VectorStore(settings.vector_db_path, settings.vector_db_collection)
    retriever = Retriever(embedder, vector_store, default_top_k=settings.rag_top_k)
    structured_engine = create_structured_engine(settings.structured_db_path)
    rag_pipeline = RagPipeline(retriever, structured_engine, RAG_CONTEXT_PROMPT_PATH, top_k=settings.rag_top_k)
    svc = ChatService(_DummyLLM(), settings.resolved_system_prompt_path, rag_pipeline)

    all_out = {}
    loaders = {
        "stage_b_smalltalk65": load_smalltalk_65,
        "stage_c_identity": load_identity_19,
        "stage_d_ood15": load_ood_15,
        "stage_e_conversational10": load_conversational_10,
        "stage_g_evidence_arbitration": load_evidence_arbitration_30,
        "stage_h_known_failure12": load_known_failure_12,
        "stage_i_rag8": load_rag8_mandatory,
    }
    for stage_name, loader in loaders.items():
        probes = loader()
        results = []
        for p in probes:
            built = build_context_for(svc, p["prompt"])
            results.append({"id": p["id"], "prompt": p["prompt"], **built})
        all_out[stage_name] = results
        print(f"{stage_name}: {len(results)} probes precomputed")

    mt_scenarios = load_multiturn_transitions()
    mt_out = []
    for sc in mt_scenarios:
        turn_contexts = []
        for turn in sc["turns"]:
            built = build_context_for(svc, turn["user"])
            turn_contexts.append({"user": turn["user"], **built})
        mt_out.append({"id": sc["id"], "description": sc["description"], "turns": turn_contexts})
    all_out["stage_j_multiturn"] = mt_out
    print(f"stage_j_multiturn: {len(mt_out)} scenarios")

    out_path = REPORTS_DIR / "phase4fc3_precomputed_contexts.json"
    out_path.write_text(json.dumps(all_out, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(v) for k, v in all_out.items() if k != "stage_j_multiturn")
    total_turns = sum(len(s["turns"]) for s in mt_out)
    print(f"TOTAL: {total} single-turn + {total_turns} multiturn turns")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
