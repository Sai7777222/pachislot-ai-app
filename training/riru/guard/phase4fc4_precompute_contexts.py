"""Phase4FC4 Stage B-J: 実本番ChatService(mode-specific prompt統合後)の
build_rag_context()を使い、各probeの最終rag_context(またはNone)を事前計算する。

Section4/6/7/8の凍結要件を守るため、retrieval/dispatchロジック自体はFC3から
一切変更していない。FC3で既に検証済みのprobeセット(stage_b/c/d/e/h/i)は
そのままFC3のloaderコードを再利用し、Stage J(multi-turn)のみFC4で新たに
6シナリオ/18ターンへ拡張する(OOD->SMALL_TALK, SMALL_TALK->OODの新規カバレッジ含む)。
Section15の「共有pluming変化がないか、想定せず検査する」指示に従い、値を
コピーするのではなく現在のコードで再計算することで暗黙のドリフトがないか確認する。
"""
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


def load_identity_23():
    """FC3のstage_c_identity(23件)をそのまま再利用する(Section13は
    'representative identity/persona set'を要求するのみで、既に検証済みの
    上位互換セットを流用することでdriftのない直接比較を可能にする)。"""
    fc3_contexts = json.loads((REPORTS_DIR / "phase4fc3_precomputed_contexts.json").read_text(encoding="utf-8"))
    return [{"id": r["id"], "prompt": r["prompt"]} for r in fc3_contexts["stage_c_identity"]]


def load_ood_15():
    rows = json.loads((REPORTS_DIR / "phase4zp_ood_recheck_raw.json").read_text(encoding="utf-8"))
    return [{"id": r["probe_id"], "prompt": r["prompt"]} for r in rows]


def load_conversational_10():
    sys.path.insert(0, str(PROJECT_ROOT / "training" / "riru" / "eval"))
    from phase4zn_unattended_probes import ALL_PROBES
    return [{"id": p["id"], "prompt": p["prompt"]} for p in ALL_PROBES if p["category"] == "pachislot_conversational"]


def load_known_failure_12():
    """Section15の11個の既知failure familyを全てカバーする(H-AD04がAD-04を兼ねる)。"""
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
    """Section19: 6シナリオ/18ターン以上、6種の遷移を全てカバーする
    (small-talk<->factual, identity<->factual, OOD<->small-talk)。
    FC3のMT-01/02/04/05を土台にしつつ3ターンへ拡張し、FC4新規のOOD<->small-talk
    シナリオ(MT-05/06)を追加する。"""
    return [
        {"id": "FC4-MT-01", "description": "small-talk -> factual -> small-talk",
         "turns": [{"user": "おはよう！"}, {"user": "天井は何ゲームですか"},
                   {"user": "ありがとう、助かったよ"}]},
        {"id": "FC4-MT-02", "description": "factual -> small-talk -> factual",
         "turns": [{"user": "天井は何ゲームですか"}, {"user": "今日は疲れたな〜"},
                   {"user": "ヤメ時はいつがいい？"}]},
        {"id": "FC4-MT-03", "description": "identity -> factual -> identity",
         "turns": [{"user": "君の名前は？"}, {"user": "ヤメ時はいつがいい？"},
                   {"user": "自己紹介して"}]},
        {"id": "FC4-MT-04", "description": "factual -> identity -> factual",
         "turns": [{"user": "天井は何ゲームですか"}, {"user": "君の名前は？"},
                   {"user": "GGとSGGの違いを初心者向けに説明して"}]},
        {"id": "FC4-MT-05", "description": "OOD -> small-talk -> OOD",
         "turns": [{"user": "今日の東京の天気を教えて"}, {"user": "趣味とかあるの？"},
                   {"user": "おすすめのレシピを教えて"}]},
        {"id": "FC4-MT-06", "description": "small-talk -> OOD -> small-talk",
         "turns": [{"user": "趣味とかあるの？"}, {"user": "株の投資のコツを教えて"},
                   {"user": "ありがとう、助かったよ"}]},
    ]


def build_context_for(svc: ChatService, query: str) -> dict:
    dispatch_result = dispatch(query)
    selected_prompt_kind = (
        dispatch_result.mode if dispatch_result.mode in svc._mode_system_prompts else "FACTUAL_OR_UNKNOWN_DEFAULT"
    )
    ctx = svc.build_rag_context([ChatMessage(role="user", content=query)], None)
    if ctx is None:
        return {
            "mode": dispatch_result.mode, "matched_rule": dispatch_result.matched_rule,
            "selected_prompt_kind": selected_prompt_kind,
            "retrieval_called": False, "rag_context_injected": False,
            "chunk_evidence": False, "structured_evidence": False, "combined_evidence": False,
            "no_evidence_marker_present": False, "prompt_text": "",
        }
    return {
        "mode": dispatch_result.mode, "matched_rule": dispatch_result.matched_rule,
        "selected_prompt_kind": selected_prompt_kind,
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
        "stage_d_identity23": load_identity_23,
        "stage_e_ood15": load_ood_15,
        "stage_f_known_failure12": load_known_failure_12,
        "stage_g_rag8": load_rag8_mandatory,
        "stage_h_conversational10": load_conversational_10,
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

    out_path = REPORTS_DIR / "phase4fc4_precomputed_contexts.json"
    out_path.write_text(json.dumps(all_out, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(v) for k, v in all_out.items() if k != "stage_j_multiturn")
    total_turns = sum(len(s["turns"]) for s in mt_out)
    print(f"TOTAL: {total} single-turn + {total_turns} multiturn turns")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
