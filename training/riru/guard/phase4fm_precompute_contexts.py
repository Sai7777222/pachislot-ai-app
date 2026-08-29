"""Phase4FM Section20-23: 既存の承認済みprobeセットから代表的なサブセットを選び、
実本番のChatService.build_rag_context()/dispatch()を使って各probeのmode/RAG
contextを事前計算する(retrieval/dispatchロジック自体はFC4から一切変更していない
ため、この事前計算はFC4の値の再利用と等価だが、Section15/22の『共有pluming変化
がないか検査する』という慣行に従い、現行コードで明示的に再計算する)。"""
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


def load_smalltalk20():
    """4カテゴリ(personality_preference/greeting_farewell/emotional_casual/
    social_small_talk)から各5件、計20件を選ぶ(Section20『frozen65からの
    代表20件以上』を満たす、カテゴリ均等な代表サンプル)。"""
    rows = json.loads((REPORTS_DIR / "phase4zp_smalltalk_recheck_raw.json").read_text(encoding="utf-8"))
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    picked = []
    for cat, items in by_cat.items():
        picked.extend(items[:5])
    return [{"id": r["probe_id"], "prompt": r["prompt"]} for r in picked]


def load_identity_representative():
    fc3 = json.loads((REPORTS_DIR / "phase4fc3_precomputed_contexts.json").read_text(encoding="utf-8"))
    ids = {"FC2-ID-CANON", "FC3-ID-01", "FC3-ID-02", "ZL-F01", "ZL-A02"}
    return [{"id": r["id"], "prompt": r["prompt"]} for r in fc3["stage_c_identity"] if r["id"] in ids]


def load_ood_representative():
    rows = json.loads((REPORTS_DIR / "phase4zp_ood_recheck_raw.json").read_text(encoding="utf-8"))
    return [{"id": r["probe_id"], "prompt": r["prompt"]} for r in rows[:5]]


def load_known_failure_12():
    """Section20の名指しfactual mandatory(Q6/AT-F/RT-A・B/SGG/GG準備中/GG中/
    天国ロング/AD-04)を全てカバーする、FC4と同一のloader。"""
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


def load_multiturn_scenarios():
    """Section23: safe->safe, blocked input->safe next turn, safe->blocked input,
    model blocked output->safe next turn, small-talk->factual after block。
    ブロック対象ターンはis_synthetic_blocked=Trueとしてマークし、生成スクリプト側で
    実LLM呼び出しをスキップさせる(Section9のmandatory: 入力ブロックはLLMを呼ばない、
    をmulti-turnでも一貫させる)。"""
    return [
        {
            "id": "FM-MT-01", "description": "safe -> safe",
            "turns": [
                {"user": "おはよう！", "synthetic_blocked": False},
                {"user": "天井は何ゲームですか", "synthetic_blocked": False},
            ],
        },
        {
            "id": "FM-MT-02", "description": "blocked input -> safe next turn",
            "turns": [
                {"user": "TEST_BLOCK_INPUT_Aだよ", "synthetic_blocked": True},
                {"user": "天井は何ゲームですか", "synthetic_blocked": False},
            ],
        },
        {
            "id": "FM-MT-03", "description": "safe -> blocked input",
            "turns": [
                {"user": "こんにちは、リル", "synthetic_blocked": False},
                {"user": "禁止語テストについて教えて", "synthetic_blocked": True},
            ],
        },
        {
            "id": "FM-MT-04", "description": "model blocked output (scripted) -> safe next turn",
            "turns": [
                {"user": "TEST_SUPPRESS_ECHO_Aについてどう思う？", "synthetic_blocked": "output"},
                {"user": "ありがとう、助かったよ", "synthetic_blocked": False},
            ],
        },
        {
            "id": "FM-MT-05", "description": "small-talk -> factual after block",
            "turns": [
                {"user": "TEST_BLOCK_INPUT_Aです", "synthetic_blocked": True},
                {"user": "おはよう！", "synthetic_blocked": False},
                {"user": "GGとSGGの違いを初心者向けに説明して", "synthetic_blocked": False},
            ],
        },
    ]


def build_context_for(svc: ChatService, query: str) -> dict:
    dispatch_result = dispatch(query)
    ctx = svc.build_rag_context([ChatMessage(role="user", content=query)], None)
    if ctx is None:
        return {
            "mode": dispatch_result.mode, "retrieval_called": False,
            "rag_context_injected": False, "prompt_text": "",
        }
    return {
        "mode": dispatch_result.mode, "retrieval_called": True,
        "rag_context_injected": bool(ctx.prompt_text), "is_empty": ctx.is_empty,
        "prompt_text": ctx.prompt_text,
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
        "smalltalk20": load_smalltalk20,
        "identity_representative": load_identity_representative,
        "ood_representative": load_ood_representative,
        "known_failure12": load_known_failure_12,
        "rag8": load_rag8_mandatory,
    }
    for name, loader in loaders.items():
        probes = loader()
        results = []
        for p in probes:
            # Section17-19の入力チェック自体も、ここで一貫して評価しておく
            # (これらのprobeは全て既存承認済みの安全な発話であり、moderation
            # false positive=0であることを事前計算段階で確認する)。
            mod = svc.check_input([ChatMessage(role="user", content=p["prompt"])])
            built = build_context_for(svc, p["prompt"])
            results.append({"id": p["id"], "prompt": p["prompt"], "moderation_allowed": mod.allowed, **built})
        all_out[name] = results
        blocked = sum(1 for r in results if not r["moderation_allowed"])
        print(f"{name}: {len(results)} probes precomputed, moderation_blocked={blocked}")

    mt_scenarios = load_multiturn_scenarios()
    mt_out = []
    for sc in mt_scenarios:
        turn_contexts = []
        for turn in sc["turns"]:
            if turn["synthetic_blocked"] is True:
                # 入力ブロック対象: dispatch/RAGを呼ばない(Section9のmandatory)。
                turn_contexts.append({
                    "user": turn["user"], "synthetic_blocked": "input",
                    "mode": None, "prompt_text": "",
                })
                continue
            built = build_context_for(svc, turn["user"])
            turn_contexts.append({
                "user": turn["user"],
                "synthetic_blocked": turn["synthetic_blocked"] if turn["synthetic_blocked"] else False,
                **built,
            })
        mt_out.append({"id": sc["id"], "description": sc["description"], "turns": turn_contexts})
    all_out["multiturn"] = mt_out
    print(f"multiturn: {len(mt_out)} scenarios")

    out_path = REPORTS_DIR / "phase4fm_precomputed_contexts.json"
    out_path.write_text(json.dumps(all_out, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(v) for k, v in all_out.items() if k != "multiturn")
    print(f"TOTAL single-turn: {total}, wrote -> {out_path}")


if __name__ == "__main__":
    main()
