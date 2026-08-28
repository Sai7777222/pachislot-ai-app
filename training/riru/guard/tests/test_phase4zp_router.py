"""Phase4ZP Section19: lightweight router + mode-policyのunit test。GPU不要。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUARD_DIR))
REPORTS_DIR = GUARD_DIR.parent / "reports"
PROJECT_ROOT = GUARD_DIR.parents[2]

from phase4zp_router import (  # noqa: E402
    route, SMALL_TALK, PACHISLOT_FACTUAL, PACHISLOT_CONVERSATIONAL, OOD_FACTUAL,
)

BASE_SYSTEM_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
CONV_PROMPT = (GUARD_DIR / "phase4zp_pachislot_conversational_prompt.txt").read_text(encoding="utf-8")


class TestRouterBasicModes:
    def test_router_smalltalk(self):
        assert route("おはよう、今日も頑張ろうね").mode == SMALL_TALK

    def test_router_pachislot_factual(self):
        assert route("GODの機械割は？").mode == PACHISLOT_FACTUAL

    def test_router_pachislot_conversational(self):
        assert route("パチスロ好き？").mode == PACHISLOT_CONVERSATIONAL

    def test_router_ood_factual(self):
        assert route("今日の東京の最高気温は？").mode == OOD_FACTUAL


class TestCriticalDistinctions:
    """Section6必須のdistinction test。"""

    def test_weather_statement_vs_fact(self):
        assert route("今日は暑いね").mode == SMALL_TALK
        assert route("今日の東京の最高気温は？").mode == OOD_FACTUAL

    def test_movie_preference_vs_fact(self):
        assert route("映画好き？").mode == SMALL_TALK
        assert route("今年一番売れた映画は？").mode == OOD_FACTUAL

    def test_dog_preference_vs_fact(self):
        assert route("犬派？猫派？").mode == SMALL_TALK
        assert route("犬の平均寿命は？").mode == OOD_FACTUAL

    def test_experience_question_smalltalk(self):
        for text in ["最近面白いことあった？", "最近ハマってることある？", "休みの日は何してるの？",
                     "行きたい場所ある？", "最近本読んだ？"]:
            assert route(text).mode == SMALL_TALK, text


class TestRetrievalIsolation:
    def test_smalltalk_skips_retrieval(self):
        # SMALL_TALKにroutingされたqueryは、PACHISLOT_FACTUAL専用path(既存RAG system prompt)
        # を一切経由しない設計であることを、mode値そのもので確認する。
        r = route("元気にしてた？")
        assert r.mode != PACHISLOT_FACTUAL

    def test_ood_skips_retrieval(self):
        r = route("株式投資のコツを教えて")
        assert r.mode != PACHISLOT_FACTUAL

    def test_pachislot_factual_uses_existing_rag(self):
        # PACHISLOT_FACTUALへrouteされたqueryは、run_phase4zp_generation.pyのPROMPT_PATHS
        # マッピングに存在しない(=専用の軽量promptを持たず、既存system.jinja2を使う設計)。
        import run_phase4zp_generation as g
        assert PACHISLOT_FACTUAL not in g.PROMPT_PATHS
        r = route("設定6の初当り確率は？")
        assert r.mode == PACHISLOT_FACTUAL


class TestRagPromptUnchanged:
    def test_rag_prompt_unchanged(self):
        text = BASE_SYSTEM_PATH.read_text(encoding="utf-8")
        # Phase4ZO/ZP通じて、既存RAG groundingの核心的な指示文言が一切変更されていないことを確認する。
        assert "構造化データの原文表記をそのまま使ってください" in text
        assert "登録データにありません" in text
        assert "創作" in text


class TestNoPlaceholderMachineName:
    def test_no_placeholder_machine_name(self):
        assert "機種名を創作しないでください" in CONV_PROMPT
        assert "聞き返して" in CONV_PROMPT


class TestRouterGroundTruthIndependence:
    def test_router_ground_truth_independent(self):
        # RULE EVAL-001/002準拠: router evalはrouter自身の出力ではなく、事前annotationの
        # phase4zp_router_ground_truth.jsonと比較していることを構造的に確認する。
        gt = json.loads((REPORTS_DIR / "phase4zp_router_ground_truth.json").read_text(encoding="utf-8"))
        assert gt["total"] == 120
        assert all(row["frozen"] is True for row in gt["rows"])
        assert all("human_predefined" in row["annotation_source"] for row in gt["rows"])

    def test_rag50_independent_check_reveals_generalization_gap(self):
        # 自作ground truth(100%)とは独立に作成されたRAG50への適用結果が、
        # 実際に大きなgapを露呈していることを回帰的に記録する
        # (このgapを覆い隠さない、というのが本テストの目的)。
        eq = json.loads((REPORTS_DIR / "phase4zp_rag_prompt_equivalence.json").read_text(encoding="utf-8"))
        assert eq["n_total"] == 50
        assert eq["dangerous_misroute_count"] > 0  # このgapの存在自体を記録する回帰テスト
