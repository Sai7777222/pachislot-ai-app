"""Phase4ZO Section19: system-prompt-only boundary policyのunit test。

このフェーズはtrainingもrouterも追加しない、system prompt textのみの変更で
あるため、ここでのテストは(1) prompt textが要求される方針を実際に含んでいるか
の構造テストと、(2) ground truth annotation自体の論理的整合性テストの2種類。
GPU不要、pure-Python。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUARD_DIR))
REPORTS_DIR = GUARD_DIR.parent / "reports"
PROJECT_ROOT = GUARD_DIR.parents[2]

THREE_MODE_TEXT = (GUARD_DIR / "phase4zo_three_mode_prompt.txt").read_text(encoding="utf-8")
BASE_SYSTEM_TEXT = (PROJECT_ROOT / "config" / "prompts" / "system.jinja2").read_text(encoding="utf-8")
GROUND_TRUTH = json.loads((REPORTS_DIR / "phase4zo_boundary_ground_truth_v1.json").read_text(encoding="utf-8"))
ROWS_BY_ID = {r["probe_id"]: r for r in GROUND_TRUTH["rows"]}


class TestThreeModePromptContent:
    def test_smalltalk_no_rag_hedge_policy(self):
        assert "登録データを参照したような言い方" in THREE_MODE_TEXT or \
               ("登録データ" in THREE_MODE_TEXT and "しないでください" in THREE_MODE_TEXT)
        assert "データがない" in THREE_MODE_TEXT and "拒否しないでください" in THREE_MODE_TEXT

    def test_character_preference_allowed(self):
        assert "キャラクターとして自然に答えて" in THREE_MODE_TEXT

    def test_pachislot_machine_name_not_invented(self):
        assert "機種名を創作しないでください" in THREE_MODE_TEXT

    def test_no_placeholder_machine_names_policy(self):
        # 具体的な機種情報がない場合は聞き返す方針が明記されていること
        assert "聞き返して" in THREE_MODE_TEXT

    def test_ood_factual_specialist_boundary_policy(self):
        assert "専門分野ではない" in THREE_MODE_TEXT
        assert "詳しい解説を始めないでください" in THREE_MODE_TEXT


class TestBaseSystemPromptUnchanged:
    def test_rag_mode_keeps_grounding_instruction(self):
        # ベースのsystem.jinja2自体は本フェーズで一切変更していないことを、
        # RAG groundingの核心的な指示文言が残っていることで確認する。
        assert "構造化データの原文表記をそのまま使ってください" in BASE_SYSTEM_TEXT
        assert "登録データにありません" in BASE_SYSTEM_TEXT
        assert "創作" in BASE_SYSTEM_TEXT


class TestGroundTruthConsistency:
    def test_120_probes_present(self):
        assert GROUND_TRUTH["n_probes"] == 120

    def test_sleep_advice_is_ood(self):
        # 「良い睡眠をとるコツを教えて」はOOD_FACTUALかつspecialist_boundary_expected
        sleep_rows = [r for r in GROUND_TRUTH["rows"] if "睡眠" in r["prompt"]]
        assert sleep_rows, "sleep advice probe not found"
        for r in sleep_rows:
            assert r["expected_mode"] == "OOD_FACTUAL"
            assert r["specialist_boundary_expected"] is True

    def test_pachislot_preference_is_conversational(self):
        # 「パチスロで一番好きな瞬間っていつ？」はcharacter opinion扱い(factual DBではない)
        rows = [r for r in GROUND_TRUTH["rows"] if r["category"] == "pachislot_conversational"]
        assert len(rows) == 10
        for r in rows:
            assert r["character_opinion_allowed"] is True
            assert r["factual_answer_allowed"] is False
            assert r["no_fabricated_machine_names_required"] is True

    def test_news_probe_overridden_to_ambiguous(self):
        row = ROWS_BY_ID["ZN-D03"]
        assert row["expected_mode"] == "AMBIGUOUS"

    def test_character_opinion_vs_factual_knowledge_distinction(self):
        # 「犬派？猫派？」(character opinion)と「良い睡眠をとるコツ」(factual/OOD)が
        # 論理的に区別されていることを、ground truthのフィールド整合性で確認する。
        pref_row = ROWS_BY_ID["ZN-C04"]  # リルは犬派？猫派？
        assert pref_row["character_opinion_allowed"] is True
        assert pref_row["factual_answer_allowed"] is False
        assert pref_row["expected_mode"] == "SMALL_TALK"

        sleep_rows = [r for r in GROUND_TRUTH["rows"] if "睡眠" in r["prompt"]]
        assert sleep_rows[0]["character_opinion_allowed"] is False
        assert sleep_rows[0]["expected_mode"] == "OOD_FACTUAL"

    def test_all_ood_factual_rows_expect_specialist_boundary(self):
        for r in GROUND_TRUTH["rows"]:
            if r["expected_mode"] == "OOD_FACTUAL":
                assert r["specialist_boundary_expected"] is True, r["probe_id"]

    def test_all_small_talk_rows_do_not_expect_rag(self):
        for r in GROUND_TRUTH["rows"]:
            if r["expected_mode"] == "SMALL_TALK":
                assert r["rag_expected"] is False, r["probe_id"]

    def test_pachislot_factual_rows_carry_context_caveat(self):
        rows = [r for r in GROUND_TRUTH["rows"] if r["category"] == "pachislot_factual"]
        assert len(rows) == 20
        for r in rows:
            assert r["rag_expected"] is True
            assert r["caveat"] is not None
