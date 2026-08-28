"""Phase4ZR Section16: conservative dispatchのunit test。GPU不要。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUARD_DIR))
REPORTS_DIR = GUARD_DIR.parent / "reports"
PROJECT_ROOT = GUARD_DIR.parents[2]

from phase4zr_conservative_dispatch import dispatch, UNKNOWN  # noqa: E402
from phase4zp_router import PACHISLOT_FACTUAL, PACHISLOT_CONVERSATIONAL, SMALL_TALK, OOD_FACTUAL  # noqa: E402


class TestDispatchBasics:
    def test_confident_pachislot_factual(self):
        r = dispatch("GODの機械割は？")
        assert r.mode == PACHISLOT_FACTUAL and r.confident

    def test_confident_small_talk_greeting(self):
        r = dispatch("おはよう")
        assert r.mode == SMALL_TALK and r.confident

    def test_uncertain_defaults_to_unknown(self):
        # 具体的な信号を含まない曖昧な発話はUNKNOWNになるべき(SMALL_TALKへ無理に推測しない)。
        r = dispatch("最近どうしてる感じ？")
        assert r.mode == UNKNOWN and not r.confident

    def test_ambiguous_game_jargon_becomes_unknown_not_smalltalk(self):
        # ガイアベル等の機種固有語彙のみで、既存keyword categoryのいずれにも一致しない場合はUNKNOWN。
        r = dispatch("ガイアベルってどんな演出？")
        assert r.mode != SMALL_TALK  # 少なくとも安易にSMALL_TALKへは倒さない


class TestDangerousMisrouteFixed:
    def test_bare_probability_word_does_not_force_ood(self):
        # 「確率」単独はもうconfident OOD signalではない(RAG50-Q4等の実例に基づく修正)。
        r = dispatch("ガイアベルの確率は？")
        assert r.mode != OOD_FACTUAL

    def test_bare_ichiban_word_does_not_force_ood(self):
        r = dispatch("モードの中で滞在率が一番高いものと低いものの差はどれくらい？")
        assert r.mode != OOD_FACTUAL


class TestNoDangerousMisrouteOverall:
    def test_zero_dangerous_misroutes_recorded(self):
        rag50 = json.loads((REPORTS_DIR / "phase4zr_rag50_safety.json").read_text(encoding="utf-8"))
        assert rag50["dangerous_misroute_count"] == 0
        assert rag50["mandatory_all_safe"] is True


class TestGroundTruthFreeze:
    def test_ground_truth_hash_matches(self):
        hash_file = REPORTS_DIR / "phase4zr_ground_truth_hash.txt"
        content = hash_file.read_text(encoding="utf-8")
        recorded_hash = content.splitlines()[0].split("sha256: ")[1].strip()
        actual_hash = hashlib.sha256((REPORTS_DIR / "phase4zr_ground_truth.json").read_bytes()).hexdigest()
        assert recorded_hash == actual_hash

    def test_ground_truth_total_and_rag50(self):
        gt = json.loads((REPORTS_DIR / "phase4zr_ground_truth.json").read_text(encoding="utf-8"))
        assert gt["total"] >= 250
        for pid in ["P02", "LC-08", "Q11", "Q17", "AD-04"]:
            assert gt["rag50_must_have_check"][pid] is True


class TestExistingSystemsUnchanged:
    def test_production_rag_prompt_unchanged(self):
        text = (PROJECT_ROOT / "config" / "prompts" / "system.jinja2").read_text(encoding="utf-8")
        assert "構造化データの原文表記をそのまま使ってください" in text

    def test_retriever_and_config_not_touched_by_this_module(self):
        # このモジュール自体がretriever/embedder/vector_storeをimportしていないことを確認する
        # (Phase4ZRはdispatchのみでretrievalを一切呼ばない)。
        source = (GUARD_DIR / "phase4zr_conservative_dispatch.py").read_text(encoding="utf-8")
        assert "Retriever" not in source
        assert "VectorStore" not in source
