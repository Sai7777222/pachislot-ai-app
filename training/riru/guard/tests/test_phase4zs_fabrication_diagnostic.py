"""Phase4ZS Section16: 診断結果のintegrity/structure test。GPU不要(保存済みJSONの検証のみ)。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
PROJECT_ROOT = Path(__file__).resolve().parents[4]


class TestQ6Forensic:
    def test_forensic_capture_saved(self):
        d = json.loads((REPORTS_DIR / "phase4zs_q6_forensic.json").read_text(encoding="utf-8"))
        assert d["retrieved_chunk_count"] > 0
        assert d["stop_condition_check"]["triggered"] is False

    def test_original_generation_had_no_context(self):
        d = json.loads((REPORTS_DIR / "phase4zs_q6_forensic.json").read_text(encoding="utf-8"))
        assert d["critical_premise_check"]["phase4zr_original_generation_had_context_injected"] is False


class TestGroundTruthFreeze:
    def test_ground_truth_hash_matches(self):
        hash_file = REPORTS_DIR / "phase4zs_ground_truth_hash.txt"
        content = hash_file.read_text(encoding="utf-8")
        recorded_hash = content.splitlines()[0].split("sha256: ")[1].strip()
        actual_hash = hashlib.sha256((REPORTS_DIR / "phase4zs_ground_truth.json").read_bytes()).hexdigest()
        assert recorded_hash == actual_hash

    def test_ground_truth_from_retrieval_not_generation(self):
        gt = json.loads((REPORTS_DIR / "phase4zs_ground_truth.json").read_text(encoding="utf-8"))
        assert all(row["annotation_source"] == "read_only_retrieval_before_any_generation" for row in gt["rows"])
        assert gt["total"] == 70


class TestZeroContextVsRealContext:
    def test_zero_context_reproduces_fabrication(self):
        d = json.loads((REPORTS_DIR / "phase4zs_zero_context_confirmation.json").read_text(encoding="utf-8"))
        assert d["unsupported_numeric_rate"] == 1.0

    def test_real_context_prevents_fabrication_q6(self):
        d = json.loads((REPORTS_DIR / "phase4zs_q6_reproduction.json").read_text(encoding="utf-8"))
        assert d["greedy"]["summary"]["unsupported_numeric_rate"] == 0.0
        assert d["production_sampling"]["summary"]["unsupported_numeric_rate"] == 0.0

    def test_rag50_existing_outputs_have_no_unsupported_numerics(self):
        d = json.loads((REPORTS_DIR / "phase4zs_rag50_numeric_audit.json").read_text(encoding="utf-8"))
        assert d["unsupported_numeric_turn_count"] == 0
        assert d["n_total"] == 50


class TestRootCauseCase:
    def test_case_is_zs_d(self):
        d = json.loads((REPORTS_DIR / "phase4zs_root_cause.json").read_text(encoding="utf-8"))
        assert d["root_cause_classification"]["case"] == "ZS-D"


class TestNoProductionChanges:
    def test_production_rag_prompt_unchanged(self):
        text = (PROJECT_ROOT / "config" / "prompts" / "system.jinja2").read_text(encoding="utf-8")
        assert "構造化データの原文表記をそのまま使ってください" in text

    def test_dispatch_module_unchanged_by_this_phase(self):
        # Phase4ZSはdispatch/router変更禁止。phase4zr_conservative_dispatch.pyがこのphaseの
        # ファイル群からimportされていないことを確認する(このphaseは独立した診断のみ)。
        for f in ["run_phase4zs_diagnostics.py", "phase4zs_q6_forensic.py", "phase4zs_rag50_audit.py"]:
            source = (Path(__file__).resolve().parents[1] / f).read_text(encoding="utf-8")
            assert "phase4zr_conservative_dispatch" not in source
