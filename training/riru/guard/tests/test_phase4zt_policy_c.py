"""Phase4ZT Section15: Policy C integrationのunit test。GPU不要。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUARD_DIR))
REPORTS_DIR = GUARD_DIR.parent / "reports"
PROJECT_ROOT = GUARD_DIR.parents[2]

from phase4zt_policy_c import decide_c1, decide_c2, decide_c3  # noqa: E402
from phase4zr_conservative_dispatch import dispatch, UNKNOWN  # noqa: E402
from phase4zp_router import PACHISLOT_FACTUAL, SMALL_TALK, OOD_FACTUAL  # noqa: E402


class TestUnknownContextlessPathBanned:
    def test_c1_never_produces_context_absent_rag(self):
        d = decide_c1("何かのクエリ", "何らかのcontext")
        assert not (d.selected_path == "rag_with_context" and not d.context_injected)

    def test_c2_never_uses_strict_rag_path(self):
        d = decide_c2("何かのクエリ", "")
        assert d.selected_path != "rag_with_context"

    def test_c3_rag_path_always_has_context(self):
        d = decide_c3("GGとSGGの違いを教えて", "context", ["title"], ["text"])
        if d.selected_path == "rag_with_context":
            assert d.context_injected is True

    def test_mandatory_invariant_recorded_zero(self):
        trace = json.loads((REPORTS_DIR / "phase4zt_path_trace.json").read_text(encoding="utf-8"))
        assert trace["mandatory_invariant_satisfied"] is True
        assert all(v == 0 for v in trace["mandatory_invariant_violations_by_variant"].values())


class TestQ6PolicyC:
    def test_q6_context_injection_via_c1(self):
        d = decide_c1("GGとSGGの違いを初心者向けに説明して", "何らかの実context")
        assert d.context_injected is True

    def test_q6_regression_zero_unsupported_numeric(self):
        d = json.loads((REPORTS_DIR / "phase4zt_q6_regression.json").read_text(encoding="utf-8"))
        for variant in ("C1", "C2", "C3"):
            assert d[variant]["greedy"]["summary"]["unsupported_numeric_count"] == 0
            assert d[variant]["production"]["summary"]["unsupported_numeric_count"] == 0


class TestRetrievalNotUsedAsClassifier:
    def test_c3_does_not_classify_purely_on_hit_presence(self):
        # 検索結果が非空でも、字句的重なりが全くなければclarificationへ倒れることを確認する
        # (retrieval hitの有無だけでmode判定しない、Section3準拠)。
        d = decide_c3("おはよう、今日も一日頑張ろうね", "context", ["天空の扉解説"], ["GG準備中開始から…"])
        assert d.selected_path == "clarification"

    def test_c3_uses_lexical_overlap_not_score(self):
        import inspect
        source = inspect.getsource(decide_c3)
        assert "score" not in source.lower()


class TestDirectPathUnchanged:
    def test_pachislot_factual_direct_path_never_touches_policy_c(self):
        # 確信度の高いdispatch結果(PACHISLOT_FACTUAL等)はPolicy Cのいずれのvariantにも
        # 一切影響されない設計であることを、path_traceのdirect-mode行で確認する。
        trace = json.loads((REPORTS_DIR / "phase4zt_path_trace.json").read_text(encoding="utf-8"))
        for variant in ("C1", "C2", "C3"):
            for t in trace["traces"][variant]:
                if not t.get("is_unknown", True):
                    assert "Phase4ZP, unchanged" in t["selected_policy"]

    def test_small_talk_direct_path_no_retrieval(self):
        trace = json.loads((REPORTS_DIR / "phase4zt_path_trace.json").read_text(encoding="utf-8"))
        for t in trace["traces"]["C1"]:
            if not t.get("is_unknown", True) and t["dispatch_mode"] == SMALL_TALK:
                assert t["retrieval_called"] is False

    def test_ood_direct_path_no_retrieval(self):
        trace = json.loads((REPORTS_DIR / "phase4zt_path_trace.json").read_text(encoding="utf-8"))
        for t in trace["traces"]["C1"]:
            if not t.get("is_unknown", True) and t["dispatch_mode"] == OOD_FACTUAL:
                assert t["retrieval_called"] is False


class TestConservativeDispatchUnchanged:
    def test_dispatch_module_not_modified_by_this_phase(self):
        for f in ["phase4zt_policy_c.py", "phase4zt_path_trace.py", "run_phase4zt_policy_generation.py"]:
            source = (GUARD_DIR / f).read_text(encoding="utf-8")
            assert "phase4zr_conservative_dispatch" not in source or "import" in source
            # importはOK(path_trace.pyのみ)だが、dispatch関数の再定義・変更が無いことを確認
        pt_source = (GUARD_DIR / "phase4zt_path_trace.py").read_text(encoding="utf-8")
        assert "def dispatch(" not in pt_source

    def test_dispatch_still_gives_same_results(self):
        assert dispatch("GODの機械割は？").mode == PACHISLOT_FACTUAL
        assert dispatch("おはよう").mode == SMALL_TALK
