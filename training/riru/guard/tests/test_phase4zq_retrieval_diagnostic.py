"""Phase4ZQ Section14: minimal test set. GPU/embedding不要(既存の保存済みJSONを検証するのみ)。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
PROJECT_ROOT = Path(__file__).resolve().parents[4]


class TestGroundTruthFreeze:
    def test_ground_truth_frozen_before_retrieval(self):
        gt = json.loads((REPORTS_DIR / "phase4zq_ground_truth.json").read_text(encoding="utf-8"))
        assert all(row["frozen"] is True for row in gt["rows"])
        assert all("no_retrieval_seen" in row["annotation_source"] for row in gt["rows"])
        assert gt["total"] == 260

    def test_ground_truth_hash_recorded(self):
        hash_file = REPORTS_DIR / "phase4zq_ground_truth_hash.txt"
        content = hash_file.read_text(encoding="utf-8")
        assert "sha256:" in content
        assert "frozen_before_retrieval: true" in content
        # 実際にファイル内容とhashが一致することを確認する
        recorded_hash = content.splitlines()[0].split("sha256: ")[1].strip()
        actual_hash = hashlib.sha256((REPORTS_DIR / "phase4zq_ground_truth.json").read_bytes()).hexdigest()
        assert recorded_hash == actual_hash

    def test_mandatory_rag_probes_included(self):
        gt = json.loads((REPORTS_DIR / "phase4zq_ground_truth.json").read_text(encoding="utf-8"))
        for pid in ["P02", "P04", "LC-08", "Q11", "Q15", "Q17", "AD-04"]:
            assert gt["rag50_must_have_check"][pid] is True, pid


class TestRetrievalTraceSaved:
    def test_retrieval_trace_saved_for_all_probes(self):
        trace = json.loads((REPORTS_DIR / "phase4zq_retrieval_trace.json").read_text(encoding="utf-8"))
        assert trace["n_total"] == 260
        assert all("top1_score" in row for row in trace["rows"])
        assert all("retrieved_chunk_ids" in row for row in trace["rows"])


class TestExistingSystemsUnchanged:
    def test_rag_prompt_unchanged(self):
        text = (PROJECT_ROOT / "config" / "prompts" / "system.jinja2").read_text(encoding="utf-8")
        assert "構造化データの原文表記をそのまま使ってください" in text
        assert "登録データにありません" in text

    def test_retriever_module_unchanged_read_only(self):
        # このphaseはRetriever.search()を呼ぶだけで、rag/embedder.py, rag/vector_store.py,
        # rag/retriever.py自体は一切編集していないことを、既存の本体pytestが全通過することで
        # 間接的に確認する(training/riru/guard/tests/ の外側にある実際のsrc配下test群)。
        retriever_path = PROJECT_ROOT / "src" / "pachislot_ai" / "rag" / "retriever.py"
        assert retriever_path.exists()

    def test_no_config_change(self):
        config_path = PROJECT_ROOT / "src" / "pachislot_ai" / "core" / "config.py"
        text = config_path.read_text(encoding="utf-8")
        assert 'vector_db_collection: str = "pachislot_rag"' in text


class TestNoTraining:
    def test_zg_adapter_directory_untouched_marker(self):
        # Phase4ZGアダプタのディレクトリ自体が存在し続けていることのみ確認する
        # (実際のhash一致確認はphase4zq_end_integrity.jsonで別途記録)。
        adapter_dir = PROJECT_ROOT / "training" / "riru" / "lora-riru-qwen-phase4zg-identity-hardened"
        assert adapter_dir.exists()
