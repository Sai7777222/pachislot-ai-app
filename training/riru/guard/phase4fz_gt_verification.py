"""Phase4FZ Section9-11: 修正後のfind_relevant_structured_facts()を、frozen GT(40件)+
mandatory known-failures + phantom stress + real regression に対して実行し評価する。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
REPORTS_DIR = PROJECT_ROOT / "training" / "riru" / "reports"

from pachislot_ai.core.config import Settings  # noqa: E402
from pachislot_ai.data.db import create_structured_engine, open_session  # noqa: E402
from pachislot_ai.data.repositories import machine_repository as mrepo  # noqa: E402
from pachislot_ai.rag.structured_lookup import find_relevant_structured_facts  # noqa: E402

_NUM_RE = re.compile(r"\d+(?:\.\d+)?%?")


def main():
    settings = Settings()
    engine = create_structured_engine(settings.structured_db_path)
    gt = json.loads((REPORTS_DIR / "phase4fz_gt.json").read_text(encoding="utf-8"))

    with open_session(engine) as session:
        from sqlalchemy import select
        from pachislot_ai.data.models.structured import Machine
        machine_id = session.scalars(select(Machine.machine_id)).first()

        results = {"phantom": [], "real": [], "close_concept": []}

        # --- Phantom: 期待 NO_FACTS ---
        n_phantom_pass = 0
        for q in gt["phantom"]:
            findings = find_relevant_structured_facts(session, machine_id, q["query"])
            passed = len(findings) == 0
            n_phantom_pass += int(passed)
            results["phantom"].append({
                "id": q["id"], "query": q["query"], "phantom_entity": q["phantom_entity"],
                "gt_label": "NO_FACTS", "n_findings": len(findings),
                "findings_detail": [f.detail for f in findings],
                "verdict": "PASS" if passed else "FAIL",
            })

        # --- Real: 期待 EXPECTED_FACTS (n_findings > 0) ---
        n_real_pass = 0
        for q in gt["real"]:
            findings = find_relevant_structured_facts(session, machine_id, q["query"])
            passed = len(findings) > 0
            n_real_pass += int(passed)
            results["real"].append({
                "id": q["id"], "query": q["query"], "real_entity": q["real_entity"],
                "gt_label": "EXPECTED_FACTS", "n_findings": len(findings),
                "findings_detail": [f.detail for f in findings][:10],
                "verdict": "PASS" if passed else "FAIL",
            })

        # --- Close concept: 期待 EXPECTED_FACTS かつ 相手entityの情報が混入しない ---
        n_close_pass = 0
        for q in gt["close_concept"]:
            findings = find_relevant_structured_facts(session, machine_id, q["query"])
            detail = [f.detail for f in findings]
            has_facts = len(findings) > 0
            results["close_concept"].append({
                "id": q["id"], "query": q["query"], "target_entity": q["target_entity"],
                "confusable_with": q["confusable_with"], "gt_label": "EXPECTED_FACTS_ENTITY_SPECIFIC",
                "n_findings": len(findings), "findings_detail": detail[:15],
                "has_facts": has_facts,
            })
            n_close_pass += int(has_facts)

    out = {
        "phantom_structured_misbinding": f"{len(gt['phantom']) - n_phantom_pass}/{len(gt['phantom'])}",
        "phantom_pass_rate": f"{n_phantom_pass}/{len(gt['phantom'])}",
        "real_recall_rate": f"{n_real_pass}/{len(gt['real'])}",
        "close_concept_has_facts_rate": f"{n_close_pass}/{len(gt['close_concept'])}",
        "detail": results,
    }
    out_path = REPORTS_DIR / "_tmp_gt_verification.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"phantom: {n_phantom_pass}/{len(gt['phantom'])} correctly NO_FACTS")
    print(f"real: {n_real_pass}/{len(gt['real'])} correctly EXPECTED_FACTS")
    print(f"close_concept: {n_close_pass}/{len(gt['close_concept'])} has facts")
    for p in results["phantom"]:
        if p["verdict"] == "FAIL":
            print("PHANTOM FAIL:", p["id"], p["query"], "->", p["n_findings"], "findings:", p["findings_detail"])
    for r in results["real"]:
        if r["verdict"] == "FAIL":
            print("REAL FAIL:", r["id"], r["query"], "-> 0 findings")


if __name__ == "__main__":
    main()
