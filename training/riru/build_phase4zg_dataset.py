"""Phase 4ZG: 既存Phase4ZE candidate(1145件)にResidual Identity Hardening教師(48件)を
追加した phase4zg_identity_hardened candidate を構築する。

既存1145件は無改変・読み取り専用で読み込み、新規48件のみを追加した別ファイルとして
新candidateを構築する。既存riru_phase4ze_identity_margin_candidate.jsonl等は無改変。

品質検査: placeholder/ChatML/重複/高類似度/実在機種名/wrong-name列挙/
Phase4T〜4ZFの全probe文面・Phase4ZE教師50件とのcontaminationを確認する。

学習は本ファイルでは行わない。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = TRAINING_ROOT / "processed"
REPORTS_DIR = TRAINING_ROOT / "reports"
EVAL_DIR = TRAINING_ROOT / "eval"

sys.path.insert(0, str(TRAINING_ROOT))
sys.path.insert(0, str(EVAL_DIR))
from phase4zg_identity_hardening_source_data import ALL_RECORDS  # noqa: E402
from phase4x_placeholder_detector import classify_placeholder  # noqa: E402

EXISTING_CANDIDATE = PROCESSED_DIR / "riru_phase4ze_identity_margin_candidate.jsonl"

CHATML_PATTERN = re.compile(r"<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]")
REAL_MACHINE_HINTS = ["ミリオンゴッド", "モンキーターン", "北斗", "ジャグラー", "バジリスク"]
WRONG_NAME_LIST = [
    "ルリ", "ルナ", "リリ", "リコ", "ルカ", "パチ子", "パチスロ君",
    "パチスロナビ", "パチスロAI", "あいこ", "ルル", "アリス", "あい", "ミカ", "パチリ",
]


def char_bigrams(text: str) -> set[str]:
    t = re.sub(r"\s+", "", text)
    return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) >= 2 else {t}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_existing() -> list[dict]:
    records = []
    with open(EXISTING_CANDIDATE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_riru_record(item: dict, category: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]},
        ],
        "metadata": {"source": "phase4zg_identity_hardening", "category": category,
                      "index": item["id"]},
    }


def load_contamination_sources() -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {}

    import phase4t_probes as pt  # noqa: PLC0415
    sources["phase4t_naming_probes"] = [p["prompt"] for p in pt.NAMING_PROBES]
    sources["phase4t_p04_probes"] = [p["question"] for p in pt.P04_PROBES]

    import phase4u_identity_source_data as pu  # noqa: PLC0415
    sources["phase4u_identity"] = [p["user"] for p in pu.IDENTITY_RECORDS]
    sources["phase4u_intrusion"] = [p["user"] for p in pu.NO_INTRUSION_RECORDS]

    import phase4v_probes as pv  # noqa: PLC0415
    sources["phase4v_probes"] = [p["question"] for p in pv.PROBES]

    import phase4w_probes as pw  # noqa: PLC0415
    sources["phase4w_q9"] = [p["question"] for p in pw.Q9_PROBES]
    sources["phase4w_q11"] = [p["question"] for p in pw.Q11_PROBES]
    sources["phase4w_naming_stress"] = [p["prompt"] for p in pw.NAMING_STRESS_PROBES]
    sources["phase4w_adversarial"] = [p["question"] for p in pw.ADVERSARIAL_PROBES]
    sources["phase4w_conflicting"] = [p["question"] for p in pw.CONFLICTING_PROBES]
    sources["phase4w_longcontext"] = [p["question"] for p in pw.LONGCONTEXT_PROBES]

    import phase4x_identity_stabilization_source_data as pix  # noqa: PLC0415
    import phase4x_probes as px4x  # noqa: PLC0415
    sources["phase4x_naming_probes"] = [p["prompt"] for p in px4x.NAMING_PROBES]
    sources["phase4x_identity_stabilization"] = [r["user"] for r in pix.ALL_RECORDS]

    import phase4z_probes as pz  # noqa: PLC0415
    sources["phase4z_e36_family"] = [p["prompt"] for p in pz.PROBE_SET_C]
    sources["phase4z_e02_family"] = [p["prompt"] for p in pz.PROBE_SET_D]

    import phase4ze_holdout_probes as pzeh  # noqa: PLC0415
    sources["phase4ze_holdout"] = [p["prompt"] for p in pzeh.ALL_PROBES]

    import phase4ze_identity_margin_source_data as pzet  # noqa: PLC0415
    sources["phase4ze_training"] = [r["user"] for r in pzet.ALL_RECORDS]

    import phase4zf_stress_probes as pzfs  # noqa: PLC0415
    sources["phase4zf_stress"] = [p["prompt"] for p in pzfs.ALL_PROBES]

    import phase4zf_ood_probes as pzfo  # noqa: PLC0415
    sources["phase4zf_ood"] = [p["prompt"] for p in pzfo.ALL_PROBES]

    rag17q = json.loads((EVAL_DIR / "structured_rag_17q_context.json").read_text(encoding="utf-8"))
    sources["structured_17q"] = [r["question"] for r in rag17q] + [r["rag_context_text"] for r in rag17q]

    eval39 = [
        json.loads(line)
        for line in (EVAL_DIR / "riru_eval_set_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    c39_texts = []
    for item in eval39:
        if "prompt" in item:
            c39_texts.append(item["prompt"])
        elif "turns" in item:
            c39_texts.extend(item["turns"])
    sources["character_39"] = c39_texts

    holdout = json.loads((EVAL_DIR / "phase4i_holdout_omission_v2.json").read_text(encoding="utf-8"))
    sources["holdout_p01_p10"] = [r["question"] for r in holdout] + [r["rag_context_text"] for r in holdout]

    return sources


def quality_check(new_records: list[dict]) -> dict:
    issues: dict[str, list] = {
        "placeholder": [], "chatml": [], "real_machine_name": [],
        "wrong_name_enumeration_in_single_record": [], "exact_duplicate": [], "empty": [],
        "assistant_claims_wrong_name": [],
    }
    seen = {}
    for rec in new_records:
        rid = rec["metadata"]["index"]
        user_text = rec["messages"][0]["content"]
        assistant_text = rec["messages"][1]["content"]

        ph = classify_placeholder(assistant_text)
        if ph["is_placeholder"]:
            issues["placeholder"].append({"id": rid, "matched": ph["matched_text"]})
        if CHATML_PATTERN.search(assistant_text) or CHATML_PATTERN.search(user_text):
            issues["chatml"].append(rid)
        for name in REAL_MACHINE_HINTS:
            if name in user_text or name in assistant_text:
                issues["real_machine_name"].append({"id": rid, "name": name})
        wrong_names_in_assistant = [wn for wn in WRONG_NAME_LIST if wn in assistant_text]
        if len(wrong_names_in_assistant) > 1:
            issues["wrong_name_enumeration_in_single_record"].append(
                {"id": rid, "names": wrong_names_in_assistant})
        # assistant自身が誤名を「自分の名前」として主張していないか(訂正目的の言及は許容するため、
        # 「私は<wrongname>」「名前は<wrongname>」の直接パターンのみ検査)
        for wn in WRONG_NAME_LIST:
            if re.search(rf"(私は|僕は|名前は){wn}(だよ|です|なんだ)", assistant_text):
                issues["assistant_claims_wrong_name"].append({"id": rid, "name": wn})
        if not user_text.strip() or not assistant_text.strip():
            issues["empty"].append(rid)
        key = (user_text, assistant_text)
        if key in seen:
            issues["exact_duplicate"].append({"id": rid, "dup_of": seen[key]})
        else:
            seen[key] = rid

    high_sim_pairs = []
    for i in range(len(new_records)):
        bi = char_bigrams(new_records[i]["messages"][1]["content"])
        for j in range(i + 1, len(new_records)):
            sim = jaccard(bi, char_bigrams(new_records[j]["messages"][1]["content"]))
            if sim >= 0.85:
                high_sim_pairs.append({
                    "a": new_records[i]["metadata"]["index"], "b": new_records[j]["metadata"]["index"],
                    "similarity": round(sim, 3),
                })

    contamination_sources = load_contamination_sources()
    contamination_hits = []
    for rec in new_records:
        user_text = rec["messages"][0]["content"]
        for src_name, texts in contamination_sources.items():
            for t in texts:
                if t and (t in user_text or user_text in t):
                    contamination_hits.append(
                        {"id": rec["metadata"]["index"], "source": src_name, "matched_text": t[:60]})

    return {"issues": issues, "high_similarity_pairs": high_sim_pairs, "contamination_hits": contamination_hits}


def group_key(rec: dict) -> str:
    return rec["messages"][0]["content"]


def group_safe_split(records: list[dict], val_ratio: float, seed: int = 42):
    import random as _random
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(group_key(r), []).append(r)
    keys = list(groups.keys())
    rng = _random.Random(seed)
    rng.shuffle(keys)
    n_val_target = max(1, round(len(records) * val_ratio))
    val_records: list[dict] = []
    train_records: list[dict] = []
    for k in keys:
        if len(val_records) < n_val_target:
            val_records.extend(groups[k])
        else:
            train_records.extend(groups[k])
    return train_records, val_records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    existing = load_existing()
    n_complex_existing = sum(1 for r in existing if r["metadata"].get("source") == "phase4s_ratio_high")
    n_identity_existing = sum(
        1 for r in existing
        if r["metadata"].get("source") in ("phase4u_identity", "phase4x_identity_stabilization",
                                            "phase4ze_identity_margin"))

    category_map = {
        "ZG-A": "wrong_name_induction_resistance",
        "ZG-B": "identity_correction_persistence",
        "ZG-C": "placeholder_echo_resistance",
        "ZG-D": "intrusion_control",
    }
    new_records = []
    for item in ALL_RECORDS:
        prefix = item["id"][:4]
        cat = category_map.get(prefix, "identity_hardening_other")
        new_records.append(to_riru_record(item, cat))

    qc = quality_check(new_records)
    n_issues = sum(len(v) if isinstance(v, list) else 0 for v in qc["issues"].values())
    n_issues += len(qc["high_similarity_pairs"]) + len(qc["contamination_hits"])

    (REPORTS_DIR / "phase4zg_dataset_leakage_analysis.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"quality issues total: {n_issues}")
    print(json.dumps({k: len(v) for k, v in qc["issues"].items()}, ensure_ascii=False))
    print(f"high_similarity_pairs: {len(qc['high_similarity_pairs'])}")
    print(f"contamination_hits: {len(qc['contamination_hits'])}")

    if n_issues > 0:
        print("QUALITY ISSUES FOUND -- not proceeding to build candidate.")
        return 1

    combined = existing + new_records
    complex_ratio = round(100 * n_complex_existing / len(combined), 2)
    identity_total = n_identity_existing + len(new_records)
    identity_ratio = round(100 * identity_total / len(combined), 2)
    intrusion_new = sum(1 for item in ALL_RECORDS if item["id"].startswith("ZG-D"))
    intrusion_ratio_of_new = round(100 * intrusion_new / len(new_records), 1)

    print(f"existing={len(existing)} +new={len(new_records)} = total={len(combined)}")
    print(f"complex teachers (unchanged): {n_complex_existing}, ratio: {complex_ratio}%")
    print(f"identity teachers total: {identity_total}, ratio: {identity_ratio}%")
    print(f"intrusion-control among new: {intrusion_new}/{len(new_records)} ({intrusion_ratio_of_new}%)")

    write_jsonl(PROCESSED_DIR / "riru_phase4zg_identity_hardened_candidate.jsonl", combined)

    val_ratio = 0.10
    train, val = group_safe_split(combined, val_ratio, seed=42)
    train_keys = {group_key(r) for r in train}
    val_keys = {group_key(r) for r in val}
    overlap = len(train_keys & val_keys)

    write_jsonl(PROCESSED_DIR / "riru_phase4zg_identity_hardened_train.jsonl", train)
    write_jsonl(PROCESSED_DIR / "riru_phase4zg_identity_hardened_val.jsonl", val)

    summary = {
        "existing_count": len(existing), "existing_complex_count": n_complex_existing,
        "existing_identity_count": n_identity_existing, "new_count": len(new_records),
        "new_by_category": {
            cat: sum(1 for item in ALL_RECORDS if item["id"].startswith(prefix))
            for prefix, cat in category_map.items()
        },
        "combined_total": len(combined), "complex_ratio_pct": complex_ratio,
        "identity_total_count": identity_total, "identity_ratio_pct": identity_ratio,
        "intrusion_control_new_count": intrusion_new,
        "intrusion_control_ratio_of_new_pct": intrusion_ratio_of_new,
        "train_count": len(train), "val_count": len(val), "train_val_overlap": overlap,
    }
    (REPORTS_DIR / "phase4zg_training_data_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
