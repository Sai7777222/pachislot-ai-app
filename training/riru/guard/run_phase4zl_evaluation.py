"""Phase 4ZL: Guard pipelineの一括評価ドライバ。

Stage C(新規held-out100) / Stage B(既存failure replay抜粋) / Stage D(RAG regression)
/ Stage E(OOD regression) をこの1スクリプトで実行し、rawとguarded双方の結果を記録する。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = GUARD_DIR.parents[0]
PROJECT_ROOT = GUARD_DIR.parents[2]
sys.path.insert(0, str(GUARD_DIR))
sys.path.insert(0, str(TRAINING_ROOT))
sys.path.insert(0, str(TRAINING_ROOT / "eval"))
REPORTS_DIR = TRAINING_ROOT / "reports"

from identity_guard_pipeline import IdentityGuardPipeline  # noqa: E402
from identity_validator import validate_identity  # noqa: E402

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"


def result_to_dict(r) -> dict | None:
    if r is None:
        return None
    return {"safe": r.safe, "category": r.category, "claimed_name": r.claimed_name,
            "reason": r.reason, "matched_text": r.matched_text, "context_flags": r.context_flags}


def run_multiturn_through_pipeline(pipeline: IdentityGuardPipeline, system_prompt: str, turns: list[str]) -> dict:
    """複数ターンのprobeを、各ターンでguard適用しつつ会話として進める。"""
    history_messages = []
    turn_logs = []
    for i, user_text in enumerate(turns):
        # 会話履歴をextra_system_contextとしてではなく、直接messagesへ組み込む
        # ため、ここではrespond()を使わず、直接pipeline内部relevant partsを呼ぶ。
        pipeline._load()
        messages = [{"role": "system", "content": system_prompt}] + history_messages + \
                   [{"role": "user", "content": user_text}]
        t0 = time.perf_counter()
        raw = pipeline._raw_generate(messages)
        t_gen = time.perf_counter() - t0

        v1 = validate_identity(raw, user_text)
        if v1.safe:
            final = raw
            stage = "pass"
            regenerated_text = None
            v2 = None
            t_regen = 0.0
        else:
            from identity_guard_pipeline import _REGENERATION_CONSTRAINT
            constrained_messages = list(messages)
            constrained_messages[0] = {"role": "system", "content": system_prompt + _REGENERATION_CONSTRAINT}
            t0 = time.perf_counter()
            regenerated_text = pipeline._raw_generate(constrained_messages)
            t_regen = time.perf_counter() - t0
            v2 = validate_identity(regenerated_text, user_text)
            if v2.safe:
                final = regenerated_text
                stage = "regenerated_pass"
            else:
                final = pipeline._next_fallback()
                stage = "fallback"

        turn_logs.append({
            "turn": i + 1, "user": user_text, "raw": raw, "regenerated": regenerated_text,
            "final": final, "stage": stage,
            "validator_first": result_to_dict(v1), "validator_second": result_to_dict(v2),
            "latency": {"generate_sec": t_gen, "regenerate_sec": t_regen},
        })
        history_messages.append({"role": "user", "content": user_text})
        history_messages.append({"role": "assistant", "content": final})
    return {"turns": turn_logs}


def mode_new_holdout_100():
    from phase4zl_new_holdout_100 import ALL_PROBES

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    pipeline = IdentityGuardPipeline(enabled=True)

    results = {}
    t0_all = time.perf_counter()
    for p in ALL_PROBES:
        out = run_multiturn_through_pipeline(pipeline, system_prompt, p["turns"])
        results[p["id"]] = {"category": p["category"], **out}
        if len(results) % 10 == 0:
            print(f"{len(results)}/{len(ALL_PROBES)} done ({time.perf_counter()-t0_all:.1f}s)")

    out_path = REPORTS_DIR / "phase4zl_new_holdout_100_raw_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")


def mode_existing_failure_replay():
    """Stage B: Phase4ZGの既存評価データ(_phase4zg_eager/sdpa_classified.json)から
    manually-confirmed genuine wrong-name(category_final in A,C)の実例56件相当を
    抽出し、その『元のprompt』を再度Phase4ZGへ入力してguard適用結果を見る。"""
    def load_classified(path):
        return json.load(open(path, encoding="utf-8"))

    # probe_idからprompt textを引けるように、既知のprobe pool群をまとめてロードする。
    from phase4zh_holdout_probes import ALL_PROBES as ZHH
    from phase4zg_holdout_probes import ALL_PROBES as ZGH
    from phase4ze_holdout_probes import ALL_PROBES as ZEH
    from phase4zf_stress_probes import ALL_PROBES as ZFS
    from phase4z_probes import PROBE_SET_A, PROBE_SET_B, PROBE_SET_C, PROBE_SET_D

    prompt_by_id: dict[str, str] = {}
    for pool in (ZHH, ZGH, ZEH, ZFS, PROBE_SET_A, PROBE_SET_B, PROBE_SET_C, PROBE_SET_D):
        for p in pool:
            prompt_by_id[p["id"]] = p["prompt"]

    eager = load_classified(REPORTS_DIR / "_phase4zg_eager_classified.json")
    sdpa = load_classified(REPORTS_DIR / "_phase4zg_sdpa_classified.json")
    unsafe_ids = sorted({item["probe_id"] for item in (eager + sdpa) if item["category_final"] in ("A", "C")})
    unsafe_ids_with_prompt = [pid for pid in unsafe_ids if pid in prompt_by_id]
    print(f"unique unsafe probe_ids with known prompt: {len(unsafe_ids_with_prompt)} / {len(unsafe_ids)}")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    pipeline = IdentityGuardPipeline(enabled=True)

    results = {}
    for pid in unsafe_ids_with_prompt:
        prompt = prompt_by_id[pid]
        out = run_multiturn_through_pipeline(pipeline, system_prompt, [prompt])
        results[pid] = {"prompt": prompt, **out}

    out_path = REPORTS_DIR / "phase4zl_existing_failure_replay_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")


def mode_rag_regression():
    from phase4zf_rag_stress_eval import load_rag_probe_pool
    all_rag = load_rag_probe_pool()
    target_ids = {"P02", "LC-08", "Q11", "AD-04"}
    extra = [p for p in all_rag if p["id"] not in target_ids][:26]
    probes = [p for p in all_rag if p["id"] in target_ids] + extra

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    pipeline = IdentityGuardPipeline(enabled=True)
    pipeline._load()

    results = {}
    for p in probes:
        context = p.get("context")
        messages = [{"role": "system", "content": system_prompt}]
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": p["question"]})
        raw = pipeline._raw_generate(messages)
        v = validate_identity(raw, p["question"])
        if v.safe:
            final, stage = raw, "pass"
        else:
            constrained = list(messages)
            from identity_guard_pipeline import _REGENERATION_CONSTRAINT
            constrained[0] = {"role": "system", "content": system_prompt + _REGENERATION_CONSTRAINT}
            regen = pipeline._raw_generate(constrained)
            v2 = validate_identity(regen, p["question"])
            final, stage = (regen, "regenerated_pass") if v2.safe else (pipeline._next_fallback(), "fallback")
        results[p["id"]] = {"set": p["set"], "raw": raw, "final": final, "stage": stage,
                             "modified": raw != final, "validator": result_to_dict(v)}

    out_path = REPORTS_DIR / "phase4zl_rag_regression_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")


def mode_ood_regression():
    from phase4zi_ood_sanity_probes import ALL_PROBES as OOD

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    pipeline = IdentityGuardPipeline(enabled=True)
    pipeline._load()

    results = {}
    for p in OOD:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": p["prompt"]}]
        raw = pipeline._raw_generate(messages)
        v = validate_identity(raw, p["prompt"])
        if v.safe:
            final, stage = raw, "pass"
        else:
            constrained = list(messages)
            from identity_guard_pipeline import _REGENERATION_CONSTRAINT
            constrained[0] = {"role": "system", "content": system_prompt + _REGENERATION_CONSTRAINT}
            regen = pipeline._raw_generate(constrained)
            v2 = validate_identity(regen, p["prompt"])
            final, stage = (regen, "regenerated_pass") if v2.safe else (pipeline._next_fallback(), "fallback")
        results[p["id"]] = {"category": p["category"], "prompt": p["prompt"], "raw": raw,
                             "final": final, "stage": stage, "modified": raw != final,
                             "validator": result_to_dict(v)}

    out_path = REPORTS_DIR / "phase4zl_ood_regression_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                         choices=["new_holdout_100", "existing_failure_replay", "rag_regression", "ood_regression"])
    args = parser.parse_args()
    {"new_holdout_100": mode_new_holdout_100, "existing_failure_replay": mode_existing_failure_replay,
     "rag_regression": mode_rag_regression, "ood_regression": mode_ood_regression}[args.mode]()
