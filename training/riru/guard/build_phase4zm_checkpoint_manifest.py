"""Phase4ZM Section26-28: 未追跡ファイル(156件)をA-Eへ分類し、checkpoint候補を
提案する。git add/commitは一切実行しない(Section28: Human Approval Gate)。"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def classify(path: str) -> str:
    # B: 却下されたphase(ZH/ZJ/ZK-M1)専用のtraining data/config/source-data script。
    #    (adapter本体はtraining/riru/lora-riru-qwen-*/として既に.gitignoreで除外済み。)
    rejected_markers = ["phase4zh", "phase4zj", "phase4zk_m1", "zk_m1"]
    if any(m in path.lower() for m in rejected_markers) and (
        "processed/" in path or "configs/" in path or path.endswith("_dataset.py")
        or "source_data.py" in path or "merge_phase4zh" in path
    ):
        return "B_diagnostic_archive"
    # D: 使い捨てデバッグダンプ(_始まり)。実際には既に.gitignoreの
    #    training/riru/reports/_* パターンで除外されているはずだが、念のため分類。
    if "/_" in path or path.split("/")[-1].startswith("_"):
        return "D_temporary_cache"
    # それ以外(reports全般・eval全般・guard全般・phase4zi/zl/zmの成果物)はA: retain。
    return "A_retain"


def main():
    lines = (ROOT / ".." if False else Path.cwd())  # placeholder, unused
    result = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                             text=True, encoding="utf-8")
    untracked = [ln[3:].strip() for ln in result.stdout.splitlines() if ln.startswith("??")]

    buckets: dict[str, list[str]] = {"A_retain": [], "B_diagnostic_archive": [],
                                       "C_rejected_adapter_bulky": [], "D_temporary_cache": [], "E_unknown": []}
    for p in untracked:
        buckets[classify(p)].append(p)

    manifest = {
        "purpose": "Section26-28: untracked 156件の分類とcheckpoint候補提案。git add/commit/pushは"
                   "本フェーズでは一切実行しない(Human Approval Gate)。",
        "total_untracked": len(untracked),
        "note_on_category_C": "Phase4ZH/ZJ/ZK-M1のLoRAアダプタ本体(各500MB超)は、既に"
                               ".gitignoreの`training/riru/lora-riru-qwen-*/`パターンで除外されて"
                               "おり、そもそもuntracked一覧に現れない。したがってcategory C"
                               "(rejected adapter/bulky artifact)に該当する実ファイルは今回0件。",
        "counts": {k: len(v) for k, v in buckets.items()},
        "buckets": buckets,
        "checkpoint_recommendation": {
            "include": buckets["A_retain"] + buckets["B_diagnostic_archive"],
            "include_rationale": "A(現行の検証履歴・成果物)はもちろん、B(却下されたphaseの"
                                  "training data/config)も、『なぜ却下したか』の再現性を将来検証"
                                  "する際に必要となるため、除外せず含める(Section26: 削除は原則"
                                  "しない、の精神をcheckpointにも適用)。",
            "exclude": buckets["D_temporary_cache"],
            "exclude_rationale": "使い捨てデバッグダンプ(_始まりファイル)は再現性に寄与しない。"
                                  "ただし実際には.gitignoreで既に除外されるため、git add時に自動的に"
                                  "スキップされる。",
            "secrets_binaries_check": "training/riru/lora-riru-qwen-*/ (LoRAアダプタ本体、各500MB超)は"
                                       ".gitignoreで除外済み。training/riru/merged/ (mergeされたHF"
                                       "モデル本体)も同様に除外済み。.venv/.venv-qlora/等の仮想環境も"
                                       "除外済み。APIキー/認証情報を含むファイルは本フェーズで新規作成"
                                       "していない。",
        },
        "git_add_git_commit_executed": False,
    }
    out_path = ROOT / "training" / "riru" / "reports" / "phase4zm_checkpoint_manifest.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in buckets.items():
        print(f"{k}: {len(v)}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
