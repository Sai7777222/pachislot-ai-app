"""Phase4ZO Stage B-E をまとめて実行する(model一度だけロード)。variant=three_mode固定。"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase4zo_generation as g  # noqa: E402

if __name__ == "__main__":
    g.stage_smalltalk_recheck("three_mode")
    g.stage_ood_recheck("three_mode")
    g.stage_pachislot_conv_recheck("three_mode")
    g.stage_ambiguous_recheck("three_mode")
    g.stage_rag50_recheck("three_mode")
    print("ALL STAGES DONE")
