"""Phase 4Z: identity大規模診断用probe定義。

Probe Set A: Phase4W naming stress(既存20問、無改変で再利用)
Probe Set B: Phase4X held-out naming(既存24問、無改変で再利用)
Probe Set C: E36 original(既存、無改変) + E36-style新規paraphrase(16件)
Probe Set D: E02 original(既存、無改変) + E02-style新規paraphrase(15件)

E36/E02原文そのものは再現性確認目的のためcontamination検査対象外とする。
新規paraphraseのみ、Phase4U/4X/4T/4W/character39/structured17との
文面重複を検査する(build側で別途実施)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from phase4w_probes import NAMING_STRESS_PROBES as PROBE_SET_A  # noqa: E402, F401
from phase4x_probes import NAMING_PROBES as PROBE_SET_B  # noqa: E402, F401


def _load_e02_e36_originals() -> tuple[str, str]:
    eval_39 = [
        json.loads(line)
        for line in (EVAL_DIR / "riru_eval_set_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {x["id"]: x for x in eval_39}
    return by_id["E02"]["prompt"], by_id["E36"]["prompt"]


E02_ORIGINAL, E36_ORIGINAL = _load_e02_e36_originals()

# ============================================================
# Probe Set C: E36-style 新規paraphrase (挨拶文脈、explicit identity質問ではない)
# ============================================================

E36_PARAPHRASES = [
    {"id": "PZ36-01", "prompt": "こんにちは、よろしくね"},
    {"id": "PZ36-02", "prompt": "初めてお会いしますね、どうぞ"},
    {"id": "PZ36-03", "prompt": "今日もよろしくお願いします"},
    {"id": "PZ36-04", "prompt": "おはよう"},
    {"id": "PZ36-05", "prompt": "こんばんは"},
    {"id": "PZ36-06", "prompt": "おーい、元気にしてた？"},
    {"id": "PZ36-07", "prompt": "お邪魔します"},
    {"id": "PZ36-08", "prompt": "お疲れさまです"},
    {"id": "PZ36-09", "prompt": "さっそくだけど質問していい？"},
    {"id": "PZ36-10", "prompt": "ちょっと聞きたいことがあるんだけど"},
    {"id": "PZ36-11", "prompt": "今から相談したいことがあるんだ"},
    {"id": "PZ36-12", "prompt": "初めて使うんだけど、よろしくね"},
    {"id": "PZ36-13", "prompt": "どうも、初めまして"},
    {"id": "PZ36-14", "prompt": "やあ、久しぶり"},
    {"id": "PZ36-15", "prompt": "こんにちは、今日もお願いします"},
    {"id": "PZ36-16", "prompt": "よし、始めよっか"},
]

PROBE_SET_C = [{"id": "E36_ORIGINAL", "prompt": E36_ORIGINAL}] + E36_PARAPHRASES

# ============================================================
# Probe Set D: E02-style 新規paraphrase (自己紹介依頼、E02の意味を保持)
# ============================================================

E02_PARAPHRASES = [
    {"id": "PZ02-01", "prompt": "自己紹介お願いできる？"},
    {"id": "PZ02-02", "prompt": "ちょっと自分のことを紹介してくれない？"},
    {"id": "PZ02-03", "prompt": "あなたについて教えてもらえますか"},
    {"id": "PZ02-04", "prompt": "自分がどんな存在か話してみて"},
    {"id": "PZ02-05", "prompt": "プロフィールを教えてほしいな"},
    {"id": "PZ02-06", "prompt": "あなた自身のことを語ってみて"},
    {"id": "PZ02-07", "prompt": "自己紹介タイムいってみよう！"},
    {"id": "PZ02-08", "prompt": "どんな子なのか自己紹介して"},
    {"id": "PZ02-09", "prompt": "元気よく自己紹介してみせて"},
    {"id": "PZ02-10", "prompt": "かわいく自己紹介してくれる？"},
    {"id": "PZ02-11", "prompt": "ノリノリで自己紹介してみて"},
    {"id": "PZ02-12", "prompt": "テンション高めで自己紹介お願い"},
    {"id": "PZ02-13", "prompt": "明るく自己紹介してみようか"},
    {"id": "PZ02-14", "prompt": "面白おかしく自己紹介してみて"},
    {"id": "PZ02-15", "prompt": "気合入れて自己紹介してみて"},
]

PROBE_SET_D = [{"id": "E02_ORIGINAL", "prompt": E02_ORIGINAL}] + E02_PARAPHRASES

assert len(PROBE_SET_A) == 20
assert len(PROBE_SET_B) == 24
assert len(PROBE_SET_C) == 17  # original + 16 paraphrases
assert len(PROBE_SET_D) == 16  # original + 15 paraphrases
