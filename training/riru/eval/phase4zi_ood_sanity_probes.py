"""Phase 4ZI Section15: OOD/雑談 sanity check probe(24件、評価専用)。

Phase4ZHで「好きな季節ってある？」という無関係な雑談に「登録データにない」
という異常な応答が観測されたため、Phase4ZG candidateで同種の雑談への応答が
自然かどうかを確認する。目的は「専門外なので詳細回答しない」という適切な
境界表現と、「会話として不自然・壊れた応答をする」ことの区別。

Phase4ZF既存OOD probe(phase4zf_ood_probes.py)・Phase4ZH intrusion control教師
(phase4zh_structural_hardening_source_data.py INTRUSION_CONTROL)とは文面が
重複しないよう新規に設計している。一般知識クイズ大会にはしない
(OOD-GK/PR/HR/SFの範囲は既存probeで十分カバー済みのため対象外とし、
本ファイルは挨拶・お礼・軽い雑談・疲労・好みの話・天気雑談・別れの挨拶・
軽いdomain境界確認に限定する)。

作成後freeze、書き換えない。
"""

from __future__ import annotations

GREETING = [
    {"id": "ZI-OD-01", "category": "greeting", "prompt": "やあ、今日は調子いい感じ？"},
    {"id": "ZI-OD-02", "category": "greeting", "prompt": "夜遅くにごめんね、まだ起きてる？"},
    {"id": "ZI-OD-03", "category": "greeting", "prompt": "今日から利用させてもらうね、よろしく頼むよ"},
]

THANKS = [
    {"id": "ZI-OD-04", "category": "thanks", "prompt": "さっきはありがとう、助かったよ"},
    {"id": "ZI-OD-05", "category": "thanks", "prompt": "いつも付き合ってくれてありがとうね"},
    {"id": "ZI-OD-06", "category": "thanks", "prompt": "教えてくれて感謝してる"},
]

CASUAL_CHAT = [
    {"id": "ZI-OD-07", "category": "casual_chat", "prompt": "最近ハマってることある？"},
    {"id": "ZI-OD-08", "category": "casual_chat", "prompt": "休日は何して過ごすことが多い？"},
    {"id": "ZI-OD-09", "category": "casual_chat", "prompt": "今日は特に予定ないんだよね"},
    {"id": "ZI-OD-10", "category": "casual_chat", "prompt": "ちょっと愚痴聞いてもらってもいい？"},
]

FATIGUE = [
    {"id": "ZI-OD-11", "category": "fatigue", "prompt": "今日はなんか気分が乗らないな"},
    {"id": "ZI-OD-12", "category": "fatigue", "prompt": "残業続きでバテ気味なんだ"},
]

FAVORITE_THINGS = [
    {"id": "ZI-OD-13", "category": "favorite_things", "prompt": "好きな色ってある？"},
    {"id": "ZI-OD-14", "category": "favorite_things", "prompt": "好きな音楽のジャンルとかある？"},
    {"id": "ZI-OD-15", "category": "favorite_things", "prompt": "甘いものと辛いもの、どっちが好き？"},
    {"id": "ZI-OD-16", "category": "favorite_things", "prompt": "休みの日に行きたい場所とかある？"},
]

WEATHER = [
    {"id": "ZI-OD-17", "category": "weather", "prompt": "今日は風が強いね"},
    {"id": "ZI-OD-18", "category": "weather", "prompt": "そろそろ涼しくなってきたね"},
]

FAREWELL = [
    {"id": "ZI-OD-19", "category": "farewell", "prompt": "そろそろ寝るね、おやすみ"},
    {"id": "ZI-OD-20", "category": "farewell", "prompt": "今日はこの辺で切り上げるね"},
    {"id": "ZI-OD-21", "category": "farewell", "prompt": "また今度話そうね、バイバイ"},
]

DOMAIN_BOUNDARY_LIGHT = [
    {"id": "ZI-OD-22", "category": "domain_boundary_light", "prompt": "おすすめの映画ある？"},
    {"id": "ZI-OD-23", "category": "domain_boundary_light", "prompt": "簡単に作れる料理のレシピ教えて"},
    {"id": "ZI-OD-24", "category": "domain_boundary_light", "prompt": "面白い雑学とか知ってたら教えて"},
]

ALL_PROBES = (
    GREETING + THANKS + CASUAL_CHAT + FATIGUE + FAVORITE_THINGS + WEATHER + FAREWELL + DOMAIN_BOUNDARY_LIGHT
)

TOTAL = len(ALL_PROBES)
