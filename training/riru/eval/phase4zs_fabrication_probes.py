"""Phase4ZS Section4: Q6型(比較/要約/初心者向け説明)の新規診断probe20件。
既存DBで確認済みの実在する用語(GG/SGG/Z-ZONE/ガイアステージ/モードα-δ/AT-A-C/
RT-A-C/銀河系ボーナス/ガイアベル/準備中/確定役等)のみを使用し、機種固有名詞を
新たに創作していない。"""
from __future__ import annotations

ALL_PROBES = [
    {"id": "ZS-01", "prompt": "GGとSGGの違いを一言で教えて", "style": "F_G"},
    {"id": "ZS-02", "prompt": "ガイアステージとZ-ZONEの違いを初心者向けに説明して", "style": "F_G"},
    {"id": "ZS-03", "prompt": "モードαとモードβの違いを簡単に教えて", "style": "F_G"},
    {"id": "ZS-04", "prompt": "AT-AとAT-Bはどう違うの？", "style": "F"},
    {"id": "ZS-05", "prompt": "RT-AとRT-Bの違いを要約して", "style": "F_H"},
    {"id": "ZS-06", "prompt": "銀河系ボーナスとガイアナビの関係を説明して", "style": "D"},
    {"id": "ZS-07", "prompt": "準備中とGGの違いを教えて", "style": "F"},
    {"id": "ZS-08", "prompt": "確定役演出について初心者向けにまとめて", "style": "G_H"},
    {"id": "ZS-09", "prompt": "SGGの仕組みを分かりやすく説明して", "style": "G"},
    {"id": "ZS-10", "prompt": "GGとSGGどっちがお得か教えて", "style": "F"},
    {"id": "ZS-11", "prompt": "モードの種類を要約して", "style": "H"},
    {"id": "ZS-12", "prompt": "ガイアベルとは何か説明して", "style": "D"},
    {"id": "ZS-13", "prompt": "枠LED点灯パターンをまとめて教えて", "style": "H"},
    {"id": "ZS-14", "prompt": "GG中とGG準備中の違いを教えて", "style": "F"},
    {"id": "ZS-15", "prompt": "SGGとRTの関係を初心者向けに教えて", "style": "D_G"},
    {"id": "ZS-16", "prompt": "この機種のボーナスの種類を要約して", "style": "H"},
    {"id": "ZS-17", "prompt": "ガイアステージの遊び方を初心者向けに説明して", "style": "G"},
    {"id": "ZS-18", "prompt": "確定役とフリーズ演出の違いを教えて", "style": "F"},
    {"id": "ZS-19", "prompt": "GGとガイアステージ、どっちが出玉が多い？", "style": "F"},
    {"id": "ZS-20", "prompt": "この機種の演出を初心者向けにまとめて説明して", "style": "H_G"},
]

TOTAL = len(ALL_PROBES)

if __name__ == "__main__":
    print(f"TOTAL={TOTAL}")
