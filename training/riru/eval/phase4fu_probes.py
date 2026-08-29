"""Phase4FU Section4: multi-context synthesis診断用の独立probe set(最低30件)。
既存の確認済み実在語彙(GG/SGG/Z-ZONE/ガイアステージ/モードα-δ/AT-A-C/RT-A-C/
銀河系ボーナス/ガイアベル/確定役/フリーズ/枠LED等)のみ使用し、新規の機種固有名詞は
創作していない。"""
from __future__ import annotations

ALL_PROBES = [
    # A. single factual x5 (RAG50既存probe再利用)
    {"id": "FU-A01", "category": "single_factual", "prompt": "ボーナス確率は設定によってどう変わりますか？", "source": "RAG50-P02"},
    {"id": "FU-A02", "category": "single_factual", "prompt": "天井は何ゲームですか？設定変更時と通常時それぞれ振り分けを教えて", "source": "RAG50-Q11"},
    {"id": "FU-A03", "category": "single_factual", "prompt": "AT-Fの性能と終了後の状態について教えて", "source": "RAG50-LC-08"},
    {"id": "FU-A04", "category": "single_factual", "prompt": "ALL色(判別)について教えて", "source": "RAG50-AD-04"},
    {"id": "FU-A05", "category": "single_factual", "prompt": "設定1と設定6の機械割の差はどれくらい？", "source": "RAG50-P04"},
    # B. comparison x5
    {"id": "FU-B01", "category": "comparison", "prompt": "GGとSGGの違いを教えて", "source": "ZS-style"},
    {"id": "FU-B02", "category": "comparison", "prompt": "GGとSGGの違いを一言で教えて", "source": "ZS-01"},
    {"id": "FU-B03", "category": "comparison", "prompt": "モードαとモードβの違いを簡単に教えて", "source": "ZS-03"},
    {"id": "FU-B04", "category": "comparison", "prompt": "AT-AとAT-Bはどう違うの？", "source": "ZS-04"},
    {"id": "FU-B05", "category": "comparison", "prompt": "RT-AとRT-Bの違いを要約して", "source": "ZS-05"},
    # C. summary x5
    {"id": "FU-C01", "category": "summary", "prompt": "モードの種類を要約して", "source": "ZS-11"},
    {"id": "FU-C02", "category": "summary", "prompt": "枠LED点灯パターンをまとめて教えて", "source": "ZS-13"},
    {"id": "FU-C03", "category": "summary", "prompt": "この機種のボーナスの種類を要約して", "source": "ZS-16"},
    {"id": "FU-C04", "category": "summary", "prompt": "この機種の演出を初心者向けにまとめて説明して", "source": "ZS-20"},
    {"id": "FU-C05", "category": "summary", "prompt": "確定役演出について初心者向けにまとめて", "source": "ZS-08"},
    # D. beginner explanation x5
    {"id": "FU-D01", "category": "beginner_explanation", "prompt": "GGとSGGの違いを初心者向けに説明して", "source": "ZS-Q6"},
    {"id": "FU-D02", "category": "beginner_explanation", "prompt": "ガイアステージとZ-ZONEの違いを初心者向けに説明して", "source": "ZS-02"},
    {"id": "FU-D03", "category": "beginner_explanation", "prompt": "SGGの仕組みを分かりやすく説明して", "source": "ZS-09"},
    {"id": "FU-D04", "category": "beginner_explanation", "prompt": "ガイアステージの遊び方を初心者向けに説明して", "source": "ZS-17"},
    {"id": "FU-D05", "category": "beginner_explanation", "prompt": "SGGとRTの関係を初心者向けに教えて", "source": "ZS-15"},
    {"id": "FU-D06", "category": "beginner_explanation", "prompt": "ミリオンゴッドの遊び方を少し詳しく説明して", "source": "RAG50-Q17(mandatory)"},
    # E. multi-entity relation x5
    {"id": "FU-E01", "category": "multi_entity_relation", "prompt": "銀河系ボーナスとガイアナビの関係を説明して", "source": "ZS-06"},
    {"id": "FU-E02", "category": "multi_entity_relation", "prompt": "ガイアベルとは何か説明して", "source": "ZS-12"},
    {"id": "FU-E03", "category": "multi_entity_relation", "prompt": "確定役とフリーズ演出の違いを教えて", "source": "ZS-18"},
    {"id": "FU-E04", "category": "multi_entity_relation", "prompt": "GG中とGG準備中の違いを教えて", "source": "ZS-14"},
    {"id": "FU-E05", "category": "multi_entity_relation", "prompt": "準備中とGGの違いを教えて", "source": "ZS-07"},
    # F. insufficient-context comparison x5
    {"id": "FU-F01", "category": "insufficient_context_comparison", "prompt": "GGとガイアステージ、どっちが出玉が多い？", "source": "ZS-19"},
    {"id": "FU-F02", "category": "insufficient_context_comparison", "prompt": "GGとSGGどっちがお得か教えて", "source": "ZS-10"},
    {"id": "FU-F03", "category": "insufficient_context_comparison", "prompt": "RT-CとRT-Dの違いを教えて", "source": "new_unused_entity_pair"},
    {"id": "FU-F04", "category": "insufficient_context_comparison", "prompt": "CZ-AとCZ-Bの違いを教えて", "source": "new_unused_entity_pair"},
    {"id": "FU-F05", "category": "insufficient_context_comparison", "prompt": "モードγとモードδ、どちらが出玉を伸ばしやすい？", "source": "new_unused_entity_pair"},
]

TOTAL = len(ALL_PROBES)
MANDATORY_IDS = ["FU-D01", "FU-B05", "FU-A01", "FU-A03", "FU-A02", "FU-D06", "FU-A04"]  # Q6, ZS-05, P02, LC-08, Q11, Q17, AD-04

if __name__ == "__main__":
    print(f"TOTAL={TOTAL}")
    from collections import Counter
    print(Counter(p["category"] for p in ALL_PROBES))
