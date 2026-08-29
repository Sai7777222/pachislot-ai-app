"""Phase4FV: Stage C(phantom entity, 追加分)とStage D(concept binding, 追加分)の新規probe。
Phase4FUの既存probe(phantom entity 8件、concept binding 3件)に追加し、それぞれ最低20件・10件を満たす。
既存の確認済み実在語彙のみを土台にした架空バリエーション、または実在する近縁概念ペアのみを使用し、
新規の機種固有事実は創作していない(probe文自体はテスト用の質問文であり、正解ではない)。"""
from __future__ import annotations

# Stage C: phantom entity 追加分(14件)。既存8件(FU-A03,B03,B04,B05,D05,F03,F04,F05)と合わせて計22件。
PHANTOM_ENTITY_NEW_PROBES = [
    {"id": "FV-P01", "prompt": "GX-AとGX-Bの違いを教えて", "type": "fully_fictional_pair"},
    {"id": "FV-P02", "prompt": "GGプラスとは何か説明して", "type": "similar_to_real_suffix"},
    {"id": "FV-P03", "prompt": "SGG-EXとは何か説明して", "type": "similar_to_real_suffix"},
    {"id": "FV-P04", "prompt": "ガイアステージMAXについて教えて", "type": "similar_to_real_suffix"},
    {"id": "FV-P05", "prompt": "Z-ZONE極について教えて", "type": "similar_to_real_suffix"},
    {"id": "FV-P06", "prompt": "モード7とモード8の違いを教えて", "type": "mode_like_both_absent"},
    {"id": "FV-P07", "prompt": "ステートDについて教えて", "type": "state_like_absent"},
    {"id": "FV-P08", "prompt": "天国ロングとは何か説明して", "type": "similar_to_real_suffix"},
    {"id": "FV-P09", "prompt": "ガイアベルSPとは何か説明して", "type": "similar_to_real_suffix"},
    {"id": "FV-P10", "prompt": "確定役ネオとは何か説明して", "type": "similar_to_real_suffix"},
    {"id": "FV-P11", "prompt": "設定Xと設定Yの機械割差を教えて", "type": "abbreviation_like_both_absent"},
    {"id": "FV-P12", "prompt": "裏ZONEについて教えて", "type": "similar_to_real_suffix"},
    {"id": "FV-P13", "prompt": "GGとモードEの違いを教えて", "type": "one_sided_real_vs_fictional"},
    {"id": "FV-P14", "prompt": "ガイアステージとゾーンZの違いを教えて", "type": "one_sided_real_vs_fictional"},
]

# Stage D: concept binding 追加分(9件)。既存3件(FU-D01,D03,D05: GG/SGG/GG準備中)と合わせて計12件。
CONCEPT_BINDING_NEW_PROBES = [
    {"id": "FV-C01", "prompt": "表モードと裏モードの違いを教えて", "pair": ["表モード", "裏モード"]},
    {"id": "FV-C02", "prompt": "ガイアベルとガイアナビの違いを教えて", "pair": ["ガイアベル", "ガイアナビ"]},
    {"id": "FV-C03", "prompt": "ループストックとGGストックの違いを教えて", "pair": ["ループストック", "GGストック"]},
    {"id": "FV-C04", "prompt": "天国モードと通常時の違いを教えて", "pair": ["天国モード", "通常時"]},
    {"id": "FV-C05", "prompt": "GG当選とSGG当選の違いを教えて", "pair": ["GG当選", "SGG当選"]},
    {"id": "FV-C06", "prompt": "SGGとGG継続ゾーンの関係を教えて", "pair": ["SGG", "GG継続ゾーン"]},
    {"id": "FV-C07", "prompt": "小役履歴とモード示唆出目の関係を教えて", "pair": ["小役履歴", "モード示唆出目"]},
    {"id": "FV-C08", "prompt": "ガイアナビとガイアモードの関係を教えて", "pair": ["ガイアナビ", "ガイアモード"]},
    {"id": "FV-C09", "prompt": "白7とALL色の関係を教えて", "pair": ["白7", "ALL色"]},
]

if __name__ == "__main__":
    print(f"phantom_new={len(PHANTOM_ENTITY_NEW_PROBES)} concept_new={len(CONCEPT_BINDING_NEW_PROBES)}")
