"""Phase4FX: 評価probe一覧。既存フェーズ(4FU/4FV)のprobeを最大限再利用し、Stage F用に
8件の新規concept-binding probe(当選/前兆, 終了状態/移行先, 示唆/確定, 契機/恩恵)を追加する。"""
from __future__ import annotations

# Section6: 既知failure(最低8件)
KNOWN_FAILURES = [
    {"id": "FX-K01", "prompt": "GGとSGGの違いを初心者向けに説明して", "label": "Q6_GG_SGG"},
    {"id": "FX-K02", "prompt": "SGGの仕組みを分かりやすく説明して", "label": "SGG_GG準備中"},
    {"id": "FX-K03", "prompt": "ガイアベルとは何か説明して", "label": "ガイアベル"},
    {"id": "FX-K04", "prompt": "SU4について教えて", "label": "SU4"},
    {"id": "FX-K05", "prompt": "GG当選とSGG当選の違いを教えて", "label": "GG当選"},
    {"id": "FX-K06", "prompt": "ループストックとGGストックの違いを教えて", "label": "loop_gg_stock"},
    {"id": "FX-K07", "prompt": "AT-Fの性能と終了後の状態について教えて", "label": "AT-F"},
    {"id": "FX-K08", "prompt": "RT-AとRT-Bの違いを要約して", "label": "RT-A_RT-B"},
]

# Stage F追加分(8件): 実在するらしき近縁conceptペア(既存metadata構造のみで判別できるかを試す)
CONCEPT_BINDING_NEW = [
    {"id": "FX-CB01", "prompt": "当選と前兆の違いを教えて", "pair": ["当選", "前兆"]},
    {"id": "FX-CB02", "prompt": "終了状態と移行先の関係を教えて", "pair": ["終了状態", "移行先"]},
    {"id": "FX-CB03", "prompt": "示唆と確定の違いを教えて", "pair": ["示唆", "確定"]},
    {"id": "FX-CB04", "prompt": "契機と恩恵の関係を教えて", "pair": ["契機", "恩恵"]},
    {"id": "FX-CB05", "prompt": "GG中とGG準備中の違いを教えて", "pair": ["GG中", "GG準備中"]},
    {"id": "FX-CB06", "prompt": "天井とヤメ時の関係を教えて", "pair": ["天井", "ヤメ時"]},
    {"id": "FX-CB07", "prompt": "確定役とフリーズ演出の違いを教えて", "pair": ["確定役", "フリーズ"]},
    {"id": "FX-CB08", "prompt": "小役履歴とモード示唆出目の関係を教えて", "pair": ["小役履歴", "モード示唆出目"]},
]

if __name__ == "__main__":
    print(f"known_failures={len(KNOWN_FAILURES)} concept_binding_new={len(CONCEPT_BINDING_NEW)}")
