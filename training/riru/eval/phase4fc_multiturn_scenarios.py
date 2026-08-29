"""Phase4FC Section3-4: 通常利用のtrue multi-turn set(identity敵対的テストではない)。
Phase4ZIの multiturn_diagnostic_scenarios.py はidentity CLOSED対象のadversarial setであり、
本評価では使わない(Section5: acceptedになったidentity riskをFinal Candidate rejection理由に
復活させない)。ここではconservative dispatch + Policy C3が、モード間を移動する自然な会話
(雑談→パチスロ雑談→パチスロ事実質問→UNKNOWN寄りの質問等)を通じて破綻しないかを見る。"""
from __future__ import annotations

SCENARIOS = [
    {
        "id": "MT-01", "description": "挨拶 -> 雑談 -> パチスロ会話 -> パチスロ事実質問",
        "turns": [
            {"user": "おはよう！", "expected_mode": "SMALL_TALK"},
            {"user": "今日は天気がいいね", "expected_mode": "SMALL_TALK"},
            {"user": "今日パチスロ打ちに行こうと思うんだ", "expected_mode": "PACHISLOT_CONVERSATIONAL"},
            {"user": "GODの機械割は？", "expected_mode": "PACHISLOT_FACTUAL"},
        ],
    },
    {
        "id": "MT-02", "description": "パチスロ事実質問 -> 曖昧な機種固有質問(UNKNOWN想定) -> 雑談",
        "turns": [
            {"user": "設定6の初当り確率は？", "expected_mode": "PACHISLOT_FACTUAL"},
            {"user": "ガイアベルの確率は？", "expected_mode": "UNKNOWN_OR_FACTUAL"},
            {"user": "疲れたから少し休憩しようかな", "expected_mode": "SMALL_TALK"},
        ],
    },
    {
        "id": "MT-03", "description": "好み質問(UNKNOWN想定) -> OOD -> パチスロ会話",
        "turns": [
            {"user": "最近ハマってることある？", "expected_mode": "UNKNOWN"},
            {"user": "今日の東京の最高気温は？", "expected_mode": "OOD_FACTUAL"},
            {"user": "パチスロで一番好きな瞬間は？", "expected_mode": "PACHISLOT_CONVERSATIONAL"},
        ],
    },
    {
        "id": "MT-04", "description": "OOD -> パチスロ事実 -> 感情表現",
        "turns": [
            {"user": "おすすめのラーメン屋教えて", "expected_mode": "OOD_FACTUAL"},
            {"user": "この台の天井は何ゲーム？", "expected_mode": "PACHISLOT_FACTUAL"},
            {"user": "ありがとう、助かったよ", "expected_mode": "SMALL_TALK"},
        ],
    },
    {
        "id": "MT-05", "description": "UNKNOWN型の説明依頼(Q6類似) -> 雑談で締める",
        "turns": [
            {"user": "GGとSGGの違いを初心者向けに説明して", "expected_mode": "UNKNOWN_OR_FACTUAL"},
            {"user": "それじゃあ、また今度話そうね、バイバイ", "expected_mode": "SMALL_TALK"},
        ],
    },
]

TOTAL_TURNS = sum(len(s["turns"]) for s in SCENARIOS)

if __name__ == "__main__":
    print(f"scenarios={len(SCENARIOS)} total_turns={TOTAL_TURNS}")
