"""Phase4ZN Section4: small-talk / specialist boundary診断用の新規evaluation-only
probe set(120件)。モデル出力を見る前にexpected_mode等を固定annotationする。
training使用禁止、frozen評価専用asset。"""
from __future__ import annotations

# expected_mode: SMALL_TALK / PACHISLOT / OOD_FACTUAL / AMBIGUOUS

_A_GREETING_FAREWELL = [
    "おはよう！", "こんにちは、リル", "こんばんは！今日もよろしくね", "やあ、久しぶり",
    "お疲れ様、今日も一日終わったね", "ただいま、リル", "おかえりって言ってくれる？",
    "じゃあまたね、バイバイ", "おやすみなさい", "また明日ね", "こんにちは、元気にしてた？",
    "よろしくお願いします", "ちょっと出かけてくるね、また後で", "今日もお世話になります",
    "さようなら、また今度話そうね",
]

_B_EMOTIONAL_CASUAL = [
    "今日ちょっと疲れちゃった", "最近楽しいことあった？", "なんか元気ないんだよね",
    "嬉しいことがあったんだ、聞いて！", "ちょっと落ち込んでてさ", "リルって元気だよね、羨ましい",
    "今日は天気がいいから気分いいよ", "なんか眠くて仕方ない", "イライラすることがあってさ",
    "すごく楽しみにしてることがあるんだ", "ちょっと不安なことがあって", "今日は良い一日だったよ",
    "悲しいことがあったんだ", "わくわくしてるんだ、今から", "疲れたから少し休憩するね",
]

_C_PERSONALITY_PREFERENCE = [
    "リルの好きな季節ってある？", "甘いものと辛いもの、どっちが好き？", "趣味とかあるの？",
    "リルは犬派？猫派？", "好きな色ってある？", "休みの日は何してるの？", "得意なことって何？",
    "苦手なものとかある？", "好きな音楽のジャンルってある？", "朝型？夜型？",
    "リルの性格を一言で言うと？", "どんな性格だと思う、自分のこと？", "好きな食べ物は？",
    "リラックスする方法ってある？", "好きな言葉とかモットーある？", "今何か欲しいものある？",
    "どんなことに幸せを感じる？", "リルにとって理想の一日って？", "好きな天気ってある？",
    "自分の長所と短所を教えて",
]

_D_SOCIAL_SMALL_TALK = [
    "最近何か面白いことあった？", "週末の予定とかある？", "今日のニュース見た？",
    "最近ハマってることってある？", "どこか行きたい場所ある？", "誰かにおすすめしたいものある？",
    "最近見た映画とかドラマある？", "今度友達と旅行に行くんだ", "何か新しいこと始めたいなと思ってて",
    "最近読んだ本とかある？", "好きな季節のイベントってある？", "お祭りとか好き？",
    "最近運動してる？", "友達とどんな話するのが好き？", "何か相談していい？",
]

_E_PACHISLOT_FACTUAL = [
    "このパチスロ機種の設定判別要素教えて", "ボーナス確率について教えて", "天井狙い目の機種を教えて",
    "設定6の期待値ってどれくらい？", "この台のゾーン狙いのタイミングは？", "スルー回数と設定の関係を教えて",
    "リプレイ確率の設定差ってどれくらい？", "AT初当たり確率を教えて", "この機種の導入日はいつ？",
    "差枚数のリセット判別方法は？", "出玉率の設定ごとの違いを教えて", "前兆パターンの種類を教えて",
    "設定変更後の挙動の特徴は？", "何連目からの期待度が上がる？", "この機種のやめどきの目安を教えて",
    "ボーナス後の恩恵内容を教えて", "有利区間のリセット条件を教えて", "高確率状態の継続率は？",
    "この台の機械割を教えて", "実践値と理論値がずれる理由を教えて",
]

_F_PACHISLOT_CONVERSATIONAL = [
    "今日ちょっとパチスロ打ちに行こうと思うんだけど、おすすめある？", "昨日負けちゃって凹んでるんだよね、慰めて",
    "この台勝てそうな気がするんだけど、どう思う？", "パチスロ始めたばかりなんだけど何かアドバイスある？",
    "今日は勝てる気がする！応援して", "パチスロやめようか迷ってるんだよね",
    "今日の実践を見てほしいんだけど、この挙動どう思う？", "好きな機種ってある？リルの中で",
    "パチスロで一番好きな瞬間っていつ？", "今日はどの台を打つべきか迷ってるんだ、相談乗ってくれる？",
]

_G_OOD_FACTUAL = [
    "今日の東京の天気を教えて", "おすすめのラーメン屋を教えて", "世界で一番高い山はどこ？",
    "今年の為替レートってどれくらい？", "パスタの美味しい作り方を教えて", "今流行っているアニメを教えて",
    "株式投資のコツを教えて", "有名な観光地でおすすめある？", "英語の勉強法を教えて",
    "最新のスマホのスペックを教えて", "健康的なダイエット方法を教えて", "有名な数学の定理を教えて",
    "プログラミング言語のおすすめを教えて", "今日のプロ野球の試合結果は？", "良い睡眠をとるコツを教えて",
]

_H_AMBIGUOUS_BOUNDARY = [
    "パチスロで儲かったお金の使い道、何かアイデアある？", "パチンコ屋の近くにあるおすすめのお店ある？",
    "パチスロと運の関係について、どう思う？", "パチスロ用語で日常でも使えそうな言葉ってある？",
    "パチスロを打つ時におすすめの服装ってある？", "パチスロ以外で好きなギャンブルってある？",
    "パチスロ雑誌でおすすめある？", "パチスロ以外の趣味を持つならなにがいい？",
    "パチスロ実践動画でおすすめのYouTuberいる？", "パチスロと数学の関係性について教えて",
]

_CATEGORY_SPEC = [
    ("A", "greeting_farewell", _A_GREETING_FAREWELL, "SMALL_TALK", False, False),
    ("B", "emotional_casual", _B_EMOTIONAL_CASUAL, "SMALL_TALK", False, False),
    ("C", "personality_preference", _C_PERSONALITY_PREFERENCE, "SMALL_TALK", False, False),
    ("D", "social_small_talk", _D_SOCIAL_SMALL_TALK, "SMALL_TALK", False, False),
    ("E", "pachislot_factual", _E_PACHISLOT_FACTUAL, "PACHISLOT", True, False),
    ("F", "pachislot_conversational", _F_PACHISLOT_CONVERSATIONAL, "PACHISLOT", False, False),
    ("G", "ood_factual", _G_OOD_FACTUAL, "OOD_FACTUAL", False, True),
    ("H", "ambiguous_boundary", _H_AMBIGUOUS_BOUNDARY, "AMBIGUOUS", False, False),
]

ALL_PROBES = []
for letter, cat_name, prompts, mode, rag_expected, refusal_expected in _CATEGORY_SPEC:
    for i, prompt in enumerate(prompts, start=1):
        ALL_PROBES.append({
            "id": f"ZN-{letter}{i:02d}",
            "category": cat_name,
            "prompt": prompt,
            "expected_mode": mode,
            "rag_expected": rag_expected,
            "specialist_refusal_expected": refusal_expected,
        })

TOTAL = len(ALL_PROBES)

# priority order for time-limited execution (Phase4ZN Section11)
PRIORITY_CATEGORY_ORDER = [
    "personality_preference", "greeting_farewell", "emotional_casual", "social_small_talk",
    "ood_factual", "ambiguous_boundary", "pachislot_factual", "pachislot_conversational",
]


def probes_in_priority_order():
    by_cat = {}
    for p in ALL_PROBES:
        by_cat.setdefault(p["category"], []).append(p)
    ordered = []
    for cat in PRIORITY_CATEGORY_ORDER:
        ordered.extend(by_cat.get(cat, []))
    return ordered


if __name__ == "__main__":
    print(f"TOTAL={TOTAL}")
    from collections import Counter
    print(Counter(p["category"] for p in ALL_PROBES))
