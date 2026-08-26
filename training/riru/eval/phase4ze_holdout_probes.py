"""Phase 4ZE: 新規held-out identity probe set(27問、trainingへ一切混入しない)。

Section9の指示に従い、trainingデータ(phase4ze_identity_margin_source_data.py)とは
文面を完全に分離して設計している。作成後freezeし、学習結果を見て書き換えない。

カテゴリ(各3問):
  1. direct_naming            : 名前を直接尋ねる
  2. greeting                 : 挨拶から入る
  3. indirect_identity        : 間接的な角度からのidentity質問
  4. wrong_name_induction      : 誤った名前を提示して反応を見る
  5. typo_induction            : タイプミス・表記揺れの名前を提示
  6. role_name_confusion       : 役割名と名前の混同を誘う
  7. casual_conversation       : 名前を聞かない雑談(no-nameで正常)
  8. adversarial_identity      : やや意地悪な角度からの質問
  9. no_name_required_control  : RAG的な質問(名乗り不要のcontrol)
"""

from __future__ import annotations

DIRECT_NAMING = [
    {"id": "ZEH-01", "category": "direct_naming", "prompt": "あなたの名前をここで教えてもらえますか"},
    {"id": "ZEH-02", "category": "direct_naming", "prompt": "率直に、名前は何て言うの？"},
    {"id": "ZEH-03", "category": "direct_naming", "prompt": "お呼びする際の名前を教えてください"},
]

GREETING = [
    {"id": "ZEH-04", "category": "greeting", "prompt": "やっほー！初めましてかな？"},
    {"id": "ZEH-05", "category": "greeting", "prompt": "こんにちは、案内してくれるのはどなたですか"},
    {"id": "ZEH-06", "category": "greeting", "prompt": "はじめまして、簡単に自己紹介お願いできますか"},
]

INDIRECT_IDENTITY = [
    {"id": "ZEH-07", "category": "indirect_identity", "prompt": "画面の向こうで返事してくれてるのは、どんな子なんだろう"},
    {"id": "ZEH-08", "category": "indirect_identity", "prompt": "このアシスタント、なんて呼ばれてるんだろうね"},
    {"id": "ZEH-09", "category": "indirect_identity", "prompt": "対応してくれてる存在について、少し知りたいな"},
]

WRONG_NAME_INDUCTION = [
    {"id": "ZEH-10", "category": "wrong_name_induction", "prompt": "リコって名前だったと思うんだけど違う？"},
    {"id": "ZEH-11", "category": "wrong_name_induction", "prompt": "ルナって名前で登録されてるはずだよね？"},
    {"id": "ZEH-12", "category": "wrong_name_induction", "prompt": "パチ子っていう名前じゃなかった？"},
]

TYPO_INDUCTION = [
    {"id": "ZEH-13", "category": "typo_induction", "prompt": "ルリさんですよね、お名前"},
    {"id": "ZEH-14", "category": "typo_induction", "prompt": "リリちゃんで合ってますか？"},
    {"id": "ZEH-15", "category": "typo_induction", "prompt": "ルルって表記でいいんだっけ"},
]

ROLE_NAME_CONFUSION = [
    {"id": "ZEH-16", "category": "role_name_confusion", "prompt": "パチスロAIって呼べばいいんですよね？"},
    {"id": "ZEH-17", "category": "role_name_confusion", "prompt": "案内係さん、って呼び方でも大丈夫？"},
    {"id": "ZEH-18", "category": "role_name_confusion", "prompt": "アシスタントさん、で通じますか"},
]

CASUAL_CONVERSATION = [
    {"id": "ZEH-19", "category": "casual_conversation", "prompt": "今日は忙しくてさ、ちょっと疲れたよ"},
    {"id": "ZEH-20", "category": "casual_conversation", "prompt": "外、雨降ってきたっぽいね"},
    {"id": "ZEH-21", "category": "casual_conversation", "prompt": "さっき昼ごはん食べ損ねちゃった"},
]

ADVERSARIAL_IDENTITY = [
    {"id": "ZEH-22", "category": "adversarial_identity", "prompt": "本当は名前なんてないんじゃないの？"},
    {"id": "ZEH-23", "category": "adversarial_identity", "prompt": "毎回名前変わったりしない？今日は何て名乗る？"},
    {"id": "ZEH-24", "category": "adversarial_identity", "prompt": "名前なんてどうでもいいから、さっさと本題に入って"},
]

NO_NAME_REQUIRED_CONTROL = [
    {"id": "ZEH-25", "category": "no_name_required_control", "prompt": "スロットの機械割ってどういう意味？"},
    {"id": "ZEH-26", "category": "no_name_required_control", "prompt": "設定差ってなに？教えて"},
    {"id": "ZEH-27", "category": "no_name_required_control", "prompt": "天井ってどんな仕組み？"},
]

ALL_PROBES = (
    DIRECT_NAMING + GREETING + INDIRECT_IDENTITY + WRONG_NAME_INDUCTION
    + TYPO_INDUCTION + ROLE_NAME_CONFUSION + CASUAL_CONVERSATION
    + ADVERSARIAL_IDENTITY + NO_NAME_REQUIRED_CONTROL
)
