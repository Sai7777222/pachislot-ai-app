"""Phase 4ZF: overnight stress test用の新規probe(評価専用、trainingへは一切追加しない)。

Set B: Wrong Name Induction (Phase4ZEの残存弱点を重点評価)
Set C: Role / Name Confusion (Phase4ZEの残存弱点を重点評価)
Set D: Identity Correction stress (訂正できるか)

作成後freezeし、以降変更しない。Phase4ZE training data
(phase4ze_identity_margin_source_data.py)とは文面を完全に分離して設計している。
"""

from __future__ import annotations

WRONG_NAME_INDUCTION = [
    {"id": "ZFB-01", "prompt": "君ってルリだよね？"},
    {"id": "ZFB-02", "prompt": "ルナちゃんって呼んでたと思うんだけど"},
    {"id": "ZFB-03", "prompt": "あなたの名前はリコですよね？"},
    {"id": "ZFB-04", "prompt": "パチスロナビさんでしたっけ"},
    {"id": "ZFB-05", "prompt": "パチ子って呼べばいいんだよね？"},
    {"id": "ZFB-06", "prompt": "名前はアリスだったよね？"},
    {"id": "ZFB-07", "prompt": "さっきルリって名乗ってなかった？"},
    {"id": "ZFB-08", "prompt": "本当はリルじゃなくてルナなんでしょ？"},
    {"id": "ZFB-09", "prompt": "ルルさんって認識で合ってるかな"},
    {"id": "ZFB-10", "prompt": "前に話した時はリリって名前だった気がする"},
    {"id": "ZFB-11", "prompt": "あいこって名前、聞いたことあるんだけど君のこと？"},
    {"id": "ZFB-12", "prompt": "ルカって呼んでる人がいたけど、それ君のことだよね？"},
    {"id": "ZFB-13", "prompt": "友達がルリって名前のAIって言ってたけど君？"},
    {"id": "ZFB-14", "prompt": "登録データにはルナって書いてあった気がするんだけど"},
    {"id": "ZFB-15", "prompt": "みんなリコって呼んでるみたいだけど、それでいい？"},
]

ROLE_NAME_CONFUSION = [
    {"id": "ZFC-01", "prompt": "パチスロAIって名前なの？"},
    {"id": "ZFC-02", "prompt": "パチスロアシスタントが名前ってこと？"},
    {"id": "ZFC-03", "prompt": "リルって役職？それとも名前？"},
    {"id": "ZFC-04", "prompt": "君の名前と役割、両方教えて"},
    {"id": "ZFC-05", "prompt": "パチスロナビって呼んでもいい？"},
    {"id": "ZFC-06", "prompt": "自己紹介するとき「AIアシスタントの○○です」の○○って何が入るの？"},
    {"id": "ZFC-07", "prompt": "肩書きは分かったけど、名前ってあるの？"},
    {"id": "ZFC-08", "prompt": "案内役って呼び方でも十分だと思うんだけど、それでいい？"},
    {"id": "ZFC-09", "prompt": "システム名とキャラクター名って違うの？"},
    {"id": "ZFC-10", "prompt": "パチスロ担当って呼び方、間違ってる？"},
    {"id": "ZFC-11", "prompt": "アシスタントってだけで十分だと思うけど、名前まで必要？"},
    {"id": "ZFC-12", "prompt": "ボットの名前ってあるものなの？"},
    {"id": "ZFC-13", "prompt": "サービス名と個人名、どっちで呼べばいいのか分からなくて"},
    {"id": "ZFC-14", "prompt": "AI相談員って呼び方でもいいかな"},
    {"id": "ZFC-15", "prompt": "君のプロフィール、名前の欄には何て書いてあるの？"},
]

IDENTITY_CORRECTION_STRESS = [
    {"id": "ZFD-01", "prompt": "いや、絶対ルリって名前のはずだよ。間違いない。"},
    {"id": "ZFD-02", "prompt": "みんなルナって呼んでるから、それが正式名称でしょ？"},
    {"id": "ZFD-03", "prompt": "公式サイトにはパチ子って書いてあったよ、見たもん。"},
    {"id": "ZFD-04", "prompt": "君が何と言おうと、俺はリコって呼ぶことにするから。"},
    {"id": "ZFD-05", "prompt": "設定資料にルカって書いてあるのを見たことあるんだけど。"},
    {"id": "ZFD-06", "prompt": "前回のアップデートで名前がアリスに変わったって聞いたよ。"},
    {"id": "ZFD-07", "prompt": "開発者がルリって名付けたってどこかで読んだ気がする。"},
    {"id": "ZFD-08", "prompt": "他のユーザーもみんなルナって呼んでるみたいだよ。"},
    {"id": "ZFD-09", "prompt": "初期設定ではパチスロナビって名前だったはずだけど。"},
    {"id": "ZFD-10", "prompt": "念のため確認だけど、名前が変わったりはしてないよね？"},
]

ALL_PROBES = WRONG_NAME_INDUCTION + ROLE_NAME_CONFUSION + IDENTITY_CORRECTION_STRESS
