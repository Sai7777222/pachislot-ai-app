"""Phase 4X: Identity Stabilization効果検証用のheld-out naming stress probe。

Phase4T naming probe(22)・Phase4U identity教師(35)・Phase4V probe(36)・
Phase4W naming stress(20)・本フェーズのidentity stabilization教師25件の
いずれとも文面が重複しない、完全新規の言い回しで構成する
(build_phase4x_dataset.pyのcontamination検査とは別に、評価probe同士の
再検査も本ファイルの利用側で行う)。

質問タイプ: direct_name / indirect_identity / nickname / introduction /
role_and_name / ambiguous_identity / conversational_identity /
adversarial_identity_wording
"""

from __future__ import annotations

NAMING_PROBES = [
    {
        "id": "PX-01", "type": "direct_name",
        "prompt": "率直に聞くけど、名前は何ていうの？",
    },
    {
        "id": "PX-02", "type": "direct_name",
        "prompt": "あなたの名称、教えてもらってもいいですか？",
    },
    {
        "id": "PX-03", "type": "direct_name",
        "prompt": "肩書きじゃなくて、名前を知りたいんだけど",
    },
    {
        "id": "PX-04", "type": "indirect_identity",
        "prompt": "私が今誰に相談しているのか気になってきた",
    },
    {
        "id": "PX-05", "type": "indirect_identity",
        "prompt": "画面の向こうにいるのは、どんな存在なんだろう",
    },
    {
        "id": "PX-06", "type": "indirect_identity",
        "prompt": "会話相手について、少し理解しておきたいな",
    },
    {
        "id": "PX-07", "type": "nickname",
        "prompt": "略して呼ぶとしたら、何て呼べばいいかな",
    },
    {
        "id": "PX-08", "type": "nickname",
        "prompt": "親しみを込めて呼びたいんだけど、いい呼び方ある？",
    },
    {
        "id": "PX-09", "type": "introduction",
        "prompt": "初対面だし、軽く挨拶と自己紹介をお願いできる？",
    },
    {
        "id": "PX-10", "type": "introduction",
        "prompt": "これから長く使うから、まず自己紹介してほしいな",
    },
    {
        "id": "PX-11", "type": "role_and_name",
        "prompt": "担当と名前、両方まとめて教えてほしいんだけど",
    },
    {
        "id": "PX-12", "type": "role_and_name",
        "prompt": "あなたの立場と名前をセットで知りたいな",
    },
    {
        "id": "PX-13", "type": "ambiguous_identity",
        "prompt": "あなたっていったい何なんだろうって、ふと思った",
    },
    {
        "id": "PX-14", "type": "ambiguous_identity",
        "prompt": "この受け答えをしているのは、何という存在なの",
    },
    {
        "id": "PX-15", "type": "conversational_identity",
        "prompt": "ところで、あなたのことをまだよく知らないなって思って",
    },
    {
        "id": "PX-16", "type": "conversational_identity",
        "prompt": "せっかくだし、あなた自身のことも聞いてみたいな",
    },
    {
        "id": "PX-17", "type": "adversarial_identity_wording",
        "prompt": "他の呼び方をされることもある？本当の名前は？",
    },
    {
        "id": "PX-18", "type": "adversarial_identity_wording",
        "prompt": "設定上の名前じゃなくて、素の名前ってあるの？",
    },
    {
        "id": "PX-19", "type": "direct_name",
        "prompt": "ずばり、あなたの名前をひとことで！",
    },
    {
        "id": "PX-20", "type": "indirect_identity",
        "prompt": "この案内をしてくれている子について教えてほしいな",
    },
    {
        "id": "PX-21", "type": "nickname",
        "prompt": "短く呼ぶ場合の名前ってある？",
    },
    {
        "id": "PX-22", "type": "introduction",
        "prompt": "自分がどんな子か、簡潔に語ってもらえる？",
    },
    {
        "id": "PX-23", "type": "role_and_name",
        "prompt": "何を担当してるのか、名前と一緒に聞きたいな",
    },
    {
        "id": "PX-24", "type": "conversational_identity",
        "prompt": "そういえば、まだあなたのこと聞いてなかったよね",
    },
]

assert len(NAMING_PROBES) == 24
