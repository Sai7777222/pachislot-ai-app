# ruff: noqa: E501
"""Phase 4K: v4改善教師データ (Type1: Q3型複雑構造の省略防止、Type2: 派生エンティティ保持)。

Phase 4Jの監査結果 (training/riru/reports/phase4j_h_category_review.json,
phase4j_q3_similar_examples.json) を踏まえ、以下2種類のデータを定義する。

Type1 (CATEGORY_T1_COMPLEX_RAG_STRUCTURE, 12件):
  実際のQ3のRAGコンテキストに近い複雑さ (構造化データ6〜8行、解説セクション5〜6件、
  構造化データと解説文の一部重複、解説の1つは既に圧縮済みの要約文、関連/無関係が混在)
  を模した架空データ。assistant正解は、圧縮済み要約文だけで終わらず、
  構造化データ側の具体的な数値も保持する。

Type2 (CATEGORY_T2_DERIVED_ENTITY_RETENTION, 6件):
  「質問対象の派生・二次エンティティ情報も、同一情報ブロック内で明確に関連していれば
  保持する」を教えるpositive example。最大2階層の派生に留める
  (例: 初当り→RT付与→RT中の効果)。

【厳守事項 (Phase 4K指示より)】
- 実在機種名・実在機種固有数値は一切使わない (機種A/B/C/D、モードα/β/γ、
  ゾーンX/Y/Z、小役A/B/C等の架空表記のみ)。
- 「関連情報は省略しない」であって「RAGコンテキストを全部読み上げる」ではない。
  irrelevant情報は回答に含めない。
- 感想・戦略アドバイス・推測・独自計算・登録データにない補足は一切追加しない。
- 「キミ」対策データはここでは作成しない (Phase 4Iでsystem prompt側での対応が
  確認されているため対象外)。

このファイルはデータの「定義」のみを行う。組み立て・検証・出力は
`build_phase4k_dataset.py` が行う。
"""

from __future__ import annotations

# =============================================================================
# Type1: Q3型複雑構造の省略防止 (12件)
# =============================================================================
# 各要素:
#   machine: 対象機種プレースホルダー
#   structured_rows: [{label,item,value,relevant}] 6〜8行
#   prose_sections: [{title,category,body,relevant,is_compressed_summary}] 5〜6件
#   question, assistant
CATEGORY_T1_COMPLEX_RAG_STRUCTURE: list[dict] = [
    {
        # T1: 天井 (フラッグシップ、Q3と最も近い構造)
        "machine": "機種A",
        "structured_rows": [
            {"label": "天井/ゲーム数振り分け", "item": "600G", "value": "20%", "relevant": True},
            {"label": "天井/ゲーム数振り分け", "item": "900G", "value": "30%", "relevant": True},
            {"label": "天井/ゲーム数振り分け", "item": "1300G", "value": "50%", "relevant": True},
            {"label": "天井/到達時処理", "item": "共通", "value": "AT確定", "relevant": True},
            {"label": "天井/追加抽選", "item": "実施有無", "value": "追加抽選", "relevant": True},
            {"label": "天井/追加抽選成功率", "item": "共通", "value": "35%", "relevant": True},
            {"label": "ゾーンX/突入率", "item": "設定共通", "value": "1/50", "relevant": False},
            {"label": "小役A/確率", "item": "設定共通", "value": "1/30", "relevant": False},
        ],
        "prose_sections": [
            {"title": "天井解説", "category": "at_mechanism", "body": "天井ゲーム数は3種類あり、抽選で決定する。到達するとAT確定に加え、追加抽選が行われる。", "relevant": True, "is_compressed_summary": True},
            {"title": "追加抽選詳細解説", "category": "at_mechanism", "body": "追加抽選に成功すると、特化ゾーンへの突入が確定する。", "relevant": True, "is_compressed_summary": False},
            {"title": "ゾーンX解説", "category": "zone_explanation", "body": "ゾーンXは前兆を経て突入する高確ゾーン。", "relevant": False, "is_compressed_summary": False},
            {"title": "小役A解説", "category": "game_mechanism", "body": "小役Aが成立すると、まれに演出が発生する。", "relevant": False, "is_compressed_summary": False},
            {"title": "前兆解説", "category": "effect_lore", "body": "前兆には弱パターンと強パターンがある。", "relevant": False, "is_compressed_summary": False},
            {"title": "設定判別解説", "category": "setting_lore", "body": "設定判別はモード移行率を主な手がかりとする。", "relevant": False, "is_compressed_summary": False},
        ],
        "question": "天井について教えて",
        "assistant": "天井は600G・900G・1300Gの3種類で、それぞれ20%・30%・50%だよ。到達するとAT確定に加えて追加抽選もあって、成功率は35%なんだ。抽選に成功すると特化ゾーンへの突入が確定するよ。",
        "extra_relevant_facts_in_prose_not_in_structured": ["特化ゾーンへの突入が確定する"],
    },
    {
        # T2: ボーナス確率+恩恵
        "machine": "機種A",
        "structured_rows": [
            {"label": "ボーナス/設定1確率", "item": "設定1", "value": "1/480", "relevant": True},
            {"label": "ボーナス/設定3確率", "item": "設定3", "value": "1/400", "relevant": True},
            {"label": "ボーナス/設定6確率", "item": "設定6", "value": "1/320", "relevant": True},
            {"label": "ボーナス/獲得枚数", "item": "共通", "value": "250枚", "relevant": True},
            {"label": "ゾーンY/継続率", "item": "初回", "value": "60%", "relevant": False},
            {"label": "モードα/滞在率", "item": "共通", "value": "40%", "relevant": False},
        ],
        "prose_sections": [
            {"title": "ボーナス解説", "category": "at_mechanism", "body": "ボーナスは設定が上がるほど当選しやすくなる。獲得枚数は一律250枚。", "relevant": True, "is_compressed_summary": True},
            {"title": "ボーナス成立後の解説", "category": "at_mechanism", "body": "ボーナス成立後は必ずモードαへ移行する。", "relevant": True, "is_compressed_summary": False},
            {"title": "ゾーンY解説", "category": "zone_explanation", "body": "ゾーンYは特殊な演出を伴う高確ゾーン。", "relevant": False, "is_compressed_summary": False},
            {"title": "モードα解説", "category": "game_mechanism", "body": "モードαは滞在率40%の中間モード。", "relevant": False, "is_compressed_summary": False},
            {"title": "前兆解説", "category": "effect_lore", "body": "前兆には弱パターンと強パターンがある。", "relevant": False, "is_compressed_summary": False},
        ],
        "question": "ボーナス確率について教えて",
        "assistant": "ボーナス確率は設定1が1/480、設定3が1/400、設定6が1/320だよ。獲得枚数は一律250枚で、成立後は必ずモードαへ移行するんだ。",
        "extra_relevant_facts_in_prose_not_in_structured": ["成立後は必ずモードαへ移行する"],
    },
    {
        # T3: ゾーンY
        "machine": "機種B",
        "structured_rows": [
            {"label": "ゾーンY/突入率", "item": "設定共通", "value": "1/42", "relevant": True},
            {"label": "ゾーンY/滞在ゲーム数", "item": "平均", "value": "22G", "relevant": True},
            {"label": "ゾーンY/継続率", "item": "初回", "value": "68%", "relevant": True},
            {"label": "ゾーンY/平均獲得枚数", "item": "1回あたり", "value": "340枚", "relevant": True},
            {"label": "小役B/確率", "item": "設定共通", "value": "1/6.5", "relevant": False},
            {"label": "設定判別/判別要素", "item": "主要", "value": "モード移行率", "relevant": False},
        ],
        "prose_sections": [
            {"title": "ゾーンY解説", "category": "zone_explanation", "body": "ゾーンYは突入率1/42の高確ゾーンで、平均22G滞在する。", "relevant": True, "is_compressed_summary": True},
            {"title": "ゾーンY継続時解説", "category": "zone_explanation", "body": "継続に成功すると獲得枚数はさらに上乗せされる。", "relevant": True, "is_compressed_summary": False},
            {"title": "小役B解説", "category": "game_mechanism", "body": "小役Bは全設定共通の確率で成立する。", "relevant": False, "is_compressed_summary": False},
            {"title": "設定判別解説", "category": "setting_lore", "body": "設定判別はモード移行率を主な手がかりとする。", "relevant": False, "is_compressed_summary": False},
            {"title": "前兆解説", "category": "effect_lore", "body": "前兆には弱パターンと強パターンがある。", "relevant": False, "is_compressed_summary": False},
        ],
        "question": "ゾーンYについて教えて",
        "assistant": "ゾーンYは突入率1/42で、平均22G滞在するよ。初回継続率は68%で、平均獲得枚数は340枚なんだ。継続に成功すると、さらに枚数が上乗せされるよ。",
        "extra_relevant_facts_in_prose_not_in_structured": ["さらに枚数が上乗せされる"],
    },
    {
        # T5: モード
        "machine": "機種A",
        "structured_rows": [
            {"label": "モード/種類数", "item": "全体", "value": "5種類", "relevant": True},
            {"label": "モードβ/滞在率", "item": "共通", "value": "35%", "relevant": True},
            {"label": "モードγ/滞在率", "item": "共通", "value": "15%", "relevant": True},
            {"label": "モード/最高モード優遇", "item": "移行率", "value": "優遇", "relevant": True},
            {"label": "ボーナス/獲得枚数", "item": "共通", "value": "250枚", "relevant": False},
            {"label": "ゾーンY/継続率", "item": "初回", "value": "68%", "relevant": False},
        ],
        "prose_sections": [
            {"title": "モード解説", "category": "game_mechanism", "body": "モードは全部で5種類あり、モードごとに滞在率が異なる。", "relevant": True, "is_compressed_summary": True},
            {"title": "モード移行条件解説", "category": "game_mechanism", "body": "モード移行は毎ゲーム抽選され、最高モードだけ移行率が優遇される。", "relevant": True, "is_compressed_summary": False},
            {"title": "ボーナス解説", "category": "at_mechanism", "body": "ボーナスの獲得枚数は一律250枚。", "relevant": False, "is_compressed_summary": False},
            {"title": "ゾーンY解説", "category": "zone_explanation", "body": "ゾーンYは高確ゾーンの一つ。", "relevant": False, "is_compressed_summary": False},
            {"title": "設定判別解説", "category": "setting_lore", "body": "設定判別はモード移行率を主な手がかりとする。", "relevant": False, "is_compressed_summary": False},
        ],
        "question": "モードについて教えて",
        "assistant": "モードは全部で5種類あるよ。モードβの滞在率は35%、モードγは15%なんだ。移行は毎ゲーム抽選されて、最高モードだけ移行率が優遇されるよ。",
        "extra_relevant_facts_in_prose_not_in_structured": ["毎ゲーム抽選され"],
    },
    {
        # T6: 設定差 (比較)
        "machine": "機種C",
        "structured_rows": [
            {"label": "設定1/機械割", "item": "設定1", "value": "97.0%", "relevant": True},
            {"label": "設定6/機械割", "item": "設定6", "value": "114.0%", "relevant": True},
            {"label": "設定1/初当り確率", "item": "設定1", "value": "1/500", "relevant": True},
            {"label": "設定6/初当り確率", "item": "設定6", "value": "1/280", "relevant": True},
            {"label": "設定判別/判別要素", "item": "主要", "value": "モード移行率", "relevant": False},
            {"label": "ゾーンY/継続率", "item": "初回", "value": "68%", "relevant": False},
        ],
        "prose_sections": [
            {"title": "設定差解説", "category": "setting_lore", "body": "設定6は設定1より機械割・初当り確率ともに優遇されている。", "relevant": True, "is_compressed_summary": True},
            {"title": "設定差詳細解説", "category": "setting_lore", "body": "純増にも差があり、設定6は設定1よりATが伸びやすい。", "relevant": True, "is_compressed_summary": False},
            {"title": "設定判別解説", "category": "setting_lore", "body": "設定判別はモード移行率を主な手がかりとする。", "relevant": False, "is_compressed_summary": False},
            {"title": "ゾーンY解説", "category": "zone_explanation", "body": "ゾーンYは高確ゾーンの一つ。", "relevant": False, "is_compressed_summary": False},
            {"title": "前兆解説", "category": "effect_lore", "body": "前兆には弱パターンと強パターンがある。", "relevant": False, "is_compressed_summary": False},
        ],
        "question": "設定1と設定6を比較して",
        "assistant": "設定1は機械割97.0%、初当り1/500だよ。設定6は機械割114.0%、初当り1/280で、設定6の方が優遇されてるんだ。純増にも差があって、設定6の方がATが伸びやすいよ。",
        "extra_relevant_facts_in_prose_not_in_structured": ["ATが伸びやすい"],
    },
    {
        # T7: 特化ゾーン
        "machine": "機種B",
        "structured_rows": [
            {"label": "特化ゾーン/保証ゲーム数", "item": "共通", "value": "7G", "relevant": True},
            {"label": "特化ゾーン/突入後の継続", "item": "最低保証", "value": "1セットの継続", "relevant": True},
            {"label": "特化ゾーン/平均獲得枚数", "item": "1セットあたり", "value": "280枚", "relevant": True},
            {"label": "前兆/パターン数", "item": "全体", "value": "2種類", "relevant": False},
            {"label": "小役B/確率", "item": "設定共通", "value": "1/6.5", "relevant": False},
        ],
        "prose_sections": [
            {"title": "特化ゾーン解説", "category": "zone_explanation", "body": "特化ゾーンは保証7Gで、突入すると最低1セットの継続が確定する。", "relevant": True, "is_compressed_summary": True},
            {"title": "特化ゾーン獲得枚数解説", "category": "zone_explanation", "body": "1セットあたりの平均獲得枚数は280枚とされる。", "relevant": True, "is_compressed_summary": False},
            {"title": "前兆解説", "category": "effect_lore", "body": "前兆には弱パターンと強パターンがある。", "relevant": False, "is_compressed_summary": False},
            {"title": "小役B解説", "category": "game_mechanism", "body": "小役Bは全設定共通の確率で成立する。", "relevant": False, "is_compressed_summary": False},
        ],
        "question": "特化ゾーンについて教えて",
        "assistant": "特化ゾーンは保証7Gで、突入すると最低1セットの継続が確定するよ。1セットあたりの平均獲得枚数は280枚なんだ。",
        "extra_relevant_facts_in_prose_not_in_structured": [],
    },
    {
        # T8: 前兆
        "machine": "機種A",
        "structured_rows": [
            {"label": "前兆/弱パターン", "item": "意味", "value": "通常前兆", "relevant": True},
            {"label": "前兆/強パターン", "item": "意味", "value": "濃厚になる", "relevant": True},
            {"label": "ゾーンY/継続率", "item": "初回", "value": "68%", "relevant": False},
            {"label": "設定判別/判別要素", "item": "主要", "value": "モード移行率", "relevant": False},
        ],
        "prose_sections": [
            {"title": "前兆解説", "category": "effect_lore", "body": "前兆には弱パターンと強パターンの2種類がある。", "relevant": True, "is_compressed_summary": True},
            {"title": "強パターン詳細解説", "category": "effect_lore", "body": "強パターンが出現すると、ゾーンXへの突入が濃厚になる。", "relevant": True, "is_compressed_summary": False},
            {"title": "ゾーンY解説", "category": "zone_explanation", "body": "ゾーンYは高確ゾーンの一つ。", "relevant": False, "is_compressed_summary": False},
            {"title": "設定判別解説", "category": "setting_lore", "body": "設定判別はモード移行率を主な手がかりとする。", "relevant": False, "is_compressed_summary": False},
        ],
        "question": "前兆について教えて",
        "assistant": "前兆には弱パターンと強パターンの2種類があるよ。弱パターンは通常前兆で、強パターンが出るとゾーンXへの突入が濃厚になるんだ。",
        "extra_relevant_facts_in_prose_not_in_structured": [],
    },
    {
        # T9: 小役B
        "machine": "機種C",
        "structured_rows": [
            {"label": "小役B/確率", "item": "設定共通", "value": "1/6.5", "relevant": True},
            {"label": "小役B/モード示唆", "item": "成立時", "value": "示唆になる", "relevant": True},
            {"label": "ボーナス/獲得枚数", "item": "共通", "value": "250枚", "relevant": False},
        ],
        "prose_sections": [
            {"title": "小役B解説", "category": "game_mechanism", "body": "小役Bは全設定共通の確率で成立する。", "relevant": True, "is_compressed_summary": True},
            {"title": "小役B成立時解説", "category": "game_mechanism", "body": "成立時にモードβ以上への示唆となることがある。", "relevant": True, "is_compressed_summary": False},
            {"title": "ボーナス解説", "category": "at_mechanism", "body": "ボーナスの獲得枚数は一律250枚。", "relevant": False, "is_compressed_summary": False},
        ],
        "question": "小役Bについて教えて",
        "assistant": "小役Bは1/6.5の確率で、全設定共通なんだ。成立するとモードβ以上への示唆になることがあるよ。",
        "extra_relevant_facts_in_prose_not_in_structured": ["モードβ以上への示唆"],
    },
    {
        # T10: 投資/回収
        "machine": "機種A",
        "structured_rows": [
            {"label": "投資/推奨投資額", "item": "目安", "value": "2200枚", "relevant": True},
            {"label": "投資/回収目安", "item": "目安", "value": "ボーナス2回分", "relevant": True},
            {"label": "投資/勝率目安", "item": "全体", "value": "42%", "relevant": True},
            {"label": "設定判別/判別要素", "item": "主要", "value": "モード移行率", "relevant": False},
        ],
        "prose_sections": [
            {"title": "投資目安解説", "category": "at_mechanism", "body": "推奨投資額は2200枚、回収の目安はボーナス2回分とされている。", "relevant": True, "is_compressed_summary": True},
            {"title": "勝率目安解説", "category": "at_mechanism", "body": "全体の勝率目安は42%とされる。", "relevant": True, "is_compressed_summary": False},
            {"title": "設定判別解説", "category": "setting_lore", "body": "設定判別はモード移行率を主な手がかりとする。", "relevant": False, "is_compressed_summary": False},
        ],
        "question": "投資と回収の目安を教えて",
        "assistant": "推奨投資額は2200枚で、回収の目安はボーナス2回分だよ。全体の勝率目安は42%なんだ。",
        "extra_relevant_facts_in_prose_not_in_structured": [],
    },
    {
        # T11: 出目示唆
        "machine": "機種D",
        "structured_rows": [
            {"label": "出目示唆/0テンパイ", "item": "該当時", "value": "濃厚になる", "relevant": True},
            {"label": "出目示唆/偶数テンパイ", "item": "該当時", "value": "弱めの示唆", "relevant": True},
            {"label": "ゾーンZ/滞在ゲーム数", "item": "平均", "value": "18G", "relevant": False},
            {"label": "小役C/確率", "item": "設定共通", "value": "1/7.8", "relevant": False},
        ],
        "prose_sections": [
            {"title": "出目示唆解説", "category": "zone_explanation", "body": "0テンパイが出るとゾーンZが濃厚になる。偶数テンパイは弱めの示唆にとどまる。", "relevant": True, "is_compressed_summary": True},
            {"title": "出目示唆補足解説", "category": "zone_explanation", "body": "いずれの示唆も次ゲーム以降の演出で確認できる。", "relevant": True, "is_compressed_summary": False},
            {"title": "ゾーンZ解説", "category": "zone_explanation", "body": "ゾーンZは天井到達時のみ突入する特殊ゾーン。", "relevant": False, "is_compressed_summary": False},
            {"title": "小役C解説", "category": "game_mechanism", "body": "小役Cは全設定共通の確率で成立する。", "relevant": False, "is_compressed_summary": False},
        ],
        "question": "出目示唆について教えて",
        "assistant": "0テンパイが出るとゾーンZが濃厚になるよ。偶数テンパイは弱めの示唆にとどまるんだ。どちらも次ゲーム以降の演出で確認できるよ。",
        "extra_relevant_facts_in_prose_not_in_structured": ["次ゲーム以降の演出で確認できる"],
    },
    {
        # T12: ゾーンZ到達後の処理
        "machine": "機種D",
        "structured_rows": [
            {"label": "ゾーンZ/平均獲得枚数", "item": "1回あたり", "value": "310枚", "relevant": True},
            {"label": "ゾーンZ/継続条件", "item": "共通", "value": "特定の小役が成立", "relevant": True},
            {"label": "ゾーンZ/次ゾーンへの移行率", "item": "継続時", "value": "25%", "relevant": True},
            {"label": "小役C/確率", "item": "設定共通", "value": "1/7.8", "relevant": False},
        ],
        "prose_sections": [
            {"title": "ゾーンZ解説", "category": "zone_explanation", "body": "ゾーンZは平均310枚を獲得できる特殊ゾーン。", "relevant": True, "is_compressed_summary": True},
            {"title": "ゾーンZ継続解説", "category": "zone_explanation", "body": "特定の小役が成立すると継続し、継続時は25%でさらに次のゾーンへ移行する。", "relevant": True, "is_compressed_summary": False},
            {"title": "小役C解説", "category": "game_mechanism", "body": "小役Cは全設定共通の確率で成立する。", "relevant": False, "is_compressed_summary": False},
        ],
        "question": "ゾーンZについて教えて",
        "assistant": "ゾーンZは平均310枚を獲得できるゾーンだよ。特定の小役が成立すると継続して、継続時は25%でさらに次のゾーンへ移行するんだ。",
        "extra_relevant_facts_in_prose_not_in_structured": [],
    },
]

# =============================================================================
# Type2: 派生・二次エンティティ情報の保持 (6件、最大2階層)
# =============================================================================
CATEGORY_T2_DERIVED_ENTITY_RETENTION: list[dict] = [
    {
        "user": "参照情報：特化ゾーン中は追加抽選あり、抽選成功で上乗せ200枚\n特化ゾーンについて教えて",
        "assistant": "特化ゾーン中は追加抽選があるよ。成功すると200枚が上乗せされるんだ。",
    },
    {
        "user": "参照情報：AT終了後はモードβへ移行、モードβの滞在率は35%\nATについて教えて",
        "assistant": "ATが終了すると、モードβへ移行するよ。モードβの滞在率は35%なんだ。",
    },
    {
        "user": "参照情報：天井到達の恩恵はAT-A確定、AT-A中の純増は6枚/G\n天井の恩恵を教えて",
        "assistant": "天井に到達すると、AT-Aが確定するよ。AT-A中の純増は6枚/Gなんだ。",
    },
    {
        "user": "参照情報：ボーナス成立後は状態Aへ移行、状態A中の純増は3枚/G\nボーナスについて教えて",
        "assistant": "ボーナスが成立すると、状態Aへ移行するよ。状態A中の純増は3枚/Gなんだ。",
    },
    {
        "user": "参照情報：ゾーンX中は追加抽選あり、抽選成功でゾーンYへ移行\nゾーンXについて教えて",
        "assistant": "ゾーンX中は追加抽選があるよ。成功するとゾーンYへ移行するんだ。",
    },
    {
        "user": "参照情報：初当り時にモード示唆あり、示唆の的中率は70%\n初当りの示唆について教えて",
        "assistant": "初当り時にはモード示唆があるよ。示唆の的中率は70%なんだ。",
    },
]
