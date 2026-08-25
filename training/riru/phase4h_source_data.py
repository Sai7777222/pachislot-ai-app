# ruff: noqa: E501
"""Phase 4H-3: 本番RAG形式に近い「重要情報省略防止」追加学習データ。

Phase 4E/4Gで確認されたQ3型問題（RAGコンテキストに存在する、質問対象に
直接関連する複数の重要情報を回答時に省略してしまう）を改善するためのデータ。

Phase 4Fの反省 (phase4h_q3_analysis.md参照):
  Phase 4FのHカテゴリは「参照情報：fact1、fact2、fact3」という単一行フラット形式
  だったため、本番のRAGコンテキスト (【対象機種】ヘッダー + 構造化箇条書き複数行
  + 複数の出典カテゴリ付き解説文セクション、うち一部だけが質問に関連) という
  より複雑な文書形式には転移しなかったと考えられる。

このカテゴリの設計方針:
  - 本番RAGコンテキストと同じ形式 (build_rag_context() が組み立てる) を使う。
  - 各項目 (structured_rows / prose_sections) には relevant フラグを持たせ、
    「質問に直接関連する項目」と「同じコンテキスト内にある無関係な項目」を
    明示的に混在させる。
  - assistant回答は、relevant=Trueの項目の値を全て含み (省略しない)、
    relevant=Falseの項目の内容は含まない (全部答えるモデルにしない) ことを、
    build_phase4h_dataset.py が自動検査する。
  - 実在機種名・実在機種固有の数値は一切使わない (「機種A」固定、ゾーン・設定は
    ギリシャ文字/アルファベットの架空名、数値も実在機種の値と異なるものを使う)。

このファイルはデータの「定義」のみを行う。組み立て・検証・出力は
`build_phase4h_dataset.py` が行う。
"""

from __future__ import annotations

MACHINE_LINE = "機種A（パチスロ） 設定判別・天井・ゾーン・解析・打ち方・ヤメ時"

# =============================================================================
# T. 本番RAG形式での「重要情報省略防止」(質問に直接関連する情報は複数あっても
#    省略しない。関連しない情報は答えに含めない)
# =============================================================================
CATEGORY_T_PRODUCTION_RAG_OMISSION: list[dict] = [
    {
        "structured_rows": [
            {"label": "天井/天井ゲーム数振り分け", "item": "ゲーム数=600G", "value": "20.0%", "relevant": True},
            {"label": "天井/天井ゲーム数振り分け", "item": "ゲーム数=900G", "value": "30.0%", "relevant": True},
            {"label": "天井/天井ゲーム数振り分け", "item": "ゲーム数=1500G", "value": "50.0%", "relevant": True},
            {"label": "ゾーンα/突入率", "item": "設定共通", "value": "1/60", "relevant": False},
        ],
        "prose_sections": [
            {"title": "天井解説", "category": "at_mechanism", "body": "天井は3種類のゲーム数があり、到達すると設定変更の可能性が高まる。", "relevant": True},
            {"title": "前兆解説", "category": "effect_lore", "body": "リール上部の点灯パターンで前兆の強さが変わる。", "relevant": False},
        ],
        "question": "天井について教えて",
        "assistant": "天井は600G・900G・1500Gの3種類あって、確率はそれぞれ20.0%・30.0%・50.0%だよ。到達すると設定変更の可能性が高まるみたい。",
    },
    {
        "structured_rows": [
            {"label": "設定X/機械割", "item": "設定X", "value": "112.3%", "relevant": True},
            {"label": "設定X/初当り確率", "item": "設定X", "value": "1/270", "relevant": True},
            {"label": "設定X/ボーナス中純増", "item": "設定X", "value": "4枚/G", "relevant": True},
            {"label": "設定Y/機械割", "item": "設定Y", "value": "98.1%", "relevant": False},
        ],
        "prose_sections": [
            {"title": "設定Xの傾向", "category": "setting_lore", "body": "設定Xは高設定域の中でも狙い目とされている。", "relevant": True},
            {"title": "ゾーンβ解説", "category": "zone_explanation", "body": "ゾーンβは特殊な演出を伴う高確ゾーン。", "relevant": False},
        ],
        "question": "設定Xについて教えて",
        "assistant": "設定Xは機械割112.3%、初当り確率1/270、ボーナス中の純増は4枚/Gだよ。高設定域の中でも狙い目とされてるみたい。",
    },
    {
        "structured_rows": [
            {"label": "ゾーンα/突入率", "item": "設定共通", "value": "1/45", "relevant": True},
            {"label": "ゾーンα/滞在ゲーム数", "item": "平均", "value": "25G", "relevant": True},
            {"label": "ゾーンα/継続率", "item": "初回", "value": "65%", "relevant": True},
            {"label": "ゾーンβ/平均獲得枚数", "item": "1回あたり", "value": "380枚", "relevant": False},
        ],
        "prose_sections": [
            {"title": "ゾーンα解説", "category": "zone_explanation", "body": "ゾーンαは前兆を経て突入する高確ゾーンで、継続すると純増が上乗せされる。", "relevant": True},
            {"title": "裏モード解説", "category": "effect_lore", "body": "特定の条件で裏モードに移行することがある。", "relevant": False},
        ],
        "question": "ゾーンαについて詳しく教えて",
        "assistant": "ゾーンαは突入率1/45、滞在ゲーム数は平均25G、初回継続率は65%だよ。前兆を経て突入する高確ゾーンで、継続すると純増が上乗せされるみたい。",
    },
    {
        "structured_rows": [
            {"label": "ゾーンβ/平均獲得枚数", "item": "1回あたり", "value": "380枚", "relevant": True},
            {"label": "ゾーンβ/突入率", "item": "設定共通", "value": "1/90", "relevant": True},
            {"label": "天井/天井到達時のゾーン突入", "item": "ゾーンβ", "value": "必ず突入", "relevant": True},
            {"label": "ゾーンα/継続率", "item": "初回", "value": "65%", "relevant": False},
        ],
        "prose_sections": [
            {"title": "ゾーンβ解説", "category": "zone_explanation", "body": "ゾーンβは特殊な演出を伴う高確ゾーンで、天井到達時に必ず突入する。", "relevant": True},
            {"title": "小役示唆解説", "category": "effect_lore", "body": "特定の小役成立でゾーン示唆が出ることがある。", "relevant": False},
        ],
        "question": "ゾーンβについて教えて",
        "assistant": "ゾーンβは平均獲得枚数380枚、突入率1/90だよ。天井に到達したときは必ず突入するみたい。",
    },
    {
        "structured_rows": [
            {"label": "AT/AT中純増", "item": "設定共通", "value": "5枚/G", "relevant": True},
            {"label": "AT/AT終了後の恩恵", "item": "RT付与", "value": "8G", "relevant": True},
            {"label": "天井/天井ゲーム数振り分け", "item": "ゲーム数=600G", "value": "20.0%", "relevant": False},
        ],
        "prose_sections": [
            {"title": "AT解説", "category": "at_mechanism", "body": "ATは純増が高く、終了後は短いRTが付与される。", "relevant": True},
            {"title": "天井解説", "category": "at_mechanism", "body": "天井は3種類のゲーム数があり、到達すると設定変更の可能性が高まる。", "relevant": False},
        ],
        "question": "ATについて教えて",
        "assistant": "ATは純増5枚/Gで、終了後は8GのRTが付与されるよ。",
    },
    {
        "structured_rows": [
            {"label": "初当り/初当り確率", "item": "設定共通", "value": "1/310", "relevant": True},
            {"label": "初当り/初当り後の恩恵", "item": "モード示唆", "value": "付与", "relevant": True},
            {"label": "ボーナス/種類数", "item": "全体", "value": "3種類", "relevant": False},
        ],
        "prose_sections": [
            {"title": "初当り解説", "category": "at_mechanism", "body": "初当りすると、モード示唆が付与されることがある。", "relevant": True},
            {"title": "ボーナス解説", "category": "at_mechanism", "body": "ボーナスは3種類あり、獲得枚数が異なる。", "relevant": False},
        ],
        "question": "初当りについて教えて",
        "assistant": "初当り確率は1/310だよ。初当りすると、モード示唆が付与されることがあるみたい。",
    },
    {
        "structured_rows": [
            {"label": "ボーナス/種類数", "item": "全体", "value": "3種類", "relevant": True},
            {"label": "ボーナス/獲得枚数", "item": "種別1", "value": "140枚", "relevant": True},
            {"label": "ボーナス/獲得枚数", "item": "種別2", "value": "230枚", "relevant": True},
            {"label": "ボーナス/獲得枚数", "item": "種別3", "value": "420枚", "relevant": True},
            {"label": "設定X/機械割", "item": "設定X", "value": "112.3%", "relevant": False},
        ],
        "prose_sections": [
            {"title": "ボーナス解説", "category": "at_mechanism", "body": "ボーナスは3種類あり、獲得枚数が異なる。", "relevant": True},
            {"title": "設定Xの傾向", "category": "setting_lore", "body": "設定Xは高設定域の中でも狙い目とされている。", "relevant": False},
        ],
        "question": "ボーナスの種類を教えて",
        "assistant": "ボーナスは3種類あって、獲得枚数はそれぞれ140枚・230枚・420枚だよ。",
    },
    {
        "structured_rows": [
            {"label": "特化ゾーン/保証ゲーム数", "item": "共通", "value": "6G", "relevant": True},
            {"label": "特化ゾーン/突入後の継続", "item": "最低保証", "value": "1セット確定", "relevant": True},
            {"label": "前兆/パターン数", "item": "全体", "value": "2種類", "relevant": False},
        ],
        "prose_sections": [
            {"title": "特化ゾーン解説", "category": "zone_explanation", "body": "特化ゾーンは保証ゲーム数があり、突入後は最低1セットの継続が確定する。", "relevant": True},
            {"title": "前兆解説", "category": "effect_lore", "body": "前兆には中パターンと上パターンの2種類がある。", "relevant": False},
        ],
        "question": "特化ゾーンについて教えて",
        "assistant": "特化ゾーンは保証6Gで、突入すると1セット確定で継続するよ。",
    },
    {
        "structured_rows": [
            {"label": "プレミア演出/発生率", "item": "全設定共通", "value": "1/8192", "relevant": True},
            {"label": "プレミア演出/該当時の確定内容", "item": "成立時", "value": "ボーナスが確定", "relevant": True},
            {"label": "ゾーンγ/継続率", "item": "初回", "value": "80%", "relevant": False},
        ],
        "prose_sections": [
            {"title": "プレミア演出解説", "category": "effect_lore", "body": "プレミア演出が成立すると、ボーナスが確定する。", "relevant": True},
            {"title": "ゾーンγ解説", "category": "zone_explanation", "body": "ゾーンγは高継続率の特殊ゾーン。", "relevant": False},
        ],
        "question": "プレミア演出について教えて",
        "assistant": "プレミア演出の発生率は1/8192で、成立するとボーナスが確定するよ。",
    },
    {
        "structured_rows": [
            {"label": "設定差/機械割差", "item": "設定1と設定6", "value": "2.1%", "relevant": True},
            {"label": "設定判別/判別要素", "item": "主要", "value": "モード移行率", "relevant": True},
        ],
        "prose_sections": [
            {"title": "設定差解説", "category": "setting_lore", "body": "設定差は機械割でおよそ2.1%程度とされている。", "relevant": True},
            {"title": "設定判別解説", "category": "setting_lore", "body": "設定判別はモード移行率を主な手がかりとする。", "relevant": True},
        ],
        "question": "設定差と、その見分け方も含めて教えて",
        "assistant": "設定差は機械割でおよそ2.1%だよ。見分け方としては、モード移行率が主な手がかりになるみたい。",
    },
    {
        "structured_rows": [
            {"label": "天井/天井恩恵", "item": "共通", "value": "AT確定", "relevant": True},
            {"label": "天井/天井ゲーム数振り分け", "item": "ゲーム数=600G", "value": "20.0%", "relevant": True},
            {"label": "天井/天井ゲーム数振り分け", "item": "ゲーム数=900G", "value": "30.0%", "relevant": True},
            {"label": "天井/天井ゲーム数振り分け", "item": "ゲーム数=1500G", "value": "50.0%", "relevant": True},
            {"label": "ゾーンα/突入率", "item": "設定共通", "value": "1/45", "relevant": False},
        ],
        "prose_sections": [
            {"title": "天井解説", "category": "at_mechanism", "body": "天井到達で必ずAT確定となる。", "relevant": True},
            {"title": "ゾーンα解説", "category": "zone_explanation", "body": "ゾーンαは前兆を経て突入する高確ゾーン。", "relevant": False},
        ],
        "question": "天井の恩恵と、天井までの目安を教えて",
        "assistant": "天井の恩恵はAT確定だよ。目安は600G・900G・1500Gの3種類で、確率はそれぞれ20.0%・30.0%・50.0%だね。",
    },
    {
        "structured_rows": [
            {"label": "リセット恩恵/電源投入時", "item": "モード", "value": "初期化", "relevant": True},
            {"label": "リセット恩恵/初当り優遇", "item": "1000G以内", "value": "優遇", "relevant": True},
            {"label": "ゾーンβ/突入率", "item": "設定共通", "value": "1/90", "relevant": False},
        ],
        "prose_sections": [
            {"title": "リセット恩恵解説", "category": "at_mechanism", "body": "電源を入れ直すとモードが初期化され、1000G以内は初当りが優遇される。", "relevant": True},
            {"title": "ゾーンβ解説", "category": "zone_explanation", "body": "ゾーンβは特殊な演出を伴う高確ゾーン。", "relevant": False},
        ],
        "question": "リセット時の恩恵について教えて",
        "assistant": "リセットするとモードが初期化されるよ。あと、1000G以内は初当りが優遇されるみたい。",
    },
    {
        "structured_rows": [
            {"label": "小役X/確率", "item": "設定共通", "value": "1/38", "relevant": True},
            {"label": "小役X/モード示唆", "item": "成立時", "value": "示唆", "relevant": True},
            {"label": "小役Y/確率", "item": "設定共通", "value": "1/6.8", "relevant": False},
        ],
        "prose_sections": [
            {"title": "小役X解説", "category": "game_mechanism", "body": "小役Xが成立すると、モード示唆になることがある。", "relevant": True},
            {"title": "小役Y解説", "category": "game_mechanism", "body": "小役Yは全設定共通の確率で成立する。", "relevant": False},
        ],
        "question": "小役Xについて教えて",
        "assistant": "小役Xは1/38の確率で、成立するとモード示唆になることがあるよ。",
    },
    {
        "structured_rows": [
            {"label": "RT/RT中のボーナス確率", "item": "共通", "value": "アップ", "relevant": True},
            {"label": "RT/RT継続ゲーム数", "item": "共通", "value": "12G", "relevant": True},
            {"label": "AT/AT中純増", "item": "設定共通", "value": "5枚/G", "relevant": False},
        ],
        "prose_sections": [
            {"title": "RT解説", "category": "at_mechanism", "body": "RT中はボーナス確率がアップし、12G継続する。", "relevant": True},
            {"title": "AT解説", "category": "at_mechanism", "body": "ATは純増が高く、終了後は短いRTが付与される。", "relevant": False},
        ],
        "question": "RTについて教えて",
        "assistant": "RT中はボーナス確率がアップして、12G継続するよ。",
    },
    {
        "structured_rows": [
            {"label": "前兆/中パターン", "item": "意味", "value": "高確前兆濃厚", "relevant": True},
            {"label": "前兆/上パターン", "item": "意味", "value": "ステージチェンジ濃厚", "relevant": True},
            {"label": "ゾーンβ/平均獲得枚数", "item": "1回あたり", "value": "380枚", "relevant": False},
        ],
        "prose_sections": [
            {"title": "前兆解説", "category": "effect_lore", "body": "前兆には中パターンと上パターンの2種類があり、それぞれ示唆内容が異なる。", "relevant": True},
            {"title": "ゾーンβ解説", "category": "zone_explanation", "body": "ゾーンβは特殊な演出を伴う高確ゾーン。", "relevant": False},
        ],
        "question": "前兆について教えて",
        "assistant": "前兆には中パターンと上パターンの2種類があるよ。中パターンは高確前兆濃厚、上パターンはステージチェンジ濃厚を意味するみたい。",
    },
    {
        "structured_rows": [
            {"label": "サブ液晶/示唆頻度", "item": "高確中", "value": "頻度が上がる", "relevant": True},
            {"label": "サブ液晶/信頼度", "item": "プレミア発展時", "value": "信頼度も上がる", "relevant": True},
            {"label": "ボーナス/種類数", "item": "全体", "value": "3種類", "relevant": False},
        ],
        "prose_sections": [
            {"title": "サブ液晶解説", "category": "effect_lore", "body": "サブ液晶は高確中に示唆の頻度が上がり、プレミアに発展すると信頼度も上がる。", "relevant": True},
            {"title": "ボーナス解説", "category": "at_mechanism", "body": "ボーナスは3種類あり、獲得枚数が異なる。", "relevant": False},
        ],
        "question": "サブ液晶の示唆について教えて",
        "assistant": "サブ液晶は高確中に示唆の頻度が上がるよ。プレミアに発展すると信頼度も上がるみたい。",
    },
    {
        "structured_rows": [
            {"label": "設定1/初当り確率", "item": "設定1", "value": "1/480", "relevant": True},
            {"label": "設定6/初当り確率", "item": "設定6", "value": "1/270", "relevant": True},
            {"label": "設定1/機械割", "item": "設定1", "value": "97.5%", "relevant": True},
            {"label": "設定6/機械割", "item": "設定6", "value": "112.3%", "relevant": True},
            {"label": "ゾーンα/継続率", "item": "初回", "value": "65%", "relevant": False},
        ],
        "prose_sections": [
            {"title": "設定比較解説", "category": "setting_lore", "body": "設定6は設定1に比べて初当りしやすく、機械割も高い。", "relevant": True},
            {"title": "ゾーンα解説", "category": "zone_explanation", "body": "ゾーンαは前兆を経て突入する高確ゾーン。", "relevant": False},
        ],
        "question": "設定1と設定6の初当りと機械割を比較して",
        "assistant": "設定1は初当り1/480、機械割97.5%だよ。設定6は初当り1/270、機械割112.3%で、設定6の方が初当りしやすく機械割も高いよ。",
    },
]
