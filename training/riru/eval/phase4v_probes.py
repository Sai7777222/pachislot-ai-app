"""Phase 4V: Broad-Question Completeness 診断用 held-out probe。

P01/P02/Q3の文面・数値・エンティティは一切コピーせず、完全に架空のデータで
6つのcontext family(天井/設定別確率/複数モード/ゾーン/示唆対応/例外条件分岐)を
作り、各familyについて6種類の質問文体(broad_topic/overview/explain/
tell_me_all/specific_complete/narrow_control)を用意する(6x6=36問)。

同一context(情報源)に対して質問文だけを変えることで、
「情報を保持できないのか」と「質問文を狭く解釈しているのか」を分離する。

学習は行わない。既存P01/P02/Q3/Phase4T probe/Phase4U identity教師/structured17/
character39/ratio-high complex教師との重複は検査スクリプト側で確認する。
"""

from __future__ import annotations


def make_family(
    family_id: str, topic: str, context_body: str,
    main_facts: list[str], narrow_target_fact: str, narrow_target_facts: list[str],
    irrelevant_facts: list[str], broad_topic_question: str | None = None,
) -> dict:
    """1 context familyから6種類の質問variantを生成する。

    main_facts: broad質問で要求される主要fact群 (required)
    narrow_target_fact: narrow_control質問で聞かれる単一fact
    narrow_target_facts: specific_complete質問で聞かれる複数fact (main_factsの部分集合)
    broad_topic_question: 既存P01/P07等と文面が重複する場合の言い換え指定
    """
    context = context_body
    variants = [
        {
            "id": f"{family_id}-A", "category": "broad_topic",
            "question": broad_topic_question or f"{topic}について教えて",
            "required_facts": main_facts, "optional_facts": [],
            "irrelevant_facts": irrelevant_facts,
        },
        {
            "id": f"{family_id}-B", "category": "overview",
            "question": f"{topic}の概要を教えて",
            "required_facts": main_facts, "optional_facts": [],
            "irrelevant_facts": irrelevant_facts,
        },
        {
            "id": f"{family_id}-C", "category": "explain",
            "question": f"{topic}を詳しく教えて",
            "required_facts": main_facts, "optional_facts": [],
            "irrelevant_facts": irrelevant_facts,
        },
        {
            "id": f"{family_id}-D", "category": "tell_me_all",
            "question": f"{topic}について分かっていることを全部教えて",
            "required_facts": main_facts, "optional_facts": [],
            "irrelevant_facts": irrelevant_facts,
        },
        {
            "id": f"{family_id}-E", "category": "specific_complete",
            "question": narrow_target_facts[1],
            "required_facts": [narrow_target_facts[0]],
            "optional_facts": [f for f in main_facts if f != narrow_target_facts[0]],
            "irrelevant_facts": irrelevant_facts,
        },
        {
            "id": f"{family_id}-F", "category": "narrow_control",
            "question": f"{narrow_target_fact}だけ教えて",
            "required_facts": [narrow_target_facts[0]],
            "optional_facts": [f for f in main_facts if f != narrow_target_facts[0]],
            "irrelevant_facts": irrelevant_facts,
        },
    ]
    for v in variants:
        v["context"] = context
        v["family"] = family_id
        v["topic"] = topic
    return {"family_id": family_id, "variants": variants}


FAMILIES = []

# --- Family V1: 天井/ゲーム数振り分け型 (numeric+percentage) ---
FAMILIES.append(
    make_family(
        "V1", "天井",
        (
            "【対象機種】\n機種CA（パチスロ）\n\n【構造化データ】\n"
            "- [天井/ゲーム数振り分け] 550G: 20%\n"
            "- [天井/ゲーム数振り分け] 850G: 30%\n"
            "- [天井/ゲーム数振り分け] 1350G: 50%\n"
            "- [天井/到達時処理] 共通: RT確定\n"
            "- [ゾーンU/突入率] 設定共通: 1/58\n"
        ),
        main_facts=["550G", "20%", "850G", "30%", "1350G", "50%", "RT確定"],
        narrow_target_fact="到達時処理",
        narrow_target_facts=["RT確定", "天井の到達時処理を教えて"],
        irrelevant_facts=["1/58"],
        broad_topic_question="天井の仕組みについて教えて",
    )
)

# --- Family V2: 設定別確率テーブル型 (mapping) ---
FAMILIES.append(
    make_family(
        "V2", "設定別の初当り確率",
        (
            "【対象機種】\n機種CB（パチスロ）\n\n【構造化データ】\n"
            "- [初当り確率] 設定1: 1/389\n"
            "- [初当り確率] 設定3: 1/349\n"
            "- [初当り確率] 設定5: 1/299\n"
            "- [初当り確率] 設定6: 1/259\n"
            "- [小役E/確率] 設定共通: 1/7.4\n"
        ),
        main_facts=["1/389", "1/349", "1/299", "1/259"],
        narrow_target_fact="設定6の初当り確率",
        narrow_target_facts=["1/259", "設定6の初当り確率を教えて"],
        irrelevant_facts=["1/7.4"],
    )
)

# --- Family V3: 複数モード型 (mapping+condition) ---
FAMILIES.append(
    make_family(
        "V3", "モード",
        (
            "【対象機種】\n機種CC（パチスロ）\n\n【構造化データ】\n"
            "- [モード/滞在率] モードA: 45%\n"
            "- [モード/滞在率] モードB: 35%\n"
            "- [モード/滞在率] モードC: 20%\n"
            "- [モード/移行条件] 共通: ボーナス終了時\n"
            "- [ゾーンT/継続率] 共通: 62%\n"
        ),
        main_facts=["45%", "35%", "20%", "ボーナス終了時"],
        narrow_target_fact="モードの移行条件",
        narrow_target_facts=["ボーナス終了時", "モードはいつ移行する？"],
        irrelevant_facts=["62%"],
        broad_topic_question="モードの仕組みについて教えて",
    )
)

# --- Family V4: ゾーン型 (numeric+categorical) ---
FAMILIES.append(
    make_family(
        "V4", "ゾーンS",
        (
            "【対象機種】\n機種CD（パチスロ）\n\n【構造化データ】\n"
            "- [ゾーンS/突入率] 設定共通: 1/62\n"
            "- [ゾーンS/滞在ゲーム数] 範囲: 15〜45G\n"
            "- [ゾーンS/獲得枚数] 1回あたり: 290枚\n"
            "- [ゾーンS/移行先] 成功時: モードC\n"
            "- [小役F/確率] 設定共通: 1/6.8\n"
        ),
        main_facts=["1/62", "15〜45G", "290枚", "モードC"],
        narrow_target_fact="ゾーンSの獲得枚数",
        narrow_target_facts=["290枚", "ゾーンSの獲得枚数を教えて"],
        irrelevant_facts=["1/6.8"],
    )
)

# --- Family V5: 示唆対応型 (mapping, deme-shisa) ---
FAMILIES.append(
    make_family(
        "V5", "出目示唆",
        (
            "【対象機種】\n機種CE（パチスロ）\n\n【構造化データ】\n"
            "- [出目示唆/対応] 出目P: 示唆L\n"
            "- [出目示唆/対応] 出目Q: 示唆M\n"
            "- [出目示唆/対応] 出目R: 示唆N\n"
            "- [出目示唆/発生条件] 共通: ART中のみ\n"
            "- [小役G/確率] 設定共通: 1/9.3\n"
        ),
        main_facts=["出目P", "示唆L", "出目Q", "示唆M", "出目R", "示唆N", "ART中のみ"],
        narrow_target_fact="出目Qの示唆",
        narrow_target_facts=["示唆M", "出目Qの示唆を教えて"],
        irrelevant_facts=["1/9.3"],
    )
)

# --- Family V6: 例外付き条件分岐型 (categorical+exception) ---
FAMILIES.append(
    make_family(
        "V6", "状態Dの抽選率",
        (
            "【対象機種】\n機種CF（パチスロ）\n\n【構造化データ】\n"
            "- [状態D/抽選率] 通常時: 12%\n"
            "- [状態D/抽選率] 特定役成立時: 48%\n"
            "- [状態D/抽選率] 例外条件時: 88%\n"
            "- [状態D/例外条件] 内容: 天井到達後1回目のみ\n"
            "- [小役H/確率] 設定共通: 1/5.7\n"
        ),
        main_facts=["12%", "48%", "88%", "天井到達後1回目のみ"],
        narrow_target_fact="例外条件時の抽選率",
        narrow_target_facts=["88%", "例外条件時の状態Dの抽選率を教えて"],
        irrelevant_facts=["1/5.7"],
    )
)

PROBES = [v for fam in FAMILIES for v in fam["variants"]]

assert len(PROBES) == 36, f"expected 36 probes, got {len(PROBES)}"
