"""Phase 4S: complex multi-fact教師データ プール生成。

Phase4Kと同じ構造(structured_rows / prose_sections)を踏襲し、パターンA〜H
(天井/ゲーム数振り分け・設定別確率テーブル・複数モードmapping・ゾーン振り分け・
示唆対応・AT/RT派生・条件分岐・例外付きmapping)について、架空エンティティ・
架空数値でrelevant>=5件・percentage/mapping含む複雑教師例のpoolを生成する。

実在機種名・実在数値のコピーは一切行わない。学習は本ファイルでは行わない
(build_phase4s_dataset.pyが読み込んでcandidate/train/valを構築する)。
"""

from __future__ import annotations

import random

MACHINES = ["機種A", "機種B", "機種C", "機種D", "機種E"]
STATES = ["状態A", "状態B", "状態C", "状態D"]
MODES = ["モードα", "モードβ", "モードγ", "モードδ", "モードε"]
ZONES = ["ゾーンX", "ゾーンY", "ゾーンZ", "ゾーンW"]
KOYAKU = ["小役A", "小役B", "小役C", "小役D", "小役E", "小役F", "小役G"]
AT_NAMES = ["AT-A", "AT-B", "AT-C", "AT-D"]
RT_NAMES = ["RT-A", "RT-B", "RT-C"]
DEME = ["出目A", "出目B", "出目C", "出目D", "出目E", "出目F"]
SHISA = ["示唆X", "示唆Y", "示唆Z", "示唆W", "示唆V"]


def split_percentages(rng: random.Random, n: int) -> list[int]:
    """n個の整数%に分割し合計100にする(各10刻み、簡易的に)。"""
    cuts = sorted(rng.sample(range(10, 100, 5), n - 1))
    bounds = [0, *cuts, 100]
    vals = [bounds[i + 1] - bounds[i] for i in range(n)]
    return [v if v > 0 else 5 for v in vals]


REAL_Q3_GAME_COUNTS = {510, 1000, 1480}


def game_counts(rng: random.Random, n: int) -> list[int]:
    pool = [v for v in range(300, 1500, 10) if v not in REAL_Q3_GAME_COUNTS]
    base = sorted(rng.sample(pool, n))
    return base


def record(
    category: str, category_code: str, index: int, question: str,
    structured_rows: list[dict], prose_sections: list[dict],
    relevant_facts: list[str], irrelevant_facts: list[str], answer: str,
) -> dict:
    return {
        "id": f"{category_code}-{index}",
        "category": category,
        "category_code": category_code,
        "question": question,
        "structured_rows": structured_rows,
        "prose_sections": prose_sections,
        "relevant_facts": relevant_facts,
        "irrelevant_facts": irrelevant_facts,
        "answer": answer,
    }


def build_context_text(
    machine: str, structured_rows: list[dict], prose_sections: list[dict]
) -> str:
    lines = [
        f"【対象機種】\n{machine}（パチスロ） 設定判別・天井・ゾーン・解析・打ち方・ヤメ時\n"
        "このセクション以下の情報はすべて上記の機種に関するものです。"
        "他の機種の名称を補完したり、機種名を推測したりしないでください。\n",
        "【構造化データ（数値・確率・設定差・天井・示唆など）】\n"
        "このセクションの数値は必ず原文表記のまま回答に使ってください。"
        "計算し直したり丸めたりしないでください。\n",
    ]
    for row in structured_rows:
        lines.append(f"- [{row['label']}] {row['key']}: {row['value']}")
    if prose_sections:
        lines.append("\n【関連する解説文章】\n")
        for sec in prose_sections:
            lines.append(f"◆ {sec['title']}（出典カテゴリ: {sec['cat']}）\n{sec['body']}\n")
    return "\n".join(lines)


# --- Pattern A: 天井/ゲーム数振り分け型 ---
def gen_pattern_a(rng: random.Random, idx: int) -> dict:
    m = MACHINES[idx % len(MACHINES)]
    counts = game_counts(rng, 3)
    pcts = split_percentages(rng, 3)
    at_bonus_pct = rng.choice([25, 30, 35, 40])
    rows = [
        {"label": "天井/ゲーム数振り分け", "key": f"{counts[0]}G", "value": f"{pcts[0]}%"},
        {"label": "天井/ゲーム数振り分け", "key": f"{counts[1]}G", "value": f"{pcts[1]}%"},
        {"label": "天井/ゲーム数振り分け", "key": f"{counts[2]}G", "value": f"{pcts[2]}%"},
        {"label": "天井/到達時処理", "key": "共通", "value": "AT確定"},
        {"label": "天井/追加抽選", "key": "実施有無", "value": "追加抽選あり"},
        {"label": "天井/追加抽選成功率", "key": "共通", "value": f"{at_bonus_pct}%"},
        {"label": "ゾーンX/突入率", "key": "設定共通", "value": "1/60"},  # irrelevant
    ]
    ceiling_body = (
        f"天井ゲーム数は3種類あり、抽選で決定する。到達すると{at_bonus_pct}%の確率で追加抽選が行われる。"
    )
    prose = [
        {
            "title": "天井解説", "cat": "at_mechanism",
            "relevant": True, "is_compressed_summary": True,
            "body": ceiling_body,
        },
        {
            "title": "ゾーンX解説", "cat": "zone_explanation",
            "relevant": False, "is_compressed_summary": False,
            "body": "ゾーンXは前兆を経て突入する高確ゾーン。",
        },
    ]
    question = "天井について教えて"
    relevant = [
        f"{counts[0]}G", f"{pcts[0]}%", f"{counts[1]}G", f"{pcts[1]}%",
        f"{counts[2]}G", f"{pcts[2]}%", "AT確定", "追加抽選", f"{at_bonus_pct}%",
    ]
    irrelevant = ["1/60", "ゾーンXは前兆"]
    style = idx % 3
    if style == 0:
        answer = (
            f"天井は{counts[0]}G・{counts[1]}G・{counts[2]}Gの3種類で、"
            f"それぞれ{pcts[0]}%・{pcts[1]}%・{pcts[2]}%だよ。"
            f"到達するとAT確定に加えて追加抽選もあって、成功率は{at_bonus_pct}%なんだ。"
        )
    elif style == 1:
        answer = (
            f"天井ゲーム数は{counts[0]}G(確率{pcts[0]}%)、{counts[1]}G(確率{pcts[1]}%)、"
            f"{counts[2]}G(確率{pcts[2]}%)の3パターンあるよ。"
            f"到達時はAT確定で、さらに{at_bonus_pct}%で追加抽選が発生するんだ。"
        )
    else:
        answer = (
            f"{counts[0]}G・{counts[1]}G・{counts[2]}Gが天井の候補で、"
            f"振り分けは{pcts[0]}%・{pcts[1]}%・{pcts[2]}%だね。"
            f"追加抽選(成功率{at_bonus_pct}%)もあって、成立するとAT確定になるよ。"
        )
    return {
        "machine": m, "rows": rows, "prose": prose, "question": question,
        "relevant": relevant, "irrelevant": irrelevant, "answer": answer,
        "category": "complex_ratio_ceiling_breakdown", "category_code": "S1",
    }


# --- Pattern B: 設定別確率テーブル型 ---
def gen_pattern_b(rng: random.Random, idx: int) -> dict:
    m = MACHINES[idx % len(MACHINES)]
    denominators = sorted(rng.sample(range(250, 550, 10), 3), reverse=True)
    settings = [1, 3, 6]
    machine_rate = rng.choice([96, 97, 98]) + rng.random() * 0 + 0  # int base
    machine_rate_high = machine_rate + rng.choice([15, 17, 18])
    rows = [
        {"label": "初当り確率", "key": f"設定{settings[0]}", "value": f"1/{denominators[0]}"},
        {"label": "初当り確率", "key": f"設定{settings[1]}", "value": f"1/{denominators[1]}"},
        {"label": "初当り確率", "key": f"設定{settings[2]}", "value": f"1/{denominators[2]}"},
        {"label": "機械割", "key": f"設定{settings[0]}", "value": f"{int(machine_rate)}.0%"},
        {"label": "機械割", "key": f"設定{settings[2]}", "value": f"{int(machine_rate_high)}.0%"},
        {"label": "小役A/確率", "key": "設定共通", "value": "1/5.5"},  # irrelevant
    ]
    setting_body = f"設定{settings[0]}と設定{settings[2]}では初当り確率・機械割ともに差がある。"
    prose = [
        {
            "title": "設定差解説", "cat": "setting_lore",
            "relevant": True, "is_compressed_summary": True,
            "body": setting_body,
        },
    ]
    question = f"設定{settings[0]}と設定{settings[2]}の初当り確率と機械割を教えて"
    relevant = [
        f"1/{denominators[0]}", f"1/{denominators[2]}",
        f"{int(machine_rate)}.0%", f"{int(machine_rate_high)}.0%",
    ]
    irrelevant = ["1/5.5"]
    r0, r2 = int(machine_rate), int(machine_rate_high)
    style = idx % 3
    if style == 0:
        answer = (
            f"設定{settings[0]}の初当り確率は1/{denominators[0]}、機械割は{r0}.0%だよ。"
            f"設定{settings[2]}だと初当り確率は1/{denominators[2]}、機械割は{r2}.0%になるんだ。"
        )
    elif style == 1:
        answer = (
            f"初当り確率は設定{settings[0]}が1/{denominators[0]}、設定{settings[2]}が1/{denominators[2]}だね。"
            f"機械割はそれぞれ{r0}.0%と{r2}.0%だよ。"
        )
    else:
        answer = (
            f"設定{settings[0]}: 初当り1/{denominators[0]}・機械割{r0}.0%、"
            f"設定{settings[2]}: 初当り1/{denominators[2]}・機械割{r2}.0%、という違いがあるんだ。"
        )
    return {
        "machine": m, "rows": rows, "prose": prose, "question": question,
        "relevant": relevant, "irrelevant": irrelevant, "answer": answer,
        "category": "complex_ratio_setting_table", "category_code": "S2",
    }


# --- Pattern C: 複数モードmapping型 ---
def gen_pattern_c(rng: random.Random, idx: int) -> dict:
    m = MACHINES[idx % len(MACHINES)]
    pcts = split_percentages(rng, 3)
    zone_irrelevant = ZONES[idx % len(ZONES)]
    modes = [MODES[(idx + i) % len(MODES)] for i in range(3)]
    rows = [
        {"label": "モード/滞在率", "key": modes[0], "value": f"{pcts[0]}%"},
        {"label": "モード/滞在率", "key": modes[1], "value": f"{pcts[1]}%"},
        {"label": "モード/滞在率", "key": modes[2], "value": f"{pcts[2]}%"},
        {"label": "モード/移行契機", "key": "共通", "value": "毎ゲーム抽選"},
        {"label": "モード/優遇", "key": "最高モード", "value": "移行率優遇"},
        {"label": f"{zone_irrelevant}/突入率", "key": "設定共通", "value": "1/45"},  # irrelevant
    ]
    prose = [
        {
            "title": "モード解説", "cat": "mode_mechanism",
            "relevant": True, "is_compressed_summary": True,
            "body": "モードは全部で3種類あり、それぞれ滞在率が異なる。",
        },
    ]
    question = "モードの種類と滞在率を教えて"
    relevant = [f"{pcts[0]}%", f"{pcts[1]}%", f"{pcts[2]}%", "毎ゲーム", "優遇"]
    irrelevant = ["1/45"]
    style = idx % 3
    if style == 0:
        answer = (
            f"モードは{modes[0]}が{pcts[0]}%、{modes[1]}が{pcts[1]}%、{modes[2]}が{pcts[2]}%だよ。"
            f"移行は毎ゲーム抽選されて、最高モードだけ移行率が優遇されるんだ。"
        )
    elif style == 1:
        answer = (
            f"モードの滞在率は{modes[0]}: {pcts[0]}%、{modes[1]}: {pcts[1]}%、"
            f"{modes[2]}: {pcts[2]}%だね。"
            f"毎ゲーム移行抽選があって、最高モードは優遇されているよ。"
        )
    else:
        answer = (
            f"{modes[0]}({pcts[0]}%)・{modes[1]}({pcts[1]}%)・{modes[2]}({pcts[2]}%)の3つのモードがあるんだ。"
            f"移行抽選は毎ゲーム行われて、最高モードだと優遇されるよ。"
        )
    return {
        "machine": m, "rows": rows, "prose": prose, "question": question,
        "relevant": relevant, "irrelevant": irrelevant, "answer": answer,
        "category": "complex_ratio_mode_mapping", "category_code": "S3",
    }


# --- Pattern D: ゾーン振り分け型 ---
def gen_pattern_d(rng: random.Random, idx: int) -> dict:
    m = MACHINES[idx % len(MACHINES)]
    zone = ZONES[idx % len(ZONES)]
    enter_denom = rng.choice([30, 40, 50, 60])
    stay_low, stay_high = sorted(rng.sample(range(5, 60, 5), 2))
    continue_pct = rng.choice([60, 65, 70, 75, 80])
    prize = rng.choice([150, 200, 250, 300, 350])
    rows = [
        {"label": f"{zone}/突入率", "key": "設定共通", "value": f"1/{enter_denom}"},
        {"label": f"{zone}/滞在ゲーム数", "key": "範囲", "value": f"{stay_low}〜{stay_high}G"},
        {"label": f"{zone}/継続率", "key": "共通", "value": f"{continue_pct}%"},
        {"label": f"{zone}/獲得枚数", "key": "1回あたり", "value": f"{prize}枚"},
        {"label": "小役B/確率", "key": "設定共通", "value": "1/8.2"},  # irrelevant
    ]
    zone_body = f"{zone}は前兆を経て突入する高確ゾーンで、継続すると獲得枚数が上乗せされる。"
    prose = [
        {
            "title": f"{zone}解説", "cat": "zone_explanation",
            "relevant": True, "is_compressed_summary": False,
            "body": zone_body,
        },
    ]
    question = f"{zone}について教えて"
    relevant = [f"1/{enter_denom}", f"{stay_low}〜{stay_high}G", f"{continue_pct}%", f"{prize}枚"]
    irrelevant = ["1/8.2"]
    style = idx % 3
    if style == 0:
        answer = (
            f"{zone}は突入率1/{enter_denom}で、滞在ゲーム数は{stay_low}〜{stay_high}Gだよ。"
            f"継続率は{continue_pct}%で、1回あたりの獲得枚数は{prize}枚なんだ。"
        )
    elif style == 1:
        answer = (
            f"{zone}の突入率は1/{enter_denom}、滞在は{stay_low}〜{stay_high}Gくらいだね。"
            f"継続率{continue_pct}%で、獲得枚数は1回{prize}枚が目安だよ。"
        )
    else:
        answer = (
            f"突入率1/{enter_denom}・滞在{stay_low}〜{stay_high}Gが{zone}の基本情報だよ。"
            f"継続率{continue_pct}%、獲得枚数{prize}枚(1回あたり)というデータもあるんだ。"
        )
    return {
        "machine": m, "rows": rows, "prose": prose, "question": question,
        "relevant": relevant, "irrelevant": irrelevant, "answer": answer,
        "category": "complex_ratio_zone_breakdown", "category_code": "S4",
    }


# --- Pattern E: 示唆対応型 ---
def gen_pattern_e(rng: random.Random, idx: int) -> dict:
    m = MACHINES[idx % len(MACHINES)]
    deme = [DEME[(idx + i) % len(DEME)] for i in range(3)]
    shisa = [SHISA[(idx * 2 + i) % len(SHISA)] for i in range(3)]
    rows = [
        {"label": "出目示唆/対応", "key": deme[0], "value": shisa[0]},
        {"label": "出目示唆/対応", "key": deme[1], "value": shisa[1]},
        {"label": "出目示唆/対応", "key": deme[2], "value": shisa[2]},
        {"label": "出目示唆/発生条件", "key": "共通", "value": "AT中のみ"},
        {"label": "小役C/確率", "key": "設定共通", "value": "1/12.3"},  # irrelevant
    ]
    prose = [
        {
            "title": "出目示唆解説", "cat": "effect_lore",
            "relevant": True, "is_compressed_summary": True,
            "body": "出目によって次回のゾーン示唆が異なり、AT中にのみ発生する。",
        },
    ]
    question = "出目と示唆の対応を教えて"
    relevant = [deme[0], shisa[0], deme[1], shisa[1], deme[2], shisa[2], "AT中のみ"]
    irrelevant = ["1/12.3"]
    style = idx % 3
    if style == 0:
        answer = (
            f"{deme[0]}が出ると{shisa[0]}、{deme[1]}なら{shisa[1]}、{deme[2]}なら{shisa[2]}の示唆になるよ。"
            f"この示唆はAT中のみ発生するんだ。"
        )
    elif style == 1:
        answer = (
            f"出目と示唆の対応は、{deme[0]}→{shisa[0]}、{deme[1]}→{shisa[1]}、{deme[2]}→{shisa[2]}だね。"
            f"AT中のみ出る示唆だよ。"
        )
    else:
        answer = (
            f"示唆の対応表: {deme[0]}なら{shisa[0]}、{deme[1]}は{shisa[1]}、"
            f"{deme[2]}は{shisa[2]}になるんだ。"
            f"発生条件はAT中のみだよ。"
        )
    return {
        "machine": m, "rows": rows, "prose": prose, "question": question,
        "relevant": relevant, "irrelevant": irrelevant, "answer": answer,
        "category": "complex_ratio_deme_shisa_mapping", "category_code": "S5",
    }


# --- Pattern F: AT/RT派生型 ---
def gen_pattern_f(rng: random.Random, idx: int) -> dict:
    m = MACHINES[idx % len(MACHINES)]
    at_name = AT_NAMES[idx % len(AT_NAMES)]
    rt_name = RT_NAMES[idx % len(RT_NAMES)]
    junzo = rng.choice([2.0, 2.5, 3.0, 4.0, 5.0])
    rt_games = rng.choice([30, 50, 70, 100])
    rt_pct = rng.choice([20, 25, 30, 35])
    rows = [
        {"label": f"{at_name}/純増", "key": "共通", "value": f"{junzo}枚/G"},
        {"label": f"{at_name}/終了後", "key": "移行先", "value": rt_name},
        {"label": f"{rt_name}/継続G数", "key": "共通", "value": f"{rt_games}G"},
        {"label": f"{rt_name}/中抽選確率", "key": "共通", "value": f"{rt_pct}%"},
        {"label": "小役D/確率", "key": "設定共通", "value": "1/9.8"},  # irrelevant
    ]
    at_body = f"{at_name}は純増が高く、終了後は{rt_name}へ移行する。"
    prose = [
        {
            "title": f"{at_name}解説", "cat": "at_mechanism",
            "relevant": True, "is_compressed_summary": False,
            "body": at_body,
        },
    ]
    question = f"{at_name}の性能と終了後の状態を教えて"
    relevant = [f"{junzo}枚/G", rt_name, f"{rt_games}G", f"{rt_pct}%"]
    irrelevant = ["1/9.8"]
    style = idx % 3
    if style == 0:
        answer = (
            f"{at_name}の純増は{junzo}枚/Gだよ。終了すると{rt_name}へ移行して、"
            f"{rt_games}G継続するんだ。{rt_name}中は{rt_pct}%で抽選があるよ。"
        )
    elif style == 1:
        answer = (
            f"{at_name}は純増{junzo}枚/Gで、終了後は{rt_name}(継続{rt_games}G)に移行するね。"
            f"{rt_name}中の抽選確率は{rt_pct}%だよ。"
        )
    else:
        answer = (
            f"純増{junzo}枚/Gの{at_name}が終わると、{rt_name}へ移るんだ。"
            f"{rt_name}は{rt_games}G継続で、中では{rt_pct}%の抽選があるよ。"
        )
    return {
        "machine": m, "rows": rows, "prose": prose, "question": question,
        "relevant": relevant, "irrelevant": irrelevant, "answer": answer,
        "category": "complex_ratio_at_rt_derivation", "category_code": "S6",
    }


# --- Pattern G: 条件分岐型 ---
def gen_pattern_g(rng: random.Random, idx: int) -> dict:
    m = MACHINES[idx % len(MACHINES)]
    koyaku = KOYAKU[idx % len(KOYAKU)]
    outcome_pool = [
        "ゾーン当選", "モード昇格", "AT当選確定", "上乗せ抽選", "小役確率優遇", "前兆濃厚",
        "継続G数上乗せ",
    ]
    outcomes = [outcome_pool[(idx + i) % len(outcome_pool)] for i in range(3)]
    rows = [
        {"label": f"{koyaku}/条件A成立時", "key": "結果", "value": outcomes[0]},
        {"label": f"{koyaku}/条件B成立時", "key": "結果", "value": outcomes[1]},
        {"label": f"{koyaku}/条件C成立時", "key": "結果", "value": outcomes[2]},
        {"label": "小役A/確率", "key": "設定共通", "value": "1/7.7"},  # irrelevant
    ]
    branch_body = f"{koyaku}が成立した状況によって、得られる恩恵が3パターンに分かれる。"
    prose = [
        {
            "title": f"{koyaku}分岐解説", "cat": "game_mechanism",
            "relevant": True, "is_compressed_summary": True,
            "body": branch_body,
        },
    ]
    question = f"{koyaku}が成立したときの条件別の結果を教えて"
    relevant = outcomes
    irrelevant = ["1/7.7"]
    style = idx % 3
    if style == 0:
        answer = (
            f"{koyaku}は、条件Aで成立すると{outcomes[0]}、条件Bだと{outcomes[1]}、"
            f"条件Cだと{outcomes[2]}になるよ。"
        )
    elif style == 1:
        answer = (
            f"{koyaku}成立時、条件別に結果が変わるよ。条件A: {outcomes[0]}、"
            f"条件B: {outcomes[1]}、条件C: {outcomes[2]}だね。"
        )
    else:
        answer = (
            f"条件Aなら{outcomes[0]}、条件Bなら{outcomes[1]}、条件Cなら{outcomes[2]}になるのが、"
            f"{koyaku}成立時のパターンなんだ。"
        )
    return {
        "machine": m, "rows": rows, "prose": prose, "question": question,
        "relevant": relevant, "irrelevant": irrelevant, "answer": answer,
        "category": "complex_ratio_condition_branch", "category_code": "S7",
    }


# --- Pattern H: 例外付きmapping型 ---
def gen_pattern_h(rng: random.Random, idx: int) -> dict:
    m = MACHINES[idx % len(MACHINES)]
    state = STATES[idx % len(STATES)]
    normal_pct = rng.choice([10, 15, 20, 25])
    special_pct = rng.choice([40, 45, 50, 55])
    exception_pct = rng.choice([80, 85, 90])
    rows = [
        {"label": f"{state}/抽選率", "key": "通常時", "value": f"{normal_pct}%"},
        {"label": f"{state}/抽選率", "key": "特定条件成立時", "value": f"{special_pct}%"},
        {"label": f"{state}/抽選率", "key": "例外条件時", "value": f"{exception_pct}%"},
        {"label": f"{state}/例外条件", "key": "内容", "value": "設定変更後1回目のみ"},
        {"label": "小役B/確率", "key": "設定共通", "value": "1/6.1"},  # irrelevant
    ]
    state_body = f"{state}中の抽選率は条件によって変動し、例外条件下ではさらに優遇される。"
    prose = [
        {
            "title": f"{state}解説", "cat": "state_mechanism",
            "relevant": True, "is_compressed_summary": True,
            "body": state_body,
        },
    ]
    question = f"{state}中の抽選率について、条件ごとの違いを教えて"
    relevant = [f"{normal_pct}%", f"{special_pct}%", f"{exception_pct}%", "設定変更後1回目のみ"]
    irrelevant = ["1/6.1"]
    style = idx % 3
    if style == 0:
        answer = (
            f"{state}中の抽選率は、通常時が{normal_pct}%、特定条件成立時が{special_pct}%だよ。"
            f"ただし、設定変更後1回目のみの例外条件だと{exception_pct}%になるんだ。"
        )
    elif style == 1:
        answer = (
            f"{state}中は通常{normal_pct}%、条件成立で{special_pct}%まで上がるね。"
            f"例外条件(設定変更後1回目のみ)だと{exception_pct}%になるよ。"
        )
    else:
        answer = (
            f"通常時{normal_pct}%・特定条件時{special_pct}%が{state}中の基本の抽選率だよ。"
            f"例外(設定変更後1回目のみ)だと{exception_pct}%まで優遇されるんだ。"
        )
    return {
        "machine": m, "rows": rows, "prose": prose, "question": question,
        "relevant": relevant, "irrelevant": irrelevant, "answer": answer,
        "category": "complex_ratio_exception_mapping", "category_code": "S8",
    }


PATTERN_GENERATORS = [
    gen_pattern_a, gen_pattern_b, gen_pattern_c, gen_pattern_d,
    gen_pattern_e, gen_pattern_f, gen_pattern_g, gen_pattern_h,
]


def build_pool(n_per_pattern: int = 14, seed: int = 42) -> list[dict]:
    """8パターン x n_per_pattern件のcomplex教師poolを生成する。"""
    rng = random.Random(seed)
    pool = []
    for gen_idx, gen in enumerate(PATTERN_GENERATORS):
        for i in range(n_per_pattern):
            item = gen(rng, i + gen_idx * 7)
            context = build_context_text(item["machine"], item["rows"], item["prose"])
            user_content = f"{context}\n{item['question']}"
            pool.append(
                {
                    "id": f"{item['category_code']}-{i}",
                    "category": item["category"],
                    "category_code": item["category_code"],
                    "user": user_content,
                    "assistant": item["answer"],
                    "relevant_facts": item["relevant"],
                    "irrelevant_facts": item["irrelevant"],
                    "question": item["question"],
                    "machine": item["machine"],
                }
            )
    return pool


if __name__ == "__main__":
    p = build_pool()
    print(f"pool size: {len(p)}")
    for x in p[:3]:
        print(x["id"], x["category"], "| relevant:", x["relevant_facts"])
