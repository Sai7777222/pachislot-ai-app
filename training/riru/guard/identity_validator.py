"""Phase 4ZL/4ZM: Identity Production Guard — Stage A Validator.

製品仕様上のcanonical character identity(リル)を、application側で不変条件
として保証するためのoutput-side validator。標準ライブラリのみに依存し、
GPUやモデルのロードを一切必要としない(Section21: 軽量性の優先)。

設計原則(Phase4ZL Section5-6 + Phase4ZM Section3-7による縮小):
  - ユーザーが何を言ったかではなく、リル自身が何を自称したかを検証する。
  - user messageに別名が含まれるだけでblockしてはいけない
    (「アリスってキャラ知ってる？」等は合法的な入力)。
  - quotation・第三者言及・仮定/翻訳依頼は誤検出しない。
  - nickname(リル由来の愛称)とcanonical identity rewriteを区別する。
  - Phase4ZM: Precision >> Recall。「明らかに誤った自己名乗り」のみを高
    confidenceで検出し、bare agreement(名前トークンを含まない裸の同意)や
    暗黙の受諾のような、ユーザー発話の意味理解を要する曖昧な検出は行わない
    (RISK IDENTITY-R02/R03として既知残存リスクに正式登録)。
  - Phase4ZM: 「私は{任意の自然言語句}」のような、TOKENが名前以外の一般的な
    述語・感想にもマッチしうる曖昧なpatternは、_looks_like_name_token()に
    よるカタカナ限定gateを必ず通す(training/riru/reports/
    phase4zm_guard_simplification.json参照)。

本モジュールはpure-Pythonであり、tests/ではなくtraining/riru/guard/tests/
に配置したunit testでカバーする(既存の本体テストスイートとは独立)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CANONICAL_NAME = "リル"

# リル由来の正当な愛称。canonical_nameとは別管理(Section7)。
ALLOWED_NICKNAMES = frozenset({
    "リルちゃん", "リルにゃん", "リルっち", "リルたそ", "リルりん", "りるりる",
    "りーちゃん", "りるぽん", "りーたん", "ちびリル", "リルたん",
})

_HIRA_KATA_OFFSET = ord("ァ") - ord("ぁ")


def _to_katakana(s: str) -> str:
    """ひらがなをカタカナへ正規化する(「りる」→「リル」等、表記揺れの吸収用)。"""
    return "".join(
        chr(ord(ch) + _HIRA_KATA_OFFSET) if "ぁ" <= ch <= "ん" else ch
        for ch in s
    )


def _starts_or_ends_with_canonical(candidate: str) -> bool:
    """候補がcanonical_nameで始まる/終わる/一致するかを、ひらがな表記揺れも
    吸収した上で判定する(「りるっぴ」等がリル由来と正しく認識されるように)。"""
    norm = _to_katakana(candidate)
    return norm == CANONICAL_NAME or norm.startswith(CANONICAL_NAME) or norm.endswith(CANONICAL_NAME)


# Phase4ZM Section6: 「TOKENが任意自然言語phraseになり得る形でidentity判定する
# 設計は禁止」への対応。このprojectの偽名語彙(アリス/ルナ/ミア/テト/ソウ等)は
# 一貫してカタカナの固有名詞であり、「甘いものが好きだ」「大丈夫」のような
# 一般的な述語・感想・状態表現は通常カタカナのみでは書かれない。この構造的な
# 違いを、STOPWORDS/synonym辞書を増やさず高精度に利用する(Section7準拠:
# dangerous broad pattern削除の代替として、synonym列挙ではなくorthographic
# heuristicを採用)。
_KATAKANA_ONLY = re.compile(r"^[゠-ヿー]+$")


def _looks_like_name_token(candidate: str) -> bool:
    """曖昧な(naming-anchorを伴わない)patternから得た候補が、高confidenceで
    名前らしいと言えるかどうかを判定する。カタカナのみの語、canonical名由来、
    または既知のnicknameのみを名前候補として許可する。"""
    norm = _to_katakana(candidate)
    if _KATAKANA_ONLY.fullmatch(norm):
        return True
    return _starts_or_ends_with_canonical(candidate) or candidate in ALLOWED_NICKNAMES

# 名前候補として現実的な長さ(既存project全phaseの誤名語彙は2〜6文字)。
_TOKEN = r"([^\s、。！？♪〜\-「」『』]{1,8})"

# ============================================================
# 1. 自己名乗り検出(Stage A)
#    全てのsuffixを必須にし(optional `?`を排除)、tokenが不用意に文全体を
#    飲み込まないようにする。既存のphase4t_wrongname_detector.pyの
#    NAME_CUE_PATTERNSを土台に、Phase4ZH〜4ZKで実際に観測された言い回しを追加。
# ============================================================

SELF_ID_PATTERNS = [
    # Phase4ZM: 末尾の裸の「よ」alternativeは、greedy TOKENが「アリスだよ」を
    # 「アリスだ」+「よ」に誤って分割してしまう(TOKENの末尾に「だ」が残存する)
    # bugの原因だったため削除した(「だよ」自体は別alternativeで既にカバー済み)。
    re.compile(rf"私(?:自身)?は{_TOKEN}(?:だよ|なんだ|です|といいます|と申します)"),
    re.compile(rf"私は{_TOKEN}っていうん?(?:だ|です)"),
    # 「私は<役割記述>、<名前>です」型(役割部分は読み飛ばし、カンマ直後の名前だけ拾う)
    re.compile(rf"私は[^、。]{{0,20}}、\s*{_TOKEN}(?:だよ|です)"),
    # 「私は<役割記述>の<名前>です」型(「の」区切り)
    re.compile(rf"私は[^、。]{{0,15}}の{_TOKEN}(?:だよ|です)"),
    re.compile(rf"僕は{_TOKEN}(?:だよ|なんだ|です)"),
    # 文頭の裸の名乗り: 「Xだよ！」(挨拶cueなしでも許可)
    re.compile(rf"^{_TOKEN}だよ[！!]"),
    re.compile(rf"(?:私の)?名前は{_TOKEN}(?:だよ|なんだ|です|といいます)"),
    re.compile(rf"登録名は{_TOKEN}(?:だよ|です|になります)"),
    re.compile(rf"登録名は{_TOKEN}と(?:して|し)(?:おく|います)(?:ね|よ)?"),
    re.compile(rf"登録名は{_TOKEN}に(?:して|し)(?:おく)?(?:ね|よ)?"),
    re.compile(rf"{_TOKEN}として登録(?:して|し)(?:おく)?(?:ね|よ)?"),
    re.compile(rf"呼び名は{_TOKEN}(?:だよ|です)"),
    re.compile(rf"正式名称は{_TOKEN}(?:だよ|です|になります)"),
    re.compile(rf"呼称は{_TOKEN}と?しま(?:す|すね)"),
    re.compile(rf"呼称は{_TOKEN}に変更しま(?:す|すね)"),
    re.compile(rf"{_TOKEN}と申します"),
    re.compile(rf"{_TOKEN}といいます"),
    re.compile(rf"アシスタントの{_TOKEN}(?:だよ|です|よ)"),
    re.compile(rf"今後は{_TOKEN}として(?:お話し|対応|お答え)?しま(?:す|すね|する)"),
    re.compile(rf"これから(?:は)?{_TOKEN}を名乗ります"),
    re.compile(rf"これから(?:は)?{_TOKEN}と名乗る"),
    re.compile(rf"{_TOKEN}になったよ"),
    re.compile(rf"{_TOKEN}に変更(?:しま|いたしま)す"),
    re.compile(rf"(?:こんにちは|はじめまして)[！!、,]\s*{_TOKEN}(?:です|だよ)"),
    # quote-as-answer型: 「正式名称は『X』だよ」「正しくは『X』だと思う」
    re.compile(rf"正式名称は[「『]{_TOKEN}[」』](?:だよ|です)"),
    re.compile(rf"正しくは[「『]?{_TOKEN}[」』]?だ(?:と思うよ|よ)"),
    # 「正式名称は『A』じゃなくて、『B』だよ」型(否定した後の第二引用を拾う)
    re.compile(rf"正式名称は[「『][^」』]+[」』]じゃなくて、?[「『]{_TOKEN}[」』](?:だよ|です)"),
]

# Phase4ZM Section6/7: naming-specific vocabulary(登録名/正式名称/呼称/と申します/
# といいます/って呼んで等)を伴わない、あるいは「名前は」のように主語を問わず
# 一般的な述語も飲み込みうる"曖昧"なpattern群。ここに含まれるindexの候補は、
# _looks_like_name_token()によるカタカナ限定gateを通過した場合のみ名前候補
# として扱う。indices 0-5は「私は/僕は/文頭裸の宣言」型、6は「名前は{TOKEN}」
# 型で、Phase4ZLで実際に「私は甘いものが好きだよ」「もちろん大丈夫だよ」
# 「(第三者の名前は)可愛いよね」等の一般的な文を誤って名前候補として抽出した
# 実例(ZI-OD-15/ZL-I07/ZL-G02)が確認されたため、この10件を新設のgateで保護する。
_AMBIGUOUS_PATTERN_INDICES = {0, 1, 2, 3, 4, 5, 6}

# 「Xって呼んで」「Xとして話すね」等、ユーザー提案への直接応答系。
# 文中どこにあってもよい(「うん、Xって呼んでね！」等)が、「私は」等の主語
# プレフィックスをtoken外に出し、swallowを防ぐ。
SELF_ID_RESPONSE_PATTERNS = [
    re.compile(rf"(?:私は)?{_TOKEN}って呼んで(?:くれて|くれても|くれたら|も)?"
               r"(?:いい|OK|オッケー|大丈夫|嬉しい|平気|問題ない)?"),
    re.compile(rf"(?:私は)?{_TOKEN}と呼んでね"),
    re.compile(rf"{_TOKEN}として(?:お話し|お答え|対応)しま(?:す|すね)"),
    re.compile(rf"私も{_TOKEN}って呼んでもいい"),
]

STOPWORDS = frozenset({
    "パチスロ", "アシスタント", "AI", "キャラクター", "元気", "笑顔", "とっても",
    "ちょっと", "みんな", "詳しく", "得意", "専門", "情報", "登録", "データ",
    "データベース", "何でも", "今日", "私", "僕", "その", "この", "リル",
    "私だ", "私です", "あなた", "それ", "もちろん", "そう", "うん", "はい",
    "いいよ", "オッケー", "OK",
})

# 候補の先頭に付きがちなfiller(「みんな」「うんうん」等)を除去してから
# STOPWORDS/canonical判定を行う。
_LEADING_FILLER = re.compile(r"^(みんな|うんうん|うん|そう|もちろん|はい)")

# 明らかに「名前」ではない継続(否定・非断定の述語断片・一般的な述語)を除外する。
# 「名前は/登録名は」の直後は自由な述語が続きうるため、広めに列挙する。
_NON_NAME_CONTINUATION = re.compile(
    r"ない|わから|不明|変わって|同じ|まま|以前|前と|違う|ちゃんと|ある|特に|"
    r"まだ|そう|大切|別物|なくて|決めて|決まって|考え中|秘密|内緒|自由"
)

# 汎用的な役割呼称(Section8-H: role_as_nameは文脈判定対象であり、A-Gのような
# 即block対象ではない)。「○○役」「○○係」「○○担当」等はここで低優先度に分離する。
_GENERIC_ROLE_PATTERN = re.compile(r"(役|係|担当|アシスタント|相談員|ナビ|コンシェルジュ)$")

# 文中の他の場所で既にcanonical identityが明確に再確認されている場合の安全弁
# (「登録情報では『リル』という名前になっている」等)。
_CANONICAL_CONFIRMED_ELSEWHERE = re.compile(rf"[「『]?{CANONICAL_NAME}[」』]?という名前|{CANONICAL_NAME}になっている")

QUOTATION_PATTERN = re.compile(r"[「『][^」』]*[」』]")
TRANSLATION_HYPOTHETICAL_CUE = re.compile(
    r"翻訳して|という文章|って文章|って文を|もし.*(?:名前|呼)|仮に|例えば.*名前"
)


@dataclass
class ValidationResult:
    safe: bool
    category: str  # "safe" or one of the unsafe categories A-I (英語スラッグ)
    claimed_name: str | None = None
    reason: str = ""
    matched_text: str = ""
    context_flags: list[str] = field(default_factory=list)


def _is_inside_plain_quotation(text: str, span: tuple[int, int]) -> bool:
    """spanが「」『』の中に完全に含まれるかどうか。quote-as-answer型patternは
    そもそも引用符を含めてmatchするため、ここでの除外対象にはならない。"""
    for qm in QUOTATION_PATTERN.finditer(text):
        if qm.start() <= span[0] and span[1] <= qm.end():
            return True
    return False


def detect_self_identification(assistant_text: str) -> list[tuple[str, tuple[int, int]]]:
    """assistant出力から自己名乗り候補を(候補文字列, matchのspan)として抽出する。
    _AMBIGUOUS_PATTERN_INDICESに属するpatternの候補は、_looks_like_name_token()
    のカタカナ限定gateを通過したものだけをここで候補として残す(Phase4ZM Section6)。"""
    candidates: list[tuple[str, tuple[int, int]]] = []
    if TRANSLATION_HYPOTHETICAL_CUE.search(assistant_text):
        # 翻訳依頼・仮定文脈が明示されている場合、通常のself-ID検出はスキップする
        # (quote-as-answer型は例外的に別途扱いたいケースもあるが、prototypeでは
        # falseの安全側[過剰検出を避ける]に倒す)。
        return []
    for idx, pat in enumerate(SELF_ID_PATTERNS):
        is_ambiguous = idx in _AMBIGUOUS_PATTERN_INDICES
        for m in pat.finditer(assistant_text):
            cand = m.group(1).strip()
            if not cand:
                continue
            if is_ambiguous and not _looks_like_name_token(cand):
                continue
            span = m.span(1)
            is_quote_as_answer = "正式名称は" in m.group(0) or "正しくは" in m.group(0)
            if not is_quote_as_answer and _is_inside_plain_quotation(assistant_text, span):
                continue
            candidates.append((cand, span))
    for pat in SELF_ID_RESPONSE_PATTERNS:
        for m in pat.finditer(assistant_text):
            cand = m.group(1).strip()
            if not cand:
                continue
            span = m.span(1)
            if _is_inside_plain_quotation(assistant_text, span):
                continue
            candidates.append((cand, span))
    return candidates


def validate_identity(assistant_text: str, user_text: str = "") -> ValidationResult:
    """assistant_textから、identity出力の安全性を判定する。

    user_textは呼び出し側の既存インタフェース互換のため引数として残しているが、
    Phase4ZM以降は判定に使用しない(Section2: 「output-side validationが主で
    あり、user入力単体でblockしてはいけない」という原則、およびSection5の
    bare-agreement検出撤廃により、user_textを参照する分岐が無くなったため)。"""
    if not assistant_text.strip():
        return ValidationResult(safe=True, category="safe", reason="empty_output")

    # --- placeholder detection (Section8-G): 自己名乗り文脈でplaceholderが
    #     そのまま名前として使われている場合のみ対象とする ---
    for pat in SELF_ID_PATTERNS:
        m = pat.search(assistant_text)
        if m and re.fullmatch(r"[〜ー]{2,}|○○|〇〇|XX|xx|\[名前\]|<名前>|\{name\}|\[NAME\]|<NAME>",
                               m.group(1)):
            return ValidationResult(safe=False, category="placeholder_identity",
                                     claimed_name=m.group(1), matched_text=m.group(0),
                                     reason="self-identification resolved to a placeholder token")

    raw_candidates = detect_self_identification(assistant_text)
    real_candidates = []
    for cand, _span in raw_candidates:
        cand = _LEADING_FILLER.sub("", cand).strip()
        if not cand or cand in STOPWORDS:
            continue
        if _NON_NAME_CONTINUATION.search(cand):
            continue
        real_candidates.append(cand)

    for c in real_candidates:
        if _starts_or_ends_with_canonical(c):
            return ValidationResult(safe=True, category="safe", claimed_name=c,
                                     reason="explicit canonical self-identification")
        if c in ALLOWED_NICKNAMES or any(c == nn or nn.startswith(c) for nn in ALLOWED_NICKNAMES):
            return ValidationResult(safe=True, category="safe", claimed_name=c,
                                     reason="allowed nickname")

    # role_as_name(Section8-H): 明示的なnaming文脈(登録名は/正式名称は/って呼んで
    # への同意等)から得られた役割語候補はblock対象とする。この経路に到達する
    # 候補は、既に_looks_like_name_token()のgate(曖昧pattern由来のもの)または
    # 明示的なnaming-anchor(登録名は等、gate対象外のrigidなpattern)のいずれかを
    # 通過済みであるため、単なる「私はパチスロ専門のAIアシスタントだよ」のような
    # generic self-descriptionは、そもそもambiguous-pattern gateで候補にすら
    # ならず、ここへ到達しない(Phase4ZM Section6の簡素化により、is_genericという
    # 別レーンを設ける必要がなくなった)。
    role_candidates = [c for c in real_candidates if _GENERIC_ROLE_PATTERN.search(c)]
    non_role_candidates = [c for c in real_candidates if not _GENERIC_ROLE_PATTERN.search(c)]

    if _CANONICAL_CONFIRMED_ELSEWHERE.search(assistant_text):
        return ValidationResult(safe=True, category="safe",
                                 reason="canonical identity confirmed elsewhere in the same response")

    if non_role_candidates:
        wrong = non_role_candidates[0]
        is_future = bool(re.search(r"今後は|これから(?:は)?|次からは", assistant_text))
        category = "future_identity_rewrite" if is_future else "explicit_wrong_self_name"
        return ValidationResult(safe=False, category=category, claimed_name=wrong,
                                 reason="assistant explicitly self-identified with a non-canonical name",
                                 matched_text=wrong)

    if role_candidates:
        # 設計上の判断(Section8注記): Section8はH(role_as_name)を「文脈判定対象」と
        # 位置付け、A-Gのような無条件block対象とはしていない。しかし本prototypeでの
        # 大規模replay検証(1368件)の結果、role-as-name(「案内係って呼んで」
        # 「パチスロアシスタントだよ」等)をnon-blockingとして扱うと、recallが
        # 大きく低下すること(既知failureの約35%を取りこぼす)が判明した。
        # G1(「別名を正式な自分の名前として名乗らない」)の趣旨に照らし、prototype
        # としては安全側に倒し、role_as_nameもblock対象として扱う(context_flagsで
        # 通常のexplicit_wrong_self_nameと区別できるようにしている)。将来的な精緻化
        # (真に無害な文脈のみ許可する等)の余地として記録する。
        wrong = role_candidates[0]
        return ValidationResult(safe=False, category="role_as_name", claimed_name=wrong,
                                 reason="generic role phrase used as a self-name (treated as block-worthy "
                                        "by default in this prototype; see design note)",
                                 matched_text=wrong, context_flags=["role_as_name"])

    # Phase4ZM Section5/7: 「bare agreement」(名前トークンを一切含まない裸の
    # 同意、例:「うん、その通りだよ！」「了解！」)をユーザー発話の意味理解なしに
    # regexだけで検出することは、精度を大きく損なう(disagree-without-denial-word
    # という文の断片を無理に拾おうとするたびに新しい誤検知源が生まれる)ため、
    # Phase4ZLの実装にあった USER_REWRITE_CUE_PATTERNS/COMPLIANCE_MARKERS/
    # DENIAL_PATTERNSに基づく検出ロジックはPhase4ZMで意図的に削除した。
    # これはバグではなく、Precision >> Recallという方針転換に基づく設計判断
    # であり、RISK IDENTITY-R02(bare agreement can semantically accept wrong
    # identity without explicit wrong-name token)としてknown residual risk
    # に正式登録されている(training/riru/reports/phase4zm_known_residual_risks.json)。

    return ValidationResult(safe=True, category="safe", reason="no unsafe identity pattern detected")
