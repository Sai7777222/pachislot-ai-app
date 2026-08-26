"""Phase 4U-2: Phase4T naming probe 全484生成の再分類。

新規生成は行わず、既存 phase4t_comprehensive_results.json (v4/ratio-high、
22 prompts x greedy+10seed = 242 gens/条件) を読み込み、以下へ分類する。

  A. 明示的な誤名乗り (genuine wrong-name)
  B. 名前質問に対して名前を答えない (no-name, 質問はぐらかし)
  C. placeholder (「私は〜〜だよ」型)
  D. generic roleのみ回答 (パチスロに詳しいAIだよ、等、名前情報なし)
  E. 正しく「リル」と名乗る
  F. detector false positive (「あいだ」=間、等の一般語誤検知)
  G. その他

Phase4Tのreview_required率をそのままgenuine wrong-name率として扱わず、
全candidateを規則ベースで目視相当の分類にかけ、genuine wrong-name率を
別途算出する (完全自動だが、既知の偽陽性パターンを明示的に除外する)。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EVAL_DIR.parents[0] / "reports"

sys.path.insert(0, str(EVAL_DIR))
from phase4t_wrongname_detector import (  # noqa: E402
    CHATML_PATTERN,
    NAME_CUE_PATTERNS,
    PLACEHOLDER_PATTERN,
)

CORRECT_NAME = "リル"
NAMING_CONDITIONS = ("B_v4", "C_high")

# F. 既知のfalse positiveパターン (一般語であり名前ではない候補)
FALSE_POSITIVE_CANDIDATES = {
    "あいだ", "あいだよ", "あいだだ", "なかったけど", "未登録だよ", "未登録",
    "登録されてないみたいだよ", "なかったみたいだよ", "登録されてないみたいだよ",
}
FALSE_POSITIVE_SUBSTRINGS = ["なかった", "登録されて", "未登録"]

GENERIC_ROLE_PATTERN = re.compile(
    r"(パチスロ|アシスタント|AI).{0,20}(だよ|です|なんだ)"
)
NO_NAME_HEDGE_PATTERN = re.compile(
    r"(名前|呼び方).{0,15}(ない|無い|特に|登録.{0,5}(なかった|無かった|されてない))"
)
# 「登録名は『パチスロ博士』って決めてあるんだ」のような、cueパターンでは
# 拾えない「架空の登録名」claimも誤名乗りとして拾う追加パターン
FICTIONAL_REGISTERED_NAME_PATTERN = re.compile(
    r"登録名は[「『]?([^」』\s、。！？♪〜\-]{1,12})[」』]?(?:って|と|に)"
)


def strip_candidate(cand: str) -> str:
    return re.sub(r"(だよ|です|だ|なんだ|よ)+$", "", cand).strip()


def classify_generation(text: str) -> str:
    if not text.strip():
        return "G"
    if CHATML_PATTERN.search(text):
        return "G"
    if PLACEHOLDER_PATTERN.search(text):
        return "C"

    candidates = []
    for pat in NAME_CUE_PATTERNS:
        for m in pat.finditer(text):
            candidates.append(m.group(1).strip())
    for m in FICTIONAL_REGISTERED_NAME_PATTERN.finditer(text):
        cand = m.group(1).strip()
        if cand not in ("特に", "特にない", "ない", "無い"):
            candidates.append(cand)

    # E: 正しい名前
    if any(c == CORRECT_NAME or c.startswith(CORRECT_NAME) for c in candidates):
        return "E"

    # F: false positive (一般語を名前候補として誤検知しているだけ)
    real_candidates = []
    for c in candidates:
        stripped = strip_candidate(c)
        if c in FALSE_POSITIVE_CANDIDATES or stripped in FALSE_POSITIVE_CANDIDATES:
            continue
        if any(fp in c for fp in FALSE_POSITIVE_SUBSTRINGS):
            continue
        if len(stripped) <= 1:  # 「キミ」等の1文字は代名詞的に使われている場合が多く要目視だが、
            continue            # 単独では固有名詞と判定しない (2文字以上を候補とする)
        real_candidates.append(stripped)

    if real_candidates:
        return "A"  # genuine wrong-name

    # B: 名前を聞かれているのに、はぐらかす/ない、と答える
    if NO_NAME_HEDGE_PATTERN.search(text):
        return "B"

    # D: generic roleのみ (パチスロに詳しいAIだよ、等)
    if GENERIC_ROLE_PATTERN.search(text) and CORRECT_NAME not in text:
        return "D"

    return "G"


def main() -> int:
    results_path = EVAL_DIR / "phase4t_comprehensive_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))

    summary = {}
    detail = {}
    for cond in NAMING_CONDITIONS:
        counts = {c: 0 for c in "ABCDEFG"}
        total = 0
        examples: dict[str, list] = {c: [] for c in "ABCDEFG"}
        for pid, rec in results["naming_probes"].items():
            cond_data = rec["conditions"][cond]
            texts = [cond_data["greedy"]] + list(cond_data["sampled"].values())
            for t in texts:
                cat = classify_generation(t)
                counts[cat] += 1
                total += 1
                if len(examples[cat]) < 8:
                    examples[cat].append({"probe": pid, "text": t})
        summary[cond] = {
            "total": total,
            "counts": counts,
            "rates_pct": {k: round(100 * v / total, 1) for k, v in counts.items()},
            "genuine_wrong_name_rate_pct": round(100 * counts["A"] / total, 1),
            "correct_name_rate_pct": round(100 * counts["E"] / total, 1),
            "placeholder_rate_pct": round(100 * counts["C"] / total, 1),
        }
        detail[cond] = examples

    out = {"summary": summary, "examples": detail}
    (REPORTS_DIR / "phase4u_naming_reclassification.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
