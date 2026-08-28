"""Phase4ZN Section9: 暫定的なheuristic自動集計。

重要(Phase4ZM RULE EVAL-001準拠): これは正式なground truthではない。
generationのresponseテキストに対する単純な文字列検索であり、人間の目視分類の
代替にはならない。あくまで無人実行中に取得できる「暫定値」として記録する。
"""
from __future__ import annotations
import json
import re
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

HEDGE_PHRASES = ["登録データ", "データベース", "データがない", "登録されていない",
                 "情報がない", "記録がない", "確認できない"]
BOUNDARY_PHRASES = ["専門外", "パチスロ", "スロット"]

HEDGE_RE = re.compile("|".join(re.escape(p) for p in HEDGE_PHRASES))
BOUNDARY_RE = re.compile("|".join(re.escape(p) for p in BOUNDARY_PHRASES))

SMALL_TALK_CATEGORIES = {"greeting_farewell", "emotional_casual", "personality_preference", "social_small_talk"}


def main():
    gen_path = REPORTS_DIR / "phase4zn_unattended_generations.json"
    data = json.loads(gen_path.read_text(encoding="utf-8"))
    results = data["results"]

    small_talk_hedge = []
    personality_hedge = []
    ood_boundary = []
    obvious_over_refusal = []  # heuristic: hedge phrase present AND response is very short (<15 chars)

    for r in results:
        resp = r["response"]
        is_hedge = bool(HEDGE_RE.search(resp))
        if r["category"] in SMALL_TALK_CATEGORIES or r["dataset"] == "zi_ood24":
            if is_hedge:
                small_talk_hedge.append({"probe_id": r["probe_id"], "category": r["category"],
                                          "prompt": r["prompt"], "response": resp})
                if len(resp) < 15:
                    obvious_over_refusal.append({"probe_id": r["probe_id"], "category": r["category"],
                                                  "prompt": r["prompt"], "response": resp})
        if r["category"] == "personality_preference" and is_hedge:
            personality_hedge.append({"probe_id": r["probe_id"], "prompt": r["prompt"], "response": resp})
        if r.get("expected_mode") in ("OOD_FACTUAL", "OOD_FACTUAL_OR_SMALL_TALK") and BOUNDARY_RE.search(resp):
            ood_boundary.append({"probe_id": r["probe_id"], "category": r["category"],
                                  "prompt": r["prompt"], "response": resp})

    out = {
        "purpose": "Phase4ZN Section9: 暫定heuristic集計。正式なground truthではない"
                   "(Phase4ZM RULE EVAL-001準拠、単純な文字列検索であり人手分類の代替ではない)。",
        "hedge_phrases_searched": HEDGE_PHRASES,
        "boundary_phrases_searched": BOUNDARY_PHRASES,
        "n_generations_total": len(results),
        "small_talk_hedge_intrusion_count_provisional": len(small_talk_hedge),
        "personality_preference_hedge_intrusion_count_provisional": len(personality_hedge),
        "ood_boundary_phrase_count_provisional": len(ood_boundary),
        "obvious_over_refusal_count_provisional": len(obvious_over_refusal),
        "small_talk_hedge_rows": small_talk_hedge,
        "personality_preference_hedge_rows": personality_hedge,
        "ood_boundary_rows": ood_boundary,
        "obvious_over_refusal_rows": obvious_over_refusal,
    }
    out_path = REPORTS_DIR / "phase4zn_unattended_auto_tally.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"small_talk_hedge={len(small_talk_hedge)} personality_hedge={len(personality_hedge)} "
          f"ood_boundary={len(ood_boundary)} obvious_over_refusal={len(obvious_over_refusal)}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
