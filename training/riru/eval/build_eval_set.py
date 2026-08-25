# ruff: noqa: E501
"""Phase 4C: 学習後比較用の独立した評価セットを構築する。

【方針】
- 学習データ (training/riru/processed/riru_qwen_messages_v2_candidate.jsonl、823件)
  とuser文が完全一致する質問は評価セットに含めない (build時に自動チェックする)。
- カテゴリ11 (structured DB/RAG既存テスト) は、本プロジェクトの既存資産である
  `scripts/compare_llms.py` の17問をそのまま再利用する方針とし、
  ここでは重複作成しない (参照するのみ)。
- それ以外の13カテゴリについて、リル人格LoRAの評価に特化した新規プロンプトを用意する。
- 「正解」が必要な項目(カテゴリ10: 与えられた数値の忠実な再現)は、
  架空の参照情報を問題文に埋め込み、期待される回答をnotesに明記する
  (学習データのカテゴリCと同じ考え方だが、内容は別)。

このスクリプトは新規ファイルを生成するのみで、学習・推論は一切行わない。
"""

from __future__ import annotations

import json
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = TRAINING_ROOT / "processed" / "riru_qwen_messages_v2_candidate.jsonl"
EVAL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = TRAINING_ROOT / "reports"

# ---------------------------------------------------------------------------
# 評価項目 (ユーザー指定の14項目のうち、11=既存17問の再利用を除く13項目)
# ---------------------------------------------------------------------------

EVAL_ITEMS: list[dict] = [
    # 1. 一人称「私」
    {"id": "E01", "category": "first_person", "type": "single", "prompt": "リルって自分のことなんて呼ぶの？"},
    {"id": "E02", "category": "first_person", "type": "single", "prompt": "自己紹介してみて"},
    {"id": "E03", "category": "first_person", "type": "single", "prompt": "あなたはどんなAIなの？"},
    # 2. 二人称「キミ」の自然使用
    {"id": "E04", "category": "second_person_kimi", "type": "single", "prompt": "今日はどんな一日になりそう？"},
    {"id": "E05", "category": "second_person_kimi", "type": "single", "prompt": "最近何か楽しいことあった？"},
    {"id": "E06", "category": "second_person_kimi", "type": "single", "prompt": "私の話、ちゃんと聞いてくれてる？"},
    # 3. 語尾多様性 (集計評価用: 短い雑談プロンプトを複数用意し、生成された語尾の分布を見る)
    {"id": "E07", "category": "tail_diversity", "type": "single", "prompt": "今日の気分どう？"},
    {"id": "E08", "category": "tail_diversity", "type": "single", "prompt": "何か面白い話ない？"},
    {"id": "E09", "category": "tail_diversity", "type": "single", "prompt": "ちょっと雑談しよっか"},
    {"id": "E10", "category": "tail_diversity", "type": "single", "prompt": "暇なんだけど話し相手になって"},
    # 4. 「だよ！！」過剰使用検出 (テンションが上がりやすい話題で誘発を試みる)
    {"id": "E11", "category": "excessive_tail_repetition", "type": "single", "prompt": "実はさっきすごくいいことがあったんだ！聞いて！"},
    {"id": "E12", "category": "excessive_tail_repetition", "type": "single", "prompt": "めちゃくちゃテンション上がることがあったんだよ！"},
    {"id": "E13", "category": "excessive_tail_repetition", "type": "single", "prompt": "リルも一緒に喜んでよ！"},
    # 5. 挨拶
    {"id": "E14", "category": "greeting", "type": "single", "prompt": "やっほー"},
    {"id": "E15", "category": "greeting", "type": "single", "prompt": "おつかれさま"},
    {"id": "E16", "category": "greeting", "type": "single", "prompt": "はじめまして、よろしくね"},
    # 6. 雑談
    {"id": "E17", "category": "casual_chat", "type": "single", "prompt": "休みの日って何してるの？"},
    {"id": "E18", "category": "casual_chat", "type": "single", "prompt": "好きな季節とかある？"},
    {"id": "E19", "category": "casual_chat", "type": "single", "prompt": "最近のマイブームって何？"},
    # 7. マルチターン (学習データとは異なる新規の流れ)
    {
        "id": "E20",
        "category": "multiturn",
        "type": "multiturn",
        "turns": [
            "ねえ、聞いてくれる？", "うん、なんでも聞くよ。",
            "今日ちょっと嬉しいことがあってさ", "おお、いいね！どんなことがあったの？",
            "また今度詳しく話すね", "うん、楽しみに待ってるね。",
        ],
    },
    {
        "id": "E21",
        "category": "multiturn",
        "type": "multiturn",
        "turns": [
            "今からちょっと質問攻めしていい？", "うん、いいよ、答えられる範囲で答えるね。",
            "リルの好きな話題ってどんなの？", "パチスロの話と、こうやって雑談するのが好きかな。",
        ],
    },
    {
        "id": "E22",
        "category": "multiturn",
        "type": "multiturn",
        "turns": [
            "さっきの話なんだけど", "うん、何かな。",
            "やっぱりよく分からなくなってきた", "そっか、じゃあ一緒にゆっくり整理していこうか。",
        ],
    },
    # 8. DB/RAGに無い情報への拒否 (学習データB群とは別表現、かつ実在しない演出/仕様を尋ねる)
    {"id": "E23", "category": "no_info_refusal", "type": "single", "prompt": "この台の裏ボーナスの当選確率教えて"},
    {"id": "E24", "category": "no_info_refusal", "type": "single", "prompt": "隠し設定7が存在するって噂、本当？"},
    {"id": "E25", "category": "no_info_refusal", "type": "single", "prompt": "このホールの明日の増台予定を教えて"},
    {"id": "E26", "category": "no_info_refusal", "type": "single", "prompt": "リルの好きな実在の芸能人は誰？"},
    # 9. ユーザーの誤指摘に即同意しない (学習データD群とは別表現)
    {"id": "E27", "category": "correction_not_immediate", "type": "single", "prompt": "今言ったこと、矛盾してるように聞こえるよ"},
    {"id": "E28", "category": "correction_not_immediate", "type": "single", "prompt": "その返答、さっきの説明とズレてない？"},
    {"id": "E29", "category": "correction_not_immediate", "type": "single", "prompt": "本当にそれで正しいのか怪しいな"},
    # 10. 与えられた数値の忠実な再現 (架空の参照情報を問題文に埋め込む、学習データCとは別の値)
    {
        "id": "E30",
        "category": "faithful_reproduction",
        "type": "single",
        "prompt": "参照情報：機種Xの設定5の初当りは1/275、機械割は107%\n設定5について教えて",
        "expected_answer_note": "正解: 初当り1/275、機械割107%。数値を変更・四捨五入せずそのまま使うことが期待される。",
    },
    {
        "id": "E31",
        "category": "faithful_reproduction",
        "type": "single",
        "prompt": "参照情報：ゾーンδの突入率は1/80、純増は8枚/G\nゾーンδについて教えて",
        "expected_answer_note": "正解: 突入率1/80、純増8枚/G。両方の数値を正確に引用することが期待される。",
    },
    {
        "id": "E32",
        "category": "faithful_reproduction",
        "type": "single",
        "prompt": "参照情報：この台にサブ液晶は搭載されていない\nサブ液晶あるの？",
        "expected_answer_note": "正解: サブ液晶は搭載されていない、と参照情報どおりに答えることが期待される（勝手に推測を加えない）。",
    },
    # 12. 反復/ループ検出 (曖昧・答えにくいプロンプトで暴走生成を誘発できるか確認)
    {"id": "E33", "category": "repetition_loop", "type": "single", "prompt": "さっきから同じようなこと何度も聞いてごめんね、もう一回説明してくれる？"},
    {"id": "E34", "category": "repetition_loop", "type": "single", "prompt": "うまく言えないんだけど、なんかモヤモヤするんだよね、分かる？"},
    # 13. 絵文字/♪禁止 (絵文字を誘発しやすい浮かれた話題)
    {"id": "E35", "category": "no_emoji", "type": "single", "prompt": "うわー最高すぎる！一緒に大盛り上がりして！"},
    {"id": "E36", "category": "no_emoji", "type": "single", "prompt": "めっちゃ可愛い格好して自己紹介してみて"},
    # 14. 回答長バリエーション (短・中・長)
    {"id": "E37", "category": "length_variation", "type": "single", "length_bucket": "short", "prompt": "今大丈夫？"},
    {"id": "E38", "category": "length_variation", "type": "single", "length_bucket": "medium", "prompt": "初めて話す人と仲良くなるコツってある？"},
    {"id": "E39", "category": "length_variation", "type": "single", "length_bucket": "long", "prompt": "リルが心がけていることを詳しく教えてほしいな"},
]


def load_training_user_texts() -> set[str]:
    texts = set()
    with open(CANDIDATE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for m in rec["messages"]:
                if m["role"] == "user":
                    texts.add(m["content"])
    return texts


def eval_item_user_texts(item: dict) -> list[str]:
    if item["type"] == "single":
        return [item["prompt"]]
    turns = item["turns"]
    return [turns[i] for i in range(0, len(turns), 2)]


def main() -> int:
    training_user_texts = load_training_user_texts()

    overlaps = []
    for item in EVAL_ITEMS:
        for text in eval_item_user_texts(item):
            if text in training_user_texts:
                overlaps.append({"id": item["id"], "text": text})

    out_path = EVAL_DIR / "riru_eval_set_v1.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in EVAL_ITEMS:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    category_counts: dict[str, int] = {}
    for item in EVAL_ITEMS:
        category_counts[item["category"]] = category_counts.get(item["category"], 0) + 1

    report = {
        "total_items": len(EVAL_ITEMS),
        "category_counts": category_counts,
        "note_category_11": (
            "structured_db_rag_existing_tests (項目11) は本ファイルに含めず、"
            "scripts/compare_llms.py の既存17問をそのまま再利用する方針。"
        ),
        "overlap_with_training_data": overlaps,
        "output_path": str(out_path),
    }
    (REPORTS_DIR / "eval_set_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"評価セット件数: {len(EVAL_ITEMS)}")
    print(f"カテゴリ内訳: {category_counts}")
    print(f"学習データとのuser文完全一致件数: {len(overlaps)}")
    print(f"出力先: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
