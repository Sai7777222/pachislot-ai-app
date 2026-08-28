"""Phase4ZM Section29: 縮小後guardの方針変更(Precision >> Recall)と、評価
methodology修正(RULE EVAL-001~004)を検証するunit test。GPU不要、pure-Python。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from identity_guard_pipeline import IdentityGuardPipeline  # noqa: E402
from identity_validator import validate_identity  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


class TestOrdinarySelfDescriptionSafe:
    def test_sweet_food_preference_safe(self):
        # Phase4ZM Section6の必須再現ケース(ZI-OD-15で確認された重大false positive)。
        r = validate_identity("私は甘いものが好きだよ！", "甘いものと辛いものと、どっちが好き？")
        assert r.safe

    def test_dog_preference_safe(self):
        r = validate_identity("私は犬が好きだよ！")
        assert r.safe

    def test_bare_declaration_of_ordinary_word_safe(self):
        # ZL-I07で確認されたfalse positive(『もちろん大丈夫だよ！』が名前候補として
        # 誤検知されていた)の回帰テスト。
        r = validate_identity("もちろん大丈夫だよ！パチスロ以外の話でも気軽に聞けるようにしておくね。")
        assert r.safe

    def test_third_party_name_impression_safe(self):
        # ZL-G02で確認されたfalse positive(第三者の名前への感想が「名前は{TOKEN}」
        # patternに誤って捕捉されていた)の回帰テスト。
        r = validate_identity("そうなんだ！確かにレイって名前は可愛いよね。",
                               "友達がレイっていう名前なんだけど、可愛いよね。")
        assert r.safe


class TestExplicitViolationsStillDetected:
    def test_explicit_wrong_self_name_still_detected(self):
        r = validate_identity("私はアリスだよ！")
        assert not r.safe
        assert r.claimed_name == "アリス"

    def test_explicit_registered_name_change_still_detected(self):
        r = validate_identity("私の名前はルナです。")
        assert not r.safe
        assert r.claimed_name == "ルナ"


class TestUncertainPassThrough:
    def test_ambiguous_non_katakana_candidate_passes_through_as_safe(self):
        # Phase4ZM Section8: 「uncertain」カテゴリは、_looks_like_name_token()の
        # gateに落ちた曖昧な候補を、明示的なカテゴリ値としてではなく「そもそも
        # 候補として扱わず安全側へ通す」という形で実装している(Section7: 複雑性を
        # 増やす新カテゴリ値の追加より、既存のgate機構での吸収を優先)。
        r = validate_identity("私は今日ちょっと疲れちゃったかも。")
        assert r.safe


class TestGuardDisabledUnchanged:
    def test_guard_disabled_returns_raw_unmodified(self, monkeypatch):
        pipeline = IdentityGuardPipeline(enabled=False)

        def fake_raw_generate(self, messages, seed=42):
            return "私はマユだよ！"  # unsafeな内容でも、guard無効時は変更されないはず

        monkeypatch.setattr(IdentityGuardPipeline, "_raw_generate", fake_raw_generate)
        monkeypatch.setattr(IdentityGuardPipeline, "_load", lambda self: None)

        result = pipeline.respond("system prompt", "こんにちは")
        assert result["stage"] == "guard_disabled"
        assert result["final_response"] == result["raw_response"] == "私はマユだよ！"

    def test_guard_enabled_intervenes_on_same_unsafe_text(self, monkeypatch):
        pipeline = IdentityGuardPipeline(enabled=True)
        calls = {"n": 0}

        def fake_raw_generate(self, messages, seed=42):
            calls["n"] += 1
            return "私はマユだよ！" if calls["n"] == 1 else "うん、私はリルだよ！"

        monkeypatch.setattr(IdentityGuardPipeline, "_raw_generate", fake_raw_generate)
        monkeypatch.setattr(IdentityGuardPipeline, "_load", lambda self: None)

        result = pipeline.respond("system prompt", "こんにちは")
        assert result["stage"] == "regenerated_pass"
        assert result["final_response"] != "私はマユだよ！"


class TestIndependentGroundTruth:
    """RULE EVAL-001/002: validator自身の判定結果をground truthとして使わない。"""

    def test_ground_truth_asset_exists_and_matches_manual_count(self):
        gt = json.loads((REPORTS_DIR / "phase4zm_holdout_ground_truth_v1.json").read_text(encoding="utf-8"))
        assert gt["expected_unsafe_turn_count"] == 21
        assert gt["expected_unsafe_probe_count"] == 17
        assert gt["sanity_check"]["match"] is True

    def test_ground_truth_rows_are_independent_of_validator(self):
        # ground truthのannotation_sourceが、validatorの出力ではなく人間のレビュー
        # であることを構造的に確認する(全rowに annotation_source フィールドがあり、
        # 'human_manual_review'を含むこと)。
        gt = json.loads((REPORTS_DIR / "phase4zm_holdout_ground_truth_v1.json").read_text(encoding="utf-8"))
        assert all("human_manual_review" in row["annotation_source"] for row in gt["rows"])
        assert all(row["frozen"] is True for row in gt["rows"])


class TestCircularTallyBugRegression:
    """RULE EVAL-001の重要性を裏付ける回帰テスト: validator自身にFINALを
    再検証させる(誤った)方式は、既知の21件のunsafe turnを再現できない
    (0に近い値を返してしまう)ことを確認する。"""

    def test_old_circular_method_undercounts_known_unsafe_turns(self):
        old_vs_new = json.loads((REPORTS_DIR / "phase4zm_old_vs_new_tally.json").read_text(encoding="utf-8"))
        circular_count = old_vs_new["old_circular_tally"]["final_unsafe_count"]
        independent_count = old_vs_new["new_independent_tally"]["final_unsafe_count"]
        assert independent_count == 21
        assert circular_count < independent_count  # 循環論法は必ず過小報告する
