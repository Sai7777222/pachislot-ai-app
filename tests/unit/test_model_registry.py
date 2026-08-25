"""Phase 3.5: LLMモデル切替レジストリの単体テスト。"""

from __future__ import annotations

import pytest

from pachislot_ai.core.config import Settings
from pachislot_ai.llm.model_registry import MODEL_REGISTRY, resolve_model_spec


def test_registry_has_both_models() -> None:
    assert "qwen" in MODEL_REGISTRY
    assert "swallow" in MODEL_REGISTRY


def test_resolve_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="Unknown LLM_MODEL_KEY"):
        resolve_model_spec("does_not_exist")


def test_qwen_and_swallow_have_independent_chat_format_and_paths() -> None:
    qwen = resolve_model_spec("qwen")
    swallow = resolve_model_spec("swallow")
    assert qwen.path != swallow.path
    # 両方とも GGUF 埋め込みテンプレートを使う設計 (どちらのテンプレートも流用していない)
    assert qwen.chat_format is None
    assert swallow.chat_format is None


def test_explicit_llm_model_path_overrides_registry() -> None:
    settings = Settings(llm_model_key="swallow", llm_model_path=r"D:\custom\model.gguf")
    assert str(settings.resolved_llm_model_path) == r"D:\custom\model.gguf"


def test_llm_model_key_resolves_when_path_not_set() -> None:
    settings = Settings(llm_model_key="swallow", llm_model_path=None)
    assert settings.resolved_llm_model_path == MODEL_REGISTRY["swallow"].path


def test_existing_qwen_env_behavior_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    """既存 .env のように LLM_MODEL_PATH を明示指定した場合、llm_model_key の値に
    関わらず常にその明示パスが使われる (Qwen運用を変更しないための保証)。
    """
    settings = Settings(
        llm_model_key="qwen",
        llm_model_path=r"D:\AI\models\llm\qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
    )
    assert "qwen2.5-14b-instruct" in str(settings.resolved_llm_model_path)
