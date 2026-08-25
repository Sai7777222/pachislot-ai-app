from __future__ import annotations

from pathlib import Path

from pachislot_ai.core.config import SYSTEM_PROMPT_PATH, Settings, get_settings


def test_settings_loads_from_env() -> None:
    settings = get_settings()
    assert settings.llm_provider == "local_llama_cpp"
    assert isinstance(settings.llm_model_path, Path)
    assert settings.llm_context_size > 0
    assert settings.api_port > 0


def test_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_system_prompt_file_exists() -> None:
    assert SYSTEM_PROMPT_PATH.is_file()


def test_settings_paths_under_d_ai() -> None:
    settings = Settings()
    for path in (settings.models_dir, settings.data_dir, settings.cache_dir):
        assert str(path).lower().startswith(r"d:\ai")
