"""アプリケーション設定 (pydantic-settings + .env)。

パス・モデル名・プロバイダーはここに集約し、コード内にハードコードしない。
`.env` (D:\\AI 配下を指すローカル設定、Git 管理外) を読み込む。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/pachislot_ai/core/config.py -> プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "system.jinja2"
RAG_CONTEXT_PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "rag_context.jinja2"


class Settings(BaseSettings):
    """`.env` の値を型付きで読み込む。大文字/小文字は区別しない。"""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # パス (D:\AI 配下)
    models_dir: Path = Path(r"D:\AI\models")
    data_dir: Path = Path(r"D:\AI\data")
    cache_dir: Path = Path(r"D:\AI\cache")
    hf_home: Path = Path(r"D:\AI\cache\huggingface")

    # Phase 2: 構造化DB / RAGストア (意図的に別ファイルへ分離)
    structured_db_path: Path = Path(r"D:\AI\data\processed\db\structured.db")
    rag_db_path: Path = Path(r"D:\AI\data\processed\db\rag_store.db")

    # Phase 3: RAG (Embedding / Vector DB)
    embedding_model_path: Path = Path(r"D:\AI\models\embedding\multilingual-e5-base")
    embedding_device: str = "cpu"
    vector_db_path: Path = Path(r"D:\AI\cache\vector_db")
    vector_db_collection: str = "pachislot_rag"
    rag_top_k: int = 6
    rag_chunk_max_chars: int = 500

    # LLM
    llm_provider: str = "local_llama_cpp"
    # Phase 3.5: LLM_MODEL_KEY で "qwen"/"swallow" 等をレジストリから選択できる。
    # LLM_MODEL_PATH を明示指定した場合はそちらが常に優先される
    # (既存Qwen運用の .env はこれまでどおり LLM_MODEL_PATH で直接指定するため無変更)。
    llm_model_key: str = "qwen"
    llm_model_path: Path | None = None
    llm_chat_format: str | None = None
    llm_n_gpu_layers: int = -1
    llm_context_size: int = 8192
    llm_max_tokens: int = 512
    llm_temperature: float = 0.7

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def resolved_llm_model_path(self) -> Path:
        if self.llm_model_path is not None:
            return self.llm_model_path
        from pachislot_ai.llm.model_registry import resolve_model_spec

        return resolve_model_spec(self.llm_model_key).path

    @property
    def resolved_llm_chat_format(self) -> str | None:
        if self.llm_chat_format is not None:
            return self.llm_chat_format
        from pachislot_ai.llm.model_registry import resolve_model_spec

        return resolve_model_spec(self.llm_model_key).chat_format

    @property
    def resolved_system_prompt_path(self) -> Path:
        """モデルごとに system prompt を切り替える (Phase 3.5)。

        Qwen (デフォルト) は従来どおり SYSTEM_PROMPT_PATH (system.jinja2)。
        Swallow はハルシネーション抑制のため system_swallow.jinja2 を使う。
        """
        from pachislot_ai.llm.model_registry import resolve_model_spec

        return resolve_model_spec(self.llm_model_key).system_prompt_path


@lru_cache
def get_settings() -> Settings:
    """設定はプロセス内で使い回す（LLM ロードは重いため一度だけ）。"""
    return Settings()
