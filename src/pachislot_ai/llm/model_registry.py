"""ローカルLLMの切替レジストリ

(Phase 3.5: Qwen/Swallow A/B比較、Phase 3.6: LLM-jp追加、Phase 3.7: ELYZA追加、
Phase 3.8: ELYZA改善プロンプト再テスト)。

`LLM_MODEL_KEY` (.env) で選択する。`LLM_MODEL_PATH` を明示的に設定した場合は
そちらが優先される (既存Qwen運用を一切変更しないため)。

chat_format=None は「GGUFに埋め込まれたテンプレートを使用する」を意味する。
各モデル用のテンプレートを他モデルに流用しないよう、各モデルは独立して定義する。

system_prompt_path も同様にモデルごとに独立させている。Swallowは再評価テストで
ハルシネーション・比較計算ミスが確認されたため、より厳格な制約を追加した
`system_swallow.jinja2` を使う。Qwen (`system.jinja2`) は変更しない。LLM-jpは
新規候補としての一次比較のため、Qwenと同一のベースラインsystem prompt
(`system.jinja2`) を使う。ELYZAは一次比較(Phase 3.7)でQ6のゾーン間数値混同・
Q7の出典ラベル捏造が確認されたため、Phase 3.8で「対象間の数値転用禁止」
「出典ラベルの自己創作禁止」等を明示した `system_elyza.jinja2` に切り替えた。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# src/pachislot_ai/llm/model_registry.py -> プロジェクトルート
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PROMPTS_DIR = _PROJECT_ROOT / "config" / "prompts"

QWEN_SYSTEM_PROMPT_PATH = _PROMPTS_DIR / "system.jinja2"
SWALLOW_SYSTEM_PROMPT_PATH = _PROMPTS_DIR / "system_swallow.jinja2"
ELYZA_SYSTEM_PROMPT_PATH = _PROMPTS_DIR / "system_elyza.jinja2"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    display_name: str
    path: Path
    chat_format: str | None  # None = GGUF埋め込みテンプレートを自動使用
    license_summary: str
    system_prompt_path: Path


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen": ModelSpec(
        key="qwen",
        display_name="Qwen2.5-14B-Instruct (Q4_K_M)",
        path=Path(r"D:\AI\models\llm\qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"),
        chat_format=None,
        license_summary="Apache License 2.0 (Qwen/Qwen2.5-14B-Instruct-GGUF)",
        system_prompt_path=QWEN_SYSTEM_PROMPT_PATH,
    ),
    "swallow": ModelSpec(
        key="swallow",
        display_name="Llama-3.1-Swallow-8B-Instruct-v0.5 (Q4_K_M)",
        path=Path(r"D:\AI\models\llm\swallow\Llama-3.1-Swallow-8B-Instruct-v0.5_Q4_K_M.gguf"),
        chat_format=None,
        license_summary=(
            "デュアルライセンス: Meta Llama 3.1 Community License "
            "+ Google Gemma Terms of Use"
        ),
        system_prompt_path=SWALLOW_SYSTEM_PROMPT_PATH,
    ),
    "llm-jp": ModelSpec(
        key="llm-jp",
        display_name="LLM-jp-3-13B-Instruct (Q4_K_M)",
        path=Path(r"D:\AI\models\llm\llm-jp-3-13b\llm-jp-3-13b-instruct-Q4_K_M.gguf"),
        chat_format=None,
        license_summary="Apache License 2.0 (llm-jp/llm-jp-3-13b-instruct)",
        system_prompt_path=QWEN_SYSTEM_PROMPT_PATH,
    ),
    "elyza": ModelSpec(
        key="elyza",
        display_name="Llama-3-ELYZA-JP-8B (Q4_K_M)",
        path=Path(r"D:\AI\models\llm\elyza\Llama-3-ELYZA-JP-8B-q4_k_m.gguf"),
        chat_format=None,
        license_summary="Meta Llama 3 Community License (elyza/Llama-3-ELYZA-JP-8B)",
        system_prompt_path=ELYZA_SYSTEM_PROMPT_PATH,
    ),
}


def resolve_model_spec(model_key: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown LLM_MODEL_KEY={model_key!r}. Known keys: {known}") from exc
