# Phase 0: ローカル開発環境セットアップ（D:\AI 配下のみ）
$ErrorActionPreference = "Stop"
$ProjectRoot = "D:\AI\projects\pachislot-ai-app"

Set-Location $ProjectRoot

# ディレクトリ作成
$dirs = @(
    "D:\AI\models\llm",
    "D:\AI\cache\huggingface",
    "D:\AI\data\raw",
    "D:\AI\data\processed"
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

# 仮想環境
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& ".venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip

# llama-cpp-python CUDA 版（セッション内環境変数のみ）
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# その他依存関係
pip install -e ".[dev]"

Write-Host "Setup complete. Run scripts/smoke_test_llm.py after model download."
