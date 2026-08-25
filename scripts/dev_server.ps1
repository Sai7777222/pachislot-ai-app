# Phase 1: 開発サーバー起動 (D:\AI 配下のみで完結)
$ErrorActionPreference = "Stop"
$ProjectRoot = "D:\AI\projects\pachislot-ai-app"

Set-Location $ProjectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "venv not found. Run scripts/setup_local.ps1 first."
}

# host/port/reload は .env (API_HOST / API_PORT) を pachislot_ai.main が読み込む
& ".venv\Scripts\python.exe" -m pachislot_ai.main
