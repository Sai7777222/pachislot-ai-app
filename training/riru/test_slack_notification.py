"""Slack通知機能の単体動作確認スクリプト。

train_qlora.py の `send_slack_notification` / `notify_training_success` /
`notify_training_failure` を、モデル・データセット・学習処理に一切触れずに
直接呼び出して動作確認するためのスタンドアロンスクリプト。

事前準備:
  - プロジェクトルートの .env に SLACK_WEBHOOK_URL=https://hooks.slack.com/services/... を
    設定しておくこと (このスクリプト・train_qlora.py ともに、URLをソースコードに
    直接書くことはない)。
  - .venv-qlora に python-dotenv がインストール済みであること
    (`./.venv-qlora/Scripts/python.exe -m pip install python-dotenv` 済み)。

実行方法 (プロジェクトルートから, .venv-qlora を使用):
    ./.venv-qlora/Scripts/python.exe training/riru/test_slack_notification.py

このスクリプトはQLoRA本学習を一切開始しない。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_qlora as tq  # noqa: E402


def main() -> int:
    webhook_configured = bool(os.environ.get("SLACK_WEBHOOK_URL", "").strip())
    print(f".env読み込み元: {tq.ENV_PATH}")
    print(f"SLACK_WEBHOOK_URL 設定状況: {'設定済み' if webhook_configured else '未設定'}")
    if not webhook_configured:
        print(
            "SLACK_WEBHOOK_URL が未設定です。プロジェクトルートの .env に\n"
            "SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...\n"
            "を設定してから再実行してください。\n"
            "(未設定のままでも send_slack_notification() は例外を出さずFalseを返すのみで、\n"
            "本学習フローには影響しません。ここでは通知の実送信を確認するため停止します。)"
        )
        return 1

    print("\n--- 成功通知テスト (notify_training_success) ---")
    tq.notify_training_success(
        duration_sec=123.4,
        epoch=3.0,
        step=279,
        train_loss=1.234,
        eval_loss=1.567,
    )

    print("\n--- 失敗通知テスト (notify_training_failure) ---")
    tq.notify_training_failure(
        "RuntimeError: これはtest_slack_notification.pyによるテスト通知です。"
    )

    print(
        "\n両方の送信呼び出しが完了しました (成否はログ上の "
        "'Slack通知を送信しました' / 'Slack通知の送信に失敗しました' を確認してください)。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
