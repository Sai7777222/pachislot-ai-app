"""FastAPI 依存性注入（DI）。

LLM モデルはアプリ起動時に一度だけロードし `app.state` に保持する
（リクエストごとの再ロードは重すぎるため）。
"""

from __future__ import annotations

from fastapi import Request

from pachislot_ai.core.config import Settings
from pachislot_ai.services.chat_service import ChatService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service
