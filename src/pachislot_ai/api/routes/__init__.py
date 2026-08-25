"""API v1 ルーター集約。"""

from fastapi import APIRouter

from pachislot_ai.api.routes import chat, health

api_router = APIRouter(prefix="/v1")
api_router.include_router(health.router, tags=["health"])
api_router.include_router(chat.router, tags=["chat"])
