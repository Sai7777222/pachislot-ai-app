"""FastAPI エントリポイント。

LLM モデルはアプリ起動時 (lifespan) に一度だけロードし、`app.state` に保持する。
リクエストごとのロードは行わない（GPU/VRAM を無駄に消費するため）。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from pachislot_ai.api.routes import api_router
from pachislot_ai.core.config import RAG_CONTEXT_PROMPT_PATH, get_settings
from pachislot_ai.core.exceptions import LLMGenerationError, PachislotAIError
from pachislot_ai.core.logging import setup_logging
from pachislot_ai.data.db import create_structured_engine
from pachislot_ai.llm.factory import create_llm_provider
from pachislot_ai.rag.embedder import Embedder
from pachislot_ai.rag.pipeline import RagPipeline
from pachislot_ai.rag.retriever import Retriever
from pachislot_ai.rag.vector_store import VectorStore
from pachislot_ai.services.chat_service import ChatService

logger = logging.getLogger(__name__)


def _try_create_rag_pipeline(settings) -> RagPipeline | None:  # noqa: ANN001
    """RAGコンポーネントの初期化に失敗しても、LLM単体でのチャットは継続できるようにする。

    Embedding モデル・Vector DB はいずれもローカルのみを参照し、
    ネットワークアクセスは行わない。
    """
    try:
        if not settings.structured_db_path.is_file():
            logger.warning(
                "structured.db not found (%s); RAG disabled.", settings.structured_db_path
            )
            return None
        embedder = Embedder(settings.embedding_model_path, device=settings.embedding_device)
        vector_store = VectorStore(settings.vector_db_path, settings.vector_db_collection)
        if vector_store.count() == 0:
            logger.warning(
                "Vector DB collection is empty (%s); RAG will run with no retrievable chunks "
                "until scripts/build_index.py is run.",
                settings.vector_db_path,
            )
        retriever = Retriever(embedder, vector_store, default_top_k=settings.rag_top_k)
        structured_engine = create_structured_engine(settings.structured_db_path)
        return RagPipeline(
            retriever,
            structured_engine,
            RAG_CONTEXT_PROMPT_PATH,
            top_k=settings.rag_top_k,
        )
    except Exception:  # noqa: BLE001 - RAG初期化失敗はチャット全体を止めない
        logger.exception("Failed to initialize RAG pipeline; continuing without RAG.")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    settings = get_settings()

    logger.info("Starting up: initializing LLM provider (%s)...", settings.llm_provider)
    llm_provider = create_llm_provider(settings)

    logger.info("Initializing RAG pipeline (Embedding + Vector DB)...")
    rag_pipeline = _try_create_rag_pipeline(settings)
    logger.info("RAG pipeline %s.", "enabled" if rag_pipeline is not None else "disabled")

    chat_service = ChatService(llm_provider, settings.resolved_system_prompt_path, rag_pipeline)

    app.state.settings = settings
    app.state.chat_service = chat_service
    logger.info(
        "Startup complete. model=%s rag_enabled=%s",
        chat_service.model_name,
        chat_service.rag_enabled,
    )

    yield

    logger.info("Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Pachislot AI API",
        description="パチスロ情報AIアプリ バックエンド (Phase 1: 最小 API + LLM 接続)",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(PachislotAIError)
    async def handle_app_error(request, exc: PachislotAIError) -> JSONResponse:  # noqa: ANN001, ARG001
        status_code = 503 if isinstance(exc, LLMGenerationError) else 500
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    app.include_router(api_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "pachislot_ai.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
