from __future__ import annotations

from fastapi import APIRouter, Depends

from pachislot_ai.api.deps import get_chat_service
from pachislot_ai.api.schemas.health import HealthResponse, LLMHealthInfo
from pachislot_ai.services.chat_service import ChatService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def get_health(
    service: ChatService = Depends(get_chat_service),
) -> HealthResponse:
    loaded = await service.health_check()

    llm_provider = service.llm
    llm_info = LLMHealthInfo(
        provider=type(llm_provider).__name__,
        model=service.model_name,
        loaded=loaded,
        gpu_offload_supported=getattr(llm_provider, "gpu_offload_supported", None),
        n_gpu_layers=getattr(llm_provider, "n_gpu_layers", None),
    )
    return HealthResponse(status="ok" if loaded else "degraded", llm=llm_info)
