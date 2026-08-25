from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class LLMHealthInfo(BaseModel):
    provider: str
    model: str
    loaded: bool
    gpu_offload_supported: bool | None = None
    n_gpu_layers: int | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    llm: LLMHealthInfo
