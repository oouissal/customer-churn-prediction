"""Health endpoint — used by orchestrators and Docker health checks."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.schemas import HealthResponse
from app.core.config import APP_VERSION
from app.services.model_service import ModelService, get_model_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health(model_service: ModelService = Depends(get_model_service)) -> HealthResponse:
    """Liveness probe: reports whether the API is up and the model loaded."""
    model_loaded = model_service.is_loaded()
    return HealthResponse(
        status="ok",
        api_version=APP_VERSION,
        model_loaded=model_loaded,
        model_version=model_service.model_version if model_loaded else None,
    )
