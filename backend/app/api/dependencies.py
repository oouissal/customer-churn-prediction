"""FastAPI dependency injection: process-wide service singletons."""

from __future__ import annotations

from functools import lru_cache

from app.services.dashboard_service import DashboardService
from app.services.explanation_service import ExplanationService
from app.services.model_service import get_model_service
from app.services.prediction_service import PredictionService


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """Singleton prediction service bound to the shared model service."""
    return PredictionService(get_model_service())


@lru_cache(maxsize=1)
def get_explanation_service() -> ExplanationService:
    """Singleton explanation service bound to the shared model service."""
    return ExplanationService(get_model_service())


@lru_cache(maxsize=1)
def get_dashboard_service() -> DashboardService:
    """Singleton dashboard service."""
    return DashboardService(get_model_service(), get_prediction_service())
