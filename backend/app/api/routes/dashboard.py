"""Dashboard endpoints — KPIs and chart data computed by the backend.

The frontend never runs ML logic itself: it renders whatever these endpoints
return. All numbers are computed from the model's predictions on the
held-out test set (see ``DashboardService``).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_dashboard_service
from app.services.dashboard_service import DashboardService

router = APIRouter(tags=["dashboard"])


@router.get(
    "/dashboard/summary",
    summary="Dashboard KPIs and distributions",
)
def dashboard_summary(
    dashboard_service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    """Totals, churn/risk/probability distributions and model version."""
    try:
        return dashboard_service.summary()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get(
    "/dashboard/drivers",
    summary="Top churn drivers (global SHAP importance)",
)
def dashboard_drivers(
    dashboard_service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    """Global SHAP feature importance computed on the hold-out test set."""
    try:
        return {
            "source": "Global SHAP importance — hold-out test set (first 400 rows)",
            "drivers": dashboard_service.top_drivers(),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
