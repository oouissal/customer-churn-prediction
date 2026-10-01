"""Model information endpoint — metadata and metrics of the deployed model."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import ModelInfoResponse, ModelMetrics
from app.services.model_service import (
    ModelNotAvailableError,
    ModelService,
    get_model_service,
)

router = APIRouter(tags=["model"])


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Deployed model information",
)
def model_info(model_service: ModelService = Depends(get_model_service)) -> ModelInfoResponse:
    """Model name, version, training date, threshold and test metrics."""
    try:
        model_service.load()
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    metrics = model_service.metrics
    test_metrics = metrics.get("test_metrics_tuned_threshold", {})
    return ModelInfoResponse(
        model_name=model_service.model_name,
        model_version=model_service.model_version,
        trained_at=model_service.artifact.get("trained_at"),
        threshold=model_service.threshold,
        feature_count=model_service.feature_count(),
        metrics=ModelMetrics(
            threshold=float(model_service.threshold),
            accuracy=float(test_metrics.get("accuracy", 0.0)),
            precision=float(test_metrics.get("precision", 0.0)),
            recall=float(test_metrics.get("recall", 0.0)),
            f1=float(test_metrics.get("f1", 0.0)),
            roc_auc=float(test_metrics.get("roc_auc", 0.0)),
        ),
        mlflow_run_id=model_service.artifact.get("mlflow_run_id"),
        mlflow_registered_model=(
            model_service.artifact.get("mlflow_registered_model")
            or metrics.get("mlflow_registered_model")
        ),
    )
