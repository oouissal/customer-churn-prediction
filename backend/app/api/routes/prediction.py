"""Prediction endpoints: single and batch scoring."""

from __future__ import annotations

import io
import logging

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from src import config as src_config

from app.api.dependencies import get_explanation_service, get_prediction_service
from app.api.schemas import (
    BatchPredictionResponse,
    BatchPredictionRow,
    PredictionRequest,
    PredictionResponse,
    TopDriver,
)
from app.core.config import BATCH_PREDICTION_LIMIT, MAX_UPLOAD_MB
from app.services.explanation_service import ExplanationService
from app.services.model_service import ModelNotAvailableError
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict churn for a single customer",
)
def predict(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
    explanation_service: ExplanationService = Depends(get_explanation_service),
) -> PredictionResponse:
    """Score one customer profile and explain the prediction with SHAP."""
    try:
        result = prediction_service.predict(request.customer.to_features_dict())
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    drivers = explanation_service.explain(
        request.customer.to_features_dict()
    )
    return PredictionResponse(
        **result,
        top_drivers=[TopDriver(**driver) for driver in drivers],
        explanation=explanation_service.business_summary(
            drivers, result["probability"]
        ),
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict churn for a batch of customers (CSV upload)",
)
def predict_batch(
    file: UploadFile = File(..., description="CSV with one customer per row"),
    format: str = Query("json", pattern="^(json|csv)$"),
    prediction_service: PredictionService = Depends(get_prediction_service),
) -> BatchPredictionResponse:
    """Score every row of an uploaded CSV and return the enriched results.

    The CSV must contain the 19 raw feature columns (``customerID`` is
    optional and carried through to the results). Use ``?format=csv`` to get
    a downloadable CSV response instead of JSON.
    """
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted for batch prediction.",
        )
    content = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not parse the CSV file: {exc}"
        ) from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="The CSV contains no rows.")
    if len(df) > BATCH_PREDICTION_LIMIT:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Batch too large: {len(df)} rows. "
                f"Limit is {BATCH_PREDICTION_LIMIT} rows per request."
            ),
        )

    missing = [
        column for column in src_config.RAW_FEATURES if column not in df.columns
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {', '.join(missing)}.",
        )

    try:
        results = prediction_service.predict_dataframe(df)
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    rows = [BatchPredictionRow(**row) for row in results]
    threshold = prediction_service.model_service.threshold
    response = BatchPredictionResponse(
        total=len(rows),
        model_version=prediction_service.model_service.model_version,
        threshold=threshold,
        results=rows,
    )

    if format == "csv":
        from fastapi.responses import Response

        csv_frame = pd.DataFrame([row.model_dump() for row in rows])
        return Response(
            content=csv_frame.to_csv(index=False),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )
    return response
