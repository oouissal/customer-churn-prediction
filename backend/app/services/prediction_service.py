"""Prediction service: single and batch inference on the deployed model.

All ML logic lives here (and in ``src``). The API layer only validates input
and forwards it to this service. The risk level is a business rule on top of
the raw probability, independent from the decision threshold used for the
binary CHURN / NO CHURN label.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from src.predict import predict_customer

from app.core import config
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)


class PredictionService:
    """Wraps the trained pipeline for single and batch predictions."""

    def __init__(self, model_service: ModelService) -> None:
        self.model_service = model_service

    # ------------------------------------------------------------------
    # Business rules
    # ------------------------------------------------------------------
    @staticmethod
    def risk_level(probability: float) -> str:
        """Map a churn probability to a business risk band.

        * ``LOW``    : probability < 0.40
        * ``MEDIUM`` : 0.40 <= probability <= 0.70
        * ``HIGH``   : probability > 0.70
        """
        if probability >= config.RISK_HIGH_ABOVE:
            return "HIGH"
        if probability >= config.RISK_LOW_BELOW:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def label(probability: float, threshold: float) -> str:
        """Binary label at the tuned decision threshold."""
        return "CHURN" if probability >= threshold else "NO_CHURN"

    @staticmethod
    def confidence(probability: float) -> float:
        """Distance from the decision boundary (max of p and 1-p)."""
        return max(probability, 1.0 - probability)

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------
    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Score one customer profile.

        Returns the raw prediction plus probability, risk level, confidence
        and the model version, ready to be shaped into the API response.
        """
        threshold = self.model_service.threshold
        raw = predict_customer(
            features,
            pipeline=self.model_service.pipeline,
            threshold=threshold,
        )
        probability = float(raw["churn_probability"])
        return {
            "prediction": self.label(probability, threshold),
            "probability": probability,
            "risk_level": self.risk_level(probability),
            "confidence": self.confidence(probability),
            "threshold": threshold,
            "model_version": self.model_service.model_version,
        }

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------
    def predict_dataframe(
        self, df: pd.DataFrame, customer_id_column: str | None = None
    ) -> list[dict[str, Any]]:
        """Score every row of a DataFrame (must contain the raw features).

        The ``customerID`` column, when present, is carried through to the
        results so callers can join predictions back to their records.
        """
        threshold = self.model_service.threshold
        results: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            features = row.to_dict()
            customer_id = features.pop("customerID", None)
            if customer_id is not None and not pd.isna(customer_id):
                customer_id = str(customer_id)
            else:
                customer_id = None
            try:
                raw = predict_customer(
                    features,
                    pipeline=self.model_service.pipeline,
                    threshold=threshold,
                )
            except Exception as exc:  # row-level robustness for batch jobs
                logger.warning("Skipping row %s: %s", customer_id, exc)
                continue
            probability = float(raw["churn_probability"])
            results.append(
                {
                    "customer_id": customer_id,
                    "churn_probability": round(probability, 6),
                    "prediction": self.label(probability, threshold),
                    "risk_level": self.risk_level(probability),
                    "confidence": round(self.confidence(probability), 6),
                }
            )
        return results
