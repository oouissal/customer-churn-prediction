"""Explanation service: SHAP-based explanations of individual predictions.

Reuses the SHAP machinery from ``src.predict`` (aggregated to the original
features so explanations read in business terms) and converts it into the
API's ``top_drivers`` representation.
"""

from __future__ import annotations

from typing import Any

from src.predict import explain_prediction, probability_shifts

from app.core import config
from app.services.model_service import ModelService

# Features whose direction is reported with a human-readable label.
_DIRECTION_LABELS = {
    "increases": "increases churn risk",
    "decreases": "decreases churn risk",
}


class ExplanationService:
    """Produces per-prediction SHAP explanations."""

    def __init__(self, model_service: ModelService) -> None:
        self.model_service = model_service

    def explain(
        self, features: dict[str, Any], top_n: int | None = None
    ) -> list[dict[str, Any]]:
        """Return the top SHAP drivers for one customer profile."""
        top_n = top_n or config.TOP_DRIVERS_COUNT
        explanation = explain_prediction(
            features, pipeline=self.model_service.pipeline
        )
        shifts = probability_shifts(explanation)
        drivers: list[dict[str, Any]] = []
        for _, row in shifts.head(top_n).iterrows():
            drivers.append(
                {
                    "feature": str(row["feature"]),
                    "value": _to_json_safe(row["value"]),
                    "impact": _DIRECTION_LABELS.get(
                        str(row["direction"]), str(row["direction"])
                    ),
                    "probability_shift": round(float(row["prob_shift"]), 4),
                }
            )
        return drivers

    def business_summary(
        self, drivers: list[dict[str, Any]], probability: float
    ) -> str:
        """One-sentence explanation of the prediction for business users."""
        if not drivers:
            return ""
        top = drivers[0]
        direction = (
            "increases"
            if top["probability_shift"] > 0
            else "decreases"
        )
        value = "" if top["value"] is None else f" = {top['value']}"
        return (
            f"The customer has a {probability:.0%} churn probability. "
            f"The strongest driver is {top['feature']}{value}, which "
            f"{direction} churn risk."
        )


def pd_isna(value: Any) -> bool:
    """pandas-aware NaN check that never crashes on arbitrary values."""
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def _to_json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars to native Python types for JSON."""
    if pd_isna(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    return str(value)
