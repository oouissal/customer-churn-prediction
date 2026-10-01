"""Dashboard service: aggregates scored on the held-out test set.

The dashboard needs real numbers (no fabricated metrics), so it scores the
test split once and caches the result. The values are honest: they are the
model's predictions on the 1,409 customers it never saw during training,
which the UI labels accordingly ("hold-out test set").
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from src import config as src_config
from src.predict import shap_values_many

from app.core import config
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

_NUM_BINS = 10  # probability histogram buckets


class DashboardService:
    """Computes the dashboard KPIs and charts data."""

    def __init__(
        self,
        model_service: ModelService,
        prediction_service: PredictionService,
    ) -> None:
        self.model_service = model_service
        self.prediction_service = prediction_service
        self._cached_scoring: tuple[float, tuple] | None = None

    # ------------------------------------------------------------------
    # Data preparation (cached per service instance)
    # ------------------------------------------------------------------
    def _scored_test_set(self) -> tuple[float, tuple]:
        """Score the test split once; returns (threshold, (probas, labels))."""
        if self._cached_scoring is not None:
            return self._cached_scoring
        if not config.TEST_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Test data not found at {config.TEST_DATA_PATH}. "
                "Run `python -m src.train` to regenerate it."
            )
        test = pd.read_csv(config.TEST_DATA_PATH)
        features = test.drop(columns=[src_config.TARGET], errors="ignore")
        probas = np.asarray(
            self.model_service.pipeline.predict_proba(features)[:, 1],
            dtype=float,
        )
        labels = (probas >= self.model_service.threshold).astype(int)
        self._cached_scoring = self.model_service.threshold, (probas, labels)
        return self._cached_scoring

    def _probas(self) -> np.ndarray:
        """Predicted churn probabilities on the hold-out test set."""
        _, (probas, _) = self._scored_test_set()
        return probas

    # ------------------------------------------------------------------
    # Endpoints payloads
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """KPIs + distribution data for the dashboard page."""
        probas = self._probas()
        threshold = self.model_service.threshold
        labels = (probas >= threshold).astype(int)
        risk = np.array(
            [self.prediction_service.risk_level(p) for p in probas]
        )

        hist_counts, hist_edges = np.histogram(
            probas, bins=_NUM_BINS, range=(0.0, 1.0)
        )
        return {
            "source": "Hold-out test set (n=1409 customers)",
            "total_customers": int(probas.size),
            "predicted_churn_rate": float(labels.mean()),
            "high_risk_customers": int((risk == "HIGH").sum()),
            "average_churn_probability": float(probas.mean()),
            "model_version": self.model_service.model_version,
            "model_name": self.model_service.model_name,
            "threshold": threshold,
            "churn_distribution": [
                {"name": "CHURN", "value": int(labels.sum())},
                {"name": "NO_CHURN", "value": int((labels == 0).sum())},
            ],
            "risk_distribution": [
                {"name": level, "value": int((risk == level).sum())}
                for level in ("LOW", "MEDIUM", "HIGH")
            ],
            "probability_distribution": [
                {
                    "bucket": f"{hist_edges[i]:.1f}-{hist_edges[i + 1]:.1f}",
                    "count": int(count),
                }
                for i, count in enumerate(hist_counts)
            ],
        }

    def top_drivers(self, top_n: int = 8) -> list[dict]:
        """Global SHAP importance on the hold-out test set (top features)."""
        if not config.TEST_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Test data not found at {config.TEST_DATA_PATH}."
            )
        test = pd.read_csv(config.TEST_DATA_PATH)
        features = test.drop(columns=[src_config.TARGET], errors="ignore")
        names, shap_values, _ = shap_values_many(
            self.model_service.pipeline, features.head(400)
        )
        importance = np.abs(shap_values).mean(axis=0)
        table = pd.DataFrame({"feature": names, "importance": importance})
        # Aggregate one-hot columns back to original features.
        table["feature"] = table["feature"].apply(_original_feature)
        table = (
            table.groupby("feature", sort=False)["importance"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        return [
            {"feature": row["feature"], "importance": round(row["importance"], 4)}
            for _, row in table.iterrows()
        ]


def _original_feature(output_name: str) -> str:
    """Map a preprocessor output column back to its original feature."""
    if output_name in src_config.NUMERICAL_FEATURES:
        return output_name
    for feature in src_config.CATEGORICAL_FEATURES:
        if output_name == feature or output_name.startswith(feature + "_"):
            return feature
    return output_name
