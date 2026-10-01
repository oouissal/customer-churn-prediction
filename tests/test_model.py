"""Tests for the trained model: loading, predictions and SHAP explanations.

These tests require the trained pipeline (`models/churn_model.joblib`),
generated with `python -m src.train`. They are skipped with a clear message
if the model has not been trained yet.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.predict import (
    explain_prediction,
    load_model,
    predict_customer,
    probability_shifts,
)

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="models/churn_model.joblib not found — run `python -m src.train` first",
)


@pytest.fixture(scope="module")
def trained():
    return load_model()


class TestModelLoading:
    def test_artifacts_exist(self):
        assert config.MODEL_PATH.exists()
        assert config.METRICS_PATH.exists()

    def test_pipeline_structure(self, trained):
        pipeline, artifact = trained
        assert hasattr(pipeline, "predict_proba")
        assert {"features", "preprocessor", "model"} <= set(pipeline.named_steps)
        assert 0.0 < artifact["threshold"] < 1.0

    def test_metrics_file_is_valid(self):
        metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
        for key in ("model_name", "threshold", "test_metrics_tuned_threshold"):
            assert key in metrics
        auc = metrics["test_metrics_tuned_threshold"]["roc_auc"]
        assert 0.5 < auc <= 1.0  # strictly better than random


class TestPrediction:
    def test_predict_proba_valid(self, trained, sample_customer):
        pipeline, _ = trained
        proba = pipeline.predict_proba(pd.DataFrame([sample_customer]))
        assert proba.shape == (1, 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert np.isclose(proba.sum(axis=1), 1.0)

    def test_predict_customer_structure(self, sample_customer):
        result = predict_customer(sample_customer)
        assert set(result) == {
            "churn_probability", "prediction_label", "prediction", "confidence",
        }
        assert 0.0 <= result["churn_probability"] <= 1.0
        assert result["prediction"] in {"HIGH CHURN RISK", "LOW CHURN RISK"}
        assert result["prediction_label"] in {0, 1}

    def test_prediction_is_deterministic(self, sample_customer):
        a = predict_customer(sample_customer)
        b = predict_customer(sample_customer)
        assert a["churn_probability"] == b["churn_probability"]
        assert a["prediction"] == b["prediction"]

    def test_threshold_extremes(self, sample_customer):
        low = predict_customer(sample_customer, threshold=1.0)
        high = predict_customer(sample_customer, threshold=0.0)
        assert low["prediction"] == "LOW CHURN RISK"
        assert high["prediction"] == "HIGH CHURN RISK"

    def test_tuned_threshold_is_used_by_default(self, sample_customer):
        _, artifact = load_model()
        result = predict_customer(sample_customer)
        expected_label = int(
            result["churn_probability"] >= artifact["threshold"]
        )
        assert result["prediction_label"] == expected_label


class TestExplainability:
    def test_explanation_table(self, sample_customer):
        explanation = explain_prediction(sample_customer)
        table = explanation["table"]
        assert {"feature", "value", "shap_value", "direction"} <= set(table.columns)
        assert table["abs_shap"].is_monotonic_decreasing
        assert set(table["direction"]) <= {"increases", "decreases"}
        assert "base_value" in explanation

    def test_probability_shifts_are_bounded(self, sample_customer):
        explanation = explain_prediction(sample_customer)
        shifts = probability_shifts(explanation)
        assert shifts["prob_shift"].between(-1.0, 1.0).all()
        assert len(shifts) == len(explanation["table"])
