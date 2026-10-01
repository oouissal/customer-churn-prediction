"""Unit tests for src/evaluate.py (metrics, threshold tuning, tables)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    compute_metrics,
    find_best_threshold,
    metrics_dataframe,
)


@pytest.fixture(scope="module")
def toy_targets() -> tuple[np.ndarray, np.ndarray]:
    """Perfectly separable predictions -> metrics are trivially checkable."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    # 2D output shaped like sklearn's predict_proba (neg | pos classes).
    y_proba = np.column_stack(
        [[0.9, 0.8, 0.7, 0.2, 0.1, 0.05], [0.1, 0.2, 0.3, 0.8, 0.9, 0.95]]
    )
    return y_true, y_proba


class TestComputeMetrics:
    def test_perfect_separation(self, toy_targets):
        y_true, y_proba = toy_targets
        metrics = compute_metrics(y_true, y_proba, threshold=0.5)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["roc_auc"] == 1.0
        assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (3, 0, 0, 3)

    def test_all_negative_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.column_stack(
            [[0.9, 0.8, 0.7, 0.6], [0.1, 0.2, 0.3, 0.4]]
        )
        metrics = compute_metrics(y_true, y_proba, threshold=0.9)
        assert metrics["recall"] == 0.0
        assert metrics["precision"] == 0.0  # zero_division handled
        assert metrics["tp"] == 0

    def test_threshold_changes_labels(self, toy_targets):
        y_true, y_proba = toy_targets
        strict = compute_metrics(y_true, y_proba, threshold=0.99)
        assert strict["tp"] == 0
        assert strict["accuracy"] == 0.5


class TestFindBestThreshold:
    def test_returns_valid_threshold(self, toy_targets):
        y_true, y_proba = toy_targets
        threshold, metrics = find_best_threshold(y_true, y_proba, metric="f1")
        assert 0.0 < threshold < 1.0
        assert metrics["f1"] == 1.0
        assert set(metrics) >= {"precision", "recall", "f1"}

    def test_unknown_metric_raises(self, toy_targets):
        y_true, y_proba = toy_targets
        with pytest.raises(KeyError):
            find_best_threshold(y_true, y_proba, metric="nonsense")


class TestMetricsDataframe:
    def test_rounds_and_selects_columns(self):
        rows = [
            {
                "model": "A",
                "accuracy": 0.123456,
                "precision": 0.5,
                "recall": 0.6,
                "f1": 0.55,
                "roc_auc": 0.9,
                "extra": "ignored",
            }
        ]
        table = metrics_dataframe(rows)
        assert list(table.columns) == ["model", "accuracy", "precision",
                                       "recall", "f1", "roc_auc"]
        assert table.loc[0, "accuracy"] == pytest.approx(0.1235, abs=1e-4)
