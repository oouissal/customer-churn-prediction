"""Tests for the model information endpoint."""

from __future__ import annotations


class TestModelInfo:
    def test_model_info_structure(self, client):
        response = client.get("/api/model/info")
        assert response.status_code == 200
        body = response.json()
        assert body["model_name"] == "Logistic Regression"
        assert body["model_version"]
        assert 0.0 < body["threshold"] < 1.0
        assert body["feature_count"] > 0

    def test_model_metrics_are_real(self, client):
        """The metrics must match the values persisted by training."""
        response = client.get("/api/model/info")
        metrics = response.json()["metrics"]
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert key in metrics
            assert 0.0 < metrics[key] < 1.0

    def test_model_info_has_training_date(self, client):
        response = client.get("/api/model/info")
        assert response.json()["trained_at"]
