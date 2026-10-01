"""Tests for the single prediction endpoint and its validation."""

from __future__ import annotations

import pytest


class TestPredictEndpoint:
    def test_predict_valid_customer(self, client, valid_customer):
        response = client.post(
            "/api/predict", json={"customer": valid_customer}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in {"CHURN", "NO_CHURN"}
        assert 0.0 <= body["probability"] <= 1.0
        assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
        assert 0.5 <= body["confidence"] <= 1.0
        assert body["model_version"]

    def test_predict_returns_top_drivers(self, client, valid_customer):
        response = client.post(
            "/api/predict", json={"customer": valid_customer}
        )
        drivers = response.json()["top_drivers"]
        assert len(drivers) > 0
        first = drivers[0]
        assert {"feature", "value", "impact", "probability_shift"} <= set(first)
        # Top drivers are sorted by absolute impact.
        shifts = [abs(driver["probability_shift"]) for driver in drivers]
        assert shifts == sorted(shifts, reverse=True)

    def test_predict_high_risk_profile_flags_churn(self, client, valid_customer):
        response = client.post(
            "/api/predict", json={"customer": valid_customer}
        )
        body = response.json()
        # Short tenure + fibre + month-to-month + electronic check: the model
        # scores this well above the 0.61 threshold.
        assert body["prediction"] == "CHURN"
        assert body["risk_level"] == "HIGH"

    def test_predict_invalid_category_rejected(self, client, valid_customer):
        payload = {**valid_customer, "Contract": "Lifetime"}
        response = client.post("/api/predict", json={"customer": payload})
        assert response.status_code == 422

    def test_predict_negative_tenure_rejected(self, client, valid_customer):
        payload = {**valid_customer, "tenure": -5}
        response = client.post("/api/predict", json={"customer": payload})
        assert response.status_code == 422

    def test_predict_missing_feature_rejected(self, client, valid_customer):
        payload = {key: value for key, value in valid_customer.items()
                   if key != "MonthlyCharges"}
        response = client.post("/api/predict", json={"customer": payload})
        assert response.status_code == 422

    def test_predict_unknown_field_rejected(self, client, valid_customer):
        payload = {**valid_customer, "favourite_colour": "blue"}
        response = client.post("/api/predict", json={"customer": payload})
        assert response.status_code == 422

    def test_predict_missing_body_rejected(self, client):
        response = client.post("/api/predict", json={})
        assert response.status_code == 422

    def test_predict_deterministic(self, client, valid_customer):
        first = client.post("/api/predict", json={"customer": valid_customer})
        second = client.post("/api/predict", json={"customer": valid_customer})
        assert first.json()["probability"] == second.json()["probability"]

    @pytest.mark.parametrize(
        "probability,expected_risk",
        [(0.05, "LOW"), (0.45, "MEDIUM"), (0.69, "MEDIUM"), (0.95, "HIGH")],
    )
    def test_risk_level_bands(self, probability, expected_risk):
        from app.services.prediction_service import PredictionService

        assert PredictionService.risk_level(probability) == expected_risk
