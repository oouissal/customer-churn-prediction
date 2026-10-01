"""Tests for the dashboard endpoints (computed on the hold-out test set)."""

from __future__ import annotations


class TestDashboard:
    def test_summary_structure(self, client):
        response = client.get("/api/dashboard/summary")
        assert response.status_code == 200
        body = response.json()
        assert body["total_customers"] == 1409
        assert 0.0 <= body["predicted_churn_rate"] <= 1.0
        assert body["high_risk_customers"] >= 0
        assert 0.0 <= body["average_churn_probability"] <= 1.0
        assert body["model_version"]

    def test_summary_distributions_are_consistent(self, client):
        body = client.get("/api/dashboard/summary").json()
        churn = body["churn_distribution"]
        risk = body["risk_distribution"]
        hist = body["probability_distribution"]
        assert sum(item["value"] for item in churn) == body["total_customers"]
        assert sum(item["value"] for item in risk) == body["total_customers"]
        assert sum(item["count"] for item in hist) == body["total_customers"]

    def test_summary_labels_source_honestly(self, client):
        body = client.get("/api/dashboard/summary").json()
        assert "test" in body["source"].lower()

    def test_drivers_structure(self, client):
        response = client.get("/api/dashboard/drivers")
        assert response.status_code == 200
        body = response.json()
        drivers = body["drivers"]
        assert len(drivers) > 0
        assert all(
            {"feature", "importance"} <= set(driver) for driver in drivers
        )
        # Sorted by importance, descending.
        values = [driver["importance"] for driver in drivers]
        assert values == sorted(values, reverse=True)

    def test_drivers_are_business_features(self, client):
        body = client.get("/api/dashboard/drivers").json()
        features = {driver["feature"] for driver in body["drivers"]}
        # Original features, not one-hot encoded columns.
        assert all("_" not in feature.split("=")[0] or feature in
                   {"tenure_group", "num_services", "avg_monthly_spend",
                    "MonthlyCharges", "TotalCharges"}
                   for feature in features)
