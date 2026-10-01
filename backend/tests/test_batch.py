"""Tests for the batch prediction endpoint."""

from __future__ import annotations

import io


class TestBatchPrediction:
    def test_batch_valid_csv(self, client, valid_batch_csv):
        response = client.post(
            "/api/predict/batch",
            files={"file": ("customers.csv", io.BytesIO(valid_batch_csv.encode()), "text/csv")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert len(body["results"]) == 2
        first = body["results"][0]
        assert first["prediction"] in {"CHURN", "NO_CHURN"}
        assert first["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
        assert 0.0 <= first["churn_probability"] <= 1.0

    def test_batch_csv_download(self, client, valid_batch_csv):
        response = client.post(
            "/api/predict/batch?format=csv",
            files={"file": ("customers.csv", io.BytesIO(valid_batch_csv.encode()), "text/csv")},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert "churn_probability" in response.text
        assert response.text.strip().count("\n") == 2  # header + 2 rows

    def test_batch_missing_columns_rejected(self, client):
        csv_content = "gender,tenure\nMale,3\n"
        response = client.post(
            "/api/predict/batch",
            files={"file": ("bad.csv", io.BytesIO(csv_content.encode()), "text/csv")},
        )
        assert response.status_code == 400
        assert "Missing required columns" in response.json()["detail"]

    def test_batch_wrong_file_type_rejected(self, client):
        response = client.post(
            "/api/predict/batch",
            files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 400
        assert "CSV" in response.json()["detail"]

    def test_batch_empty_file_rejected(self, client):
        response = client.post(
            "/api/predict/batch",
            files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
        )
        assert response.status_code == 400

    def test_batch_garbage_csv_rejected(self, client):
        response = client.post(
            "/api/predict/batch",
            files={"file": ("garbage.csv", io.BytesIO(b"\x00\x01\x02"), "text/csv")},
        )
        assert response.status_code in (400, 500)

    def test_batch_over_limit_rejected(self, client, valid_batch_csv):
        from app.core.config import BATCH_PREDICTION_LIMIT

        header, row = valid_batch_csv.split("\n", 1)
        oversized = "\n".join([header] + [row] * (BATCH_PREDICTION_LIMIT + 1))
        response = client.post(
            "/api/predict/batch",
            files={"file": ("big.csv", io.BytesIO(oversized.encode()), "text/csv")},
        )
        assert response.status_code == 413
