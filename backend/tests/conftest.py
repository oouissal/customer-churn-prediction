"""Shared fixtures for the backend API tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Make the repository root importable (backend/app imports src).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A TestClient bound to the FastAPI app (real model, real pipeline)."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def valid_customer() -> dict:
    """A complete, valid customer profile (high-risk)."""
    return {
        "gender": "Male",
        "SeniorCitizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 268.5,
    }


@pytest.fixture(scope="module")
def valid_batch_csv() -> str:
    """A minimal CSV with the 19 required columns (two rows)."""
    header = ",".join(config.RAW_FEATURES)
    row1 = ",".join(
        [
            "Male", "No", "No", "No", "3", "Yes", "No", "Fiber optic",
            "No", "No", "No", "No", "Yes", "Yes", "Month-to-month", "Yes",
            "Electronic check", "89.5", "268.5",
        ]
    )
    row2 = ",".join(
        [
            "Female", "No", "Yes", "No", "48", "Yes", "No", "DSL",
            "Yes", "Yes", "No", "No", "No", "No", "Two year", "No",
            "Bank transfer (automatic)", "52.35", "2512.8",
        ]
    )
    return "\n".join([header, row1, row2])
