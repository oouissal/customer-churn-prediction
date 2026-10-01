"""Backend configuration, driven by environment variables.

Every value has a sensible local default and can be overridden through the
environment (see ``.env.example``). No secrets are hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path

from app import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
APP_NAME: str = os.getenv("APP_NAME", "Customer Churn Prediction API")
APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")
API_PREFIX: str = "/api"

# ---------------------------------------------------------------------------
# Model & data paths
# ---------------------------------------------------------------------------
MODEL_PATH: Path = Path(
    os.getenv("CHURN_MODEL_PATH", PROJECT_ROOT / "models" / "churn_model.joblib")
)
METRICS_PATH: Path = Path(
    os.getenv("CHURN_METRICS_PATH", PROJECT_ROOT / "models" / "metrics.json")
)
TEST_DATA_PATH: Path = Path(
    os.getenv("CHURN_TEST_DATA_PATH", PROJECT_ROOT / "data" / "processed" / "test.csv")
)

# ---------------------------------------------------------------------------
# Risk level bands (business rule, independent of the decision threshold)
# ---------------------------------------------------------------------------
RISK_LOW_BELOW: float = float(os.getenv("RISK_LOW_BELOW", "0.4"))
RISK_HIGH_ABOVE: float = float(os.getenv("RISK_HIGH_ABOVE", "0.7"))

# ---------------------------------------------------------------------------
# API behaviour
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
BATCH_PREDICTION_LIMIT: int = int(os.getenv("BATCH_PREDICTION_LIMIT", "5000"))
TOP_DRIVERS_COUNT: int = int(os.getenv("TOP_DRIVERS_COUNT", "8"))

# ---------------------------------------------------------------------------
# CORS (comma-separated list of allowed origins)
# ---------------------------------------------------------------------------
CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://localhost:8080",
    ).split(",")
    if origin.strip()
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
