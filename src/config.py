"""Central configuration for the customer churn prediction project.

Every path is derived from ``PROJECT_ROOT`` with ``pathlib``, so no absolute
path is hard-coded and the scripts work from any working directory and on any
machine. Experiment settings (random state, split sizes, threshold metric)
live here too, making the pipeline easy to reconfigure without touching code.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "Telco-Customer-Churn.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
COMPARISON_CSV_PATH = REPORTS_DIR / "model_comparison.csv"

# ---------------------------------------------------------------------------
# MLflow experiment tracking & model registry
# ---------------------------------------------------------------------------
# Default: local SQLite tracking store inside the repository (the file-based
# store is deprecated in MLflow 3.x). Override with the
# MLFLOW_TRACKING_URI environment variable (e.g. a shared tracking server).
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(PROJECT_ROOT / 'mlruns' / 'mlflow.db').as_posix()}",
)
MLFLOW_EXPERIMENT_NAME = "customer-churn-prediction"
MLFLOW_REGISTERED_MODEL_NAME = "customer-churn-model"

# Where MLflow stores model/artefact files for local runs.
os.environ.setdefault(
    "MLFLOW_DEFAULT_ARTIFACT_ROOT",
    (PROJECT_ROOT / "mlruns" / "mlartifacts").as_uri(),
)

# Model version reported by the API when the MLflow registry is unavailable
# (e.g. the model was produced before registry integration).
FALLBACK_MODEL_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

TEST_SIZE = 0.20        # share of the data held out as final test set
VALIDATION_SIZE = 0.25  # share of the remaining data used for validation
                        # -> effective split 60% train / 20% validation / 20% test

CV_FOLDS = 5            # folds of the stratified cross-validation
THRESHOLD_METRIC = "f1" # metric optimised when tuning the decision threshold
POSITIVE_CLASS = 1      # 1 == churner

# ---------------------------------------------------------------------------
# Dataset columns (IBM Telco Customer Churn)
# ---------------------------------------------------------------------------
TARGET = "Churn"
ID_COLUMN = "customerID"

# Raw feature columns as they appear in the CSV (customerID excluded).
RAW_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

# Add-on services counted by the engineered "num_services" feature.
SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Final categorical features after feature engineering (tenure_group included).
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "tenure_group",
]

# Final numerical features after feature engineering.
NUMERICAL_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "num_services", "avg_monthly_spend",
]

# Features created by src.feature_engineering.FeatureEngineer.
ENGINEERED_FEATURES = ["tenure_group", "num_services", "avg_monthly_spend"]


def ensure_directories() -> None:
    """Create the data/models/reports directories used by the pipeline."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
