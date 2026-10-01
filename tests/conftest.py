"""Shared pytest fixtures for the churn project test suite."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the project root importable when pytest is launched from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


@pytest.fixture(scope="session")
def sample_customer() -> dict:
    """A representative (high-risk) customer, as entered in the Streamlit form."""
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
def sample_df() -> pd.DataFrame:
    """Synthetic dataset mimicking the raw Telco schema, with edge cases:

    * tenure == 0 rows with blank TotalCharges (the IBM data quirk),
    * one duplicated row,
    * all tenure bands, "No internet service" values, both churn classes.
    """
    n = 24
    base = {
        "customerID": [f"{i:04d}-TEST" for i in range(n)],
        "gender": ["Male", "Female"] * (n // 2),
        "SeniorCitizen": [0, 1] * (n // 2),
        "Partner": ["Yes", "No"] * (n // 2),
        "Dependents": ["No", "Yes"] * (n // 2),
        "tenure": [0, 1, 5, 12, 13, 24, 25, 48, 49, 72] * 2 + [0, 1, 2, 3],
        "PhoneService": ["Yes"] * n,
        "MultipleLines": ["No", "Yes"] * (n // 2),
        "InternetService": ["DSL", "Fiber optic", "No"] * (n // 3),
        "OnlineSecurity": ["No", "Yes", "No internet service"] * (n // 3),
        "OnlineBackup": ["Yes", "No", "No internet service"] * (n // 3),
        "DeviceProtection": ["No", "Yes", "No internet service"] * (n // 3),
        "TechSupport": ["Yes", "No", "No internet service"] * (n // 3),
        "StreamingTV": ["No", "Yes", "No internet service"] * (n // 3),
        "StreamingMovies": ["Yes", "No", "No internet service"] * (n // 3),
        "Contract": ["Month-to-month", "One year", "Two year"] * (n // 3),
        "PaperlessBilling": ["Yes", "No"] * (n // 2),
        "PaymentMethod": [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ] * (n // 4),
        "MonthlyCharges": [round(20 + i * 3.1, 2) for i in range(n)],
        # first two rows are brand-new customers with no billing history
        "TotalCharges": ["" if i < 2 else round(50 + i * 10.5, 2) for i in range(n)],
        "Churn": ["No", "Yes"] * (n // 2),
    }
    df = pd.DataFrame(base)
    # one intentional duplicate
    df = pd.concat([df, df.iloc[[3]]], ignore_index=True)
    return df


@pytest.fixture(scope="module")
def cleaned_df(sample_df) -> pd.DataFrame:
    """The sample dataset after deterministic cleaning."""
    from src.data_preprocessing import clean_data

    return clean_data(sample_df)


@pytest.fixture(scope="module")
def engineered_df(cleaned_df) -> pd.DataFrame:
    """The cleaned sample dataset with engineered features added."""
    from src.feature_engineering import add_features

    return add_features(cleaned_df.drop(columns=[config.TARGET, config.ID_COLUMN]))
