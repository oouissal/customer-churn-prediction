"""Unit tests for src/validation.py (Pandera data validation)."""

from __future__ import annotations

import pandas as pd
import pytest
from src.validation import DataValidationError, validate_raw_data


@pytest.fixture(scope="module")
def valid_raw_df() -> pd.DataFrame:
    """A tiny raw-format DataFrame that satisfies the schema."""
    return pd.DataFrame(
        {
            "customerID": ["0001-A", "0002-B"],
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["No", "Yes"],
            "Dependents": ["No", "No"],
            "tenure": [1, 48],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "No"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "No"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "No"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 70.70],
            "TotalCharges": ["29.85", "3393.6"],
            "Churn": ["Yes", "No"],
        }
    )


class TestValidationPasses:
    def test_valid_dataframe_passes(self, valid_raw_df):
        result = validate_raw_data(valid_raw_df)
        assert result.shape == (2, 21)

    def test_real_raw_dataset_passes(self):
        from src.data_preprocessing import load_raw_data

        raw = load_raw_data()
        assert validate_raw_data(raw).shape[0] == 7043


class TestValidationFails:
    def test_missing_column_detected(self, valid_raw_df):
        df = valid_raw_df.drop(columns=["Churn"])
        with pytest.raises(DataValidationError, match="Churn"):
            validate_raw_data(df)

    def test_bad_category_detected(self, valid_raw_df):
        df = valid_raw_df.copy()
        df.loc[0, "Contract"] = "Lifetime"
        with pytest.raises(DataValidationError, match="Contract"):
            validate_raw_data(df)

    def test_out_of_range_value_detected(self, valid_raw_df):
        df = valid_raw_df.copy()
        df.loc[0, "MonthlyCharges"] = -10.0
        with pytest.raises(DataValidationError):
            validate_raw_data(df)

    def test_negative_tenure_detected(self, valid_raw_df):
        df = valid_raw_df.copy()
        df.loc[0, "tenure"] = -1
        with pytest.raises(DataValidationError):
            validate_raw_data(df)

    def test_duplicate_rows_detected(self, valid_raw_df):
        df = pd.concat([valid_raw_df, valid_raw_df.iloc[[0]]], ignore_index=True)
        with pytest.raises(DataValidationError, match="duplicate"):
            validate_raw_data(df)

    def test_unparseable_total_charges_detected(self, valid_raw_df):
        df = valid_raw_df.copy()
        df.loc[0, "TotalCharges"] = "not-a-number"
        with pytest.raises(DataValidationError):
            validate_raw_data(df)

    def test_blank_total_charges_for_existing_customer_detected(self, valid_raw_df):
        df = valid_raw_df.copy()
        df.loc[1, "TotalCharges"] = ""  # tenure is 48 — not a new customer
        with pytest.raises(DataValidationError):
            validate_raw_data(df)

    def test_blank_total_charges_for_new_customer_allowed(self, valid_raw_df):
        df = valid_raw_df.copy()
        df.loc[0, "tenure"] = 0
        df.loc[0, "TotalCharges"] = ""
        assert validate_raw_data(df).shape[0] == 2

    def test_duplicate_customer_ids_detected(self, valid_raw_df):
        df = valid_raw_df.copy()
        df.loc[1, "customerID"] = "0001-A"
        with pytest.raises(DataValidationError):
            validate_raw_data(df)
