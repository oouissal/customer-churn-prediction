"""Unit tests for src/feature_engineering.py."""

import pandas as pd
import pytest
from src import config
from src.feature_engineering import (
    FeatureEngineer,
    add_features,
    tenure_to_group,
)


class TestTenureGroup:
    @pytest.mark.parametrize(
        "tenure, expected",
        [
            (0, "0-12"),
            (12, "0-12"),
            (13, "13-24"),
            (24, "13-24"),
            (25, "25-48"),
            (48, "25-48"),
            (49, "49+"),
            (72, "49+"),
        ],
    )
    def test_boundaries(self, tenure, expected):
        assert tenure_to_group(tenure) == expected


class TestNumServices:
    def test_counts_only_yes(self):
        row = pd.DataFrame(
            [{
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No internet service",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "Yes",
                "tenure": 10,
                "TotalCharges": 500.0,
            }]
        )
        assert add_features(row)["num_services"].iloc[0] == 3

    def test_no_internet_counts_as_zero_services(self):
        row = pd.DataFrame(
            [{col: "No internet service" for col in config.SERVICE_COLUMNS}
             | {"tenure": 5, "TotalCharges": 200.0}]
        )
        assert add_features(row)["num_services"].iloc[0] == 0


class TestAvgMonthlySpend:
    @staticmethod
    def _row(**overrides) -> pd.DataFrame:
        data = {col: "No" for col in config.SERVICE_COLUMNS}
        data.update({"tenure": 10, "TotalCharges": 500.0})
        data.update(overrides)
        return pd.DataFrame([data])

    def test_normal_case(self):
        row = self._row()
        assert add_features(row)["avg_monthly_spend"].iloc[0] == pytest.approx(50.0)

    def test_tenure_zero_falls_back_to_monthly_charges(self):
        row = self._row(tenure=0, MonthlyCharges=70.0, TotalCharges=70.0)
        result = add_features(row)["avg_monthly_spend"].iloc[0]
        assert result == pytest.approx(70.0)
        assert pd.notna(result)


class TestFeatureEngineerTransformer:
    def test_adds_engineered_columns_and_keeps_raw(self, cleaned_df):
        X = cleaned_df.drop(columns=[config.TARGET, config.ID_COLUMN])
        out = FeatureEngineer().fit_transform(X)
        for col in config.ENGINEERED_FEATURES:
            assert col in out.columns
        for col in config.RAW_FEATURES:
            assert col in out.columns

    def test_get_feature_names_out(self):
        expected = config.RAW_FEATURES + config.ENGINEERED_FEATURES
        assert FeatureEngineer().get_feature_names_out() == expected
