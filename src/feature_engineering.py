"""Business-driven feature engineering for the churn models.

The features created here are all deterministic, row-wise transformations —
no statistics are learned from the data — so the transformer can be placed
inside the modelling pipeline without any risk of data leakage.

Why these features?
-------------------
* ``tenure_group``: the relationship between tenure and churn is strongly
  non-linear (churn risk collapses after the first 1–2 years). Binning
  tenure lets linear models capture that shape.
* ``num_services``: the number of subscribed add-on services proxies
  "customer stickiness" — every extra service is a reason to stay.
* ``avg_monthly_spend``: ``TotalCharges / tenure`` is a cleaner measure of
  spending level than ``MonthlyCharges`` alone, which can be discounted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

TENURE_BINS = [-1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49+"]


def tenure_to_group(tenure: float | int) -> str:
    """Map a tenure value (in months) to one of the business bands."""
    tenure = float(tenure)
    if tenure <= 12:
        return "0-12"
    if tenure <= 24:
        return "13-24"
    if tenure <= 48:
        return "25-48"
    return "49+"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered features to a DataFrame (functional wrapper)."""
    df = df.copy()
    df["tenure_group"] = df["tenure"].apply(tenure_to_group)
    df["num_services"] = (
        df[config.SERVICE_COLUMNS] == "Yes"
    ).sum(axis=1)
    # Customers with tenure == 0 have no billing history: fall back to the
    # current monthly charge instead of dividing by zero.
    df["avg_monthly_spend"] = df["TotalCharges"] / np.maximum(df["tenure"], 1)
    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """scikit-learn transformer that adds the engineered features.

    The transformer only *adds* columns (``tenure_group``, ``num_services``,
    ``avg_monthly_spend``); the raw columns are kept so the downstream
    ``ColumnTransformer`` can select its own final feature lists.
    """

    def fit(self, X: pd.DataFrame, y=None) -> FeatureEngineer:
        """No statistics are learned — kept for the scikit-learn API."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the DataFrame with the engineered features appended."""
        return add_features(X)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return the full output schema of the transformer."""
        base = list(input_features) if input_features is not None else config.RAW_FEATURES
        return base + config.ENGINEERED_FEATURES


if __name__ == "__main__":
    from src.data_preprocessing import clean_data, load_raw_data

    df = clean_data(load_raw_data()).drop(columns=[config.TARGET, config.ID_COLUMN])
    engineered = add_features(df)
    print(engineered[["tenure", "tenure_group", "num_services", "avg_monthly_spend"]].head(10))
    print("\nTenure group counts:\n", engineered["tenure_group"].value_counts())
