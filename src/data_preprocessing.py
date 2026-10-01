"""Data loading and preprocessing for the Telco Customer Churn dataset.

This module implements

1. ``load_raw_data`` / ``clean_data`` — deterministic, row-wise business
   cleaning (no statistics learned from the data, therefore no leakage), and
2. ``build_preprocessor`` — the scikit-learn ``ColumnTransformer`` that turns
   the cleaned DataFrame into a numeric matrix.

Design principles
-----------------
* Every statistic (medians, category lists, scaler parameters) is learned
  inside the preprocessor with ``.fit()`` **on training data only**, so
  nothing leaks from the validation/test sets.
* The preprocessor is reusable at inference time: unknown categories are
  ignored by the ``OneHotEncoder`` (``handle_unknown="ignore"``), which makes
  the Streamlit app robust to new user inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Allow running this file directly (`python src/data_preprocessing.py`) as
# well as importing it as part of the package (`python -m src.train`).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402


# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------
def load_raw_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw Telco dataset from CSV.

    Parameters
    ----------
    path : optional path override; defaults to ``config.RAW_DATA_PATH``.

    Raises
    ------
    FileNotFoundError
        If the raw file is missing, with instructions on how to obtain it.
    """
    path = Path(path) if path is not None else config.RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download it with the commands "
            "documented in data/README.md."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, row-wise cleaning to the raw dataset.

    Steps
    -----
    1. Strip whitespace from column names.
    2. Convert empty strings to ``NaN``.
    3. Coerce ``TotalCharges`` to numeric (the CSV stores it as text).
    4. Map ``SeniorCitizen`` from 0/1 to "No"/"Yes" so it is treated as a
       categorical feature like every other flag.
    5. Drop duplicate records if any (the raw dataset contains none, but the
       step keeps the pipeline robust).
    6. Fill ``TotalCharges`` for brand-new customers (``tenure == 0``) with
       their current ``MonthlyCharges``. A customer with zero tenure has no
       billing history, so the blank is not missing-at-random; using the
       current monthly charge keeps downstream features such as
       ``avg_monthly_spend`` meaningful.

    All steps are row-wise business rules — no dataset statistics are used,
    so nothing can leak between train and test.
    """
    df = df.copy()

    df.columns = df.columns.str.strip()
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Data types
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = (
        df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"])
    )

    # Duplicates
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)

    # Business rule for brand-new customers
    new_customers = (df["tenure"] == 0) & df["TotalCharges"].isna()
    df.loc[new_customers, "TotalCharges"] = df.loc[new_customers, "MonthlyCharges"]

    return df


def encode_target(df: pd.DataFrame) -> pd.Series:
    """Encode the churn target as 0/1 (1 = churner, the positive class)."""
    return (df[config.TARGET].astype(str).str.strip() == "Yes").astype(int)


# ---------------------------------------------------------------------------
# scikit-learn preprocessing pipeline
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Build the leak-free feature preprocessing pipeline.

    * numerical features   -> median imputation + ``StandardScaler``
    * categorical features -> most-frequent imputation + one-hot encoding
      (``handle_unknown="ignore"`` so unseen categories do not break
      inference).

    All transformers are combined in a single ``ColumnTransformer`` that is
    fitted on training data only.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    validation_size: float = config.VALIDATION_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data into train / validation / test sets.

    The split is stratified on the target so the churn rate is preserved in
    every set. With the default sizes the result is a 60/20/20 split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        stratify=y_train,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    raw = load_raw_data()
    cleaned = clean_data(raw)
    y = encode_target(cleaned)
    print(f"Raw shape      : {raw.shape}")
    print(f"Cleaned shape  : {cleaned.shape}")
    print(f"Churn rate     : {y.mean():.2%}")
    missing = cleaned.isna().sum()
    print(f"Missing values after cleaning:\n{missing[missing > 0]}")

