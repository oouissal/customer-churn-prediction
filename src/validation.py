"""Automated data validation for the IBM Telco Customer Churn dataset.

The raw CSV is validated **before** any cleaning or training happens
(``python -m src.train`` refuses to start on invalid data, and the DVC
pipeline has a dedicated ``validate`` stage). The schema is implemented with
`Pandera <https://www.union.ai/pandera>`_ because it is a lightweight,
code-first validation library that fits the existing pandas workflow.

Checks covered
--------------
* required columns (and only the expected ones),
* data types (``tenure`` int, ``MonthlyCharges`` float, ...),
* allowed categorical values (contract types, service flags, ...),
* numerical ranges (tenure >= 0, positive charges, ...),
* missing values (only ``TotalCharges`` may be blank, and only for
  brand-new customers with ``tenure == 0``),
* duplicate rows (none allowed).

If validation fails, a :class:`DataValidationError` is raised with a
human-readable summary of every failing check, so training stops with a
meaningful error instead of silently producing a broken model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



class DataValidationError(ValueError):
    """Raised when the raw dataset fails schema validation."""


# ---------------------------------------------------------------------------
# Allowed values (documented per feature for reference)
# ---------------------------------------------------------------------------
YES_NO = ("No", "Yes")
GENDER = ("Male", "Female")
PHONE_FLAG = ("No", "Yes", "No phone service")
INTERNET_FLAG = ("No", "Yes", "No internet service")
INTERNET_SERVICE = ("DSL", "Fiber optic", "No")
CONTRACT = ("Month-to-month", "One year", "Two year")
PAYMENT_METHOD = (
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
)


class RawTelcoSchema(pa.DataFrameModel):
    """Pandera schema of the raw Telco CSV (before any cleaning)."""

    customerID: Series[str]
    gender: Series[str] = pa.Field(isin=GENDER)
    SeniorCitizen: Series[int] = pa.Field(isin=[0, 1])
    Partner: Series[str] = pa.Field(isin=YES_NO)
    Dependents: Series[str] = pa.Field(isin=YES_NO)
    tenure: Series[int] = pa.Field(ge=0, le=100)
    PhoneService: Series[str] = pa.Field(isin=YES_NO)
    MultipleLines: Series[str] = pa.Field(isin=PHONE_FLAG)
    InternetService: Series[str] = pa.Field(isin=INTERNET_SERVICE)
    OnlineSecurity: Series[str] = pa.Field(isin=INTERNET_FLAG)
    OnlineBackup: Series[str] = pa.Field(isin=INTERNET_FLAG)
    DeviceProtection: Series[str] = pa.Field(isin=INTERNET_FLAG)
    TechSupport: Series[str] = pa.Field(isin=INTERNET_FLAG)
    StreamingTV: Series[str] = pa.Field(isin=INTERNET_FLAG)
    StreamingMovies: Series[str] = pa.Field(isin=INTERNET_FLAG)
    Contract: Series[str] = pa.Field(isin=CONTRACT)
    PaperlessBilling: Series[str] = pa.Field(isin=YES_NO)
    PaymentMethod: Series[str] = pa.Field(isin=PAYMENT_METHOD)
    MonthlyCharges: Series[float] = pa.Field(gt=0, le=250)
    # Stored as text in the CSV; must be blank or a parseable number.
    TotalCharges: Series[str] = pa.Field(nullable=True)
    Churn: Series[str] = pa.Field(isin=YES_NO)

    class Config:
        strict = True
        unique = ["customerID"]
        coerce = False
        add_missing_columns = False

    @pa.dataframe_check
    def total_charges_parseable(cls, df: pd.DataFrame) -> Series[bool]:
        """Every non-blank TotalCharges value must be a parseable number."""
        numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
        blank = df["TotalCharges"].isna() | (df["TotalCharges"].astype(str).str.strip() == "")
        return blank | numeric.notna()

    @pa.dataframe_check
    def missing_total_charges_only_for_new_customers(cls, df: pd.DataFrame) -> Series[bool]:
        """Blank TotalCharges is a business rule only for tenure == 0."""
        blank = df["TotalCharges"].isna() | (df["TotalCharges"].astype(str).str.strip() == "")
        return ~blank | (df["tenure"] == 0)

    @pa.dataframe_check
    def no_duplicate_rows(cls, df: pd.DataFrame) -> Series[bool]:
        """The raw dataset must not contain duplicated records."""
        return pd.Series(not df.duplicated().any(), index=[0])


RAW_SCHEMA = RawTelcoSchema.to_schema()


def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw dataset and return it unchanged if it passes.

    Parameters
    ----------
    df : raw DataFrame as loaded from the CSV.

    Raises
    ------
    DataValidationError
        With a readable summary of every failing check.
    """
    try:
        return RAW_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise DataValidationError(
            "Raw dataset failed validation. Training was stopped.\n"
            f"{exc.failure_cases.to_string(index=False)}"
        ) from exc


if __name__ == "__main__":
    from src.data_preprocessing import load_raw_data

    raw = load_raw_data()
    validated = validate_raw_data(raw)
    print(f"Data validation passed: {validated.shape[0]} rows, "
          f"{validated.shape[1]} columns, schema OK.")
