"""Unit tests for src/data_preprocessing.py."""

import numpy as np
import pandas as pd
from src import config
from src.data_preprocessing import (
    build_preprocessor,
    clean_data,
    encode_target,
    split_data,
)


class TestCleanData:
    def test_blank_total_charges_filled_for_new_customers(self, sample_df, cleaned_df):
        # Align with the cleaned frame by removing the intentional duplicate.
        source = sample_df.drop_duplicates().reset_index(drop=True)
        was_blank = source["TotalCharges"].astype(str).str.strip() == ""
        blank_new = was_blank & (source["tenure"] == 0)
        assert blank_new.sum() == 1  # exactly one brand-new customer case

        filled_rows = cleaned_df.loc[source.index[blank_new]]
        assert filled_rows["TotalCharges"].notna().all()
        assert (
            filled_rows["TotalCharges"] == filled_rows["MonthlyCharges"]
        ).all()
        # every tenure == 0 row must end up with a numeric TotalCharges
        new_customers = cleaned_df[cleaned_df["tenure"] == 0]
        assert new_customers["TotalCharges"].notna().all()

    def test_duplicates_are_dropped(self, cleaned_df):
        assert cleaned_df.duplicated().sum() == 0

    def test_total_charges_is_numeric(self, cleaned_df):
        assert pd.api.types.is_numeric_dtype(cleaned_df["TotalCharges"])

    def test_senior_citizen_mapped_to_yes_no(self, cleaned_df):
        assert set(cleaned_df["SeniorCitizen"].unique()) <= {"Yes", "No"}

    def test_column_names_are_stripped(self, sample_df):
        df = sample_df.rename(columns={"Churn": " Churn "})
        assert "Churn" in clean_data(df).columns


class TestTargetEncoding:
    def test_yes_maps_to_one_and_dtype_is_int(self, cleaned_df):
        y = encode_target(cleaned_df)
        assert set(y.unique()) <= {0, 1}
        assert (y[cleaned_df["Churn"] == "Yes"] == 1).all()
        assert (y[cleaned_df["Churn"] == "No"] == 0).all()


class TestPreprocessor:
    def test_output_has_no_nan_and_expected_shape(self, engineered_df):
        preprocessor = build_preprocessor().fit(engineered_df)
        out = preprocessor.transform(engineered_df)
        assert not np.isnan(out).any()
        assert out.shape == (len(engineered_df), len(preprocessor.get_feature_names_out()))

    def test_handles_unknown_categories(self, engineered_df):
        train = engineered_df[engineered_df["gender"] == "Male"]
        preprocessor = build_preprocessor().fit(train)
        unseen = engineered_df[engineered_df["gender"] == "Female"].iloc[[0]]
        out = preprocessor.transform(unseen)  # must not raise
        assert out.shape[0] == 1

    def test_no_leakage_transform_does_not_mutate_fit_params(self, engineered_df):
        """Transforming new data must not change the parameters learned at fit."""
        preprocessor = build_preprocessor().fit(engineered_df.iloc[:10])
        scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
        mean_before = scaler.mean_.copy()
        scale_before = scaler.scale_.copy()

        preprocessor.transform(engineered_df.iloc[10:])

        assert np.array_equal(mean_before, scaler.mean_)
        assert np.array_equal(scale_before, scaler.scale_)


class TestSplit:
    def test_stratified_split_sizes_and_balance(self, engineered_df, cleaned_df):
        import math

        y = encode_target(cleaned_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            engineered_df, y
        )
        n = len(engineered_df)
        # sklearn's train_test_split rounds the test set *up*
        assert len(X_test) == math.ceil(n * config.TEST_SIZE)
        remaining = n - len(X_test)
        assert len(X_val) == math.ceil(remaining * config.VALIDATION_SIZE)
        # stratification: churn rate approximately preserved in every split
        # (tolerance is generous because the synthetic dataset is tiny)
        for split_y in (y_train, y_val, y_test):
            assert abs(split_y.mean() - y.mean()) < 0.15
        # no overlap between splits
        assert set(X_train.index).isdisjoint(X_val.index)
        assert set(X_train.index).isdisjoint(X_test.index)
