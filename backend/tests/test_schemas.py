"""Tests for the Pydantic schemas themselves (unit level)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas import CustomerFeatures, PredictionRequest


class TestCustomerSchema:
    def test_valid_customer_accepted(self, valid_customer):
        model = CustomerFeatures(**valid_customer)
        assert model.tenure == 3
        assert model.Contract == "Month-to-month"

    def test_unknown_field_rejected(self, valid_customer):
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**valid_customer, "extra": 1})

    def test_bad_enum_value_rejected(self, valid_customer):
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**valid_customer, "gender": "Other"})

    def test_negative_monthly_charges_rejected(self, valid_customer):
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**valid_customer, "MonthlyCharges": -1.0})

    def test_tenure_out_of_range_rejected(self, valid_customer):
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**valid_customer, "tenure": 500})

    def test_to_features_dict_matches_input(self, valid_customer):
        model = CustomerFeatures(**valid_customer)
        assert model.to_features_dict() == valid_customer


class TestPredictionRequestSchema:
    def test_requires_customer_key(self):
        with pytest.raises(ValidationError):
            PredictionRequest()

    def test_accepts_customer(self, valid_customer):
        request = PredictionRequest(customer=valid_customer)
        assert request.customer.tenure == 3
