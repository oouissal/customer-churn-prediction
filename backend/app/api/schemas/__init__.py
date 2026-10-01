"""Pydantic request/response schemas for the prediction API.

Input validation happens here, before any ML code runs: unknown fields are
rejected, categorical values are restricted to the dataset vocabulary and
numerical values are range-checked. This guarantees the model only ever
receives well-formed input.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Vocabulary of the IBM Telco dataset (mirrors src.validation)
# ---------------------------------------------------------------------------
YesNo = Literal["Yes", "No"]
NoPhoneService = Literal["Yes", "No", "No phone service"]
NoInternetService = Literal["Yes", "No", "No internet service"]
InternetServiceType = Literal["DSL", "Fiber optic", "No"]
ContractType = Literal["Month-to-month", "One year", "Two year"]
PaymentMethodType = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class CustomerFeatures(BaseModel):
    """One customer profile — exactly the raw features the pipeline expects."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Male", "Female"]
    SeniorCitizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: NoPhoneService
    InternetService: InternetServiceType
    OnlineSecurity: NoInternetService
    OnlineBackup: NoInternetService
    DeviceProtection: NoInternetService
    TechSupport: NoInternetService
    StreamingTV: NoInternetService
    StreamingMovies: NoInternetService
    Contract: ContractType
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethodType
    MonthlyCharges: float = Field(gt=0, le=250)
    TotalCharges: float = Field(ge=0, le=10000)

    def to_features_dict(self) -> dict[str, Any]:
        """Return the profile as the plain feature dict used by ``src``."""
        return self.model_dump()


class PredictionRequest(BaseModel):
    """Payload of ``POST /predict``."""

    customer: CustomerFeatures = Field(description="Customer profile to score")


class RiskLevel(StrEnum):
    """Business risk bands derived from the churn probability."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PredictionLabel(StrEnum):
    CHURN = "CHURN"
    NO_CHURN = "NO_CHURN"


class TopDriver(BaseModel):
    """One SHAP-explained feature contribution."""

    feature: str = Field(description="Original feature name")
    value: Any = Field(description="Customer value for that feature")
    impact: str = Field(
        description="Human-readable direction: 'increases' / 'decreases' risk"
    )
    probability_shift: float = Field(
        description="Approximate isolated impact on churn probability (pp)"
    )


class PredictionResponse(BaseModel):
    """Response of ``POST /predict``."""

    prediction: PredictionLabel
    probability: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    confidence: float = Field(ge=0.5, le=1)
    threshold: float = Field(ge=0, le=1)
    model_version: str
    top_drivers: list[TopDriver]
    explanation: str | None = Field(
        default=None,
        description="One-sentence business explanation of the prediction",
    )


class BatchPredictionRow(BaseModel):
    """One row of the batch prediction result."""

    customer_id: str | None = None
    churn_probability: float
    prediction: PredictionLabel
    risk_level: RiskLevel
    confidence: float


class BatchPredictionResponse(BaseModel):
    """Response of ``POST /predict/batch``."""

    total: int
    model_version: str
    threshold: float
    results: list[BatchPredictionRow]


class ModelMetrics(BaseModel):
    """Evaluation metrics of the deployed model (tuned threshold)."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


class ModelInfoResponse(BaseModel):
    """Response of ``GET /model/info``."""

    model_name: str
    model_version: str
    trained_at: str | None
    threshold: float
    feature_count: int
    metrics: ModelMetrics
    mlflow_run_id: str | None = None
    mlflow_registered_model: str | None = None


class HealthResponse(BaseModel):
    """Response of ``GET /health``."""

    status: str
    api_version: str
    model_loaded: bool
    model_version: str | None = None


class ErrorResponse(BaseModel):
    """Standard error payload."""

    detail: str
