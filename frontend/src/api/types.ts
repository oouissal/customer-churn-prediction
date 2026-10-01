// API types mirroring the FastAPI schemas (backend/app/api/schemas).

export type YesNo = "Yes" | "No";
export type NoPhoneService = "Yes" | "No" | "No phone service";
export type NoInternetService = "Yes" | "No" | "No internet service";
export type InternetService = "DSL" | "Fiber optic" | "No";
export type Contract = "Month-to-month" | "One year" | "Two year";
export type PaymentMethod =
  | "Electronic check"
  | "Mailed check"
  | "Bank transfer (automatic)"
  | "Credit card (automatic)";

export interface CustomerFeatures {
  gender: "Male" | "Female";
  SeniorCitizen: YesNo;
  Partner: YesNo;
  Dependents: YesNo;
  tenure: number;
  PhoneService: YesNo;
  MultipleLines: NoPhoneService;
  InternetService: InternetService;
  OnlineSecurity: NoInternetService;
  OnlineBackup: NoInternetService;
  DeviceProtection: NoInternetService;
  TechSupport: NoInternetService;
  StreamingTV: NoInternetService;
  StreamingMovies: NoInternetService;
  Contract: Contract;
  PaperlessBilling: YesNo;
  PaymentMethod: PaymentMethod;
  MonthlyCharges: number;
  TotalCharges: number;
}

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";
export type PredictionLabel = "CHURN" | "NO_CHURN";

export interface TopDriver {
  feature: string;
  value: string | number | null;
  impact: string;
  probability_shift: number;
}

export interface PredictionResponse {
  prediction: PredictionLabel;
  probability: number;
  risk_level: RiskLevel;
  confidence: number;
  threshold: number;
  model_version: string;
  top_drivers: TopDriver[];
  explanation: string | null;
}

export interface BatchPredictionRow {
  customer_id: string | null;
  churn_probability: number;
  prediction: PredictionLabel;
  risk_level: RiskLevel;
  confidence: number;
}

export interface BatchPredictionResponse {
  total: number;
  model_version: string;
  threshold: number;
  results: BatchPredictionRow[];
}

export interface ModelInfoResponse {
  model_name: string;
  model_version: string;
  trained_at: string | null;
  threshold: number;
  feature_count: number;
  metrics: {
    threshold: number;
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    roc_auc: number;
  };
  mlflow_run_id: string | null;
  mlflow_registered_model: string | null;
}

export interface HealthResponse {
  status: string;
  api_version: string;
  model_loaded: boolean;
  model_version: string | null;
}

export interface DashboardSummary {
  source: string;
  total_customers: number;
  predicted_churn_rate: number;
  high_risk_customers: number;
  average_churn_probability: number;
  model_version: string;
  model_name: string;
  threshold: number;
  churn_distribution: { name: string; value: number }[];
  risk_distribution: { name: string; value: number }[];
  probability_distribution: { bucket: string; count: number }[];
}

export interface DashboardDrivers {
  source: string;
  drivers: { feature: string; importance: number }[];
}
