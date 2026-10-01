// Form definition for the customer prediction page. Values are restricted to
// the dataset vocabulary; the backend re-validates everything with Pydantic.
import type { CustomerFeatures } from "../api/types";

export interface SelectField {
  name: keyof CustomerFeatures;
  label: string;
  type: "select";
  options: string[];
}

export interface NumberField {
  name: keyof CustomerFeatures;
  label: string;
  type: "number";
  min?: number;
  max?: number;
  step?: number;
}

export type FormField = SelectField | NumberField;

export const FORM_FIELDS: FormField[] = [
  { name: "gender", label: "Gender", type: "select", options: ["Male", "Female"] },
  { name: "SeniorCitizen", label: "Senior citizen", type: "select", options: ["No", "Yes"] },
  { name: "Partner", label: "Partner", type: "select", options: ["No", "Yes"] },
  { name: "Dependents", label: "Dependents", type: "select", options: ["No", "Yes"] },
  { name: "tenure", label: "Tenure (months)", type: "number", min: 0, max: 100 },
  { name: "PhoneService", label: "Phone service", type: "select", options: ["No", "Yes"] },
  {
    name: "MultipleLines",
    label: "Multiple lines",
    type: "select",
    options: ["No", "Yes", "No phone service"],
  },
  {
    name: "InternetService",
    label: "Internet service",
    type: "select",
    options: ["DSL", "Fiber optic", "No"],
  },
  {
    name: "OnlineSecurity",
    label: "Online security",
    type: "select",
    options: ["No", "Yes", "No internet service"],
  },
  {
    name: "OnlineBackup",
    label: "Online backup",
    type: "select",
    options: ["No", "Yes", "No internet service"],
  },
  {
    name: "DeviceProtection",
    label: "Device protection",
    type: "select",
    options: ["No", "Yes", "No internet service"],
  },
  {
    name: "TechSupport",
    label: "Tech support",
    type: "select",
    options: ["No", "Yes", "No internet service"],
  },
  {
    name: "StreamingTV",
    label: "Streaming TV",
    type: "select",
    options: ["No", "Yes", "No internet service"],
  },
  {
    name: "StreamingMovies",
    label: "Streaming movies",
    type: "select",
    options: ["No", "Yes", "No internet service"],
  },
  {
    name: "Contract",
    label: "Contract",
    type: "select",
    options: ["Month-to-month", "One year", "Two year"],
  },
  { name: "PaperlessBilling", label: "Paperless billing", type: "select", options: ["No", "Yes"] },
  {
    name: "PaymentMethod",
    label: "Payment method",
    type: "select",
    options: [
      "Electronic check",
      "Mailed check",
      "Bank transfer (automatic)",
      "Credit card (automatic)",
    ],
  },
  { name: "MonthlyCharges", label: "Monthly charges ($)", type: "number", min: 0.01, max: 250, step: 0.01 },
  { name: "TotalCharges", label: "Total charges ($)", type: "number", min: 0, max: 10000, step: 0.01 },
];

/** A realistic high-risk profile (short tenure, fibre, month-to-month). */
export const SAMPLE_HIGH_RISK: CustomerFeatures = {
  gender: "Male",
  SeniorCitizen: "No",
  Partner: "No",
  Dependents: "No",
  tenure: 3,
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "Yes",
  StreamingMovies: "Yes",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 89.5,
  TotalCharges: 268.5,
};

/** A realistic low-risk profile (long tenure, two-year contract, DSL). */
export const SAMPLE_LOW_RISK: CustomerFeatures = {
  gender: "Female",
  SeniorCitizen: "No",
  Partner: "Yes",
  Dependents: "No",
  tenure: 48,
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "DSL",
  OnlineSecurity: "Yes",
  OnlineBackup: "Yes",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  Contract: "Two year",
  PaperlessBilling: "No",
  PaymentMethod: "Bank transfer (automatic)",
  MonthlyCharges: 52.35,
  TotalCharges: 2512.8,
};

export function emptyForm(): CustomerFeatures {
  return {
    gender: "Male",
    SeniorCitizen: "No",
    Partner: "No",
    Dependents: "No",
    tenure: 12,
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "DSL",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "Month-to-month",
    PaperlessBilling: "No",
    PaymentMethod: "Electronic check",
    MonthlyCharges: 50,
    TotalCharges: 600,
  };
}
