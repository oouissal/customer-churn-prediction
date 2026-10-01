// Thin API client. The React app talks to the ML model exclusively through
// the FastAPI REST API — no ML logic lives in the frontend.
import type {
  CustomerFeatures,
  DashboardDrivers,
  DashboardSummary,
  HealthResponse,
  ModelInfoResponse,
  PredictionResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

/** Error carrying the API detail message for display in the UI. */
export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string | unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) {
        detail = (body.detail as { msg?: string }[])
          .map((item) => item.msg ?? "")
          .filter(Boolean)
          .join("; ");
      }
    } catch {
      // non-JSON error body — keep the generic message
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>("/health"),

  modelInfo: () => request<ModelInfoResponse>("/model/info"),

  predict: (customer: CustomerFeatures) =>
    request<PredictionResponse>("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ customer }),
    }),

  predictBatchCsv: async (file: File): Promise<string> => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(`${API_BASE}/predict/batch?format=csv`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      let detail = `Request failed with status ${response.status}`;
      try {
        const body = (await response.json()) as { detail?: string };
        if (typeof body.detail === "string") detail = body.detail;
      } catch {
        // ignore
      }
      throw new ApiError(response.status, detail);
    }
    return response.text();
  },

  dashboardSummary: () => request<DashboardSummary>("/dashboard/summary"),

  dashboardDrivers: () => request<DashboardDrivers>("/dashboard/drivers"),
};
