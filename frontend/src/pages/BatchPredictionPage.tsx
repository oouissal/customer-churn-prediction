import { useState } from "react";
import { api } from "../api/client";
import type { BatchPredictionRow } from "../api/types";
import RiskBadge from "../components/RiskBadge";
import { Card, ErrorBanner, PageHeader } from "../components/ui";

const REQUIRED_COLUMNS = [
  "gender",
  "SeniorCitizen",
  "Partner",
  "Dependents",
  "tenure",
  "PhoneService",
  "MultipleLines",
  "InternetService",
  "OnlineSecurity",
  "OnlineBackup",
  "DeviceProtection",
  "TechSupport",
  "StreamingTV",
  "StreamingMovies",
  "Contract",
  "PaperlessBilling",
  "PaymentMethod",
  "MonthlyCharges",
  "TotalCharges",
];

export default function BatchPredictionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [rows, setRows] = useState<BatchPredictionRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloaded, setDownloaded] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!file) {
      setError("Please choose a CSV file first.");
      return;
    }
    setLoading(true);
    setError(null);
    setDownloaded(false);
    try {
      const csv = await api.predictBatchCsv(file);
      // Parse the returned CSV so the table can be rendered.
      const lines = csv.trim().split(/\r?\n/);
      const header = lines[0].split(",");
      const parsed: BatchPredictionRow[] = lines.slice(1).map((line) => {
        const cells = line.split(",");
        const record: Record<string, string> = {};
        header.forEach((name, index) => {
          record[name] = cells[index] ?? "";
        });
        return {
          customer_id: record["customer_id"] || null,
          churn_probability: Number(record["churn_probability"]),
          prediction:
            record["prediction"] === "CHURN" ? "CHURN" : "NO_CHURN",
          risk_level: (record["risk_level"] ?? "LOW") as
            | "LOW"
            | "MEDIUM"
            | "HIGH",
          confidence: Number(record["confidence"]),
        };
      });
      setRows(parsed);
      // Trigger a download of the enriched CSV.
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "churn_predictions.csv";
      anchor.click();
      URL.revokeObjectURL(url);
      setDownloaded(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
      setRows([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Batch Prediction"
        subtitle="Upload a CSV of customers and download the scored file"
      />

      <Card
        title="Upload CSV"
        subtitle={`Required columns: ${REQUIRED_COLUMNS.join(", ")} (customerID optional)`}
      >
        <form onSubmit={submit} className="flex flex-wrap items-center gap-3">
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-600 file:mr-3 file:rounded-md file:border-0 file:bg-indigo-50 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-indigo-700 hover:file:bg-indigo-100"
          />
          <button
            type="submit"
            disabled={loading || !file}
            className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
          >
            {loading ? "Scoring…" : "Score file"}
          </button>
          {file && (
            <p className="text-xs text-slate-500">
              {file.name} ({(file.size / 1024).toFixed(1)} KB)
            </p>
          )}
        </form>
      </Card>

      {downloaded && (
        <p className="mt-3 text-xs text-slate-500">
          ✓ The enriched CSV (churn_predictions.csv) was downloaded
          automatically.
        </p>
      )}

      {error && (
        <div className="mt-4">
          <ErrorBanner message={error} />
        </div>
      )}

      {rows.length > 0 && (
        <div className="mt-6">
          <Card title={`Results (${rows.length} customers)`}>
            <div className="max-h-96 overflow-auto">
              <table className="w-full text-left text-sm">
                <thead className="sticky top-0 bg-slate-50 text-xs uppercase text-slate-500">
                  <tr>
                    <th className="px-3 py-2">Customer ID</th>
                    <th className="px-3 py-2">Prediction</th>
                    <th className="px-3 py-2">Probability</th>
                    <th className="px-3 py-2">Risk</th>
                    <th className="px-3 py-2">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {rows.map((row, index) => (
                    <tr key={`${row.customer_id ?? "row"}-${index}`}>
                      <td className="px-3 py-2 text-slate-600">
                        {row.customer_id ?? "—"}
                      </td>
                      <td className="px-3 py-2">
                        <span
                          className={`font-semibold ${
                            row.prediction === "CHURN"
                              ? "text-rose-600"
                              : "text-emerald-600"
                          }`}
                        >
                          {row.prediction}
                        </span>
                      </td>
                      <td className="px-3 py-2">
                        {(row.churn_probability * 100).toFixed(1)}%
                      </td>
                      <td className="px-3 py-2">
                        <RiskBadge level={row.risk_level} />
                      </td>
                      <td className="px-3 py-2 text-slate-600">
                        {(row.confidence * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}

