import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import type { CustomerFeatures, PredictionResponse } from "../api/types";
import RiskBadge from "../components/RiskBadge";
import { Card, ErrorBanner, PageHeader } from "../components/ui";
import {
  FORM_FIELDS,
  SAMPLE_HIGH_RISK,
  SAMPLE_LOW_RISK,
  emptyForm,
} from "./predictionForm";

export default function PredictionPage() {
  const [form, setForm] = useState<CustomerFeatures>(emptyForm);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const setValue = (name: keyof CustomerFeatures, value: string | number) => {
    setForm((previous) => ({ ...previous, [name]: value }));
  };

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const prediction = await api.predict(form);
      setResult(prediction);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const isChurn = result?.prediction === "CHURN";

  return (
    <div>
      <PageHeader
        title="Customer Prediction"
        subtitle="Score one customer and explain the prediction with SHAP"
      />

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-5">
        <Card title="Customer profile" subtitle="All fields are validated by the API">
          <form onSubmit={submit} className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              {FORM_FIELDS.map((field) => (
                <label key={field.name} className="block">
                  <span className="mb-1 block text-xs font-medium text-slate-600">
                    {field.label}
                  </span>
                  {field.type === "select" ? (
                    <select
                      className="w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-sm focus:border-indigo-500 focus:outline-none"
                      value={String(form[field.name])}
                      onChange={(event) => setValue(field.name, event.target.value)}
                    >
                      {field.options.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="number"
                      step={field.step ?? 1}
                      min={field.min}
                      max={field.max}
                      className="w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-sm focus:border-indigo-500 focus:outline-none"
                      value={form[field.name]}
                      onChange={(event) =>
                        setValue(field.name, Number(event.target.value))
                      }
                    />
                  )}
                </label>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <button
                type="submit"
                disabled={loading}
                className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
              >
                {loading ? "Predicting…" : "Predict churn"}
              </button>
              <button
                type="button"
                onClick={() => setForm(SAMPLE_HIGH_RISK)}
                className="rounded-lg border border-slate-300 px-3 py-2 text-xs font-medium text-slate-600 hover:bg-slate-100"
              >
                Sample: high risk
              </button>
              <button
                type="button"
                onClick={() => setForm(SAMPLE_LOW_RISK)}
                className="rounded-lg border border-slate-300 px-3 py-2 text-xs font-medium text-slate-600 hover:bg-slate-100"
              >
                Sample: low risk
              </button>
            </div>
          </form>
        </Card>

        <div className="space-y-6 xl:col-span-3">
          {error && <ErrorBanner message={error} />}

          {result && (
            <>
              <div
                className={`rounded-xl border p-6 text-white shadow-sm ${
                  isChurn
                    ? "border-rose-300 bg-gradient-to-r from-rose-500 to-rose-600"
                    : "border-emerald-300 bg-gradient-to-r from-emerald-500 to-emerald-600"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wide opacity-80">
                      Prediction
                    </p>
                    <p className="text-4xl font-bold">
                      {isChurn ? "CHURN" : "NO CHURN"}
                    </p>
                  </div>
                  <RiskBadge level={result.risk_level} />
                </div>
                <div className="mt-4 grid grid-cols-3 gap-4 text-center">
                  <div>
                    <p className="text-xs uppercase opacity-80">Probability</p>
                    <p className="text-xl font-semibold">
                      {(result.probability * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs uppercase opacity-80">Confidence</p>
                    <p className="text-xl font-semibold">
                      {(result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs uppercase opacity-80">Model</p>
                    <p className="text-xl font-semibold">
                      {result.model_version}
                    </p>
                  </div>
                </div>
              </div>

              {result.explanation && (
                <p className="text-sm text-slate-600">{result.explanation}</p>
              )}

              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <Card
                  title="Top contributing factors"
                  subtitle="SHAP — original business features"
                >
                  <ul className="space-y-2.5">
                    {result.top_drivers.map((driver) => (
                      <li
                        key={driver.feature}
                        className="flex items-center justify-between gap-2 text-sm"
                      >
                        <div className="min-w-0">
                          <p className="truncate font-medium text-slate-800">
                            {driver.feature}
                            {driver.value !== null && (
                              <span className="text-slate-500">
                                {" "}
                                = {String(driver.value)}
                              </span>
                            )}
                          </p>
                          <p
                            className={`text-xs ${
                              driver.probability_shift > 0
                                ? "text-rose-600"
                                : "text-emerald-600"
                            }`}
                          >
                            {driver.impact}
                          </p>
                        </div>
                        <span
                          className={`shrink-0 rounded-md px-2 py-0.5 text-xs font-semibold ${
                            driver.probability_shift > 0
                              ? "bg-rose-100 text-rose-700"
                              : "bg-emerald-100 text-emerald-700"
                          }`}
                        >
                          {driver.probability_shift > 0 ? "+" : ""}
                          {(driver.probability_shift * 100).toFixed(1)} pp
                        </span>
                      </li>
                    ))}
                  </ul>
                </Card>

                <Card
                  title="Explanation waterfall"
                  subtitle="Approximate impact of each feature on the churn probability"
                >
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart
                      data={[...result.top_drivers].reverse()}
                      layout="vertical"
                      margin={{ left: 30 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                      <XAxis
                        type="number"
                        tick={{ fontSize: 10 }}
                        tickFormatter={(value: number) =>
                          `${(value * 100).toFixed(0)}%`
                        }
                      />
                      <YAxis
                        type="category"
                        dataKey="feature"
                        width={120}
                        tick={{ fontSize: 11 }}
                      />
                      <Tooltip
                        formatter={(value) => [
                          `${((Number(value) as number) * 100).toFixed(1)} pp`,
                          "Impact",
                        ]}
                      />
                      <ReferenceLine x={0} stroke="#64748b" />
                      <Bar dataKey="probability_shift" radius={[0, 3, 3, 0]}>
                        {result.top_drivers.map((driver) => (
                          <Cell
                            key={driver.feature}
                            fill={
                              driver.probability_shift > 0
                                ? "#f43f5e"
                                : "#10b981"
                            }
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </div>
            </>
          )}

          {!result && !error && (
            <div className="flex h-full min-h-64 items-center justify-center rounded-xl border border-dashed border-slate-300 bg-white text-sm text-slate-400">
              Fill in the customer profile and press “Predict churn”.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

