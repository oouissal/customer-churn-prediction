import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ModelInfoResponse } from "../api/types";
import StatCard from "../components/StatCard";
import { Card, ErrorBanner, LoadingBlock, PageHeader } from "../components/ui";

const METRIC_LABELS: Record<string, string> = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1: "F1 score",
  roc_auc: "ROC-AUC",
};

export default function ModelInfoPage() {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .modelInfo()
      .then(setInfo)
      .catch((err: Error) => setError(err.message));
  }, []);

  if (error) return <ErrorBanner message={error} />;
  if (!info) return <LoadingBlock>Loading model information…</LoadingBlock>;

  return (
    <div>
      <PageHeader
        title="Model Information"
        subtitle="The model currently deployed behind the API"
      />

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <StatCard label="Model" value={info.model_name} />
        <StatCard label="Version" value={info.model_version} />
        <StatCard
          label="Training date"
          value={
            info.trained_at
              ? new Date(info.trained_at).toLocaleDateString()
              : "—"
          }
        />
        <StatCard
          label="Decision threshold"
          value={`${(info.threshold * 100).toFixed(0)}%`}
          hint="F1-tuned on validation"
        />
        <StatCard label="Feature count" value={String(info.feature_count)} />
      </div>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card
          title="Test metrics"
          subtitle="Hold-out test set at the tuned threshold"
        >
          <div className="space-y-3">
            {Object.entries(info.metrics).map(([key, value]) => (
              <div key={key}>
                <p className="text-xs text-slate-500">
                  {METRIC_LABELS[key] ?? key}
                </p>
                <p className="text-xl font-semibold text-slate-900">
                  {key === "threshold"
                    ? `${(value * 100).toFixed(0)}%`
                    : (value * 100).toFixed(1) + "%"}
                </p>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Lifecycle" subtitle="Model registry metadata (MLflow)">
          <div className="space-y-3 text-sm">
            <div>
              <p className="text-xs text-slate-500">Registered model</p>
              <p className="font-medium text-slate-800">
                {info.mlflow_registered_model ?? "—"}
              </p>
            </div>
            <div>
              <p className="text-xs text-slate-500">Registry version</p>
              <p className="font-medium text-slate-800">{info.model_version}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500">MLflow run ID</p>
              <p className="break-all font-mono text-xs text-slate-800">
                {info.mlflow_run_id ?? "—"}
              </p>
            </div>
            <div>
              <p className="text-xs text-slate-500">Stage</p>
              <p className="font-medium text-slate-800">
                Development → Validation → <b>Production</b> (simulated)
              </p>
            </div>
          </div>
        </Card>

        <Card
          title="About the metrics"
          subtitle="Why these numbers matter for churn"
        >
          <ul className="space-y-2 text-sm text-slate-600">
            <li>
              <b>Recall</b> measures how many real churners were caught — the
              key metric for a retention campaign.
            </li>
            <li>
              <b>Precision</b> measures how many flagged customers actually
              churn — avoids wasting retention effort.
            </li>
            <li>
              <b>ROC-AUC</b> ranks customers by risk regardless of threshold.
            </li>
            <li>
              The threshold is tuned on the validation set to maximise F1 for
              the churn class, not fixed at 0.5.
            </li>
          </ul>
        </Card>
      </div>
    </div>
  );
}
