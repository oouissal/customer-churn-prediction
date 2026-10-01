import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import type { DashboardDrivers, DashboardSummary } from "../api/types";
import StatCard from "../components/StatCard";
import { Card, ErrorBanner, LoadingBlock, PageHeader } from "../components/ui";

const RISK_COLORS: Record<string, string> = {
  LOW: "#10b981",
  MEDIUM: "#f59e0b",
  HIGH: "#f43f5e",
};

const PIE_COLORS = ["#f43f5e", "#10b981"];

export default function DashboardPage() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [drivers, setDrivers] = useState<DashboardDrivers | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.dashboardSummary(), api.dashboardDrivers()])
      .then(([summaryData, driversData]) => {
        setSummary(summaryData);
        setDrivers(driversData);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  if (error) return <ErrorBanner message={error} />;
  if (!summary || !drivers) return <LoadingBlock>Loading dashboard…</LoadingBlock>;

  return (
    <div>
      <PageHeader
        title="Dashboard"
        subtitle={`Model predictions on the ${summary.source}`}
      />

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <StatCard
          label="Total customers"
          value={summary.total_customers.toLocaleString()}
          hint="hold-out test set"
        />
        <StatCard
          label="Predicted churn rate"
          value={`${(summary.predicted_churn_rate * 100).toFixed(1)}%`}
          hint={`threshold ${(summary.threshold * 100).toFixed(0)}%`}
        />
        <StatCard
          label="High-risk customers"
          value={summary.high_risk_customers.toLocaleString()}
          hint="probability > 70%"
        />
        <StatCard
          label="Avg churn probability"
          value={`${(summary.average_churn_probability * 100).toFixed(1)}%`}
        />
        <StatCard
          label="Model"
          value={summary.model_version}
          hint={summary.model_name}
        />
      </div>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="Churn distribution" subtitle="Predicted classes at the tuned threshold">
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie
                data={summary.churn_distribution}
                dataKey="value"
                nameKey="name"
                innerRadius={60}
                outerRadius={95}
                paddingAngle={2}
              >
                {summary.churn_distribution.map((entry, index) => (
                  <Cell key={entry.name} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Risk distribution" subtitle="LOW / MEDIUM / HIGH business risk bands">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={summary.risk_distribution}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                {summary.risk_distribution.map((entry) => (
                  <Cell key={entry.name} fill={RISK_COLORS[entry.name] ?? "#6366f1"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Churn probability distribution" subtitle="Histogram of predicted probabilities">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={summary.probability_distribution}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="bucket" tick={{ fontSize: 10 }} interval={1} />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Top churn drivers" subtitle={drivers.source}>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart
              data={drivers.drivers}
              layout="vertical"
              margin={{ left: 30 }}
            >
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis
                type="category"
                dataKey="feature"
                width={130}
                tick={{ fontSize: 11 }}
              />
              <Tooltip />
              <Bar dataKey="importance" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}
