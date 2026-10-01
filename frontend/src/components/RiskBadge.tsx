import type { RiskLevel } from "../api/types";

const STYLES: Record<RiskLevel, string> = {
  LOW: "bg-emerald-100 text-emerald-700",
  MEDIUM: "bg-amber-100 text-amber-700",
  HIGH: "bg-rose-100 text-rose-700",
};

export default function RiskBadge({ level }: { level: RiskLevel }) {
  return (
    <span
      className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold ${STYLES[level]}`}
    >
      {level}
    </span>
  );
}
