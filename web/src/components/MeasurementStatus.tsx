import type { RunDetail } from "../lib/api";

export function MeasurementStatus({ run }: { run: RunDetail }) {
  const lanes = new Set((run.lifecycle?.token_budget_remeasurements || [])
    .filter(batch => batch.status === "running" || batch.status === "paused")
    .map(batch => batch.lane === "held_out_test" ? "Test" : "Validation"));
  if (!lanes.size) return null;
  return <p className="mb-3 text-sm text-dim" role="status">
    {[...lanes].join(" and ")} remeasurement in progress. Showing previously published results;
    pending iterations will appear when the comparison is complete.
  </p>;
}
