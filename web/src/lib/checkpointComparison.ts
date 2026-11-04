import type { IterationHistoryEntry, MidTrainingValidation } from "./api";

export type ComparisonPoint = {
  id: string;
  label: string;
  score: number;
  metrics: Record<string, number>;
};

export function comparisonPoints(data: MidTrainingValidation, baseline?: IterationHistoryEntry): ComparisonPoint[] {
  return [
    ...(baseline ? [{ id: "baseline", label: "Baseline", score: baseline.score,
      metrics: baseline.validation_scores || {} }] : []),
    ...data.measurements.map((row) => ({ id: `step:${row.step}`, label: `Step ${row.step.toLocaleString()}`,
      score: row.score, metrics: row.metrics })),
  ];
}

/** Missing benchmarks stay missing: they must never become zero-score regressions. */
export function comparisonRows(left: ComparisonPoint, right: ComparisonPoint) {
  return [
    { name: "Suite average", left: left.score, right: right.score },
    ...[...new Set([...Object.keys(left.metrics), ...Object.keys(right.metrics)])].map((name) => ({
      name, left: left.metrics[name], right: right.metrics[name],
    })),
  ].map((row) => ({ ...row,
    delta: Number.isFinite(row.left) && Number.isFinite(row.right) ? row.right - row.left : null,
  }));
}
