import type { ExecutionEventDTO, MidTrainingValidation } from "./api";

/** Validation uses the Train workload, but its row counters are not optimizer steps. */
export function isMidTrainingValidationEvent(event: ExecutionEventDTO): boolean {
  const extras = event.extras || {};
  return !!(extras.benchmark_name || extras.benchmark_id || extras.adaptive_request || extras.adaptive_report)
    || "benchmark_index" in extras || "benchmark_total" in extras
    || event.phase === "adaptive_validation" || event.phase.startsWith("scoring")
    || extras.phase === "model_judge" || extras.phase === "benchmark_complete";
}

export function trainingActivity(
  events: ExecutionEventDTO[], data: MidTrainingValidation | null | undefined,
  status: string, paused = false,
): string | null {
  if (!["running", "waiting_external", "repairing"].includes(status)) return null;
  if (paused) return "Training paused";
  if (data?.awaiting_decision) return "Awaiting training decision";
  const latest = [...events].filter((event) => event.extras?.execution_purpose === "workload")
    .sort((a, b) => a.ts.localeCompare(b.ts)).at(-1);
  if (!latest) return null;
  if (latest.phase.startsWith("scoring") || latest.extras?.phase === "model_judge"
      || latest.extras?.phase === "benchmark_complete") return "Mid-training Validation · Scoring";
  if (isMidTrainingValidationEvent(latest) && latest.phase.startsWith("generate"))
    return "Mid-training Validation · Inference";
  if (latest.phase === "adaptive_validation") return "Mid-training Validation";
  if (latest.phase === "adaptive_training" || latest.phase === "train" || latest.phase === "training"
      || Number(latest.extras?.loss ?? latest.loss) >= 0) return "Training";
  return null;
}
