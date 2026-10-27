import type { ExecutionEventDTO, MidTrainingValidation } from "./api";

/** Validation uses the Train workload, but its row counters are not optimizer steps. */
export function isMidTrainingValidationEvent(event: ExecutionEventDTO): boolean {
  const extras = event.extras || {};
  return !!(extras.benchmark_name || extras.benchmark_id || extras.adaptive_request || extras.adaptive_report)
    || "benchmark_index" in extras || "benchmark_total" in extras
    || event.phase === "adaptive_checkpoint" || event.phase === "adaptive_validation" || event.phase === "adaptive_evaluation" || event.phase.startsWith("scoring")
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
  if (latest.phase === "adaptive_checkpoint") return "Awaiting checkpoint validation";
  if (latest.phase === "adaptive_evaluation") return "Awaiting checkpoint evaluation";
  if (latest.phase.startsWith("scoring") || latest.extras?.phase === "model_judge"
      || latest.extras?.phase === "benchmark_complete") return "Mid-training Validation · Scoring";
  if (isMidTrainingValidationEvent(latest) && latest.phase.startsWith("generate"))
    return "Mid-training Validation · Inference";
  if (latest.phase === "adaptive_validation") return "Mid-training Validation";
  if (latest.phase === "adaptive_training" || latest.phase === "train" || latest.phase === "training"
      || Number(latest.extras?.loss ?? latest.loss) >= 0) return "Training";
  return null;
}

/** Keep checkpoint resumes on one global-step timeline. A replay or rollback
 * replaces the old suffix, so abandoned steps cannot survive a restart. */
export function continuousTrainingEvents(events: ExecutionEventDTO[]): ExecutionEventDTO[] {
  let result: ExecutionEventDTO[] = [];
  const seenAttempts = new Set<string>();
  for (const event of [...events].sort((a, b) => a.ts.localeCompare(b.ts))) {
    if (isMidTrainingValidationEvent(event)) continue;
    const loss = event.extras?.loss ?? event.loss;
    const isTrainingReading = event.event_type === "progress" && event.current_step > 0
      && typeof loss === "number" && loss >= 0
      && event.extras?.eval_loss == null && event.extras?.validation_loss == null;
    if (isTrainingReading && event.attempt_id && !seenAttempts.has(event.attempt_id)) {
      seenAttempts.add(event.attempt_id);
      result = result.filter((prior) => prior.event_type !== "progress"
        || prior.current_step < event.current_step);
    }
    result.push(event);
  }
  return result;
}
