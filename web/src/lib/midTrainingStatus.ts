import type { ExecutionEventDTO, HeartbeatDTO, TicketDetail } from "./api";
import { trainingActivity, isMidTrainingValidationEvent } from "./trainingTelemetry";

export function midTrainingExecution(ticket: TicketDetail) {
  const events = [...ticket.execution_events].filter((event) => event.extras?.execution_purpose === "workload")
    .sort((a, b) => a.ts.localeCompare(b.ts));
  let start = 0;
  events.forEach((event, index) => {
    if (["train", "training", "adaptive_training"].includes(event.phase)) start = index + 1;
  });
  const current = events.slice(start);
  const activity = trainingActivity(events, ticket.mid_training_validation, "running");
  const phase = activity === "Mid-training Validation · Inference" ? "inference"
    : activity === "Mid-training Validation · Scoring" ? "scoring" : null;
  return { events: current, activity, phase };
}

export function validationStatus(ticket: TicketDetail, paused: boolean): string {
  if (ticket.mid_training_validation?.complete) return "succeeded";
  const { phase } = midTrainingExecution(ticket);
  if (phase && ["failed", "cancelled"].includes(ticket.status)) return ticket.status;
  if (paused) return "paused";
  if (phase) return "running";
  if (ticket.mid_training_validation?.awaiting_decision) return "succeeded";
  if (ticket.status === "cancelled") return "cancelled";
  return "queued";
}

export function decisionStatus(activations: HeartbeatDTO[], paused: boolean, awaitingDecision = false): string {
  if (paused && (awaitingDecision || activations.some((item) => item.is_live))) return "paused";
  if (activations.some((item) => item.is_live)) return "running";
  const latest = activations.at(-1);
  if (latest && latest.exit_code !== 0) return "failed";
  if (awaitingDecision) return "queued";
  if (latest?.mid_training?.action) return "succeeded";
  return "queued";
}

export function inferenceProgress(events: ExecutionEventDTO[]) {
  const latest = events.filter((event) => event.phase.startsWith("generate")).at(-1);
  if (!latest) return null;
  const suiteTotal = Number(latest.extras?.suite_rows_total);
  const suiteDone = Number(latest.extras?.suite_rows_completed);
  const suite = Number.isFinite(suiteTotal) && suiteTotal > 0 && Number.isFinite(suiteDone);
  return { name: String(latest.extras?.benchmark_name || "Validation"),
    completed: suite ? suiteDone : latest.current_step, total: suite ? suiteTotal : latest.total_steps,
    suite, workers: Number(latest.extras?.parallel_workers || 0) };
}

/** Reveal the branch on its first real validation, then retain its history. */
export function midTrainingHasStarted(ticket: TicketDetail | undefined, activationCount = 0): boolean {
  if (activationCount > 0) return true;
  if (!ticket) return false;
  const validation = ticket.mid_training_validation;
  if (validation?.measurements?.length || validation?.awaiting_decision || validation?.complete) return true;
  return ticket.execution_events.some((event) => event.extras?.execution_purpose === "workload"
    && isMidTrainingValidationEvent(event));
}
