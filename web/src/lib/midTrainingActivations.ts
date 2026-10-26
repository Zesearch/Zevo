import type { HeartbeatDTO } from "./api";

export function isMidTrainingActivation(heartbeat: HeartbeatDTO): boolean {
  return !!heartbeat.mid_training && !heartbeat.child_ticket_id
    && !["emit_ticket", "mark_done", "mark_failed"].includes(heartbeat.action)
    && heartbeat.activation_phase !== "instruction";
}

export function midTrainingActivationLabel(heartbeat: HeartbeatDTO, activations: HeartbeatDTO[]): string {
  const context = heartbeat.mid_training;
  if (!context) return "Orchestrator";
  const peers = activations.filter((item) => item.mid_training?.train_ticket_id === context.train_ticket_id
    && item.mid_training?.step === context.step);
  const attempt = peers.findIndex((item) => item.id === heartbeat.id) + 1;
  const retry = attempt > 1 ? ` · Retry ${attempt}` : "";
  const outcome = context.action === "continue" ? `Continue → ${context.next_step?.toLocaleString()}`
    : context.action === "stop" ? "Stop"
    : heartbeat.is_live ? "Deciding…"
    : heartbeat.exit_code !== 0 ? "Failed" : "No decision recorded";
  return `Step ${context.step.toLocaleString()}${retry} · ${outcome}`;
}
