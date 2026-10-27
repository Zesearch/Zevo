import type { HeartbeatDTO } from "./api";

export function isMidTrainingActivation(heartbeat: HeartbeatDTO): boolean {
  return !!heartbeat.mid_training && heartbeat.activation_phase !== "instruction";
}

export function midTrainingActivationLabel(heartbeat: HeartbeatDTO, activations: HeartbeatDTO[]): string {
  const context = heartbeat.mid_training;
  if (!context) return "Orchestrator";
  const peers = activations.filter((item) => item.mid_training?.train_ticket_id === context.train_ticket_id
    && item.mid_training?.step === context.step
    && (item.mid_training?.phase || "decision") === (context.phase || "decision"));
  const attempt = peers.findIndex((item) => item.id === heartbeat.id) + 1;
  const retry = attempt > 1 ? ` · Retry ${attempt}` : "";
  const phaseLabel = context.phase === "inference" ? "Start inference"
    : context.phase === "evaluation" ? "Start evaluation"
    : context.phase === "repair_inference" ? "Repair inference"
    : context.phase === "repair_evaluation" ? "Repair evaluation" : "";
  const outcome = phaseLabel ? `${phaseLabel}${heartbeat.is_live ? "…" : heartbeat.exit_code !== 0 ? " · Failed" : ""}`
    : context.action === "continue" ? `Continue → ${context.next_step?.toLocaleString()}`
    : context.action === "stop" ? "Stop"
    : heartbeat.is_live ? "Deciding…"
    : heartbeat.exit_code !== 0 ? "Failed" : "No decision recorded";
  return `Step ${context.step.toLocaleString()}${retry} · ${outcome}`;
}

export function midTrainingActionLabel(heartbeat?: HeartbeatDTO): string {
  if (!heartbeat) return "Plan validation";
  const context = heartbeat.mid_training;
  if (context?.action === "continue") return "Resume training";
  if (context?.action === "stop") return "Stop training";
  if (heartbeat.action === "mark_failed") return "Stop Run";
  if (heartbeat.action === "wait") return "Wait for dependencies";
  if (context?.phase === "inference") return "Start Inference";
  if (context?.phase === "evaluation") return "Start Evaluation";
  if (context?.phase === "repair_inference") return "Repair Inference";
  if (context?.phase === "repair_evaluation") return "Repair Evaluation";
  return heartbeat.is_live ? "Decide next training step…" : "Review validation";
}
