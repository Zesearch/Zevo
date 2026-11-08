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
  const phaseLabel = context.phase === "handoff_failed" ? "Review handoff"
    : context.phase === "inference" ? "Start inference"
    : context.phase === "evaluation" ? "Start evaluation"
    : context.phase === "repair_inference" ? "Start inference"
    : context.phase === "repair_evaluation" ? "Start evaluation" : "";
  const outcome = phaseLabel ? `${phaseLabel}${heartbeat.is_live ? "…" : heartbeat.exit_code !== 0 ? " · Failed" : ""}`
    : context.action === "continue" ? `Continue → ${context.next_step?.toLocaleString()}`
    : context.action === "stop" ? "Stop"
    : heartbeat.is_live ? "Deciding…"
    : heartbeat.exit_code !== 0 ? "Failed" : "No decision recorded";
  const repair = ["handoff_failed", "repair_inference", "repair_evaluation"].includes(context.phase || "")
    || heartbeat.activation_phase === "repair";
  return `Step ${context.step.toLocaleString()}${retry} · ${outcome}${repair ? " · Repair" : ""}`;
}

export function midTrainingActionLabel(heartbeat?: HeartbeatDTO): string {
  if (!heartbeat) return "Plan validation";
  const context = heartbeat.mid_training;
  if (context?.action === "continue") return "Resume training";
  if (context?.action === "stop") return "Stop training";
  if (context?.phase === "handoff_failed") return "Review handoff · Repair";
  if (heartbeat.action === "mark_failed") return "Stop Run";
  if (heartbeat.action === "wait") return "Wait for dependencies";
  if (context?.phase === "inference") return "Start Inference";
  if (context?.phase === "evaluation") return "Start Evaluation";
  if (context?.phase === "repair_inference") return "Start Inference · Repair";
  if (context?.phase === "repair_evaluation") return "Start Evaluation · Repair";
  return heartbeat.is_live ? "Decide next training step…" : "Review validation";
}
