import type { HeartbeatDTO } from "./api";

export function executionPhase(h: HeartbeatDTO): string {
  if (["remeasure", "remeasure_collect", "instruction", "continue"].includes(h.activation_phase)) return h.activation_phase;
  return h.execution_phase || h.activation_phase;
}
