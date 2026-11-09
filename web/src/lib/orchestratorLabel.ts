import type { HeartbeatDTO } from "./api";

/** Keep the existing action wording; identify repair using backend provenance. */
export function withOrchestratorRepairLabel(heartbeat: HeartbeatDTO, action: string): string {
  const source = heartbeat.activation_phase === "repair"
    ? "orchestrator" : heartbeat.repair_source_agent_id;
  if (!source) return action;
  const label = source.charAt(0).toUpperCase() + source.slice(1);
  return `Repair · ${label} → ${action}`;
}
