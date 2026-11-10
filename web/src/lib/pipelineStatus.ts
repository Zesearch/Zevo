import type { HeartbeatDTO, InfraInstanceDTO, RunDetail, RunInstructionDTO } from "./api";
import { instructionBlocks, stageStatus, ticketRepairing } from "./stageStatus";
import { timelineTicketStatus } from "./timelineStatus";

// The iteration header and loop share exactly five lamp meanings.
export const PIPELINE_LAMPS = {
  idle: { color: "#3B4C5B", dot: "bg-slate-600", label: "idle" },
  waiting: { color: "#5FB0DD", dot: "bg-skyx-400", label: "waiting" },
  active: { color: "#E08D38", dot: "bg-brass-500", label: "running" },
  done: { color: "#4FD1B5", dot: "bg-phosphor-400", label: "done" },
  failed: { color: "#F26F58", dot: "bg-coral-400", label: "failed" },
} as const;
export type PipelineState = keyof typeof PIPELINE_LAMPS;

export function pipelineState(status: string): PipelineState {
  if (["running", "repairing"].includes(status)) return "active";
  if (["queued", "planning", "awaiting_input", "waiting_external", "needs_input", "paused"].includes(status)) return "waiting";
  if (["succeeded", "success", "skipped"].includes(status)) return "done";
  if (["failed", "degraded"].includes(status)) return "failed";
  return "idle";
}

type Ticket = Omit<RunDetail["tickets"][number], "status"> & { status: string };
type Resource = Pick<InfraInstanceDTO, "ticket_id" | "created_at" | "meta">;
type Activation = Pick<HeartbeatDTO, "ticket_id" | "is_live" | "started_at" | "activation_phase">;
type Instruction = Pick<RunInstructionDTO, "status" | "target_ticket_id">;

/** Resolve the visible state once for rows, iteration lamps and loop stations.
 * In particular, a finished submit Agent does not mean its remote work is idle. */
export function pipelineTicketStatuses(
  tickets: Ticket[], resources: Resource[], heartbeats: Activation[], instructions: Instruction[],
): Map<string, string> {
  const blocking = instructions.filter((i) => instructionBlocks(i.status));
  const needsInput = blocking.some((i) => i.status === "needs_input");
  const paused = new Set(blocking.map((i) => i.target_ticket_id).filter(Boolean));
  if (blocking.length) {
    const optimization = tickets.filter((t) => t.lane !== "held_out_test" && t.agent_id !== "orchestrator");
    for (const ticket of optimization) {
      if (["running", "repairing"].includes(ticket.status)) paused.add(ticket.id);
    }
    if (!paused.size) {
      const waiting = optimization.filter((t) => ["queued", "awaiting_input", "waiting_external"].includes(t.status))
        .sort((a, b) => a.created_at.localeCompare(b.created_at)).at(-1);
      if (waiting) paused.add(waiting.id);
    }
  }
  const latestResources = new Map<string, Resource>();
  for (const resource of resources) {
    if (!resource.ticket_id || !resource.meta.resource_request || resource.meta.superseded_by_instruction_id) continue;
    const prior = latestResources.get(resource.ticket_id);
    if (!prior || resource.created_at > prior.created_at) latestResources.set(resource.ticket_id, resource);
  }
  const live = new Map<string, Activation>();
  for (const heartbeat of heartbeats) {
    const prior = live.get(heartbeat.ticket_id);
    if (heartbeat.is_live && (!prior || heartbeat.started_at > prior.started_at)) live.set(heartbeat.ticket_id, heartbeat);
  }
  return new Map(tickets.map((ticket) => {
    const scheduler = String(latestResources.get(ticket.id)?.meta.scheduler_state || "");
    const isPaused = paused.has(ticket.id);
    const phase = live.get(ticket.id)?.activation_phase;
    const visible = stageStatus(ticket.status, isPaused, scheduler, isPaused && needsInput, phase);
    return [ticket.id, ["running", "waiting_external"].includes(visible) && ticketRepairing(ticket, phase)
      ? "repairing" : visible];
  }));
}

/** A station spans iterations; the latest work in each lane/stage supersedes
 * its earlier attempts, while concurrent held-out work remains independent. */
export function stationStateFromTickets(tickets: Ticket[]): PipelineState {
  return pipelineState(timelineTicketStatus(tickets.map((ticket) => ({ ...ticket, iteration: 0 }))));
}
