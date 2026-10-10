type StageTicket = {
  agent_id: string;
  iteration: number;
  lane: string;
  operation: string;
  model_source: string;
  test_set_name: string;
  created_at: string;
  status: string;
};

function stageKey(ticket: StageTicket): string {
  return JSON.stringify([
    ticket.iteration, ticket.lane, ticket.agent_id, ticket.operation,
    ticket.test_set_name,
  ]);
}

/** Historical attempts remain visible; only the newest ticket in each
 * iteration/lane/stage contributes to its current status. */
export function latestTimelineTickets<T extends StageTicket>(tickets: T[]): T[] {
  const latest = new Map<string, T>();
  for (const ticket of tickets) {
    const key = stageKey(ticket);
    const previous = latest.get(key);
    if (!previous || ticket.created_at > previous.created_at) latest.set(key, ticket);
  }
  return Array.from(latest.values());
}

export function timelineTicketStatus(tickets: StageTicket[]): string {
  const current = latestTimelineTickets(tickets);
  if (current.some((t) => ["running", "repairing"].includes(t.status))) return "running";
  if (current.some((t) => t.status === "failed")) return "failed";
  if (current.some((t) => t.status === "degraded")) return "degraded";
  if (current.length > 0 && current.every((t) => ["succeeded", "skipped"].includes(t.status))) {
    return "succeeded";
  }
  if (current.some((t) => t.status === "cancelled")
    && current.every((t) => ["succeeded", "skipped", "cancelled"].includes(t.status))) {
    return "cancelled";
  }
  return "pending";
}
