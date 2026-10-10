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
    ticket.model_source, ticket.test_set_name,
  ]);
}

/** Summarize effective stage outcomes while preserving historical Ticket rows.
 * Only a later success in the same stage can recover an unsuccessful attempt. */
export function timelineTicketStatus(tickets: StageTicket[]): string {
  const successes = new Map<string, string>();
  for (const ticket of tickets) {
    if (ticket.status !== "succeeded") continue;
    const key = stageKey(ticket);
    if (ticket.created_at > (successes.get(key) || "")) {
      successes.set(key, ticket.created_at);
    }
  }
  const current = tickets.filter((ticket) => {
    if (!["failed", "degraded", "cancelled"].includes(ticket.status)) return true;
    const recoveredAt = successes.get(stageKey(ticket));
    return !recoveredAt || recoveredAt <= ticket.created_at;
  });
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
