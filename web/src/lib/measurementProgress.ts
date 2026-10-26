import type { TicketDetail } from "./api";

export function measurementProgress(ticket: TicketDetail) {
  const attempt = ticket.measurement_attempt;
  if (!attempt?.revision || !attempt.started_at) return null;
  const names = new Set(attempt.active_members || []);
  const done = new Set<string>();
  for (const event of ticket.execution_events) {
    if (Date.parse(event.ts) < Date.parse(attempt.started_at)) continue;
    if (event.extras?.execution_purpose !== "workload") continue;
    const name = String(event.extras?.benchmark_name || "");
    if (names.has(name) && event.extras?.benchmark_complete === true) done.add(name);
  }
  if (ticket.status === "succeeded") for (const name of names) done.add(name);
  return { completed: done.size, total: names.size };
}
