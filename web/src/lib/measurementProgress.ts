import type { TicketDetail, ExecutionEventDTO } from "./api";
import { benchmarkCounts } from "./executionProgress";

export function measurementProgress(ticket: TicketDetail, events: ExecutionEventDTO[] = ticket.execution_events,
  useCurrentAttempt = true) {
  const attempt = useCurrentAttempt ? ticket.measurement_attempt : undefined;
  const members = attempt?.revision && attempt.started_at ? attempt.active_members : undefined;
  const scoped = events.filter((event) => event.extras?.execution_purpose === "workload"
    && (!members || Date.parse(event.ts) >= Date.parse(attempt!.started_at!)));
  const assigned = ticket.measurement_members;
  if (assigned) {
    const expected = assigned.filter(member => !members || members.includes(member.name));
    const byId = new Map(expected.map(member => [member.id, member]));
    const done = new Set<string>();
    for (const event of scoped) {
      if (event.extras?.benchmark_complete !== true) continue;
      if (["preflight", "test", "mock", "diagnostic"].includes(String(event.extras.progress_scope))) continue;
      const id = String(event.extras.benchmark_id || "");
      // IDs take precedence; a foreign ID cannot borrow an assigned name.
      const member = id ? byId.get(id)
        : expected.find(item => item.name === event.extras.benchmark_name);
      if (member) done.add(member.id);
    }
    return expected.length ? { completed: done.size, total: expected.length } : null;
  }
  return benchmarkCounts(scoped, members);
}
