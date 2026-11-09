import type { TicketDetail, ExecutionEventDTO } from "./api";
import { benchmarkCounts, benchmarkExecutionStarted } from "./executionProgress";

export function measurementProgress(ticket: TicketDetail, events: ExecutionEventDTO[] = ticket.execution_events,
  useCurrentAttempt = true) {
  const workloadEvents = events.filter((event) => event.extras?.execution_purpose === "workload"
    && !["preflight", "test", "mock", "diagnostic"].includes(String(event.extras.progress_scope))
    && event.extras.benchmark_id !== "probe");
  const started = Math.min(...workloadEvents.map(event => Date.parse(event.ts)));
  const attempt = ticket.measurement_attempts?.length
    ? [...ticket.measurement_attempts].filter(item => workloadEvents.length > 0
      && item.started_at && Date.parse(item.started_at) <= started)
      .sort((a, b) => Date.parse(a.started_at!) - Date.parse(b.started_at!)).at(-1)
    : useCurrentAttempt ? ticket.measurement_attempt : undefined;
  const members = attempt?.revision && attempt.started_at ? attempt.active_members : undefined;
  const scoped = workloadEvents.filter(event => !members || Date.parse(event.ts) >= Date.parse(attempt!.started_at!)
      && (!event.extras.benchmark_name || members.includes(String(event.extras.benchmark_name))));
  const assigned = ticket.measurement_members;
  if (assigned) {
    const expected = assigned.filter(member => !members || members.includes(member.name));
    const byId = new Map(expected.map(member => [member.id, member]));
    const assignedEvents = scoped.filter(event => {
      const id = String(event.extras.benchmark_id || "");
      const name = String(event.extras.benchmark_name || "");
      return id ? byId.has(id) : name ? expected.some(member => member.name === name) : true;
    });
    if (!benchmarkExecutionStarted(assignedEvents, "inference")) return null;
    const done = new Set<string>();
    for (const event of assignedEvents) {
      if (event.extras?.benchmark_complete !== true) continue;
      const id = String(event.extras.benchmark_id || "");
      // IDs take precedence; a foreign ID cannot borrow an assigned name.
      const member = id ? byId.get(id)
        : expected.find(item => item.name === event.extras.benchmark_name);
      if (member) done.add(member.id);
    }
    return expected.length ? { completed: done.size, total: expected.length } : null;
  }
  if (!benchmarkExecutionStarted(scoped, "inference")) return null;
  return benchmarkCounts(scoped, members);
}
