import type { TicketDetail, ExecutionEventDTO } from "./api";
import { benchmarkCounts } from "./executionProgress";

export function measurementProgress(ticket: TicketDetail, events: ExecutionEventDTO[] = ticket.execution_events,
  useCurrentAttempt = true) {
  const attempt = useCurrentAttempt ? ticket.measurement_attempt : undefined;
  const members = attempt?.revision && attempt.started_at ? attempt.active_members : undefined;
  const scoped = events.filter((event) => event.extras?.execution_purpose === "workload"
    && (!members || Date.parse(event.ts) >= Date.parse(attempt!.started_at!)));
  return benchmarkCounts(scoped, members);
}
