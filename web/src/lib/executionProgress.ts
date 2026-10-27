import type { ExecutionEventDTO, HeartbeatDTO } from "./api";

export function latestExecution(events: ExecutionEventDTO[]): ExecutionEventDTO[] {
  const ordered = [...events].sort((a, b) => a.ts.localeCompare(b.ts));
  const identity = ordered.at(-1)?.extras?.execution_id;
  return identity ? ordered.filter((event) => event.extras?.execution_id === identity) : ordered;
}

/** Bound history before the next activation; collect/continue can refer to the
 * same remote execution, while a repair with no workload must not inherit it. */
export function activationExecution(events: ExecutionEventDTO[], heartbeat?: HeartbeatDTO,
  activations: HeartbeatDTO[] = []): ExecutionEventDTO[] {
  if (!heartbeat) return [];
  const next = [...activations].sort((a, b) => a.started_at.localeCompare(b.started_at))
    .find((item) => item.started_at > heartbeat.started_at);
  const available = events.filter((event) => !next || event.ts < next.started_at);
  const own = available.filter((event) => event.heartbeat_id === heartbeat.id
    || event.ts >= heartbeat.started_at);
  const selected = latestExecution(own);
  const identity = selected.at(-1)?.extras?.execution_id;
  if (identity) return available.filter((event) => event.extras?.execution_id === identity)
    .sort((a, b) => a.ts.localeCompare(b.ts));
  if (selected.length) return selected;
  return ["collect", "continue", "resume"].includes(heartbeat.activation_phase)
    ? latestExecution(available) : [];
}

export function benchmarkCounts(events: ExecutionEventDTO[], members?: string[], stage: "inference" | "evaluation" = "inference") {
  const done = new Set<string>();
  let total = members?.length || 0;
  for (const event of events) {
    if (stage === "evaluation" && !event.phase?.startsWith("scoring")
      && event.phase !== "completed"
      && !["model_judge", "benchmark_complete"].includes(String(event.extras?.phase))) continue;
    const scoring = /^scoring (\d+)\/(\d+): /.exec(event.phase || "");
    if (!members && scoring) total = Math.max(total, Number(scoring[2]));
    const name = String(event.extras?.benchmark_name || "");
    if (members && !members.includes(name)) continue;
    if (!members) total = Math.max(total, Number(event.extras?.benchmark_total) || 0);
    if (stage === "evaluation" ? event.extras?.phase === "benchmark_complete"
      : event.extras?.benchmark_complete === true) {
      const key = String(event.extras?.benchmark_id || name);
      if (key) done.add(key);
    }
  }
  if (!total && stage === "evaluation" && events.some((event) => ["scoring", "completed"].includes(event.phase))) {
    total = 1;
    if (events.some((event) => event.phase === "completed")) done.add("single");
  }
  return total > 0 ? { completed: Math.min(done.size, total), total } : null;
}

/** Lifecycle events after generation carry no benchmark label. Keep the latest
 * benchmark metadata within the already selected execution. */
export function latestBenchmarkEvent(events: ExecutionEventDTO[]): ExecutionEventDTO | undefined {
  return [...events].reverse().find((event) =>
    typeof event.extras?.benchmark_name === "string" && !!event.extras.benchmark_name
    || Array.isArray(event.extras?.active_benchmarks) && event.extras.active_benchmarks.length > 0);
}
