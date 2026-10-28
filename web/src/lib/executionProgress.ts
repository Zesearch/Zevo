import type { ExecutionEventDTO, HeartbeatDTO } from "./api";

/** Markers use second-resolution timestamps. Preserve stdout order within
 * an execution even when the API returns tied rows in a different order. */
function compareEvents(a: ExecutionEventDTO, b: ExecutionEventDTO): number {
  const time = a.ts.localeCompare(b.ts);
  if (time) return time;
  if (!a.extras?.execution_id || a.extras.execution_id !== b.extras?.execution_id) return 0;
  const left = Number(a.extras.telemetry_sequence);
  const right = Number(b.extras.telemetry_sequence);
  if (left > 0 && right > 0) return left - right;
  // Historical preparation markers have no sequence. Only resolve the known
  // setup/loading tie; do not infer ordering for retries or other phases.
  const rank = (phase?: string | null) =>
    /^(validate_config|preflight|setup|environment)$/.test(phase || "") ? 1
    : phase === "load_model" ? 2 : 0;
  const x = rank(a.phase), y = rank(b.phase);
  return x && y ? x - y : 0;
}

export function latestExecution(events: ExecutionEventDTO[]): ExecutionEventDTO[] {
  const ordered = [...events].sort(compareEvents);
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
  if (identity) return events.filter((event) => event.extras?.execution_id === identity)
    .sort(compareEvents);
  if (selected.length) return selected;
  return ["collect", "continue", "resume"].includes(heartbeat.execution_phase || heartbeat.activation_phase)
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

/** Preparation is displayed without inventing a percentage before generation. */
export function inferencePreparation(events: ExecutionEventDTO[]) {
  const ordered = [...events].sort(compareEvents);
  if (ordered.some(e => e.extras?.benchmark_complete === true || /^generat/.test(e.phase || ""))) return null;
  const event = ordered.at(-1);
  if (!event) return null;
  const phase = event.phase || "";
  const label = phaseLabel(phase, "inference");
  const start = ordered.find(e => e.phase === phase)?.ts || event.ts;
  return { label, startedAt: start };
}

/** Shared vocabulary; unknown emitted phases stay visible instead of being
 * mislabelled as inference preparation. */
export function phaseLabel(phase: string, agent: string): string {
  const labels: Record<string, string> = {
    resolve_model: "Resolving model architecture", initialize_engine: "Starting inference engine",
    load_weights: "Loading model weights", load_model: "Loading model", loading_model: "Loading model",
    prepare_runtime: "Preparing runtime cache", prepare_dependencies: "Preparing dependencies",
    restore_cache: "Restoring runtime cache", publish_cache: "Saving runtime cache",
    setup: "Checking environment", preflight: "Checking environment", environment: "Checking environment",
    validate_config: "Checking configuration", train: "Training", training: "Training",
    resume: "Restoring checkpoint", resume_checkpoint: "Restoring checkpoint", loading_checkpoint: "Restoring checkpoint",
    saving_model: "Saving model", save_checkpoint: "Saving checkpoint", saving_checkpoint: "Saving checkpoint",
    download: "Downloading data", downloading_data: "Downloading data", download_data: "Downloading data",
    generate: agent === "data" ? "Generating training data" : "Generating predictions",
    filter: "Filtering data", filtering: "Filtering data", validating_data_artifacts: "Validating data",
    writing_artifacts: "Saving artifacts", write_outputs: "Saving outputs",
    loading_predictions: "Loading predictions", scoring: "Scoring predictions", aggregate: "Aggregating results",
  };
  if (labels[phase]) return labels[phase];
  if (/calibrat/.test(phase)) return "Calibrating generation length";
  if (/compil/.test(phase)) return agent === "inference" ? "Compiling inference kernels" : "Compiling kernels";
  if (/warmup/.test(phase)) return agent === "inference" ? "Warming up inference" : "Warming up model";
  const words = phase.replace(/[_-]+/g, " ");
  return words ? words[0].toUpperCase() + words.slice(1) : "Preparing workload";
}

/** Input must already be scoped to the selected activation/execution. */
export function executionPhase(events: ExecutionEventDTO[], agent: string) {
  const ordered = [...events].sort(compareEvents).filter(e =>
    !!e.phase && ["phase", "progress", "lifecycle"].includes(e.event_type));
  const event = ordered.at(-1);
  if (agent === "inference") {
    if (event && /^(done|completed|failed|cancelled|exited|succeeded)$/.test(event.phase)) return null;
    if (!event || !/^(write_outputs|writing_artifacts|publish_cache)$/.test(event.phase)) {
      return inferencePreparation(events);
    }
  }
  if (!event || /^(done|completed|failed|cancelled|exited|succeeded)$/.test(event.phase)) return null;
  // Evaluation's existing benchmark/judge card already describes scoring.
  if (agent === "evaluation" && benchmarkCounts(events, undefined, "evaluation")) return null;
  let start = ordered.length - 1;
  while (start > 0 && ordered[start - 1].phase === event.phase) start--;
  return { label: phaseLabel(event.phase, agent), startedAt: ordered[start].ts };
}
