import { activationExecution, benchmarkCounts } from "../lib/executionProgress";
import { StageProgress } from "./StageProgress";
import type { ExecutionEventDTO, HeartbeatDTO } from "../lib/api";

export function evaluationAttemptLabel(phase: string, sequence: number, lane: string): string {
  const suite = lane === "held_out_test" ? "Held-Out" : "Validation";
  if (phase === "repair") return `Retry ${suite} Evaluation`;
  return sequence > 1 ? `Re-evaluate ${suite}` : `Evaluate ${suite}`;
}

export function EvaluationProgress({ events, heartbeat, lane, paused = false, benchmarkName = "", scopedEvents, activations }: {
  events: ExecutionEventDTO[];
  activations?: HeartbeatDTO[];
  scopedEvents?: ExecutionEventDTO[];
  heartbeat?: HeartbeatDTO;
  lane: string;
  paused?: boolean;
  benchmarkName?: string;
}) {
  // Progress belongs to the selected execution, never a previous retry or
  // the Run's aggregate scores. A new execution starts its own progress.
  const scoped = scopedEvents ?? activationExecution(events, heartbeat, activations);
  const count = benchmarkCounts(scoped, undefined, "evaluation");
  let current = "";
  let judge: ExecutionEventDTO | undefined;
  for (const event of scoped) {
    const scoring = /^scoring (\d+)\/(\d+): (.+)$/.exec(event.phase);
    if (scoring) {
      judge = undefined;
      current = scoring[3];
    }
    if (event.event_type === "progress" && event.extras?.phase === "model_judge") {
      judge = event;
    }
  }
  if (!count) return null;
  const { completed: done, total } = count;
  if (!current) current = benchmarkName || (lane === "held_out_test" ? "Final Test" : "Validation");
  const judgeModel = typeof judge?.extras?.model === "string" ? judge.extras.model : "";
  const judgeCached = Number(judge?.extras?.cached || 0);
  const suite = lane === "held_out_test" ? "Final Test" : "Validation";
  return (
    <StageProgress completed={done} total={total} label={`${suite} evaluation benchmarks`}>
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1.5 font-mono text-xs">
        <span className="min-w-0 text-slate-200">
          <span className="text-brass-300">Evaluation</span>
          {current && <span> · {current}</span>}
          {judge && (
            <span className="ml-3 text-[10px] text-sky-300" title={judgeModel || undefined}>
              Model Judge · {judge.current_step.toLocaleString()} / {judge.total_steps.toLocaleString()} responses judged
              {judgeCached > 0 && <span className="text-dim"> · {judgeCached.toLocaleString()} cached</span>}
            </span>
          )}
          {paused && !heartbeat?.finished_at && <span className="ml-3 text-dim">Paused</span>}
        </span>
        <span className="tabular-nums text-brass-300">{suite} {done} / {total} <span className="text-dim">evaluated</span></span>
      </div>
    </StageProgress>
  );
}
