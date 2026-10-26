import { Fragment, useLayoutEffect, useRef, useState } from "react";
import type { ExecutionEventDTO, MidTrainingValidation } from "../lib/api";

function ExpandableReason({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const [overflow, setOverflow] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element || expanded) return;
    const measure = () => setOverflow(element.scrollWidth > element.clientWidth);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [text, expanded]);
  return <div className="w-64 max-w-full">
    <div ref={ref} className={expanded ? "whitespace-pre-wrap break-words" : "truncate"}>{text}</div>
    {(overflow || expanded) && <button type="button" aria-expanded={expanded}
      className="mt-1 text-sky-300 hover:underline" onClick={() => setExpanded(!expanded)}>
      {expanded ? "Show less" : "Show more"}
    </button>}
  </div>;
}

export function MidTrainingValidationPanel({ data, events = [], activity }: {
  data?: MidTrainingValidation | null; events?: ExecutionEventDTO[]; activity?: string | null;
}) {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());
  const progress = [...events].filter((event) => event.extras?.execution_purpose === "workload"
    && event.extras?.benchmark_name).sort((a, b) => a.ts.localeCompare(b.ts)).at(-1);
  const generating = activity === "Mid-training Validation · Inference";
  const progressNote = <div className="mb-3 text-xs text-sky-300">
    {activity || (data?.complete ? "Validation complete" : data ? "Validation results" : "Waiting for the first validation checkpoint")}
    {generating && progress && <span> · {String(progress.extras?.benchmark_name)} · {progress.current_step.toLocaleString()} / {progress.total_steps.toLocaleString()} responses</span>}
  </div>;
  if (!data) return progressNote;
  return (
    <div>
      {progressNote}
      <div className="mb-3 flex flex-wrap gap-4 text-xs text-slate-300">
        <span>{data.complete ? "Selected checkpoint" : "Best checkpoint so far"}: step {data.best_step.toLocaleString()}</span>
        {data.next_step != null && <span>Next evaluation: step {data.next_step.toLocaleString()}</span>}
        {data.awaiting_decision && activity !== "Training paused" && <span>Awaiting Orchestrator decision</span>}
        <span>Validation only · {data.direction === "max" ? "Higher" : "Lower"} is better</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="border-b border-slate-700 text-slate-400">
            <tr>{["Step", "Progress", "Suite average", "Decision", "Reason", "Benchmarks"].map((title, i) =>
              <th key={i} className="px-3 py-2 font-medium whitespace-nowrap">{title}</th>)}
            </tr>
          </thead>
          <tbody>
            {data.measurements.map((row) => (
              <Fragment key={row.step}><tr className="border-b border-slate-800 align-top">
                <td className="px-3 py-2 whitespace-nowrap">{row.step.toLocaleString()}
                  {row.step === data.best_step && <span className="ml-2 text-sky-300">Best</span>}
                </td>
                <td className="px-3 py-2">{(100 * row.step / data.planned_steps).toFixed(1)}%</td>
                <td className="px-3 py-2 font-mono">{row.score.toFixed(4)}</td>
                <td className="px-3 py-2 whitespace-nowrap">{row.plan_completed ? "Plan completed" : row.decision?.action === "stop" ? "Stop" : row.decision?.action === "continue" ? `Continue → step ${row.decision.next_step?.toLocaleString()}` : "Decision not recorded"}</td>
                <td className="px-3 py-2"><ExpandableReason text={row.decision?.reason || (row.plan_completed ? "Full training plan completed; selected the best measured checkpoint." : "—")} /></td>
                <td className="px-3 py-2 whitespace-nowrap"><button type="button" className="text-sky-300 hover:underline"
                  aria-expanded={expandedSteps.has(row.step)} onClick={() => setExpandedSteps((previous) => {
                    const next = new Set(previous);
                    if (next.has(row.step)) next.delete(row.step); else next.add(row.step);
                    return next;
                  })}>{expandedSteps.has(row.step) ? "Hide benchmarks" : "Benchmarks"} · {Object.keys(row.metrics).length}</button></td>
              </tr>
              {expandedSteps.has(row.step) && <tr className="border-b border-slate-800">
                <td colSpan={6} className="px-3 py-3">
                  <div className="max-h-80 overflow-y-auto">
                    <table className="w-full text-left text-xs">
                      <thead><tr className="text-slate-400"><th className="py-1 font-medium">Benchmark</th><th className="py-1 font-medium">Score</th></tr></thead>
                      <tbody>{Object.entries(row.metrics).map(([name, score]) => <tr key={name}>
                        <td className="py-1 pr-4 break-words">{name}</td><td className="py-1 font-mono">{score.toFixed(4)}</td>
                      </tr>)}</tbody>
                    </table>
                  </div>
                </td>
              </tr>}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
