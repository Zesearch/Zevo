import { useState } from "react";
import useSWR from "swr";
import type { TicketDetail } from "../lib/api";
import { trainingActivity } from "../lib/trainingTelemetry";

/** Inference/scoring are checkpoint executions within Train, not new agent wakes. */
export function MidTrainingPhaseDetail({ ticketId, phase, paused }: {
  ticketId: string; phase: "inference" | "scoring"; paused: boolean;
}) {
  const { data: ticket } = useSWR<TicketDetail>(`/api/tickets/${ticketId}`, { refreshInterval: 3000 });
  const [selected, setSelected] = useState<number | "current">();
  if (!ticket) return <div className="p-4 text-xs text-dim">Loading validation…</div>;
  const validation = ticket.mid_training_validation;
  const measurements = validation?.measurements ?? [];
  const activity = trainingActivity(ticket.execution_events, validation, ticket.status, paused);
  const generating = activity === "Mid-training Validation · Inference";
  const scoring = activity === "Mid-training Validation · Scoring";
  const current = generating || scoring;
  const choice = (selected === "current" && !current ? undefined : selected)
    ?? (current ? "current" : measurements.at(-1)?.step);
  const row = measurements.find((item) => item.step === choice);
  const label = phase === "inference" ? "Inference" : "Scoring";
  // Completed measurements prove both stages succeeded. Only use live counters
  // for the current checkpoint; never attach them to a historical measurement.
  const liveEvents = ticket.execution_events.filter((event) => event.extras?.execution_purpose === "workload"
    && (phase === "inference" ? event.phase.startsWith("generate")
      : event.phase.startsWith("scoring") || event.extras?.phase === "model_judge" || event.extras?.phase === "benchmark_complete"))
    .sort((a, b) => a.ts.localeCompare(b.ts));
  const latest = liveEvents.at(-1);
  return <div className="rounded-bezel border border-hair bg-panel/40 p-4">
    <h3 className="mb-3 text-sm font-medium">Mid-training Validation · {label}</h3>
    <div className="mb-4 flex flex-wrap gap-2" aria-label={`${label} checkpoints`}>
      {measurements.map((item) => <button key={item.step} type="button" aria-pressed={choice === item.step}
        className={`rounded-full border px-3 py-1 text-xs ${choice === item.step ? "border-sky-400 text-sky-300" : "border-hair text-dim"}`}
        onClick={() => setSelected(item.step)}>Step {item.step.toLocaleString()} · Completed</button>)}
      {current && <button type="button" aria-pressed={choice === "current"} onClick={() => setSelected("current")}
        className={`rounded-full border px-3 py-1 text-xs ${choice === "current" ? "border-sky-400 text-sky-300" : "border-hair text-dim"}`}>
        {validation?.next_step ? `Step ${validation.next_step.toLocaleString()}` : "Current checkpoint"}
        {phase === "inference" && scoring ? " · Completed" : phase === "scoring" && generating ? " · Waiting" : " · Running"}
      </button>}
    </div>
    {row ? <>
      <p className="mb-3 text-xs text-slate-300">{phase === "inference"
        ? "Predictions generated for all validation benchmarks."
        : `Suite average: ${row.score.toFixed(4)}`}</p>
      <table className="w-full text-left text-xs"><thead className="border-b border-hair text-dim"><tr>
        <th className="py-2 font-medium">Benchmark</th><th className="py-2 font-medium">{phase === "inference" ? "Status" : "Score"}</th>
      </tr></thead><tbody>{Object.entries(row.metrics).map(([name, score]) => <tr key={name} className="border-b border-hair">
        <td className="py-2 pr-3">{name}</td><td className="py-2">{phase === "inference" ? "Completed" : score.toFixed(4)}</td>
      </tr>)}</tbody></table>
    </> : current && choice === "current" ? <div className="text-sm text-sky-300">
      {phase === "scoring" && generating ? "Waiting for inference to complete."
        : phase === "inference" && scoring ? "Inference complete; scoring is running."
        : <>{latest?.extras?.benchmark_name ? String(latest.extras.benchmark_name) : label}
          {latest && latest.total_steps > 0 && <> · {latest.current_step.toLocaleString()} / {latest.total_steps.toLocaleString()}</>}</>}
    </div> : <p className="text-xs text-dim">{paused ? "Training paused." : "No completed checkpoint validation yet."}</p>}
  </div>;
}
