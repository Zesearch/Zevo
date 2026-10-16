import type { MidTrainingValidation } from "../lib/api";

export function MidTrainingValidationPanel({ data }: { data: MidTrainingValidation }) {
  const members = [...new Set(data.measurements.flatMap((row) => Object.keys(row.metrics)))];
  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-4 text-xs text-slate-300">
        <span>{data.complete ? "Selected checkpoint" : "Best checkpoint so far"}: step {data.best_step.toLocaleString()}</span>
        {data.next_step != null && <span>Next evaluation: step {data.next_step.toLocaleString()}</span>}
        {data.awaiting_decision && <span>Awaiting Orchestrator decision</span>}
        <span>Validation only · {data.direction === "max" ? "Higher" : "Lower"} is better</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="border-b border-slate-700 text-slate-400">
            <tr>{["Step", "Progress", ...members, "Suite average", "Decision", "Reason"].map((title, i) =>
              <th key={i} className="px-3 py-2 font-medium whitespace-nowrap">{title}</th>)}
            </tr>
          </thead>
          <tbody>
            {data.measurements.map((row) => (
              <tr key={row.step} className="border-b border-slate-800 align-top">
                <td className="px-3 py-2 whitespace-nowrap">{row.step.toLocaleString()}
                  {row.step === data.best_step && <span className="ml-2 text-brass-300">Best</span>}
                </td>
                <td className="px-3 py-2">{(100 * row.step / data.planned_steps).toFixed(1)}%</td>
                {members.map((name) => <td key={name} className="px-3 py-2 font-mono">{row.metrics[name]?.toFixed(4) ?? "—"}</td>)}
                <td className="px-3 py-2 font-mono">{row.score.toFixed(4)}</td>
                <td className="px-3 py-2 whitespace-nowrap">{row.plan_completed ? "Plan completed" : row.decision?.action === "stop" ? "Stop" : row.decision?.action === "continue" ? `Continue → step ${row.decision.next_step?.toLocaleString()}` : "Decision not recorded"}</td>
                <td className="min-w-64 px-3 py-2 whitespace-pre-wrap">{row.decision?.reason || (row.plan_completed ? "Full training plan completed; selected the best measured checkpoint." : "—")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
