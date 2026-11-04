import { useState } from "react";
import type { IterationHistoryEntry, MidTrainingValidation } from "../lib/api";
import { comparisonPoints, comparisonRows } from "../lib/checkpointComparison";
import { fmtScore } from "../lib/format";

export function CheckpointComparison({ data, baseline }: {
  data: MidTrainingValidation; baseline?: IterationHistoryEntry;
}) {
  const points = comparisonPoints(data, baseline);
  const [leftId, setLeftId] = useState(points[0]?.id);
  const [rightId, setRightId] = useState(points.at(-1)?.id);
  const left = points.find((point) => point.id === leftId) || points[0];
  const right = points.find((point) => point.id === rightId) || points.at(-1);
  if (!left || !right || points.length < 2) return <p className="text-sm text-dim">Comparison needs two completed Validation results.</p>;
  return <div className="space-y-4">
    <div className="grid grid-cols-[1fr_auto_1fr] items-end gap-3">
      {(["Reference", "Compare to"] as const).map((label, index) => <div key={label} className={index ? "col-start-3 row-start-1" : ""}>
        <label className="mb-2 block text-xs text-dim" htmlFor={`comparison-${index}`}>{label}</label>
        <select id={`comparison-${index}`} value={index ? right.id : left.id}
          onChange={(event) => (index ? setRightId : setLeftId)(event.target.value)}
          className="w-full rounded-md border border-hair bg-panel px-3 py-2 text-sm">
          {points.map((point) => <option key={point.id} value={point.id}>{point.label}</option>)}
        </select>
      </div>)}
      <button type="button" aria-label="Swap comparison" className="btn col-start-2 row-start-1"
        onClick={() => { setLeftId(right.id); setRightId(left.id); }}>⇄</button>
    </div>
    <p className="text-xs text-dim">Validation only · Change = Compare to − Reference, in percentage points.</p>
    <div className="overflow-x-auto"><table className="w-full text-left text-sm">
      <thead><tr className="border-b border-hair text-dim">
        <th className="py-3 pr-4 font-medium">Benchmark</th>
        <th className="px-3 py-3 text-right font-medium">{left.label}</th>
        <th className="px-3 py-3 text-right font-medium">{right.label}</th>
        <th className="py-3 pl-3 text-right font-medium">Change</th>
      </tr></thead>
      <tbody>{comparisonRows(left, right).map((row, index) => {
        const direction = index === 0 ? data.direction : baseline?.validation_metric_directions?.[row.name] || data.direction;
        const roundedDelta = row.delta == null ? null : Number((row.delta * 100).toFixed(1));
        const improved = roundedDelta != null && (direction === "min" ? roundedDelta < 0 : roundedDelta > 0);
        return <tr key={`${index}:${row.name}`} className={`border-b border-hair ${index === 0 ? "font-medium" : ""}`}>
          <td className="py-3 pr-4">{row.name}</td>
          <td className="px-3 py-3 text-right font-mono">{fmtScore(row.left, "validation_score")}</td>
          <td className="px-3 py-3 text-right font-mono">{fmtScore(row.right, "validation_score")}</td>
          <td className={`py-3 pl-3 text-right font-mono whitespace-nowrap ${roundedDelta == null || roundedDelta === 0 ? "text-dim" : improved ? "text-emerald-300" : "text-coral-300"}`}>
            {roundedDelta == null ? "—" : `${roundedDelta > 0 ? "+" : ""}${roundedDelta.toFixed(1)} pp`}
          </td>
        </tr>;
      })}</tbody>
    </table></div>
  </div>;
}
