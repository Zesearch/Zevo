import { useId, useState } from "react";
import type { RunDetail } from "../lib/api";
import { MidTrainingPredictions } from "./MidTrainingPredictions";

export function CheckpointPredictions({ runId, tickets, initialStep }: {
  runId: string; tickets: RunDetail["tickets"]; initialStep?: number;
}) {
  const selectorId = useId();
  const checkpoints = [...tickets].sort((a, b) =>
    (a.mid_training_step ?? 0) - (b.mid_training_step ?? 0) || a.created_at.localeCompare(b.created_at));
  const [selectedId, setSelectedId] = useState(() =>
    checkpoints.filter((item) => item.mid_training_step === initialStep).at(-1)?.id || checkpoints.at(-1)?.id);
  const selected = checkpoints.find((item) => item.id === selectedId) || checkpoints.at(-1);
  if (!selected) return <p className="text-sm text-dim">No checkpoint inference has started yet.</p>;
  return <div className="space-y-4">
    <div>
      <label htmlFor={selectorId} className="mb-2 block text-xs text-dim">Checkpoint</label>
      <select id={selectorId} value={selected.id} onChange={(event) => setSelectedId(event.target.value)}
        className="w-full rounded-md border border-hair bg-panel px-3 py-2 text-sm">
        {checkpoints.map((item) => <option key={item.id} value={item.id}>
          Step {item.mid_training_step?.toLocaleString() ?? "—"} · {item.status}
        </option>)}
      </select>
    </div>
    <MidTrainingPredictions key={selected.id} runId={runId} ticketId={selected.id} />
  </div>;
}
