import useSWR from "swr";
import { midTrainingActionLabel } from "../lib/midTrainingActivations";
import type { HeartbeatDTO, TicketDetail, RunDetail } from "../lib/api";
import { midTrainingHasStarted, midTrainingDecisionHasStarted } from "../lib/midTrainingStatus";
import { StatusBadge, statusToneFor } from "./StatusBadge";

export function MidTrainingTimelineBranch({ ticketId, supervisorId, activations, selectedId, onSelect, evaluations = [], inferences = [] }: {
  evaluations?: RunDetail["tickets"];
  inferences?: RunDetail["tickets"];
  ticketId: string; supervisorId: string; activations: HeartbeatDTO[]; selectedId: string;
  onSelect: (selection: string) => void; paused: boolean;
}) {
  const { data: ticket } = useSWR<TicketDetail>(`/api/tickets/${ticketId}`, { refreshInterval: 3000 });
  if (!inferences.length && !evaluations.length && !midTrainingHasStarted(ticket, activations.length)) return null;
  const latest = activations.at(-1);
  const inference = inferences.find((item) => item.id === selectedId) || inferences.at(-1);
  const evaluation = evaluations.find((item) => item.id === selectedId) || evaluations.at(-1);
  const children = [
    ...(midTrainingDecisionHasStarted(ticket, activations.length) ? [{ label: midTrainingActionLabel(latest), selection: `${supervisorId}#mid:${ticketId}`,
      status: "",
      description: latest?.mid_training ? `Step ${latest.mid_training.step.toLocaleString()}` : "Checkpoint orchestration" }] : []),
    ...(inference ? [{label: "Inference", selection: inference.id,
      status: inference.status, description: `Checkpoint ${inference.mid_training_step} · Validation predictions`}] : []),
    ...(evaluation ? [{label: "Evaluation", selection: evaluation.id,
      status: evaluation.status, description: `Checkpoint ${evaluation.mid_training_step} · Validation scoring`}] : []),
  ];
  return <li className="relative">
    <span className="absolute left-[18px] top-0 h-5 w-4 rounded-bl-[10px] border-b border-l border-hair" />
    <div className="ml-[38px] pb-1 pt-1.5">
      <button type="button" onClick={() => onSelect(`${ticketId}#validation:overview`)}
        aria-pressed={selectedId === `${ticketId}#validation:overview`}
        className={`rounded px-2 py-1 text-left text-[11px] font-medium uppercase tracking-wider text-skyx-300/70 hover:bg-raised/40 ${selectedId === `${ticketId}#validation:overview` ? "bg-skyx-500/10" : ""}`}>Mid-training Validation</button>
      <ol className="relative">{children.map((child, index) => {
        const orchestrator = child.selection.includes("#mid:");
        const active = selectedId === child.selection;
        const tone = statusToneFor(child.status);
        return <li key={child.selection} className="relative">
          {index < children.length - 1 && <span className="absolute left-[14px] top-6 h-[calc(100%-12px)] w-px bg-hair/60" />}
          <button type="button" onClick={() => onSelect(child.selection)}
            className={`flex w-full items-start gap-2.5 rounded-lg px-2 py-1 text-left transition ${active ? orchestrator ? "bg-lilac-500/10" : "bg-skyx-500/10" : "hover:bg-raised/40"}`}>
            <span className="relative mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center">
              {!orchestrator && tone.live && <span className={`absolute inline-flex h-full w-full animate-ping ${orchestrator ? "rotate-45 bg-lilac-400" : `rounded-full ${tone.dot}`} opacity-60`} />}
              <span className={`relative h-2 w-2 ${orchestrator ? "rotate-45 bg-lilac-400/70" : `rounded-full ${tone.dot}`}`} />
            </span>
            <span className="min-w-0 flex-1">
              <span className="flex items-center justify-between gap-2">
                <span className={`text-[13px] ${orchestrator ? active ? "italic text-lilac-200" : "italic text-lilac-300/75" : active ? "text-skyx-200" : "text-skyx-300/80"}`}>{child.label}</span>
                {!orchestrator && <StatusBadge status={child.status} />}
              </span>
              <span className="block truncate font-mono text-[11px] text-dim">{child.description}</span>
            </span>
          </button>
        </li>;
      })}</ol>
    </div>
  </li>;
}
