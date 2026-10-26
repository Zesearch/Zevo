import useSWR from "swr";
import type { TicketDetail } from "../lib/api";
import { inferenceProgress, midTrainingExecution, validationStatus } from "../lib/midTrainingStatus";
import { MidTrainingValidationPanel } from "./MidTrainingValidationPanel";
import { EvaluationProgress } from "./EvaluationProgress";
import { StageProgress } from "./StageProgress";
import { StatusBadge } from "./StatusBadge";

export function MidTrainingPhaseDetail({ ticketId, paused }: { ticketId: string; paused: boolean }) {
  const { data: ticket } = useSWR<TicketDetail>(`/api/tickets/${ticketId}`, { refreshInterval: 3000 });
  if (!ticket) return <div className="p-4 text-xs text-dim">Loading validation…</div>;
  const status = validationStatus(ticket, paused);
  const { events, activity, phase } = midTrainingExecution(ticket);
  const progress = inferenceProgress(events);
  const showProgress = ["running", "paused"].includes(status);
  return <div className="rounded-bezel border border-hair bg-panel/40">
    <div className="flex items-center justify-between gap-3 border-b border-hair p-4">
      <h3 className="text-sm font-medium">Inference &amp; Scoring</h3><StatusBadge status={status} />
    </div>
    <div className="space-y-4 p-4">
      {showProgress && phase === "inference" && progress && progress.total > 0 && <StageProgress
        completed={progress.completed} total={progress.total} label="Mid-training inference responses">
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1.5 font-mono text-xs">
          <span className="text-slate-200"><span className="text-brass-300">Inference</span> · {progress.name}
            {progress.workers > 1 && <span className="ml-2 text-dim">· {progress.workers} replicas</span>}
            {paused && <span className="ml-2 text-dim">Paused</span>}</span>
          <span className="text-brass-300">{progress.suite ? "Validation" : progress.name} {progress.completed.toLocaleString()} / {progress.total.toLocaleString()} <span className="text-dim">responses</span></span>
        </div>
      </StageProgress>}
      {showProgress && phase === "scoring" && <EvaluationProgress scopedEvents={events} events={[]} lane="optimization" paused={paused} />}
      <MidTrainingValidationPanel data={ticket.mid_training_validation} events={events}
        activity={paused ? "Training paused" : activity} showExecutionProgress={false} />
    </div>
  </div>;
}
