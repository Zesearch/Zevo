import useSWR from "swr";
import type { TicketDetail } from "../lib/api";
import { pendingCheckpointStep } from "../lib/midTrainingStatus";
import { MidTrainingValidationPanel } from "./MidTrainingValidationPanel";

export function MidTrainingPhaseDetail({ ticketId }: { ticketId: string }) {
  const { data: ticket } = useSWR<TicketDetail>(`/api/tickets/${ticketId}`, { refreshInterval: 3000 });
  if (!ticket) return <div className="p-4 text-xs text-dim">Loading validation…</div>;
  return <div className="rounded-bezel border border-hair bg-panel/40">
    <div className="p-4">
      {!ticket.checkpoint_handoff_failure && !ticket.mid_training_validation && pendingCheckpointStep(ticket, []) !== null &&
        <p className="mb-4 text-sm text-dim">{ticket.summary?.includes("waiting for Orchestrator")
          ? "Checkpoint verified · preparing Validation" : "Validate checkpoint"}</p>}
      {ticket.checkpoint_handoff_failure && <div role="status" className="mb-4 text-sm text-amber-200">
        <p>Checkpoint handoff blocked · Orchestrator notified</p>
        <p className="mt-1 text-xs text-dim">Validation is waiting for checkpoint verification.</p>
        <pre className="mt-2 whitespace-pre-wrap break-words text-xs">{ticket.checkpoint_handoff_failure.error}</pre>
      </div>}
      <MidTrainingValidationPanel data={ticket.mid_training_validation} events={ticket.execution_events}
        activity={null} showExecutionProgress={false} showHeading={true} />
    </div>
  </div>;
}
