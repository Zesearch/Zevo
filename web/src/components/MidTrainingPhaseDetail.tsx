import useSWR from "swr";
import type { TicketDetail } from "../lib/api";
import { MidTrainingValidationPanel } from "./MidTrainingValidationPanel";

export function MidTrainingPhaseDetail({ ticketId }: { ticketId: string }) {
  const { data: ticket } = useSWR<TicketDetail>(`/api/tickets/${ticketId}`, { refreshInterval: 3000 });
  if (!ticket) return <div className="p-4 text-xs text-dim">Loading validation…</div>;
  return <div className="rounded-bezel border border-hair bg-panel/40">
    <div className="p-4">
      <MidTrainingValidationPanel data={ticket.mid_training_validation} events={ticket.execution_events}
        activity={null} showExecutionProgress={false} showHeading={true} />
    </div>
  </div>;
}
