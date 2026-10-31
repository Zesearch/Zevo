import { useEffect, useState } from "react";
import { executionPhase, workloadCanProgress } from "../lib/executionProgress";
import type { ExecutionEventDTO, InfraInstanceDTO } from "../lib/api";

/** Stage-independent version of the inference phase/elapsed-time card. */
export function ExecutionPhase({ events, agent, active, workload }: {
  events: ExecutionEventDTO[]; agent: string; active: boolean; workload?: InfraInstanceDTO;
}) {
  const [now, setNow] = useState(Date.now);
  const phase = workloadCanProgress(workload) ? executionPhase(events, agent) : null;
  useEffect(() => {
    if (!active || !phase) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [active, phase?.startedAt]);
  if (!active || !phase) return null;
  const started = Date.parse(phase.startedAt);
  const elapsed = Number.isFinite(started) ? Math.max(0, Math.floor((now - started) / 1000)) : 0;
  const duration = `${elapsed >= 3600 ? `${Math.floor(elapsed / 3600)}h ` : ""}${Math.floor(elapsed / 60) % 60}m ${elapsed % 60}s`;
  return <div role="status" className="mb-3 rounded-bezel border border-hair p-3 font-mono text-xs text-slate-300">
    {phase.label}<span className="ml-3 tabular-nums text-dim">{duration}</span>
  </div>;
}
