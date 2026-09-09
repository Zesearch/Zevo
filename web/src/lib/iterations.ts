// Single source of truth for partitioning a run's tickets into
// Baseline + Iteration 1, 2, … — used by the timeline, the Per-ticket tab, and
// the Artifacts panel so all three agree on which iteration each ticket (and
// its artifacts) belongs to.

export function agentIdOf(t: { agent_id?: string }): string {
  return t.agent_id || "";
}

export type IterInfo = {
  key: string;
  label: string;
  kind: "baseline" | "iteration";
  /** This ticket is the run-wide supervisor: it belongs to no single block. */
  spansRun?: boolean;
  num: number;
};

export type MinimalTicket = {
  id: string; agent_id?: string; created_at?: string; status?: string;
  lane?: "optimization" | "held_out_test";
  iteration: number;
};

/** Use the Ticket envelope's explicit iteration. Agent role and creation order
 *  never infer stage ownership. The one supervisor Ticket spans the run. */
export function assignIterations(tickets: MinimalTicket[]): Map<string, IterInfo> {
  const map = new Map<string, IterInfo>();
  for (const t of tickets) {
    const num = Math.max(0, Number(t.iteration));
    const info: IterInfo = num === 0
      ? { key: "baseline", label: "Iteration 0", kind: "baseline", num }
      : { key: `iter-${num}`, label: `Iteration ${num}`, kind: "iteration", num };
    map.set(t.id, agentIdOf(t) === "orchestrator" ? { ...info, spansRun: true } : info);
  }
  return map;
}

/** Distinct iterations in display order (Baseline, Iteration 1, 2, …), each
 *  paired with the ordered ids of the tickets that fall inside it. */
export function iterationOrder(tickets: MinimalTicket[]): Array<IterInfo & { ticketIds: string[] }> {
  const sorted = [...tickets].sort((a, b) => (a.created_at || "").localeCompare(b.created_at || ""));
  const info = assignIterations(tickets);
  const out: Array<IterInfo & { ticketIds: string[] }> = [];
  const byKey = new Map<string, IterInfo & { ticketIds: string[] }>();
  for (const t of sorted) {
    const gi = info.get(t.id);
    if (!gi) continue;
    let g = byKey.get(gi.key);
    if (!g) {
      g = { ...gi, ticketIds: [] };
      byKey.set(gi.key, g);
      out.push(g);
    }
    g.ticketIds.push(t.id);
  }
  return out.sort((a, b) => a.num - b.num);
}

/**
 * The key of the iteration currently being worked on, or "" when nothing is.
 *
 * Every panel that groups by iteration wants the same thing: show me the block
 * that is running and get the finished ones out of the way. Deriving it here
 * keeps the timeline, the per-ticket list and the artifacts list agreeing on
 * which block that is.
 */
export function liveIterationKey(tickets: MinimalTicket[]): string {
  const info = assignIterations(tickets);
  const live = tickets.filter((t) => t.status === "running" || t.status === "repairing");
  if (!live.length) return "";
  // Latest one wins: a straggler from an earlier iteration should not drag the
  // view back once the next iteration has started.
  const keys = live.map((t) => info.get(t.id)?.key).filter(Boolean) as string[];
  if (!keys.length) return "";
  const order = iterationOrder(tickets).map((g) => g.key);
  return keys.sort((a, b) => order.indexOf(a) - order.indexOf(b)).pop() || "";
}
