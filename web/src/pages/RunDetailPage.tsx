import { useState, useMemo, useEffect, Fragment } from "react";
import { useParams, Link, useNavigate, useSearchParams } from "react-router-dom";
import useSWR from "swr";
import { StopCircle, ChevronRight, ChevronDown, Clock, ListChecks, Package, ArrowLeft, Layers3, ScrollText } from "lucide-react";
import { TrainingMonitor } from "../components/TrainingMonitor";
import { StepTimeline } from "../components/StepTimeline";
import type { EventEnvelope } from "../components/LiveTranscript";

import type { CostBreakdown, HeartbeatDTO, InfraInstanceDTO, RunDetail, TicketDetail } from "../lib/api";
import { StatusBadge, statusToneFor } from "../components/StatusBadge";
import { LiveTranscript } from "../components/LiveTranscript";
import { IterationChart, RunJournal } from "../components/IterationHistoryPanel";
import { CancelRunDialog } from "../components/CancelRunDialog";
import { ArtifactsPanel } from "../components/ArtifactsPanel";
import { IterationDetailsPanel } from "../components/IterationDetailsPanel";
import { fmtScore, fmtMetric, fmtEvalSource, fmtCost, fmtDate, fmtDuration, fmtTokens, isPercentageMetric, shortModel } from "../lib/format";
import { useRunWebsocket } from "../lib/ws";
import { agentIdOf, assignIterations, iterationOrder, liveIterationKey } from "../lib/iterations";
import { Bezel, Gauge, Kicker, Note, PageHead, Readout } from "../components/zevo/primitives";
import { LOOP_STATIONS, PipelineRing, ticketAtStation, type Station, type StationState } from "../components/zevo/PipelineRing";

// Aggregate a run's tickets into the six loop stations by Ticket.agent_id.
function stationsFrom(
  detail: RunDetail | undefined,
  wakesByAgent: Record<string, number>,
): Station[] {
  return LOOP_STATIONS.map((st) => {
    const t = (detail?.tickets ?? []).filter((x) => ticketAtStation(x, st));
    let state: StationState = "idle";
    if (t.length) {
      if (t.some((x) => x.status === "running" || x.status === "repairing")) state = "active";
      else if (t.some((x) => x.status === "failed")) state = "failed";
      else if (t.some((x) => x.status === "degraded")) state = "degraded";
      else if (t.every((x) => ["succeeded", "skipped"].includes(x.status))) state = "done";
      else if (t.every((x) => ["succeeded", "skipped", "cancelled"].includes(x.status))) state = "cancelled";
      else state = "idle";
    }
    // WAKES, not tickets. The caption always said "how many times this stage's
    // agent has been called", and the number underneath was how many tickets it
    // owned — the same thing only while every ticket runs exactly once. A stage
    // that was re-woken to fix something, or a supervisor that wakes once per
    // child, counted 1 for an agent that had run twenty times.
    const wakes = t.reduce((n, x) => n + (wakesByAgent[x.id] ?? 0), 0);
    return { ...st, state, count: wakes || undefined };
  });
}

/** `empty` is what to say when there are none. The default reads as "not yet",
 *  which is right for a stage still running and wrong for a finished activation
 *  that simply never emitted any — those are different facts and the panel
 *  should not report the second as the first. */
function TicketCard({ ticketId }: { ticketId: string }) {
  // Stop polling once the ticket reaches a terminal status. Before this
  // tweak we re-fetched every TicketCard on the page every 2s — on a
  // run with 30+ tickets that meant ~15 req/s indefinitely. Now we poll
  // at 10s while live and 0 (one-shot) once the row stops changing.
  const { data: t } = useSWR<TicketDetail>(`/api/tickets/${ticketId}`, {
    refreshInterval: (latest) =>
      latest && ["succeeded", "failed", "cancelled", "skipped", "degraded"].includes(latest.status)
        ? 0
        : 10000,
  });
  const nav = useNavigate();
  if (!t) return null;

  // `<agent>-<run8>-<nnn>`, or `<agent>-<nnn>` for ids issued before numbering
  // was scoped to the run. The prefix is the station that did the work and the
  // trailing number is the serial; the run segment in the middle is what makes
  // the id unique across runs and says nothing on a slip that is already inside
  // one run, so it is dropped. Without that it read "DATA-D83476D1".
  const m = /^(.*?)(?:-[0-9a-f]{8})?-(\d+)$/.exec(t.id);
  const series = (m ? m[1] : t.agent_id).toUpperCase();
  const serial = m ? m[2] : "";

  const tone = statusToneFor(t.status);
  const [date, time] = fmtDate(t.updated_at).split(", ");
  return (
    <div className="group relative pt-6">
      {/* Hook: a short wire from the rail above down to the slip's punch hole. */}
      <span className="absolute left-1/2 top-0 h-6 w-px -translate-x-1/2 bg-hair" />
      <div
        onClick={() => nav(`/tickets/${t.id}`)}
        title={`${t.id} · ${t.status}`}
        style={ZIGZAG}
        className="flex cursor-pointer select-none flex-col items-center bg-panel/70 px-3 pb-5 pt-3 text-center transition group-hover:bg-raised"
      >
        {/* Punch hole — the slip hangs by this. */}
        <span className="h-2 w-2 rounded-full bg-canvas ring-1 ring-hair" />

        {/* Issuing station + serial. A compound station gets a line per part —
            `HOLDOUT-DATA` on one line either overflows the slip or shrinks the
            whole column to fit its longest member, and the two words are two
            facts anyway: which lane, then which stage. */}
        <div className="mt-3 break-all font-mono text-xs uppercase leading-tight tracking-[0.18em] text-ink">
          {series.split("-").map((part, i) => (
            <div key={i} className={i === 0 && series.includes("-") ? "text-dim" : ""}>{part}</div>
          ))}
        </div>
        <div className="mt-0.5 font-mono text-[1.6rem] leading-none text-ink tabular-nums">
          {serial || t.id}
        </div>

        {/* Status, stamped in the colour of the issuing desk */}
        <div className={`mt-2.5 flex items-center gap-1.5 font-mono text-2xs uppercase tracking-[0.16em] ${tone.text}`}>
          <span className={`h-1.5 w-1.5 rounded-full ${tone.dot}`} />
          {t.status}
        </div>

        {/* Tear line, then the stamp */}
        <div className="mt-3 w-full border-t border-dashed border-hair" />
        <div className="mt-2 font-mono text-2xs uppercase tracking-[0.16em] text-slate-600">updated</div>
        <div className="whitespace-nowrap font-mono text-2xs text-dim tabular-nums">{date}</div>
        {time && <div className="whitespace-nowrap font-mono text-2xs text-slate-500 tabular-nums">{time}</div>}
      </div>
    </div>
  );
}

/** Torn bottom edge. Two mask layers: a zigzag band along the bottom, and the
 *  rest of the slip left whole. A plain border can't do this — it would draw a
 *  straight line across the teeth — so the slip carries no border and reads by
 *  its fill instead. */
const ZIGZAG: React.CSSProperties = {
  WebkitMask:
    "conic-gradient(from -45deg at bottom, #0000, #000 1deg 89deg, #0000 90deg) bottom/12px 6px repeat-x, linear-gradient(#000 0 0) top/100% calc(100% - 6px) no-repeat",
  mask:
    "conic-gradient(from -45deg at bottom, #0000, #000 1deg 89deg, #0000 90deg) bottom/12px 6px repeat-x, linear-gradient(#000 0 0) top/100% calc(100% - 6px) no-repeat",
};

/** Master-detail pipeline timeline: a vertical list of stage nodes on the
 *  left (the running one pulses), and the live, full detail of the selected /
 *  running stage blown up on the right. */
type TLTicket = RunDetail["tickets"][number];

/** Keep the specialist identity stable; describe its current activity below. */
function stageLabel(t: TLTicket): string {
  const names: Record<string, string> = {
    infrastructure: "Infrastructure",
    data: "Data",
    train: "Train",
    inference: "Inference",
    evaluation: "Evaluation",
    registry: "Registry",
    orchestrator: "Orchestrator",
  };
  const agent = agentIdOf(t);
  return names[agent] || agent;
}

function operationCaption(t: {
  agent_id?: string;
  lane?: string;
  iteration?: number;
  operation?: string;
  model_source?: string;
  test_set_name?: string;
}, heartbeatOperation = "", activationPhase = ""): string {
  const op = heartbeatOperation || t.operation || "";
  const phase = activationPhase.trim().toLowerCase();
  const inferenceName = t.lane === "held_out_test"
    ? `Held-Out Inference${t.test_set_name ? ` · ${t.test_set_name}` : ""}`
    : t.model_source === "base_model"
      ? "Baseline Inference"
      : "Candidate Inference";
  const stageName = ({
    infrastructure: "Infrastructure",
    data: "Data",
    train: "Train",
    inference: "Inference",
    evaluation: "Evaluation",
    registry: "Registry",
    orchestrator: "Orchestrator",
  } as Record<string, string>)[t.agent_id || ""] || t.agent_id || "Stage";

  if (op === "run_inference") {
    if (phase === "submit") return `Submit ${inferenceName}`;
    if (phase === "collect") return `Collect ${inferenceName} Results`;
    if (phase === "repair") return `Repair ${inferenceName}`;
    if (phase === "continue") return `Continue ${inferenceName}`;
  }
  if (op === "train") {
    if (phase === "submit") return "Submit Candidate Training";
    if (phase === "collect") return "Collect Candidate Training Results";
    if (phase === "repair") return "Repair Candidate Training";
    if (phase === "continue") return "Continue Candidate Training";
  }
  if (phase === "submit") return `Submit ${stageName} Job`;
  if (phase === "collect") return `Collect ${stageName} Results`;
  if (phase === "repair") return `Repair ${stageName}`;
  if (phase === "continue") return `Continue ${stageName}`;

  const labels: Record<string, string> = {
    provision: "Provision compute",
    prepare_run_data: "Prepare Run data and Validation setup",
    prepare_holdout_data: `Prepare questions-only held-out data${
      t.test_set_name ? ` · ${t.test_set_name}` : ""
    }`,
    train: "Select config and train candidate",
    run_inference: t.lane === "held_out_test"
      ? `Run held-out inference${t.test_set_name ? ` · ${t.test_set_name}` : ""}`
      : t.model_source === "base_model"
        ? "Run Baseline Inference"
        : "Run Candidate Inference",
    release: "Release resources",
  };
  if (labels[op]) return labels[op];
  if (agentIdOf(t) === "evaluation") {
    return t.lane === "held_out_test"
      ? `Run held-out evaluator via Bash${t.test_set_name ? ` · ${t.test_set_name}` : ""}`
      : "Run validation evaluator via Bash";
  }
  if (agentIdOf(t) === "registry") return "Select and save the best model";
  return op ? op.replaceAll("_", " ") : "Execute stage";
}

function orchestratorLabel(h: HeartbeatDTO, tickets: Map<string, TLTicket>): string {
  const child = h.child_ticket_id ? tickets.get(h.child_ticket_id) : undefined;
  const children = child ? [child] : [];
  if (h.action === "emit_ticket") {
    const child = children[0];
    if (child?.iteration === 0 && child.operation === "run_inference") {
      return "Start baseline execution";
    }
    const earlierExecution = child && Array.from(tickets.values()).some((ticket) =>
      ticket.iteration === child.iteration
      && ticket.created_at < child.created_at
      && ["prepare_run_data", "train", "run_inference"].includes(ticket.operation)
    );
    if (
      child && child.iteration > 0
      && ["prepare_run_data", "train"].includes(child.operation)
      && !earlierExecution
    ) {
      return `Start iteration ${child.iteration} execution`;
    }
    return child ? `Start ${stageLabel(child)}` : "Advance pipeline";
  }
  if (h.action === "mark_done") return "Finalize Run";
  if (h.action === "mark_failed") return "Stop Run";
  if (h.action === "wait") return "Wait for dependencies";
  return h.is_live ? "Orchestrator deciding…" : "Orchestrator";
}

type StageGroup = {
  key: string;
  label: string;
  kind: "baseline" | "iteration";
  num: number;
  tickets: TLTicket[];       // stage tickets that ran in this block
  /** The held-out measurement of this round, kept apart from `tickets`.
   *  Nothing on this list was planned, emitted or read by the orchestrator —
   *  the harness raises it against the test set once the round's model exists.
   *  Listed inline it read as more of the loop's own work, which is the one
   *  thing it must not look like. */
  mirror: TLTicket[];
};

/** Partition actual stage tickets into Baseline + Iteration 1, 2, ….
 *
 * The run-wide Orchestrator ticket is deliberately excluded from creating a
 * group. At the end of iteration N it is woken to decide whether iteration N+1
 * should exist, so its mutable envelope may temporarily say N+1 even when the
 * decision is to stop. Only a Specialist ticket proves that a round started.
 * Orchestrator heartbeats are associated with those real groups below. */
function groupByIteration(tickets: TLTicket[]): StageGroup[] {
  const sorted = [...tickets].sort((a, b) => (a.created_at || "").localeCompare(b.created_at || ""));
  const info = assignIterations(tickets);
  const groups: StageGroup[] = [];
  const byKey = new Map<string, StageGroup>();
  for (const t of sorted) {
    if (agentIdOf(t) === "orchestrator") continue;
    const gi = info.get(t.id);
    if (!gi) continue;
    let g = byKey.get(gi.key);
    if (!g) {
      g = { key: gi.key, label: gi.label, kind: gi.kind, num: gi.num, tickets: [], mirror: [] };
      byKey.set(gi.key, g);
      groups.push(g);
    }
    if (t.lane === "held_out_test") g.mirror.push(t);
    else g.tickets.push(t);
  }
  // Keep the first Orchestrator wake visible while it is still deciding which
  // Baseline stage to create. Once a child exists, that child owns the group.
  if (groups.length === 0 && sorted.some((t) => agentIdOf(t) === "orchestrator")) {
    groups.push({
      key: "baseline", label: "Iteration 0", kind: "baseline", num: 0,
      tickets: [], mirror: [],
    });
  }
  // Ticket creation time controls order inside a group, never group order.
  return groups.sort((a, b) => a.num - b.num);
}

/** One collapsible block of TicketCards under a coloured header. */
function CollapsibleTicketBlock({
  blockKey, label, tone, noun, ticketIds, expanded, onToggle,
}: {
  blockKey: string;
  label: string;
  tone: "orchestrator" | "baseline" | "iteration";
  noun: string;
  ticketIds: string[];
  expanded: Set<string>;
  onToggle: (k: string) => void;
}) {
  const isCollapsed = !expanded.has(blockKey);
  const color = tone === "orchestrator" ? "text-lilac-300" : "text-brass-300";
  return (
    <div>
      <button
        onClick={() => onToggle(blockKey)}
        className="mb-2 flex w-full items-center gap-2 text-left"
      >
        {isCollapsed ? (
          <ChevronRight size={14} className="text-dim" />
        ) : (
          <ChevronDown size={14} className="text-dim" />
        )}
        <span className={`font-mono text-xs font-semibold uppercase tracking-[0.14em] ${color}`}>{label}</span>
        <span className="font-mono text-[13px] text-dim">
          {ticketIds.length} {noun}{ticketIds.length === 1 ? "" : "s"}
        </span>
      </button>
      {!isCollapsed && (
        // A rail with the slips hanging off it, left to right.
        <div className="relative">
          <div className="absolute inset-x-0 top-0 h-px bg-hair" />
          <div className="grid grid-cols-[repeat(auto-fill,minmax(9.5rem,1fr))] gap-x-4 gap-y-6">
            {ticketIds.map((id) => (
              <TicketCard key={id} ticketId={id} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

type RunTab = "iterations" | "journal" | "timeline" | "artifacts" | "tickets";
const RUN_TABS: readonly RunTab[] = ["iterations", "journal", "timeline", "artifacts", "tickets"];
const isRunTab = (v: string | null): v is RunTab => RUN_TABS.includes(v as RunTab);

/** Per-ticket tab: the orchestrator's supervisor wake-tickets pulled into a
 *  single block at the top, then the stage tickets grouped under collapsible
 *  Baseline / Iteration 1, 2, … blocks so it lines up with the timeline. */
function PerTicketGroups({ run }: { run: RunDetail }) {
  const orchestrateIds = useMemo(
    () =>
      run.tickets
        .filter((t) => agentIdOf(t) === "orchestrator")
        .sort((a, b) => (a.created_at || "").localeCompare(b.created_at || ""))
        .map((t) => t.id),
    [run.tickets],
  );
  // Iteration boundaries are driven by data/train, so dropping orchestrate
  // tickets from the input doesn't shift them.
  const groups = useMemo(
    () => iterationOrder(run.tickets.filter((t) => agentIdOf(t) !== "orchestrator")),
    [run.tickets],
  );
  // Track EXPANDED blocks; empty by default → every block starts collapsed.
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  // Follow the run: the block being worked on opens, and the one before it
  // closes as soon as work moves on. A manual toggle still wins until the run
  // moves to the next block.
  const liveKey = liveIterationKey(run.tickets);
  useEffect(() => {
    if (liveKey) setExpanded(new Set([liveKey]));
  }, [liveKey]);

  const toggle = (k: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(k) ? next.delete(k) : next.add(k);
      return next;
    });
  return (
    <div className="space-y-4">
      {orchestrateIds.length > 0 && (
        <CollapsibleTicketBlock
          blockKey="orchestrator"
          label="Orchestrator"
          tone="orchestrator"
          noun="ticket"
          ticketIds={orchestrateIds}
          expanded={expanded}
          onToggle={toggle}
        />
      )}
      {groups.map((g) => (
        <CollapsibleTicketBlock
          key={g.key}
          blockKey={g.key}
          label={g.label}
          tone={g.kind}
          noun="ticket"
          ticketIds={g.ticketIds}
          expanded={expanded}
          onToggle={toggle}
        />
      ))}
    </div>
  );
}

function PipelineTimeline({ run, heartbeats }: { run: RunDetail; heartbeats: HeartbeatDTO[] }) {
  const groups = useMemo(() => groupByIteration(run.tickets), [run.tickets]);
  // A run has ONE supervisor ticket with one heartbeat per activation, spanning
  // every iteration. groupByIteration can only file it under one block, so the
  // other blocks' markers had no ticket to point at and simply did nothing when
  // clicked. Resolve it run-wide instead, and number the activations run-wide
  // too: the wake above a stage is the one that emitted it, wherever that stage
  // sits in the run.
  const supervisor = useMemo(
    () => run.tickets.find((t) => agentIdOf(t) === "orchestrator"),
    [run.tickets],
  );
  const supervisorId = supervisor?.id ?? "";
  const orchestratorWakes = useMemo(
    () => heartbeats
      .filter((h) => h.agent_id === "orchestrator" && h.ticket_id === supervisorId)
      .sort((a, b) => a.started_at.localeCompare(b.started_at)),
    [heartbeats, supervisorId],
  );
  const wakeIndex = useMemo(
    () => new Map(orchestratorWakes.map((heartbeat, index) => [heartbeat.id, index])),
    [orchestratorWakes],
  );
  const ticketById = useMemo(
    () => new Map(run.tickets.map((ticket) => [ticket.id, ticket])),
    [run.tickets],
  );
  const heartbeatsByTicket = useMemo(() => {
    const rows = new Map<string, HeartbeatDTO[]>();
    for (const heartbeat of heartbeats) {
      const current = rows.get(heartbeat.ticket_id) || [];
      current.push(heartbeat);
      rows.set(heartbeat.ticket_id, current);
    }
    for (const current of rows.values()) {
      current.sort((a, b) => a.started_at.localeCompare(b.started_at));
    }
    return rows;
  }, [heartbeats]);
  const orchestratorByGroup = useMemo(() => {
    const byGroup = new Map<string, HeartbeatDTO[]>();
    const iterationInfo = assignIterations(run.tickets);
    const stages = run.tickets
      .filter((ticket) => agentIdOf(ticket) !== "orchestrator" && ticket.lane !== "held_out_test")
      .sort((a, b) => a.created_at.localeCompare(b.created_at));
    for (const heartbeat of orchestratorWakes) {
      let key = "";
      const child = heartbeat.child_ticket_id
        ? ticketById.get(heartbeat.child_ticket_id)
        : undefined;
      if (child) key = iterationInfo.get(child.id)?.key || "";
      if (!key) {
        const preceding = stages.filter((ticket) => ticket.created_at <= heartbeat.started_at).at(-1);
        key = preceding ? iterationInfo.get(preceding.id)?.key || "baseline" : "baseline";
      }
      const rows = byGroup.get(key) || [];
      rows.push(heartbeat);
      byGroup.set(key, rows);
    }
    return byGroup;
  }, [orchestratorWakes, run.tickets, ticketById]);
  const allTickets = groups.flatMap((g) => [...g.tickets, ...g.mirror]);
  const running = allTickets.find((t) => t.status === "running" || t.status === "repairing");
  const runningOrchestrator = orchestratorWakes.find((heartbeat) => heartbeat.is_live);
  const [sel, setSel] = useState<string>("");
  // While the run is LIVE the right panel follows the running stage — a click
  // earlier in the run should not pin the view to a stage that finished ten
  // minutes ago. Clearing the selection when the running stage changes is what
  // makes it follow; without it the auto-pick below only ever applied once,
  // before the first click.
  useEffect(() => {
    if (running?.id) setSel("");
  }, [running?.id]);
  // Auto-show the running stage while the run is live; once it finishes, the
  // right panel stays blank until the user clicks a stage on the left.
  const selectedId = sel || running?.id || (
    runningOrchestrator
      ? `${supervisorId}#${wakeIndex.get(runningOrchestrator.id) ?? 0}`
      : ""
  );
  // Track EXPANDED groups; empty by default → the timeline starts fully collapsed.
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  // Follow the run: the block being worked on opens, and the one before it
  // closes as soon as work moves on. A manual toggle still wins until the run
  // moves to the next block.
  const liveKey = liveIterationKey(run.tickets);
  useEffect(() => {
    if (liveKey) setExpanded(new Set([liveKey]));
  }, [liveKey]);

  const toggle = (k: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(k) ? next.delete(k) : next.add(k);
      return next;
    });

  if (allTickets.length === 0 && orchestratorWakes.length === 0) {
    return <div className="text-xs text-dim">No stages yet.</div>;
  }

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[300px_1fr]">
      {/* Left: iteration outline */}
      <div className="space-y-2">
        {groups.map((g) => {
          const open = expanded.has(g.key);
          const groupTickets = [...g.tickets, ...g.mirror];
          const groupOrchestrator = orchestratorByGroup.get(g.key) || [];
          const groupRunning = groupTickets.some((t) => t.status === "running" || t.status === "repairing")
            || groupOrchestrator.some((heartbeat) => heartbeat.is_live);
          const groupFailed = groupTickets.some((t) => t.status === "failed")
            || groupOrchestrator.some((heartbeat) => heartbeat.action === "mark_failed");
          const groupDegraded = groupTickets.some((t) => t.status === "degraded");
          const groupDone = groupTickets.length > 0
            && groupTickets.every((t) => ["succeeded", "skipped"].includes(t.status));
          const headDot = groupRunning
            ? "bg-brass-400"
            : groupFailed
              ? "bg-coral-500"
              : groupDegraded
                ? "bg-lilac-400"
                : groupDone
                  ? "bg-phosphor-400"
                  : "bg-slate-600";
          return (
            <Fragment key={g.key}>
              <div className="rounded-bezel border border-hair bg-panel/40">
                {/* Iteration header */}
                <button
                  onClick={() => toggle(g.key)}
                  className="flex w-full items-center gap-2 px-3 py-2.5 text-left"
                >
                  <span className="font-mono text-[13px] text-dim">{open ? "▾" : "▸"}</span>
                  <span className={`h-2 w-2 rounded-full ${headDot} ${groupRunning ? "animate-lamp-pulse" : ""}`} />
                  <span className="inline-flex items-baseline gap-1.5 font-mono text-xs font-semibold uppercase tracking-[0.14em] text-brass-300">
                    <span>Iteration</span>
                    <span>{g.num}</span>
                  </span>
                  <span className="ml-auto font-mono text-[13px] text-dim">
                    {g.tickets.length + g.mirror.length}
                  </span>
                </button>
                {/* Real Orchestrator heartbeat actions and sequential Specialist
                    Tickets share one chronological rail. */}
                {open && (g.tickets.length > 0 || groupOrchestrator.length > 0 || g.mirror.length > 0) && (() => {
                  const planRow = (heartbeat: HeartbeatDTO) => {
                    const wake = wakeIndex.get(heartbeat.id) ?? 0;
                    const selId = supervisorId ? `${supervisorId}#${wake}` : "";
                    const active = !!selId && selectedId === selId;
                    const live = heartbeat.is_live;
                    const activationFailed = !!heartbeat.finished_at && heartbeat.exit_code !== 0;
                    // Preserve every activation in the Timeline, but render its
                    // settled outcome: a failed activation immediately followed
                    // by a successful repair of the same supervisor Ticket is
                    // successful here. The raw exit/error remains in its detail.
                    const nextWake = orchestratorWakes[wake + 1];
                    const recovered = heartbeat.action !== "mark_failed"
                      && activationFailed
                      && nextWake?.ticket_id === heartbeat.ticket_id
                      && !!nextWake.finished_at
                      && nextWake.exit_code === 0;
                    const failed = heartbeat.action === "mark_failed"
                      || (activationFailed && !recovered);
                    // A parse failure can erase the projected action/child from
                    // this Heartbeat even though the child Ticket was committed.
                    // Recover that display metadata from the one normal-lane
                    // Ticket actually created before the repair wake, so this
                    // row still says e.g. "Start Evaluation" instead of the
                    // uninformative "Orchestrator" fallback.
                    const committedChild = recovered && !heartbeat.action
                      ? Array.from(ticketById.values())
                        .filter((ticket) => (
                          agentIdOf(ticket) !== "orchestrator"
                          && ticket.lane !== "held_out_test"
                          && ticket.created_at >= heartbeat.started_at
                          && ticket.created_at < nextWake.started_at
                        ))
                        .sort((a, b) => a.created_at.localeCompare(b.created_at))[0]
                      : undefined;
                    const displayedHeartbeat = committedChild
                      ? {
                        ...heartbeat,
                        action: "emit_ticket",
                        child_ticket_id: committedChild.id,
                      }
                      : heartbeat;
                    const displayedSummary = heartbeat.action_summary || (
                      committedChild
                        ? `Created ${committedChild.id}${
                          committedChild.summary ? ` · ${committedChild.summary}` : ""
                        }`
                        : ""
                    );
                    return (
                      <li key={`orchestrator-${heartbeat.id}`} className="relative">
                        <span className="absolute left-[18px] top-6 h-[calc(100%-12px)] w-px bg-hair" />
                        <button
                          onClick={() => selId && setSel(selId)}
                          className={`group flex w-full items-start gap-3 rounded-lg px-2 py-1.5 text-left transition ${
                            active ? "bg-lilac-500/10" : "hover:bg-raised/40"
                          }`}
                        >
                          <span className="relative mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center">
                            {live && (
                              <span className="absolute inline-flex h-3.5 w-3.5 animate-ping rotate-45 bg-lilac-400 opacity-50" />
                            )}
                            <span
                              className={`relative h-2 w-2 rotate-45 ${
                                failed
                                  ? "bg-coral-500"
                                  : recovered
                                    ? "bg-phosphor-400 shadow-glow-phosphor"
                                    : live
                                      ? "bg-lilac-300 animate-lamp-pulse"
                                      : "bg-lilac-400/70"
                              } ${live ? "shadow-[0_0_8px] shadow-lilac-500/60" : ""}`}
                            />
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className={`block text-sm italic ${
                              live || active ? "text-lilac-200" : "text-lilac-300/75 group-hover:text-lilac-200"
                            }`}>
                              {orchestratorLabel(displayedHeartbeat, ticketById)}
                            </span>
                            {displayedSummary && (
                              <span className="block truncate text-[12px] text-dim">{displayedSummary}</span>
                            )}
                          </span>
                        </button>
                      </li>
                    );
                  };

                  const rows: JSX.Element[] = [];
                  const mainItems: Array<
                    | { kind: "ticket"; ts: string; ticket: TLTicket }
                    | { kind: "orchestrator"; ts: string; heartbeat: HeartbeatDTO }
                  > = [
                    ...g.tickets.map((ticket) => ({ kind: "ticket" as const, ts: ticket.created_at, ticket })),
                    ...groupOrchestrator.map((heartbeat) => ({
                      kind: "orchestrator" as const, ts: heartbeat.started_at, heartbeat,
                    })),
                  ].sort((a, b) => a.ts.localeCompare(b.ts));
                  mainItems.forEach((item) => {
                    if (item.kind === "orchestrator") {
                      rows.push(planRow(item.heartbeat));
                      return;
                    }
                    const t = item.ticket;
                    const activationCaptions = Array.from(new Set(
                      (heartbeatsByTicket.get(t.id) || [])
                        .map((heartbeat) => operationCaption(
                          t, heartbeat.operation, heartbeat.activation_phase,
                        ))
                        .filter(Boolean),
                    ));
                    const activity = activationCaptions.length > 1
                      ? activationCaptions.join(" → ")
                      : activationCaptions[0] || operationCaption(t);
                    const active = selectedId === t.id;
                    const isRunning = t.status === "running" || t.status === "repairing";
                    const dot = statusToneFor(t.status).dot;
                    rows.push(
                      <li key={t.id} className="relative">
                        <span className="absolute left-[18px] top-7 h-[calc(100%-12px)] w-px bg-hair" />
                        <button
                          onClick={() => setSel(t.id)}
                          className={`flex w-full items-start gap-3 rounded-lg px-2 py-1.5 text-left transition ${
                            active ? "bg-brass-500/10" : "hover:bg-raised/50"
                          }`}
                        >
                          <span className="relative mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center">
                            {isRunning && (
                              <span className={`absolute inline-flex h-full w-full animate-ping rounded-full ${dot} opacity-60`} />
                            )}
                            <span className={`relative h-3 w-3 rounded-full ${dot} ${isRunning ? "animate-lamp-pulse" : ""}`} />
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="flex items-center justify-between gap-2">
                              <span className={`text-sm font-medium ${active ? "text-brass-300" : "text-slate-200"}`}>
                                {stageLabel(t)}
                              </span>
                              <StatusBadge status={t.status} />
                            </span>
                            <span className="flex min-w-0 items-center gap-1.5">
                              <span className="truncate font-mono text-[12px] text-dim">{activity}</span>
                            </span>
                            <span className="block truncate font-mono text-[11px] text-slate-600">{t.id}</span>
                          </span>
                        </button>
                      </li>,
                    );
                  });
                  // The held-out lane hangs OFF the round, it is not a step in
                  // it. The orchestrator did not plan these tickets, cannot
                  // read them, and does not act on what they measure — drawn in
                  // the main column they read as two more stages the loop ran,
                  // which is exactly the confusion the whole arrangement exists
                  // to avoid. So: an elbow out of the main line, and the lane
                  // continues beside it.
                  if (g.mirror.length > 0) {
                    rows.push(
                      <li key={`mirror-${g.key}`} className="relative">
                        {/* Down, then out: the elbow is the branch. */}
                        <span className="absolute left-[18px] top-0 h-5 w-4 rounded-bl-[10px] border-b border-l border-hair" />
                        <div className="ml-[38px] pb-1 pt-1.5">
                          <div className="pl-2 text-[11px] font-medium uppercase tracking-wider text-skyx-300/70">
                            held out
                          </div>
                          <ol className="relative">
                            {g.mirror.map((t, i) => {
                              const active = selectedId === t.id;
                              const isRunning = t.status === "running" || t.status === "repairing";
                              const dot = statusToneFor(t.status).dot;
                              return (
                                <li key={t.id} className="relative">
                                  {i < g.mirror.length - 1 && (
                                    <span className="absolute left-[14px] top-6 h-[calc(100%-12px)] w-px bg-hair/60" />
                                  )}
                                  <button
                                    onClick={() => setSel(t.id)}
                                    className={`flex w-full items-start gap-2.5 rounded-lg px-2 py-1 text-left transition ${
                                      active ? "bg-skyx-500/10" : "hover:bg-raised/40"
                                    }`}
                                  >
                                    <span className="relative mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center">
                                      {isRunning && (
                                        <span className={`absolute inline-flex h-full w-full animate-ping rounded-full ${dot} opacity-60`} />
                                      )}
                                      <span className={`relative h-2 w-2 rounded-full ${dot} ${isRunning ? "animate-lamp-pulse" : ""}`} />
                                    </span>
                                    <span className="min-w-0 flex-1">
                                      <span className="flex items-center justify-between gap-2">
                                        <span className={`text-[13px] ${active ? "text-skyx-200" : "text-skyx-300/80"}`}>
                                          {stageLabel(t)}
                                        </span>
                                        <StatusBadge status={t.status} />
                                      </span>
                                      <span className="block truncate font-mono text-[11px] text-dim">{operationCaption(t)}</span>
                                      <span className="block truncate font-mono text-[10px] text-slate-600">{t.id}</span>
                                    </span>
                                  </button>
                                </li>
                              );
                            })}
                          </ol>
                        </div>
                      </li>,
                    );
                  }
                  return <ol className="relative border-t border-hair/60 p-2">{rows}</ol>;
                })()}
              </div>
            </Fragment>
          );
        })}
      </div>

      {/* Right: blown-up live detail of the selected / running stage. The id
          may carry a wake suffix (see planRow) — split it back apart here. */}
      <StageDetail
        ticketId={selectedId.split("#")[0]}
        wake={selectedId.includes("#") ? Number(selectedId.split("#")[1]) : undefined}
      />
    </div>
  );
}

function StageDetail({ ticketId, wake }: {
  ticketId: string;
  wake?: number;
}) {
  const { data: t } = useSWR<TicketDetail>(
    ticketId ? `/api/tickets/${ticketId}` : null, {
    refreshInterval: (latest) =>
      latest && ["succeeded", "failed", "cancelled", "skipped", "degraded"].includes(latest.status) ? 0 : 3000,
  });
  // EVERY wake of this stage, not just the latest. A ticket gets one heartbeat
  // per wake, and the orchestrator is re-woken after each child ticket
  // finishes — so `limit=1` showed one wake's tokens and cost while the run
  // header summed them all, and the stages never added up to the total.
  const { data: heartbeats = [] } = useSWR<HeartbeatDTO[]>(
    ticketId ? `/api/heartbeats?ticket_id=${encodeURIComponent(ticketId)}&limit=50` : null,
    { refreshInterval: 3000 },
  );
  // A cluster allocation is recorded immediately after `sbatch`, before Slurm
  // grants a node. Poll the active rows so the selected stage can show its live
  // queue wait independently from the Run duration (which deliberately pauses
  // while queued).
  const { data: infraInstances = [] } = useSWR<InfraInstanceDTO[]>(
    t?.run_id ? `/api/infra/instances?run_id=${encodeURIComponent(t.run_id)}` : null,
    { refreshInterval: 3000 },
  );
  // The API returns heartbeats newest-first; the timeline counts activations
  // oldest-first, so flip before indexing. With no wake given (a normal stage,
  // which has one), show the newest.
  // The feed LiveTranscript is showing, lifted so the Overview above can list
  // the same steps without opening a second socket. Both hooks stay ABOVE the
  // early returns: React counts hooks per render and requires the same count
  // every time.
  const [events, setEvents] = useState<EventEnvelope[]>([]);
  const [seek, setSeek] = useState<{ ts: string; n: number }>({ ts: "", n: 0 });
  const [selectedActivation, setSelectedActivation] = useState<number | undefined>(wake);
  const chronological = [...heartbeats].sort((a, b) => a.started_at.localeCompare(b.started_at));
  const effectiveWake = selectedActivation;
  const picked = effectiveWake !== undefined ? chronological[effectiveWake] : heartbeats[0];
  const hbId = picked?.id;
  useEffect(() => {
    setSelectedActivation(wake);
  }, [ticketId, wake]);
  // Clear on switch. LiveTranscript resets its own feed per heartbeat, but this
  // copy would otherwise keep the previous activation's steps on screen until
  // the new one produced its first event.
  useEffect(() => {
    setEvents([]);
    setSeek({ ts: "", n: 0 });
  }, [hbId]);

  if (!ticketId) {
    return (
      <div className="flex h-full min-h-[8rem] items-center justify-center rounded-bezel border border-dashed border-hair bg-panel/20 p-4 text-center text-xs text-dim">
        Select a stage on the left to view its transcript.
      </div>
    );
  }
  if (!t) {
    return <div className="rounded-bezel border border-hair bg-panel/40 p-4 text-xs text-dim">loading…</div>;
  }
  // Overview must describe the SAME activation the transcript below shows.
  // `t.execution_events` and `t.summary` are ticket-level and accumulate across every
  // wake, so on a supervisor ticket with 26 activations they always described
  // the latest one no matter which marker you clicked. Execution events carry `ts`, so
  // scope them to the picked wake's window; the summary has no per-wake
  // equivalent, so it is shown only when you are looking at the latest wake.
  const executionEvents = (() => {
    // Scope to `picked` whenever there IS one -- NOT only when a wake index was
    // passed. Clicking a stage on the timeline gives no wake, and `picked` then
    // falls back to the newest heartbeat, which is the one the transcript below
    // renders. Returning the unscoped ticket-level list there broke the very
    // invariant this block exists for: the overview listed every wake's execution events
    // while the transcript showed one wake's events, so a supervisor ticket on
    // its tenth wake dumped 27 stale headings above the first event.
    if (!picked) return t.execution_events;
    // Attribute by when the remote process emitted the marker, not by which
    // heartbeat happened to observe it. A cluster collect/repair activation may
    // read the completed Slurm log; those replayed markers carry the new
    // heartbeat id but retain their original training timestamps. Trusting only
    // heartbeat_id therefore draws the same loss curve under Collect/Repair.
    // The interval below puts both the live reading and any replay back in the
    // activation during which the work actually happened; TrainingMonitor then
    // de-duplicates equal steps.
    const from = picked.started_at;
    // With no wake index, `picked` is the newest heartbeat and nothing follows
    // it, so the window runs to the end.
    const to = effectiveWake === undefined
      ? "\uffff"
      : chronological[effectiveWake + 1]?.started_at ?? "\uffff";
    return t.execution_events.filter((p) => p.ts >= from && p.ts < to);
  })();
  const pendingSlurmJob = infraInstances.find((instance) =>
    instance.provider === "cluster"
    && instance.ticket_id === ticketId
    && instance.status === "provisioning"
    && !instance.ready_at
    && !instance.released_at,
  );
  return (
    <Bezel className="overflow-hidden">
      <div className="flex items-center justify-between border-b border-hair p-4">
        <div className="flex items-center gap-2 font-mono text-sm">
          <Link to={`/tickets/${t.id}`} className="text-slate-200 hover:text-brass-300">{t.id}</Link>
          <StatusBadge status={t.status} />
        </div>
        <div className="flex items-center gap-3">
          <Link to={`/tickets/${t.id}`} className="font-mono text-[13px] uppercase tracking-wider text-dim hover:text-brass-300">
            open full →
          </Link>
        </div>
      </div>
      {/* The supervisor is already represented by one labelled row per wake in
          the timeline on the left. Repeating every wake as an Activation strip
          here adds control-plane noise without helping the reader understand a
          model stage. Specialist tickets retain the selector because their
          multiple executions can carry different runtime evidence. */}
      {t.agent_id !== "orchestrator" && chronological.length > 1 && (
        <div className="flex flex-wrap items-center gap-2 border-b border-hair px-4 py-2.5">
          <span className="font-mono text-[11px] uppercase tracking-wider text-dim">Activations</span>
          {chronological.map((heartbeat, index) => {
            const active = heartbeat.id === picked?.id;
            const label = heartbeat.operation
              ? operationCaption(t, heartbeat.operation, heartbeat.activation_phase)
              : `Activation ${index + 1}`;
            return (
              <button
                key={heartbeat.id}
                type="button"
                onClick={() => setSelectedActivation(index)}
                className={`rounded-full border px-2.5 py-1 font-mono text-[11px] transition ${
                  active
                    ? "border-brass-500/60 bg-brass-500/10 text-brass-300"
                    : "border-hair text-slate-400 hover:border-brass-500/30 hover:text-slate-200"
                }`}
              >
                {label}
              </button>
            );
          })}
        </div>
      )}
      <div className="space-y-4 p-4">
        <div className="rounded-bezel border border-hair bg-canvas/40 p-3">
          <div className="mb-2"><Kicker strong>Overview</Kicker></div>
          {pendingSlurmJob && <SlurmQueueWait instance={pendingSlurmJob} />}
          {/* What it did, in order, read off the feed below — so a step appears
              because it happened, not because a script remembered to say so. */}
          <StepTimeline
            events={events}
            executionEvents={executionEvents}
            empty="No activity recorded for this heartbeat yet."
            onSeek={hbId ? (ts) => setSeek((s) => ({ ts, n: s.n + 1 })) : undefined}
          />
          {/* Loss chart is a TRAIN-only thing (inference has no loss); guard on the
              agent so a stray/contaminated loss row can't surface it elsewhere. */}
          {t.agent_id === "train" && executionEvents.some(
            (p) => p.event_type === "attempt" || p.loss > 0,
          ) && (
            <div className="mt-3 rounded-bezel border border-hair bg-canvas/40 p-2">
              <TrainingMonitor executionEvents={executionEvents} />
            </div>
          )}
        </div>
        {hbId ? (
          <div>
            <div className="mb-1.5"><Kicker strong>Live activity</Kicker></div>
            <LiveTranscript
              heartbeatId={hbId}
              height="max-h-[55vh]"
              onEvents={setEvents}
              seekTs={seek.ts}
              seekNonce={seek.n}
            />
          </div>
        ) : (
          <div className="text-xs text-dim">No activity yet for this stage.</div>
        )}
      </div>
    </Bezel>
  );
}

function SlurmQueueWait({ instance }: { instance: InfraInstanceDTO }) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [instance.id, instance.created_at]);

  const submittedAt = Date.parse(instance.created_at);
  const waitedSeconds = Number.isFinite(submittedAt)
    ? Math.max(0, Math.floor((now - submittedAt) / 1000))
    : 0;

  return (
    <div className="mb-3 rounded-bezel border border-brass-500/30 bg-brass-500/[0.06] px-3 py-2.5">
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1.5">
        <span className="flex items-center gap-2 font-mono text-xs uppercase tracking-[0.12em] text-brass-300">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-brass-400 opacity-60" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-brass-400" />
          </span>
          Waiting for GPU
        </span>
        <span className="font-mono text-xs text-slate-300 tabular-nums">
          <span className="mr-1.5 text-dim">waited</span>{fmtDuration(waitedSeconds)}
        </span>
      </div>
      {instance.instance_id && (
        <div className="mt-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-dim">
          Slurm job {instance.instance_id}
        </div>
      )}
    </div>
  );
}

function CancelRunButton({ run, onCancelled }: { run: RunDetail; onCancelled: () => void }) {
  const [open, setOpen] = useState(false);
  // Same folder the registry stage uses, so a rescued model sits beside a
  // registered one.
  const defaultDir = `data/runs/${run.id}/models/M-${run.id.slice(0, 8)}`;
  return (
    <>
      {open && (
        <CancelRunDialog runId={run.id} defaultDir={defaultDir}
          onClose={() => setOpen(false)} onCancelled={onCancelled} />
      )}
      <button
        onClick={() => setOpen(true)}
        className="btn border-coral-500/40 text-coral-300 hover:border-coral-500/70 hover:text-coral-200"
      >
        <StopCircle size={14} /> cancel run
      </button>
    </>
  );
}

export function RunDetailPage() {
  const { runId = "" } = useParams();
  const { data: run, error, mutate } = useSWR<RunDetail>(`/api/runs/${runId}`, {
    refreshInterval: 3000,
  });
  // Wakes per ticket, for the counts on the ring. A ticket is not a unit of
  // work here — the supervisor owns ONE ticket and wakes once per child — so
  // the number that means "how often has this agent run" has to come from the
  // heartbeats. `limit` is generous: a long run's supervisor alone passes 25.
  const { data: heartbeats = [] } = useSWR<HeartbeatDTO[]>(
    runId ? `/api/heartbeats?run_id=${encodeURIComponent(runId)}&limit=500` : null,
    { refreshInterval: 5000 },
  );
  const wakesByTicket = useMemo(() => {
    const m: Record<string, number> = {};
    for (const h of heartbeats) m[h.ticket_id] = (m[h.ticket_id] ?? 0) + 1;
    return m;
  }, [heartbeats]);
  const supervisorWakes = useMemo(
    () => heartbeats.filter((h) => h.agent_id === "orchestrator").length,
    [heartbeats],
  );
  // The hub's lamp, by the same rule the rim stations use — the supervisor is
  // an agent and can be running, done or failed like any other.
  const supervisorState: StationState = useMemo(() => {
    const ts = (run?.tickets ?? []).filter((x) => agentIdOf(x) === "orchestrator");
    if (!ts.length) return "idle";
    if (ts.some((x) => x.status === "running" || x.status === "repairing")) return "active";
    if (ts.some((x) => x.status === "failed")) return "failed";
    if (ts.some((x) => x.status === "degraded")) return "degraded";
    if (ts.every((x) => ["succeeded", "skipped"].includes(x.status))) return "done";
    if (ts.every((x) => ["succeeded", "skipped", "cancelled"].includes(x.status))) return "cancelled";
    return "idle";
  }, [run?.tickets]);
  const ws = useRunWebsocket(runId);
  // Keep the active tab in the URL (?tab=...) so navigating into a ticket and
  // hitting "back" returns to the SAME tab (e.g. Per-ticket) instead of resetting
  // to the default. `replace` keeps tab switches out of the history stack.
  const [searchParams, setSearchParams] = useSearchParams();
  const rawTab = searchParams.get("tab");
  const tab: RunTab = isRunTab(rawTab) ? rawTab : "iterations";
  const setTab = (t: RunTab) =>
    setSearchParams(
      (prev) => {
        prev.set("tab", t);
        return prev;
      },
      { replace: true },
    );
  // Mutate the SWR cache when the WS pushes an update -- causes per-ticket
  // panes (each on their own SWR key) to revalidate on the next tick. In an
  // effect, not at render time: calling mutate() in the component body
  // re-triggered a state update on every render the socket had a message for.
  useEffect(() => {
    if (ws) void mutate();
  }, [ws, mutate]);

  if (error) {
    const status = (error as any).status;
    return (
      <div className="w-full px-[max(1.5rem,1.5vw)] py-8">
        <Link to="/runs" className="inline-flex items-center gap-1.5 text-sm text-slate-400 transition hover:text-brass-300">
          <ArrowLeft size={15} /> Runs
        </Link>
        <div className="mt-6 rounded-bezel border border-coral-500/30 bg-coral-500/10 p-4 text-sm text-coral-300">
          {status === 404 ? `No run with id ${runId}.` : String(error.message || error)}
        </div>
      </div>
    );
  }
  if (!run) {
    return <div className="w-full px-[max(1.5rem,1.5vw)] py-8 text-sm text-dim">Acquiring telemetry…</div>;
  }

  const live = !["success", "degraded", "failed", "halted", "cancelled"].includes(run.status);
  const stations = stationsFrom(run, wakesByTicket);
  const measuredCandidates = (run.history ?? []).filter(
    (entry) => entry.test_scores && Object.keys(entry.test_scores).length > 1,
  );
  const trainedCandidates = measuredCandidates.filter((entry) => entry.source !== "baseline");
  const championCandidates = trainedCandidates.length ? trainedCandidates : measuredCandidates;
  const championBreakdown = championCandidates.reduce<
    (typeof championCandidates)[number] | undefined
  >(
    (best, entry) => {
      if (!best) return entry;
      return run.validation_metric_direction === "min"
        ? entry.score < best.score ? entry : best
        : entry.score > best.score ? entry : best;
    },
    undefined,
  );

  return (
    <div className="w-full px-[max(1.5rem,1.5vw)] py-8">
      <Link to="/runs" className="inline-flex items-center gap-1.5 text-sm text-slate-400 transition hover:text-brass-300">
        <ArrowLeft size={15} /> Runs
      </Link>

      <div className="mt-3">
        <PageHead
          // The run's own name, with the id beside it — the id is what you
          // paste into a URL or grep a log for, so it stays on the page even
          // when the run has a name of its own.
          title={
            run.run_name ? (
              <span className="flex flex-wrap items-baseline gap-3">
                <span className="min-w-0 break-all">{run.run_name}</span>
                <span className="font-mono text-base font-normal text-slate-500" title={run.id}>
                  {run.id.slice(0, 8)}
                </span>
              </span>
            ) : (
              run.id.slice(0, 8)
            )
          }
          // The problem, not the problem plus three sentences repeating the
          // settings shown in this very page. The composed string is what the
          // orchestrator was handed, so it stays one hover away.
          subtitle={run.task_objective}
          subtitleTitle={run.agent_objective}
          right={live && !run.cancelling ? <CancelRunButton run={run} onCancelled={() => void mutate()} /> : undefined}
        />
      </div>

      {/* ── Instrument header strip: identity + top-line telemetry ── */}
      <Bezel className="mb-5 flex flex-wrap items-center gap-x-12 gap-y-4 px-6 py-4">
        <StatusBadge status={run.status} />
        {/* No rule between this and the telemetry: the strip reads as one row
            of facts. The task name is the first of them — the id moved up into
            the title, so what the run executes is what this position says. */}
        <HeaderTelemetry run={run} runId={runId} />
      </Bezel>

      {run.cancelling ? (
        <div className="mb-5 flex items-center gap-2 rounded-bezel border border-brass-500/30 bg-brass-500/10 p-3 text-sm text-brass-300">
          <span className="lamp lamp-running" />
          cancelling: {run.cancel_policy.weights === "hf"
            ? `pushing the champion checkpoint to ${run.cancel_policy.hf_repo_id}`
            : "copying the champion checkpoint off the box"} before the GPU is released…
        </div>
      ) : run.halted_reason ? (
        <div className="mb-5 rounded-bezel border border-coral-500/30 bg-coral-500/10 p-3 text-sm text-coral-300">
          halted: {run.halted_reason}
          {run.cancel_outcome.hf_url && (
            <> · <a href={run.cancel_outcome.hf_url} target="_blank" rel="noreferrer" className="underline">open on the Hub</a></>
          )}
        </div>
      ) : null}

      {/* ── Hero: what this run achieved (left) beside the loop that produced
          it (right). The numbers and the task-score trace take two thirds and
          lead, because they are what you open a run to read; the schematic is
          context and sits in the narrower column. */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        {/* One dashboard, not six cards: the five readouts and the curve are
            read together, so they share a panel and are separated by rules
            rather than by gaps and six sets of borders. */}
        <Bezel className="flex flex-col p-6 lg:col-span-2">
          <div className="grid flex-[2] grid-cols-1 content-center divide-y divide-hair sm:grid-cols-3 sm:divide-x sm:divide-y-0">
            {/* Keep this heading on the same top line as Iterations and the
                other readouts; the dial begins below that shared baseline. */}
            <div className="pb-5 sm:pb-0 sm:pr-6">
              <Kicker strong className="!text-sm">Champion test score</Kicker>
              <div className="mt-2 flex justify-center">
                <Gauge
                  value={run.champion_test_score ?? 0}
                  display={fmtScore(run.champion_test_score, run.metric)}
                  tone="phosphor"
                  size={168}
                  bounded={isPercentageMetric(run.metric)}
                />
              </div>
              {championBreakdown?.test_scores && (
                <div className="mx-auto mt-2 w-full max-w-[15rem] divide-y divide-hair border-y border-hair">
                  {Object.entries(championBreakdown.test_scores).map(([name, value]) => (
                    <div key={name} className="flex items-center justify-between gap-3 py-1.5 font-mono text-2xs">
                      <span className="min-w-0 truncate text-slate-400" title={name}>{name}</span>
                      <span className="shrink-0 tabular-nums text-phosphor-300">
                        {fmtScore(value, championBreakdown.test_metrics?.[name] ?? "")}
                        <span className="ml-1 text-slate-600">
                          {championBreakdown.test_metric_directions?.[name] === "min" ? "↓" : "↑"}
                        </span>
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-x-6 gap-y-5 pt-5 sm:col-span-2 sm:pl-6 sm:pt-0">
              <Readout
                size="card" strongLabel label="Iterations"
                value={run.iteration_budget <= 0 ? String(run.iterations_completed || 0) : `${Math.min(run.iteration_budget, run.iterations_completed || 0)}`}
                unit={run.iteration_budget <= 0 ? "/ \u221e" : `/ ${run.iteration_budget}`}
                tone="brass"
              />
              {/* An Auto run has no metric until scoping settles one; the
                  placeholder direction the server sends meanwhile is not a
                  fact worth printing. Once settled, say where the eval came
                  from — a public benchmark or one Zevo synthesized. */}
              {run.scoring_settled === false ? (
                <Readout
                  size="card" strongLabel label="Test Metric · Target"
                  value={<span className="text-2xl text-slate-500">Scoping…</span>}
                  tone="sky"
                  hint="choosing the metric and held-out eval"
                />
              ) : (
                <Readout
                  size="card" strongLabel label="Test Metric · Target"
                  value={`${fmtMetric(run.metric)} · ${run.metric_direction === "min" ? "Min" : "Max"}`}
                  tone="sky"
                  hint={fmtEvalSource(run.eval_source) || undefined}
                />
              )}
              {/* This iteration's held-out score, beside the best one on the
                  dial. Validation is not a readout: it is the set the loop
                  tuned on, so its only honest place is beside test in the
                  per-iteration figure, where the gap between them is visible. */}
              <Readout
                size="card" strongLabel label="Best Validation Score"
                value={fmtScore(run.best_validation_score, run.validation_metric)}
                tone="phosphor"
              />
              {/* Split the way the dashboard splits it: the two halves answer
                  different questions, and a single total hides which one is
                  running away. The note explains the $0 that a run on the
                  user's own hardware legitimately shows. */}
              <Readout
                size="card" strongLabel label="Cost" value={fmtCost(run.cost_usd)}
                // Read as "spent of allowed", the same shape as Iterations
                // beside it. Both are budgets, and showing one against its cap
                // while the other stood alone made the pair look unrelated.
                unit={run.max_cost_usd > 0 ? `/ ${fmtCost(run.max_cost_usd)}` : "/ ∞"}
                tone="coral"
                hint={
                  <span className="flex flex-wrap items-center gap-x-3 gap-y-0.5">
                    <span>Tokens {fmtCost(run.agent_cost_usd ?? 0)}</span>
                    <span>GPU {fmtCost(run.gpu_cost_usd ?? 0)}</span>
                  </span>
                }
              />
            </div>
          </div>

          {run.history?.length > 0 && (
            <div className="mt-6 flex flex-[3] flex-col border-t border-hair py-5">
              <IterationChart history={run.history} metricDirection={run.validation_metric_direction} metric={run.validation_metric} />
            </div>
          )}
        </Bezel>

        {/* The loop fills the column and centres the orbit in whatever height
            the taller column beside it sets, rather than sitting at the top
            with dead space underneath. */}
        <Bezel className="flex flex-col p-5">
          {/* The panel is titled where every other panel is — above it, in the
              same key as "Best score" — rather than inside the diagram. A name
              drawn at the hub competed with the ring for the middle and left
              the one place that means something (what drives the loop) spent on
              a label. */}
          <Kicker strong className="!text-sm">Zevo improvement loop</Kicker>
          <div className="flex flex-1 items-center">
            <PipelineRing
              stations={stations}
              centerCount={supervisorWakes || undefined}
              centerState={supervisorState}
              countNote={
                <Note size={16}>
                  The number shows how many times this Agent has heartbeated so far.
                </Note>
              }
              stationNotes={{
                evaluation: (
                  <Note size={16} tooltipClassName="w-64">
                    Evaluation is deterministic Bash, not an Agent call.
                  </Note>
                ),
              }}
            />
          </div>
          {/* The station order is already drawn by the ring itself, so the
              legend only has to explain the lamp colours. */}
          <div className="mt-3 flex flex-wrap items-center justify-center gap-x-4 gap-y-2 text-sm text-slate-400">
            <span className="flex items-center gap-1.5"><span className="lamp lamp-running" /> running</span>
            <span className="flex items-center gap-1.5"><span className="lamp lamp-idle" /> idle</span>
            <span className="flex items-center gap-1.5"><span className="lamp lamp-live" /> done</span>
            <span className="flex items-center gap-1.5"><span className="lamp lamp-coral" /> failed</span>
          </div>
          <CostStrip run={run} runId={runId!} />
        </Bezel>
      </div>

      {/* ── Data views ── */}
      <div className="mt-8 flex flex-wrap items-center gap-2">
        <button onClick={() => setTab("iterations")} className={`btn ${tab === "iterations" ? "btn-brass" : ""}`}>
          <Layers3 size={14} /> Iterations
        </button>
        <button onClick={() => setTab("journal")} className={`btn ${tab === "journal" ? "btn-brass" : ""}`}>
          <ScrollText size={14} /> Journal
        </button>
        <button onClick={() => setTab("timeline")} className={`btn ${tab === "timeline" ? "btn-brass" : ""}`}>
          <Clock size={14} /> Timeline
        </button>
        <button onClick={() => setTab("artifacts")} className={`btn ${tab === "artifacts" ? "btn-brass" : ""}`}>
          <Package size={14} /> Artifacts
        </button>
        <button onClick={() => setTab("tickets")} className={`btn ${tab === "tickets" ? "btn-brass" : ""}`}>
          <ListChecks size={14} /> Per-ticket ({run.tickets.length})
        </button>
      </div>
      <div className="mt-4">
        {tab === "timeline" ? (
          <PipelineTimeline run={run} heartbeats={heartbeats} />
        ) : tab === "journal" ? (
          <JournalPanel run={run} />
        ) : tab === "tickets" ? (
          <PerTicketGroups run={run} />
        ) : tab === "artifacts" ? (
          <ArtifactsPanel runId={runId!} tickets={run.tickets} />
        ) : (
          <IterationDetailsPanel run={run} />
        )}
      </div>
    </div>
  );
}

/** Active duration. It freezes while a managed cluster allocation is pending. */
function useDuration(
  startedAt: string, finishedAt: string | null, queueWaitSeconds: number,
  durationSeconds: number | null, queueWaiting: boolean,
): string {
  const ticking = !finishedAt && !queueWaiting;
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!ticking) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [ticking]);
  if ((finishedAt || queueWaiting) && durationSeconds != null) {
    return fmtDuration(durationSeconds);
  }
  const start = startedAt ? new Date(startedAt).getTime() : NaN;
  const end = finishedAt ? new Date(finishedAt).getTime() : now;
  return isFinite(start)
    ? fmtDuration(Math.max(0, (end - start) / 1000 - queueWaitSeconds))
    : "—";
}

/** Compact instrument readout row for the header strip: started, duration,
 *  tokens/heartbeats (from the live heartbeat feed) and the authoritative cost. */
function HeaderTelemetry({ run, runId }: { run: RunDetail; runId: string }) {
  // Ask the server for THIS run's heartbeats. This used to pull the newest 500
  // across every run and filter by ticket id in the browser, which is correct
  // only while that window still reaches this run — once the table grows past
  // it, an older run's tokens quietly drop to zero while its cost (which comes
  // off the run row) stays right, and the two readings disagree.
  const { data: hbs = [] } = useSWR<HeartbeatDTO[]>(
    `/api/heartbeats?run_id=${encodeURIComponent(runId)}&limit=500`,
    { refreshInterval: 5000 }
  );
  const summary = useMemo(() => {
    let cost = 0, inp = 0, out = 0, n = 0;
    for (const h of hbs as any[]) {
      cost += Number(h.estimated_cost_usd || 0);
      inp += Number(h.input_tokens || 0);
      out += Number(h.output_tokens || 0);
      n += 1;
    }
    return { cost, inp, out, n };
  }, [hbs]);
  const duration = useDuration(
    run.started_at, run.finished_at, Number(run.queue_wait_seconds || 0),
    run.duration_s, !!run.queue_waiting,
  );
  const durationLimit = run.max_runtime_hours > 0
    ? fmtDuration(run.max_runtime_hours * 3600)
    : "∞";

  // One neutral tone across the strip, and the items share the row's full width
  // instead of bunching against the left edge with dead space on the right.
  const Item = ({ label, value, title }: { label: string; value: string; title?: string }) => (
    <div className="min-w-0">
      <div className="font-mono text-2xs uppercase tracking-[0.16em] text-slate-500">{label}</div>
      <div className="readout mt-0.5 truncate text-sm text-ink" title={title}>{value}</div>
    </div>
  );

  return (
    // Spread across the strip so the last reading ends flush with the panel's
    // right edge, which is what keeps it aligned with the cards below.
    <div className="flex flex-1 flex-wrap items-center justify-between gap-x-10 gap-y-3">
      <Item label="Task name" value={run.task_name || "—"} title={run.task_name} />
      <Item label="Created time" value={fmtDate(run.started_at)} />
      <Item
        label="Duration"
        value={`${duration} / ${durationLimit}`}
        title="Active Run time; managed-Slurm PENDING time is excluded"
      />
      {/* Tokens and Heartbeats stay behind their own guard — both are read off
          the heartbeat feed and a run with none has nothing to report, which is
          different from reporting zero. Harness sits between them rather than
          after, so it is not stranded past a pair that can vanish. */}
      {summary.n > 0 && <Item label="# Tokens" value={fmtTokens(summary.inp + summary.out)} />}
      {/* The harness, not the trained model. Cost used to sit here and now
          lives in the card below, where it reads against its own budget the
          way Iterations does; repeating it here said the number twice and
          left the strip with nothing about WHAT ran the pipeline. */}
      <Item label="Harness" value={shortModel(run.harness_model) || "—"} title={run.harness_model} />
      {summary.n > 0 && <Item label="Heartbeats" value={String(summary.n)} />}
      {/* The stable Registry tag is both the stored key and displayed name. */}
      {run.registry_version_tag && (
        <Item label="Registry" value={run.registry_version_tag} />
      )}
    </div>
  );
}


// Display names for the roles a run's cost is grouped by. Anything unmapped
// (a future role) falls back to its raw agent_id, capitalised.
const AGENT_LABELS: Record<string, string> = {
  orchestrator: "Orchestrator",
  data: "Data",
  infrastructure: "Infrastructure",
  train: "Train",
  inference: "Inference",
  evaluation: "Evaluation",
  registry: "Registry",
};

function agentLabel(id: string): string {
  return AGENT_LABELS[id] ?? (id ? id.charAt(0).toUpperCase() + id.slice(1) : "—");
}

/** Cost by agent as one stacked bar under the improvement loop: the run's
 *  total split by which role spent it, plus rented GPU, so the agent LLM bill
 *  (which routinely dwarfs GPU) is visible per role, beside the ring that
 *  shows how often each role ran, without its own card.
 *  Sourced from the same per-heartbeat cost the budget meter sums; GPU is $0
 *  on a run that used the operator's own hardware. */
function CostStrip({ run, runId }: { run: RunDetail; runId: string }) {
  const { data } = useSWR<CostBreakdown>(
    runId ? `/api/runs/${encodeURIComponent(runId)}/cost-breakdown` : null,
    { refreshInterval: 5000 },
  );
  const gpu = data?.gpu_cost_usd ?? run.gpu_cost_usd ?? 0;
  // GPU leads: it is the one line that is not an agent, and the only one
  // that can legitimately be $0.
  const segments: Array<{ key: string; label: string; cost: number; className: string }> = [
    ...(gpu > 0 ? [{ key: "__gpu__", label: "GPU", cost: gpu, className: "bg-coral-500/70" }] : []),
    ...(data?.agents ?? [])
      .filter((a) => a.cost_usd > 0)
      .map((a, i) => ({
        key: a.agent_id,
        label: agentLabel(a.agent_id),
        cost: a.cost_usd,
        className: AGENT_SEGMENT_TONES[i % AGENT_SEGMENT_TONES.length],
      })),
  ];
  const total = segments.reduce((sum, seg) => sum + seg.cost, 0);
  if (total <= 0) return null;

  return (
    <div className="mt-4 border-t border-hair pt-4">
      <div className="flex h-1.5 w-full gap-px overflow-hidden rounded-full bg-raised">
        {segments.map((seg) => (
          <div
            key={seg.key}
            className={seg.className}
            style={{ width: `${(seg.cost / total) * 100}%` }}
            title={`${seg.label} ${fmtCost(seg.cost)}`}
          />
        ))}
      </div>
      <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5">
        {segments.map((seg) => (
          <div key={seg.key} className="flex items-center gap-1.5 text-xs">
            <span className={`h-2 w-2 shrink-0 rounded-full ${seg.className}`} />
            <span className="flex-1 whitespace-nowrap text-slate-400">{seg.label}</span>
            <span className="readout whitespace-nowrap text-ink">{fmtCost(seg.cost)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Agents cycle the palette's non-coral tones; GPU alone is coral, the same
// split the cost readout above draws.
const AGENT_SEGMENT_TONES = [
  "bg-brass-500/80", "bg-skyx-500/70", "bg-phosphor-500/70",
  "bg-brass-300/50", "bg-skyx-300/50", "bg-phosphor-300/50",
];

/** What the system tried, round by round, as a timeline. The iteration / best /
 *  target readouts that used to head this are already in the instrument column
 *  above, so the tab is just the journal. */
function JournalPanel({ run }: { run: RunDetail }) {
  // The badge says what the LOOP decided, so it has to read the number the loop
  // decided on: `stop_threshold` is compared against the VALIDATION score,
  // both in the orchestrator's stop table and in `zevo.engine.method.loop_policy`. Read
  // against the held-out score instead — as this did — and the badge and the
  // run disagree in both directions: a run that stopped saying "target hit"
  // shows no badge, and a run still climbing sprouts one.
  //
  // The held-out score remains the final report, but it never determines this
  // optimization stop condition.
  const targetReached = run.stop_threshold != null && run.best_validation_score != null && (
    run.validation_metric_direction === "min"
      ? run.best_validation_score <= run.stop_threshold
      : run.best_validation_score >= run.stop_threshold
  );
  return (
    <Bezel className="p-6">
      <div className="mb-4 flex items-center justify-between gap-2">
        <Kicker strong className="!text-sm">Run journal</Kicker>
        {targetReached && (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-phosphor-500/30 bg-phosphor-500/10 px-2.5 py-0.5 font-mono text-2xs uppercase tracking-wider text-phosphor-300">
            <span className="lamp lamp-live" /> target reached
          </span>
        )}
      </div>
      <RunJournal history={run.history} metricDirection={run.validation_metric_direction} />
    </Bezel>
  );
}
