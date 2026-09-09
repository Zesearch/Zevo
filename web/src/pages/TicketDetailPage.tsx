import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  ChevronLeft, MessageSquare, PlayCircle, StopCircle,
  RotateCw, GitCommit, AlertTriangle,
} from "lucide-react";

import { TrainingMonitor } from "../components/TrainingMonitor";
import { SettingsPanel } from "../components/SettingsPanel";
import { StepTimeline } from "../components/StepTimeline";
import type { EventEnvelope } from "../components/LiveTranscript";

import { StatusBadge } from "../components/StatusBadge";
import { LiveTranscript } from "../components/LiveTranscript";
import { relPath, scrubPaths } from "../lib/format";
import { assignIterations, iterationOrder } from "../lib/iterations";
import { api } from "../lib/api";
import type { HeartbeatDTO, TicketDetail, RunDetail } from "../lib/api";
import { Bezel, Kicker } from "../components/zevo/primitives";

export function TicketDetailPage() {
  const { ticketId = "" } = useParams();
  const navigate = useNavigate();
  // Go back to wherever we came from (a run's Per-ticket view, an agent page,
  // …). With no history — a direct link — fall back to Runs: a ticket is a step
  // of a run, and the Tickets list it used to fall back to is gone.
  const goBack = () => {
    if (window.history.length > 1) navigate(-1);
    else navigate("/runs");
  };
  const { data: tk, error, mutate } = useSWR<TicketDetail>(
    `/api/tickets/${encodeURIComponent(ticketId)}`,
    { refreshInterval: 3000 }
  );
  const { data: heartbeats = [] } = useSWR<HeartbeatDTO[]>(
    ticketId ? `/api/heartbeats?ticket_id=${encodeURIComponent(ticketId)}&limit=200` : null,
    { refreshInterval: 3000 }
  );
  // For an orchestrate ticket, label each heartbeat tab with the child ticket
  // that wake emitted (so the transcript tabs line up with the "Emitted by
  // orchestrator" panel) instead of an opaque #N. A child is created DURING the
  // wake that emitted it, so we match by created_at falling in the heartbeat's
  // [started, finished] window.
  const isOrchestrate = tk?.agent_id === "orchestrator";
  // Fetched for every ticket, not just orchestrate ones: the emitted-children
  // panel and the heartbeat controls read the run, and the message box needs its
  // status to know whether a message is an instruction or just a note.
  const { data: runForHb } = useSWR<RunDetail>(
    tk?.run_id ? `/api/runs/${encodeURIComponent(tk.run_id)}` : null,
    { refreshInterval: 3000 }
  );
  const runFinished = !!runForHb
    && ["success", "degraded", "failed", "cancelled", "halted"].includes(runForHb.status);
  // How many orchestrate tickets this run has. Newer runs use ONE ticket for the
  // whole run (all wakes reuse it), so its transcript == the whole run; older
  // runs split the orchestrator into one ticket per iteration loop.
  const orchTicketCount = runForHb
    ? runForHb.tickets.filter((t) => t.agent_id === "orchestrator").length
    : 0;
  const emittedByHb = useMemo(() => {
    const map = new Map<string, { id: string; agent_id: string }>();
    if (!isOrchestrate || !runForHb) return map;
    const kids = runForHb.tickets.filter((t) => t.agent_id !== "orchestrator" && t.created_at);
    for (const k of kids) {
      const c = new Date(k.created_at).getTime();
      const hb = heartbeats.find((h) => {
        const s = new Date(h.started_at).getTime();
        const f = h.finished_at ? new Date(h.finished_at).getTime() : Date.now();
        return c >= s && c <= f + 3000;
      });
      if (hb && !map.has(hb.id)) map.set(hb.id, { id: k.id, agent_id: k.agent_id });
    }
    return map;
  }, [isOrchestrate, runForHb, heartbeats]);
  // Invert emittedByHb: child ticket id -> the heartbeat that emitted it (so a
  // click on a payload tab can select the matching heartbeat, and vice-versa).
  const hbByChild = useMemo(() => {
    const m = new Map<string, string>();
    for (const [hbId, child] of emittedByHb) m.set(child.id, hbId);
    return m;
  }, [emittedByHb]);
  // Chronological #index per heartbeat — a stable label for wakes that emitted
  // nothing (look / wait / mark_done).
  const hbSeq = useMemo(() => {
    const m = new Map<string, number>();
    [...heartbeats]
      .sort((a, b) => (a.started_at || "").localeCompare(b.started_at || ""))
      .forEach((h, i) => m.set(h.id, i + 1));
    return m;
  }, [heartbeats]);
  // Heartbeats grouped by iteration — via the child each wake emitted, empty
  // wakes carrying forward the current iteration — so the Transcript tab rows
  // line up with the Emitted-payloads panel above it.
  const hbGroups = useMemo(() => {
    const sorted = [...heartbeats].sort((a, b) => (a.started_at || "").localeCompare(b.started_at || ""));
    const childInfo = assignIterations(runForHb?.tickets || []);
    const order: string[] = [];
    const buckets = new Map<string, { key: string; label: string; kind: string; items: HeartbeatDTO[] }>();
    // Same seed the shared grouper uses, so the wake tabs here and the blocks
    // on the run view agree on what the first round is called.
    let cur = { key: "baseline", label: "Iteration 0", kind: "baseline" };
    for (const h of sorted) {
      const child = emittedByHb.get(h.id);
      if (child) {
        const gi = childInfo.get(child.id);
        if (gi) cur = { key: gi.key, label: gi.label, kind: gi.kind };
      }
      if (!buckets.has(cur.key)) {
        buckets.set(cur.key, { key: cur.key, label: cur.label, kind: cur.kind, items: [] });
        order.push(cur.key);
      }
      buckets.get(cur.key)!.items.push(h);
    }
    return order.map((k) => buckets.get(k)!);
  }, [heartbeats, emittedByHb, runForHb]);
  // Default selection: the first wake that actually emitted a child, so both
  // panels open on something meaningful.
  const firstEmittingHbId = useMemo(() => {
    const sorted = [...heartbeats].sort((a, b) => (a.started_at || "").localeCompare(b.started_at || ""));
    return sorted.find((h) => emittedByHb.has(h.id))?.id;
  }, [heartbeats, emittedByHb]);
  // The feed LiveTranscript is showing, lifted so the Overview can list the same
  // steps without opening a second socket for the same heartbeat.
  const [events, setEvents] = useState<EventEnvelope[]>([]);
  const [seek, setSeek] = useState<{ ts: string; n: number }>({ ts: "", n: 0 });
  const [message, setMessage] = useState("");
  const [messageBusy, setMessageBusy] = useState(false);
  const [hbMsg, setHbMsg] = useState<string | null>(null);
  const [hbBusy, setHbBusy] = useState(false);
  const [selectedHbId, setSelectedHbId] = useState<string>("");
  // The currently-selected heartbeat, shared by the Transcript tabs AND the
  // Emitted-payloads panel so clicking either side syncs the other.
  const curHbId = selectedHbId || firstEmittingHbId || heartbeats[0]?.id || "";
  // Clear on switch, or the previous activation's steps stay on screen until
  // the newly selected one produces its first event.
  useEffect(() => {
    setEvents([]);
    setSeek({ ts: "", n: 0 });
  }, [curHbId]);
  const renderHbTab = (h: HeartbeatDTO) => {
    const active = curHbId === h.id;
    const emitted = emittedByHb.get(h.id);
    const operationLabel: Record<string, string> = {
      provision: "Provision Compute",
      prepare_run_data: "Prepare Run Data And Validation Setup",
      prepare_holdout_data: "Prepare Questions-Only Held-Out Data",
      train: "Select Configuration And Train Candidate",
      run_inference: tk?.lane === "held_out_test"
        ? "Run Held-Out Inference"
        : tk?.payload?.model_source === "base_model"
          ? "Run Baseline Inference"
          : "Run Candidate Inference",
      release: "Release Resources",
    };
    const phase = (h.activation_phase || "").toLowerCase();
    const inferenceName = tk?.lane === "held_out_test"
      ? "Held-Out Inference"
      : tk?.payload?.model_source === "base_model"
        ? "Baseline Inference"
        : "Candidate Inference";
    let activationLabel = operationLabel[h.operation]
      || `Activation ${hbSeq.get(h.id) ?? "?"}`;
    if (h.operation === "run_inference" && phase === "submit") {
      activationLabel = `Submit ${inferenceName}`;
    } else if (h.operation === "run_inference" && phase === "collect") {
      activationLabel = `Collect ${inferenceName} Results`;
    } else if (h.operation === "run_inference" && phase === "repair") {
      activationLabel = `Repair ${inferenceName}`;
    } else if (h.operation === "train" && phase === "submit") {
      activationLabel = "Submit Candidate Training";
    } else if (h.operation === "train" && phase === "collect") {
      activationLabel = "Collect Candidate Training Results";
    } else if (h.operation === "train" && phase === "continue") {
      activationLabel = "Continue Candidate Training";
    } else if (h.operation === "train" && phase === "repair") {
      activationLabel = "Repair Candidate Training";
    } else if (phase === "submit") {
      activationLabel = `Submit ${tk?.agent_id || "Stage"} Job`;
    } else if (phase === "collect") {
      activationLabel = `Collect ${tk?.agent_id || "Stage"} Results`;
    } else if (phase === "repair") {
      activationLabel = `Repair ${tk?.agent_id || "Stage"}`;
    } else if (phase === "continue") {
      activationLabel = `Continue ${tk?.agent_id || "Stage"}`;
    }
    return (
      <button
        key={h.id}
        onClick={() => setSelectedHbId(h.id)}
        className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 font-mono text-2xs transition ${
          active
            ? "border-brass-500/60 bg-brass-500/10 text-brass-300"
            : "border-hair bg-panel text-slate-300 hover:border-brass-500/30"
        }`}
      >
        {emitted
          ? <>→ {emitted.id}</>
          : activationLabel}
        <span className="text-slate-500">{h.driver}/{h.model.slice(0, 12)}</span>
        {h.is_live ? (
          <span className="h-1.5 w-1.5 animate-lamp-pulse rounded-full bg-phosphor-400 shadow-glow-phosphor" />
        ) : (
          <span className={`text-[13px] ${h.exit_code === 0 ? "text-phosphor-300" : "text-coral-300"}`}>
            {h.exit_code === 0 ? "ok" : `e${h.exit_code}`}
          </span>
        )}
      </button>
    );
  };

  if (error) {
    const status = (error as any).status;
    return (
      <div className="w-full px-[max(1.5rem,1.5vw)] py-8">
        <button onClick={goBack} className="text-xs text-slate-500 hover:text-brass-300">← Back</button>
        <div className="mt-6 rounded-lg border border-coral-500/30 bg-coral-500/10 p-4 text-sm text-coral-300">
          {status === 404 ? `No ticket with id ${ticketId}.` : String(error.message || error)}
        </div>
      </div>
    );
  }
  if (!tk) return <div className="p-8 text-sm text-slate-500">Acquiring work-ticket telemetry…</div>;

  const resolvedConfig = tk.resolved_configs[curHbId]
    || Object.values(tk.resolved_configs).at(-1)
    || {};

  // The newest clarification request, with its machine tag stripped. The backend
  // writes `__CLARIFY__` at the head of the body "so it stands out in the
  // timeline" — nothing ever rendered it, so it stood out as literal text and
  // nowhere else. It marks the message; it is not part of the question.
  const clarification = [...(tk.messages || [])]
    .reverse()
    .find((c) => c.body.startsWith("__CLARIFY__"))
    ?.body.replace(/^__CLARIFY__\s*/, "")
    .trim();

  async function postMessage() {
    if (!message.trim()) return;
    setMessageBusy(true);
    try {
      await api(`/tickets/${encodeURIComponent(ticketId)}/messages`, {
        method: "POST",
        body: JSON.stringify({ body: message, author: "user" }),
      });
      setMessage("");
      void mutate();
    } catch (e) {
      setHbMsg(String((e as Error).message || e));
    } finally {
      setMessageBusy(false);
    }
  }

  async function runHeartbeat() {
    setHbBusy(true);
    setHbMsg(null);
    try {
      const body = await api<{ status?: string }>(
        `/tickets/${encodeURIComponent(ticketId)}/heartbeat`,
        { method: "POST", body: JSON.stringify({}) },
      );
      setHbMsg(body.status === "queued" ? "Heartbeat queued." : JSON.stringify(body));
      void mutate();
    } catch (e) {
      setHbMsg(String((e as Error).message || e));
    } finally {
      setHbBusy(false);
    }
  }

  return (
    <div className="w-full px-[max(1.5rem,1.5vw)] py-8">
      <button onClick={goBack} className="inline-flex items-center gap-1 font-mono text-2xs text-slate-500 hover:text-brass-300">
        <ChevronLeft size={15} /> Back
      </button>

      <Bezel className="mt-3 flex flex-wrap items-start justify-between gap-4 p-6 animate-zevo-in">
        <div className="min-w-0">
          <Kicker strong>Work Ticket</Kicker>
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <h1 className="font-mono text-xl text-ink">{tk.id}</h1>
            <StatusBadge status={tk.status} />
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-slate-400">
            <span className="font-mono text-2xs uppercase tracking-wider text-slate-500">
              {tk.agent_id === "evaluation" ? "runner" : "agent"}
            </span>
            {tk.agent_id === "evaluation" ? (
              <span className="font-mono text-slate-300">evaluation</span>
            ) : (
              <Link to={`/agents/${tk.agent_id}`} className="font-mono text-slate-300 hover:text-brass-300">
                {tk.agent_id}
              </Link>
            )}
            {/* The stage chip lived here; the agent already names the stage,
                and abbreviating it a second time in another colour only made
                the line harder to read. */}
            <span className="text-hair">·</span>
            <span className="font-mono text-2xs uppercase tracking-wider text-slate-500">run</span>
            <Link to={`/runs/${tk.run_id}`} className="font-mono text-slate-300 hover:text-brass-300">
              {tk.run_id.slice(0, 8)}
            </Link>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={runHeartbeat} disabled={hbBusy} className="btn btn-brass disabled:opacity-50">
            <PlayCircle size={14} /> {hbBusy ? "running…" : "Run heartbeat"}
          </button>
          {(tk.status === "running" || tk.status === "repairing") && (
            <CancelButton ticketId={tk.id} onCancelled={() => void mutate()} />
          )}
          {(tk.status === "failed" || tk.status === "cancelled") && (
            <RerunButton ticketId={tk.id} onRerun={() => void mutate()} />
          )}
        </div>
      </Bezel>
      {hbMsg && <div className="mt-3 rounded-lg border border-hair bg-panel/60 px-3 py-2 text-xs text-slate-300">{hbMsg}</div>}

      {tk.repair_attempts > 0 && (
        <div className="mt-3 flex items-center gap-2 rounded-lg border border-lilac-500/30 bg-lilac-500/10 px-3 py-2 text-xs text-lilac-200">
          <RotateCw size={13} className={tk.status === "repairing" ? "animate-spin" : ""} />
          <span>
            Automatic repair {tk.repair_attempts}/3
            {tk.repair_route ? ` · ${tk.repair_route}` : ""}
          </span>
        </div>
      )}

      {tk.error_message && (
        <div className="mt-4 space-y-2">
          <div className="rounded-lg border border-coral-500/30 bg-coral-500/10 p-3 text-sm text-coral-300">
            {tk.error_message}
          </div>
          <FailureClassificationPanel errorMessage={tk.error_message} />
        </div>
      )}

      {tk.agent_id === "orchestrator" && (
        <EmittedTicketsPanel
          tickets={runForHb?.tickets || []}
          emittedByHb={emittedByHb}
          hbByChild={hbByChild}
          selectedHbId={curHbId}
          onSelectHb={setSelectedHbId}
          className="mt-6"
        />
      )}

      {/* Overview: what it reached, then what it did to get there — the steps
          are read off the transcript below, so they cannot disagree with it. */}
      <Section title="Overview" className="mt-6">
        <StepTimeline
          events={events}
          executionEvents={tk.execution_events || []}
          empty="No activity recorded yet."
          onSeek={(ts) => setSeek((s) => ({ ts, n: s.n + 1 }))}
        />
      </Section>

      {/* The assignment and its inputs are one thought — what was sent, and
          which of it points at another ticket's output. Side by side, because
          you read the ref list against the payload it came from. */}
      <Section title="Assignment" className="mt-4">
        <div className="grid grid-cols-1 items-stretch gap-4 xl:grid-cols-2">
          <div className="flex min-w-0 flex-col">
            <div className="mb-1.5">
              <Kicker>{tk.agent_id === "orchestrator"
                ? "Payload (orchestrator's own input)"
                : `Payload from orchestrator to ${tk.agent_id}`}</Kicker>
            </div>
            <pre className="bezel-flat max-h-[22rem] flex-1 overflow-auto p-3 font-mono text-xs text-slate-300">
              {scrubPaths(JSON.stringify(tk.payload, null, 2))}
            </pre>
          </div>
          <div className="flex min-w-0 flex-col">
            <div className="mb-1.5"><Kicker>Upstream tickets, resolved to paths</Kicker></div>
            {Object.keys(tk.inputs).length === 0 ? (
              <div className="bezel-flat flex flex-1 items-center justify-center p-3 text-center text-sm text-slate-400">
                This stage reads nothing from an upstream ticket.
              </div>
            ) : (
              <ul className="bezel-flat max-h-[22rem] flex-1 space-y-2 overflow-auto p-3 font-mono text-xs">
                {Object.entries(tk.inputs).map(([k, raw]) => {
                  const binding = raw as Record<string, unknown>;
                  const origin = String(binding.source_ticket_id || "");
                  const path = String(binding.path || "");
                  return (
                    <li key={k}>
                      <div className="text-slate-500">
                        {k}
                        {origin && (
                          <>
                            {" "}
                            <Link to={`/tickets/${origin}`} className="text-brass-400 hover:text-brass-300">
                              {origin}
                            </Link>
                          </>
                        )}
                      </div>
                      {/* An empty ref is a decision, not a failure: the orchestrator
                          leaves it blank when there is no upstream to point at — a
                          baseline inference runs before any training exists. Drawing
                          the arrow anyway pointed it at nothing and read as broken. */}
                      {path ? (
                        <div className="text-slate-300"><span className="text-brass-400">→</span> {relPath(path)}</div>
                      ) : (
                        <div className="text-slate-500">not set: nothing upstream to resolve</div>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>
      </Section>

      {/* What this stage actually ran with, resolved. Its own block, beside the
          training monitor rather than buried on the run view's ticket card —
          the numbers you check are on the page you opened to check them. */}
      {!["train", "inference"].includes(tk.agent_id) &&
        Object.keys(resolvedConfig).length > 0 && (
        <Section title="Resolved settings" className="mt-4">
          <SettingsPanel config={resolvedConfig} />
        </Section>
      )}

      {/* The Training monitor holds ONLY the loss/metric charts. */}
      {tk.agent_id === "train" &&
        tk.execution_events.some(
          (p) => p.event_type === "attempt" || p.loss > 0 || (p.extras as any)?.loss !== undefined,
        ) && (
          <Section title="Training monitor" className="mt-4">
            <TrainingMonitor executionEvents={tk.execution_events} />
          </Section>
        )}

      <Section
        title={`Transcript · ${heartbeats.length} heartbeat(s)${
          tk.agent_id === "orchestrator"
            ? orchTicketCount > 1
              ? " · this supervisor ticket only"
              : " · whole run"
            : ""
        }`}
        className="mt-4"
      >
        {heartbeats.length === 0 ? (
          <WaitingPanel ticketId={tk.id} agentId={tk.agent_id} />
        ) : (
          <>
            {tk.agent_id === "orchestrator" ? (
              // Grouped by iteration, mirroring the Emitted-payloads panel.
              <div className="mb-3 space-y-2">
                {hbGroups.map((g) => (
                  <div key={g.key} className="flex flex-wrap items-center gap-1.5">
                    <span
                      className={`mr-1 shrink-0 font-mono text-2xs font-semibold uppercase tracking-wider ${
                        g.kind === "baseline" ? "text-skyx-300" : "text-brass-300"
                      }`}
                    >
                      {g.label}
                    </span>
                    {g.items.map((h) => renderHbTab(h))}
                  </div>
                ))}
              </div>
            ) : (
              <div className="mb-3 flex flex-wrap gap-1.5">
                {heartbeats.map((h) => renderHbTab(h))}
              </div>
            )}
            <LiveTranscript
              heartbeatId={curHbId}
              height="max-h-[36rem]"
              onEvents={setEvents}
              seekTs={seek.ts}
              seekNonce={seek.n}
            />
          </>
        )}
        {/* The question, when the agent stopped to ask one.
            Comments are not otherwise shown — that was deliberate, the feed
            above says what happened. But a clarification is different in kind:
            the run is HALTED on it and only a reply restarts it, so leaving it
            in a channel nothing renders meant the pipeline stopped dead with the
            reason stored where nobody could read it. Shown only while blocked,
            and only the asking message. */}
        {tk.status === "awaiting_input" && clarification && (
          <div className="mt-3 rounded-bezel border border-skyx-400/50 bg-skyx-500/10 p-3">
            <div className="mb-1 font-mono text-2xs uppercase tracking-wider text-skyx-300">
              waiting on you
            </div>
            <div className="whitespace-pre-wrap text-sm text-ink">{clarification}</div>
            <div className="mt-1 text-xs text-dim">Answer below to resume the run.</div>
          </div>
        )}
        {/* Talking to the agent belongs with its transcript, not in a separate
            section further down: you write here BECAUSE of what you just read. */}
        <div className="mt-3 flex gap-2">
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") postMessage(); }}
            placeholder={runFinished ? "Add a note…" : "Send a message…"}
            className="flex-1 rounded-lg border border-hair bg-canvas px-3 py-2 text-sm text-ink placeholder:text-slate-600 focus:border-brass-500/50"
          />
          <button onClick={postMessage} disabled={messageBusy || !message.trim()} className="btn disabled:opacity-50">
            <MessageSquare size={12} /> post
          </button>
        </div>
      </Section>

    </div>
  );
}

type RunTicketSummary = {
  id: string;
  agent_id: string;
  lane: "optimization" | "held_out_test";
  iteration: number;
  status: string;
  created_at: string;
};

/** On an orchestrate ticket: the child tickets the orchestrator emitted, each
 *  tab showing the payload it SENT. Grouped by iteration (Baseline / Iteration
 *  1, 2, …) to mirror the Transcript below, and CONTROLLED by the same selected
 *  heartbeat — clicking a payload tab selects its emitting heartbeat (and vice
 *  versa), so the two rows stay in lock-step. */
function EmittedTicketsPanel({
  tickets, emittedByHb, hbByChild, selectedHbId, onSelectHb, className = "",
}: {
  tickets: RunTicketSummary[];
  emittedByHb: Map<string, { id: string; agent_id: string }>;
  hbByChild: Map<string, string>;
  selectedHbId: string;
  onSelectHb: (hbId: string) => void;
  className?: string;
}) {
  const children = tickets.filter((t) => t.agent_id !== "orchestrator");
  // The selected child == whatever the shared selected heartbeat emitted. If
  // that wake emitted nothing (look / wait / mark_done), there's no child.
  const selectedChild = emittedByHb.get(selectedHbId);
  const displayChildId = selectedChild?.id || "";
  const { data: child } = useSWR<TicketDetail>(
    displayChildId
      ? `/api/tickets/${encodeURIComponent(displayChildId)}`
      : null,
    { refreshInterval: 3000 },
  );

  // Group the emitted child tickets by iteration, in display order.
  const order = iterationOrder(tickets);
  const byId = new Map(children.map((t) => [t.id, t]));
  const groups = order
    .map((g) => ({
      label: g.label,
      kind: g.kind,
      items: g.ticketIds.map((id) => byId.get(id)).filter((t): t is RunTicketSummary => !!t),
    }))
    .filter((g) => g.items.length > 0);

  return (
    <Section title={`Emitted by orchestrator, payloads sent to agents · whole run (${children.length})`} className={className}>
      {children.length === 0 ? (
        <div className="text-xs text-slate-500">No child tickets emitted yet.</div>
      ) : (
        <>
          <div className="mb-3 space-y-2">
            {groups.map((g) => (
              <div key={g.label} className="flex flex-wrap items-center gap-1.5">
                <span
                  className={`mr-1 shrink-0 font-mono text-2xs font-semibold uppercase tracking-wider ${
                    g.kind === "baseline" ? "text-skyx-300" : "text-brass-300"
                  }`}
                >
                  {g.label}
                </span>
                {g.items.map((t) => {
                  const active = selectedChild?.id === t.id;
                  const hb = hbByChild.get(t.id);
                  return (
                    <button
                      key={t.id}
                      onClick={() => hb && onSelectHb(hb)}
                      disabled={!hb}
                      title={hb ? "" : "no matching heartbeat found for this ticket"}
                      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 font-mono text-2xs transition ${
                        active
                          ? "border-brass-500/60 bg-brass-500/10 text-brass-300"
                          : hb
                            ? "border-hair bg-panel text-slate-300 hover:border-brass-500/30"
                            : "border-hair bg-panel/50 text-slate-500 cursor-default"
                      }`}
                    >
                      {t.id}
                      <span className="text-slate-500">→ {t.agent_id}</span>
                      <StatusBadge status={t.status} />
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
          {displayChildId ? (
            <>
              <div className="mb-2 text-2xs text-slate-400">
                orchestrator sent this payload to{" "}
                {child?.agent_id === "evaluation" ? (
                  <span className="font-mono text-brass-300">evaluation runner</span>
                ) : (
                  <Link to={`/agents/${child?.agent_id || ""}`} className="font-mono text-brass-300 hover:underline">
                    {child?.agent_id || "…"}
                  </Link>
                )}
                {" "}(ticket{" "}
                <Link to={`/tickets/${displayChildId}`} className="font-mono text-slate-300 hover:underline">{displayChildId}</Link>
                , agent {child?.agent_id || "…"})
              </div>
              <pre className="bezel-flat overflow-auto p-3 font-mono text-xs text-slate-300">
                {child ? scrubPaths(JSON.stringify(child.payload, null, 2)) : "loading…"}
              </pre>
            </>
          ) : (
            <div className="bezel-flat p-3 text-2xs text-slate-500">
              The selected heartbeat emitted no child ticket (look / wait / mark_done). Pick a
              highlighted tab to see the payload it sent.
            </div>
          )}
        </>
      )}
    </Section>
  );
}

// Fixed colors for the well-known metrics; anything else cycles the palette.

function Section({ title, children, className = "" }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <Bezel className={className}>
      <div className="px-4 pb-1 pt-3">
        <Kicker strong>{title}</Kicker>
      </div>
      <div className="px-4 pb-4 pt-2">{children}</div>
    </Bezel>
  );
}

type Wakeup = {
  id: string;
  status: string;        // queued | running | completed | failed | coalesced
  source: string;
  reason: string;
  created_at: string;
  heartbeat_run_id: string | null;
};

function WaitingPanel({ ticketId, agentId }: { ticketId: string; agentId: string }) {
  const { data: wakeups = [] } = useSWR<Wakeup[]>(
    `/api/wakeups?ticket_id=${encodeURIComponent(ticketId)}&limit=10`,
    { refreshInterval: 1000 }
  );
  const latest = wakeups[0]; // newest first per backend ordering
  const queued = wakeups.filter((w) => w.status === "queued").length;
  const running = wakeups.filter((w) => w.status === "running").length;
  const done = wakeups.filter((w) => w.status === "completed").length;
  const failed = wakeups.filter((w) => w.status === "failed").length;

  if (wakeups.length === 0) {
    return (
      <div className="text-xs text-slate-500">
        No heartbeats yet, and no wakeup queued. Click <em>Run heartbeat</em>{" "}
        above, or post a message to wake the agent.
      </div>
    );
  }

  const phase =
    latest?.status === "queued"
      ? "queued, waiting for daemon to pick up"
      : latest?.status === "running"
      ? "running, daemon is launching the heartbeat"
      : latest?.status === "completed"
      ? "wakeup completed (heartbeat should appear momentarily)"
      : latest?.status === "failed"
      ? "wakeup failed, see queue"
      : `status: ${latest?.status || "?"}`;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3 rounded-md border border-amber-500/30 bg-amber-500/5 p-3">
        <span className="h-2 w-2 animate-pulse rounded-full bg-amber-300" />
        <div className="flex-1 text-sm text-amber-200">
          {phase}
        </div>
        <div className="font-mono text-[14px] text-slate-400">
          agent <span className="text-slate-200">{agentId}</span>
        </div>
      </div>
      <div className="flex gap-2 font-mono text-[14px] text-slate-500">
        <span className="rounded bg-slate-800 px-1.5 py-0.5">queued: {queued}</span>
        <span className="rounded bg-slate-800 px-1.5 py-0.5">running: {running}</span>
        <span className="rounded bg-slate-800 px-1.5 py-0.5">done: {done}</span>
        {failed > 0 && (
          <span className="rounded bg-rose-500/20 px-1.5 py-0.5 text-rose-300">
            failed: {failed}
          </span>
        )}
      </div>
      <div className="text-[15px] text-slate-500">
        Queued work is picked up within a second or two, one heartbeat per agent
        at a time. If this sits at <em>queued</em> for more than a few seconds,{" "}
        <code>{agentId}</code> is probably still busy with earlier work.
      </div>
    </div>
  );
}

function CancelButton({
  ticketId,
  onCancelled,
}: {
  ticketId: string;
  onCancelled: () => void;
}) {
  const [busy, setBusy] = useState(false);
  async function cancel() {
    if (!confirm("Cancel this ticket? Its local Agent process and exact remote task will be stopped.")) return;
    setBusy(true);
    try {
      await api(`/tickets/${encodeURIComponent(ticketId)}/cancel`, { method: "POST" });
      onCancelled();
    } catch {
      /* the ticket keeps polling; a failed cancel shows as still running */
    } finally {
      setBusy(false);
    }
  }
  return (
    <button
      onClick={cancel}
      disabled={busy}
      className="btn border-coral-500/40 bg-coral-500/10 text-coral-300 hover:bg-coral-500/20 hover:text-coral-200 disabled:opacity-50"
    >
      <StopCircle size={14} /> {busy ? "cancelling…" : "cancel"}
    </button>
  );
}


type RetryStatus = {
  ticket_id: string;
  ticket_status: string;
  verdict: "transient" | "structural" | "cancelled" | "unknown";
  retryable: boolean;
  reason: string;
  code: string;
  description: string;
  recovery: string;
  last_error: string;
  last_exit_code: number;
};

function RerunButton({
  ticketId,
  onRerun,
}: {
  ticketId: string;
  onRerun: () => void;
}) {
  const { data: cls } = useSWR<RetryStatus>(
    `/api/tickets/${encodeURIComponent(ticketId)}/retry-status`,
    { refreshInterval: 0 },
  );
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  async function rerun(strategy: "fresh" | "from_checkpoint", force = false) {
    if (!confirm(`Rerun this ticket with strategy=${strategy}${force ? " (FORCE)" : ""}?`)) return;
    setBusy(true); setErr(null);
    try {
      await api(`/tickets/${encodeURIComponent(ticketId)}/rerun`, {
        method: "POST",
        body: JSON.stringify({ strategy, actor: "ui", force }),
      });
      setMenuOpen(false);
      onRerun();
    } catch (e) {
      setErr(String((e as Error).message || e));
    } finally {
      setBusy(false);
    }
  }

  // Verdict drives the visible affordance + tooltip.
  const isStructural = cls?.verdict === "structural";
  const verdictColor =
    cls?.verdict === "transient" ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300" :
    cls?.verdict === "cancelled" ? "border-sky-500/40 bg-sky-500/10 text-sky-300" :
    isStructural ? "border-amber-500/40 bg-amber-500/10 text-amber-300" :
    "border-slate-700 bg-slate-900 text-slate-300";

  return (
    <div className="relative">
      <button
        onClick={() => setMenuOpen((v) => !v)}
        disabled={busy}
        title={cls ? `verdict: ${cls.verdict}, because ${cls.reason}` : "loading classification…"}
        className={`inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs uppercase tracking-wider transition hover:opacity-80 disabled:opacity-50 ${verdictColor}`}
      >
        <RotateCw size={14} className={busy ? "animate-spin" : ""} />
        {busy ? "rerunning…" : "rerun"}
      </button>
      {menuOpen && !busy && (
        <div
          onMouseLeave={() => setMenuOpen(false)}
          className="absolute right-0 top-full z-10 mt-1 w-80 rounded-md border border-slate-700 bg-slate-950 p-2 shadow-xl"
        >
          {cls && (
            <div className="mb-2 rounded border border-slate-800 bg-slate-900/40 p-2 text-[15px]">
              <span className="font-mono">{cls.verdict}</span>
              <span className="ml-1 font-mono text-slate-500">{cls.code}</span>, because <span className="text-slate-400">{cls.reason}</span>
            </div>
          )}
          {isStructural && (
            <div className="mb-2 flex items-start gap-1.5 rounded border border-amber-500/30 bg-amber-500/5 p-2 text-[15px] text-amber-200">
              <AlertTriangle size={11} className="mt-0.5 flex-shrink-0" />
              <span>
                Structural failure, so rerunning the same inputs will reproduce
                the failure. Edit the upstream / payload first, OR use force.
              </span>
            </div>
          )}
          <button
            onClick={() => rerun("fresh", isStructural)}
            className="block w-full rounded px-2 py-1.5 text-left text-xs text-slate-200 hover:bg-slate-800"
          >
            <RotateCw size={11} className="inline mr-1.5" />
            Fresh rerun
            {isStructural && <span className="ml-1 text-amber-300">(force)</span>}
            <div className="text-[14px] text-slate-500">drops any WorkProduct, starts clean</div>
          </button>
          <button
            onClick={() => rerun("from_checkpoint", isStructural)}
            className="mt-1 block w-full rounded px-2 py-1.5 text-left text-xs text-slate-200 hover:bg-slate-800"
          >
            <GitCommit size={11} className="inline mr-1.5" />
            Rerun from checkpoint
            {isStructural && <span className="ml-1 text-amber-300">(force)</span>}
            <div className="text-[14px] text-slate-500">keeps prior WorkProduct; Agent re-emits result on top</div>
          </button>
          {err && (
            <div className="mt-2 rounded border border-rose-500/30 bg-rose-500/10 p-1.5 text-[14px] text-rose-300">
              {err}
            </div>
          )}
        </div>
      )}
    </div>
  );
}


// ─────────────────── L.2 — failure-mode classification panel ────────────────

type ClassifyResponse = {
  verdict: "transient" | "structural" | "cancelled" | "unknown";
  retryable: boolean;
  reason: string;
  code: string;
  matched_pattern: string;
  catalog_entry: null | {
    code: string;
    title: string;
    verdict: string;
    pattern: string;
    description: string;
    recovery: string;
  };
};

function FailureClassificationPanel({ errorMessage }: { errorMessage: string }) {
  const { data, error } = useSWR<ClassifyResponse>(
    errorMessage ? `failure-classify::${errorMessage.slice(0, 200)}` : null,
    () => api<ClassifyResponse>("/failure-modes/classify", {
      method: "POST",
      body: JSON.stringify({ error_message: errorMessage }),
    }),
    { revalidateOnFocus: false },
  );

  const [expanded, setExpanded] = useState(false);

  if (error || !data) return null;

  const pillColor =
    data.verdict === "transient"  ? "border-amber-500/40 bg-amber-500/10 text-amber-300" :
    data.verdict === "cancelled"  ? "border-sky-500/40 bg-sky-500/10 text-sky-300" :
    data.verdict === "structural" ? "border-rose-500/40 bg-rose-500/10 text-rose-300" :
                                    "border-slate-700 bg-slate-900/60 text-slate-300";

  const hasCatalog = !!data.catalog_entry;

  return (
    <div className="rounded-md border border-slate-800 bg-slate-900/40 p-3">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center justify-between gap-2 text-left"
      >
        <div className="flex items-center gap-2 text-xs">
          <span className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[14px] uppercase tracking-wider ${pillColor}`}>
            {data.verdict}
          </span>
          <span className="text-slate-300">{data.reason}</span>
          {hasCatalog && (
            <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[14px] text-slate-400">
              {data.catalog_entry!.code}
            </span>
          )}
        </div>
        {hasCatalog && (
          <span className="flex-shrink-0 text-[15px] text-slate-500">
            {expanded ? "▾ hide playbook" : "▸ recovery playbook"}
          </span>
        )}
      </button>

      {expanded && hasCatalog && (
        <div className="mt-3 border-t border-slate-800 pt-3 text-xs text-slate-300">
          {data.catalog_entry!.description && (
            <div className="mb-2 whitespace-pre-wrap leading-relaxed">
              {data.catalog_entry!.description}
            </div>
          )}
          <div className="rounded border border-slate-800 bg-slate-950/40 p-3">
            <div className="mb-1 text-[14px] uppercase tracking-wider text-slate-500">
              Recovery
            </div>
            <pre className="whitespace-pre-wrap break-words font-mono text-[15px] text-slate-300">
{data.catalog_entry!.recovery}
            </pre>
          </div>
          <div className="mt-2 text-[14px] text-slate-500">
            Classifier rule: <code className="font-mono">{data.code}</code>
          </div>
        </div>
      )}
    </div>
  );
}
