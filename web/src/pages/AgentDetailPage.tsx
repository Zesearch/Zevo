import { Fragment, useState } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import useSWR from "swr";
import {
  ChevronLeft,
  ChevronRight,
  PlayCircle,
  Search,
  X,
} from "lucide-react";

import { AgentConfigCard } from "../components/AgentConfigCard";
import { Modal } from "../components/Modal";
import { Markdown } from "../components/Markdown";
import { StatusBadge } from "../components/StatusBadge";
import { fireCommand } from "../lib/commands";
import { LiveTranscript } from "../components/LiveTranscript";
import { PageHead, Bezel, Kicker, Readout } from "../components/zevo/primitives";
import { fmtDate, fmtDuration } from "../lib/format";
import { useRowsPerPage } from "../lib/useRowsPerPage";
import { Pager } from "../components/zevo/Pager";
import type { AgentDTO, HeartbeatDTO } from "../lib/api";

/** The subset GET /agents/{id}/heartbeats serves (agents.py HeartbeatDTO). */
type Heartbeat = Pick<
  HeartbeatDTO,
  "id" | "ticket_id" | "driver" | "model" | "started_at" | "finished_at"
  | "exit_code" | "error_message"
>;

type TicketSummary = {
  id: string;
  run_id: string;
  input_format: "typed" | "freeform";
  lane: "optimization" | "held_out_test";
  iteration: number;
  status: string;
  summary: string;
  created_at: string;
};

/** GET /agents/{id}: the shared AgentDTO plus the page's own detail block. */
type AgentDetail = AgentDTO & {
  stats: {
    total_heartbeats: number;
    success_count: number;
    failure_count: number;
    last_finished_at: string;
    open_ticket_count: number;
    ticket_count: number;
  };
  recent_heartbeats: Heartbeat[];
  recent_tickets: TicketSummary[];
};

type Tab = "dashboard" | "instructions" | "configuration" | "tickets";

export function AgentDetailPage() {
  const { agentId = "" } = useParams();
  const isSystemRunner = agentId === "evaluation";
  const [tab, setTab] = useState<Tab>("dashboard");
  // Owned here so a heartbeat row nested inside the tickets table can open it.
  const [transcriptId, setTranscriptId] = useState<string>("");

  const { data: agent, error, mutate } = useSWR<AgentDetail>(
    isSystemRunner ? null : `/api/agents/${encodeURIComponent(agentId)}`,
    { refreshInterval: 3000 }
  );

  if (isSystemRunner) return <Navigate to="/agents" replace />;

  if (error) {
    const status = (error as any).status;
    return (
      <div className="w-full px-[max(1.5rem,1.5vw)] py-8">
        <Link to="/agents" className="inline-flex items-center gap-1.5 text-sm text-slate-400 transition hover:text-brass-300">
          <ChevronLeft size={15} /> Agents
        </Link>
        <Bezel className="mt-6 border-coral-500/30 p-4 text-sm text-coral-300">
          {status === 404 ? `No agent with id ${agentId}.` : String(error.message || error)}
        </Bezel>
      </div>
    );
  }
  if (!agent) {
    return <div className="p-8 text-sm text-slate-500">Acquiring station telemetry…</div>;
  }


  const online = agent.stats.open_ticket_count > 0;
  const configured = !!(agent.default_driver && agent.default_model);

  return (
    <div className="flex w-full flex-1 flex-col px-[max(1.5rem,1.5vw)] py-8">
      <Link to="/agents" className="inline-flex items-center gap-1.5 text-sm text-slate-400 transition hover:text-brass-300">
        <ChevronLeft size={15} /> Agents
      </Link>

      <div className="mt-2">
        <PageHead
          title={agent.name}
          subtitle={
            <span className="flex flex-wrap items-center gap-2.5">
              <StatusBadge status={online ? "running" : "idle"} />
              {/* The agent id used to repeat here in gold, directly under the
                  title it is the slug of. */}
              <span className="text-2xs text-ink">{agent.title}</span>
              {!configured && (
                <span className="font-mono text-2xs uppercase tracking-[0.14em] text-coral-300">unconfigured</span>
              )}
            </span>
          }
          right={
            <button onClick={() => fireCommand("open-new-run-single", agentId)} className="btn btn-brass">
              <PlayCircle size={14} /> Call this agent
            </button>
          }
        />
      </div>


      {/* console tabs */}
      <div className="mb-6 flex flex-wrap gap-1 border-b border-hair">
        {([
          ["dashboard", "Dashboard"],
          ["instructions", "Instructions"],
          ["configuration", "Configuration"],
          ["tickets", "Tickets & heartbeats"],
        ] as [Tab, string][]).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={[
              "-mb-px px-3.5 py-2.5 font-mono text-2xs uppercase tracking-[0.14em] transition",
              tab === key
                ? "border-b-2 border-brass-500 text-brass-200"
                : "border-b-2 border-transparent text-slate-500 hover:text-slate-300",
            ].join(" ")}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="flex flex-1 flex-col">
        {tab === "dashboard" && <DashboardTab agent={agent} onTab={setTab} />}
        {tab === "instructions" && <InstructionsTab agentId={agent.id} />}
        {tab === "configuration" && <ConfigurationTab agent={agent} onUpdated={() => void mutate()} />}
        {tab === "tickets" && <TicketsTab agentId={agent.id} onOpenTranscript={setTranscriptId} />}
      </div>

      <Modal open={!!transcriptId} title="Heartbeat transcript" onClose={() => setTranscriptId("")} width="max-w-4xl">
        <LiveTranscript heartbeatId={transcriptId} height="max-h-[60vh]" />
      </Modal>
    </div>
  );
}

// ---------- shared instruments ----------

/** Duration in seconds between two ISO stamps, or null. */
function durationS(a?: string, b?: string): number | null {
  if (!a || !b) return null;
  const t0 = new Date(a).getTime();
  const t1 = new Date(b).getTime();
  if (!isFinite(t0) || !isFinite(t1) || t1 < t0) return null;
  return (t1 - t0) / 1000;
}

/** A heartbeat sparkline: one bar per recent beat, height ∝ duration, colour by exit. */
function HeartbeatSparkline({ beats }: { beats: Heartbeat[] }) {
  // Oldest → newest, left to right.
  const ordered = [...beats].reverse();
  const durs = ordered.map((h) => durationS(h.started_at, h.finished_at) ?? 0);
  const max = Math.max(1, ...durs);
  if (ordered.length === 0) {
    return <div className="text-2xs text-slate-600">no heartbeats to plot</div>;
  }
  // Bar height was minutes-on-task with nothing saying so. Print the number
  // under each bar and the scale beside the title, so the shape is readable
  // instead of decorative.
  return (
    <div className="flex items-end gap-3">
      {ordered.map((h, i) => {
        const frac = Math.max(0.08, durs[i] / max);
        const tone =
          h.exit_code === 0 ? "bg-phosphor-400"
          : h.exit_code === -1 ? "bg-brass-400 animate-lamp-pulse"
          : "bg-coral-400";
        return (
          // capped so six calls read as bars, not slabs, in a wide panel
          <div key={h.id} className="flex min-w-0 max-w-[4.5rem] flex-1 flex-col items-center gap-1.5">
            <div className="flex h-16 w-full items-end">
              <div
                className={`w-full rounded-sm ${tone}`}
                style={{ height: `${frac * 100}%` }}
                title={`${h.id.slice(0, 8)} · ${h.exit_code === 0 ? "success" : h.exit_code === -1 ? "running" : "failed"}`}
              />
            </div>
            <span className="font-mono text-[0.68rem] text-slate-400">{durs[i] ? fmtDuration(durs[i]) : "—"}</span>
          </div>
        );
      })}
    </div>
  );
}

// ---------- tabs ----------

function DashboardTab({ agent, onTab }: { agent: AgentDetail; onTab: (t: Tab) => void }) {
  const total = agent.stats.total_heartbeats || 0;
  return (
    <div className="space-y-5">
      {/* Three readings, not five: the successes/failures split belongs inside
          the heartbeat count it splits, and the duration chart earns a tile. */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
        <button onClick={() => onTab("tickets")} className="text-left">
          <Bezel className="h-full p-5 transition hover:border-brass-500/40">
            <Readout
              label="Heartbeats" value={agent.stats.total_heartbeats} strongLabel size="lg"
              hint={
                total > 0 ? (
                  <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                    <span className="text-phosphor-300">{agent.stats.success_count} done</span>
                    <span className="text-slate-600">·</span>
                    <span className={agent.stats.failure_count > 0 ? "text-coral-300" : "text-slate-500"}>
                      {agent.stats.failure_count} failed
                    </span>
                  </span>
                ) : "lifetime"
              }
            />
          </Bezel>
        </button>
        <button onClick={() => onTab("tickets")} className="text-left">
          <Bezel className="h-full p-5 transition hover:border-brass-500/40">
            <Readout label="Tickets" value={agent.stats.ticket_count} strongLabel size="lg"
              hint={`${agent.stats.open_ticket_count} in flight`} />
          </Bezel>
        </button>

        {/* how long each of the last few calls took */}
        <Bezel className="p-5 lg:col-span-2">
          <div className="flex items-center justify-between">
            <Kicker strong className="!text-sm">Time per heartbeat</Kicker>
            <span className="text-2xs text-slate-500">last {agent.recent_heartbeats.length}</span>
          </div>
          <div className="mt-4">
            <HeartbeatSparkline beats={agent.recent_heartbeats} />
          </div>
          <div className="mt-3 flex items-center gap-3 text-[0.68rem] text-slate-500">
            <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-sm bg-phosphor-400" /> done</span>
            <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-sm bg-coral-400" /> failed</span>
            <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-sm bg-brass-400" /> running</span>
          </div>
        </Bezel>
      </div>

      {/* recent tickets */}
      <Bezel>
        <div className="border-b border-hair px-5 py-4">
          <Kicker strong>Recent tickets</Kicker>
        </div>
        {agent.recent_tickets.length === 0 ? (
          <div className="p-6 text-2xs text-slate-500">No tickets yet for {agent.id}.</div>
        ) : (
          <ul className="stagger divide-y divide-hair">
            {agent.recent_tickets.slice(0, 5).map((t) => (
              <li key={t.id}>
                <Link to={`/tickets/${t.id}`} className="flex items-center justify-between gap-4 px-5 py-3.5 transition hover:bg-raised">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2.5">
                      <StatusBadge status={t.status} />
                      <span className="font-mono text-sm text-slate-300">{t.id}</span>
                      <span className="text-sm text-slate-500">in run</span>
                      <span className="font-mono text-sm text-brass-300">{t.run_id.slice(0, 8)}</span>
                    </div>
                    <div className="mt-1.5 truncate text-sm text-slate-400">{t.summary || "(no summary)"}</div>
                  </div>
                  <div className="shrink-0 text-sm text-slate-500">{fmtDate(t.created_at)}</div>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </Bezel>
    </div>
  );
}

function InstructionsTab({ agentId }: { agentId: string }) {
  const { data, error } = useSWR<{ content: string; identity_path: string; input_format: string }>(
    `/api/agents/${encodeURIComponent(agentId)}/instructions`
  );
  const [raw, setRaw] = useState(false);
  if (error) return <div className="text-2xs text-coral-300">Failed to load instructions.</div>;
  if (!data) return <div className="text-2xs text-slate-500">loading…</div>;
  return (
    <Bezel className="overflow-auto">
      <div className="flex items-center justify-between gap-3 border-b border-hair px-5 py-3">
        <Kicker strong>Instructions</Kicker>
        <div className="flex items-center gap-1 rounded-md border border-hair bg-canvas p-1">
          {([["preview", false], ["raw", true]] as [string, boolean][]).map(([label, val]) => (
            <button
              key={label}
              onClick={() => setRaw(val)}
              className={`rounded px-2.5 py-1 font-mono text-2xs transition ${
                raw === val ? "bg-brass-500/15 text-brass-300" : "text-slate-500 hover:text-slate-300"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      <div className="p-5">
        {raw ? (
          <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-slate-300">{data.content}</pre>
        ) : (
          <Markdown>{data.content}</Markdown>
        )}
      </div>
    </Bezel>
  );
}

function ConfigurationTab({ agent, onUpdated }: { agent: AgentDetail; onUpdated: () => void }) {
  return (
    <div className="space-y-5">
      <AgentConfigCard
        agentId={agent.id}
        current={{ default_driver: agent.default_driver, default_model: agent.default_model, sandbox: agent.sandbox }}
        onUpdated={onUpdated}
      />
      <Bezel className="p-5">
        <div className="flex items-baseline gap-2">
          <h3 className="section-title">Skills pool</h3>
          <span className="font-mono text-2xs text-slate-500">{agent.skills.length}</span>
        </div>
        {agent.skills.length === 0 ? (
          <div className="mt-3 text-2xs text-slate-500">(none)</div>
        ) : (
          <SkillsPool
            cards={agent.skill_cards.length
              ? agent.skill_cards
              : agent.skills.map((n) => ({ name: n, method: "", description: "" }))}
          />
        )}
      </Bezel>
    </div>
  );
}

/** The method pool: the chips as they were, and ONE description panel under
 *  them for whichever chip is selected.
 *
 *  Giving every skill its own expander turned a compact row into a stack of
 *  accordions, and opening two of them pushed the rest off the card. Sharing a
 *  single panel keeps the pool readable at a glance and still answers "what is
 *  this one for" — which is a paragraph, so it is set at reading size rather
 *  than at the 13px the chips use.
 */
function SkillsPool({ cards }: { cards: { name: string; method: string; description: string }[] }) {
  const [sel, setSel] = useState("");
  const active = cards.find((c) => c.name === sel);
  return (
    <>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {cards.map((c) => (
          <button
            key={c.name}
            onClick={() => setSel(sel === c.name ? "" : c.name)}
            disabled={!c.description}
            className={`bezel-flat px-3 py-1.5 font-mono text-sm transition ${
              sel === c.name
                ? "!border-brass-500/50 bg-brass-500/10 text-brass-200"
                : "text-slate-300 hover:border-brass-500/30 hover:text-ink"
            } disabled:cursor-default disabled:opacity-60`}
          >
            {c.name}
          </button>
        ))}
      </div>
      {active && (
        <div className="mt-3 rounded-bezel border border-hair bg-canvas/60 p-4">
          <span className="font-mono text-sm text-brass-300">{active.name}</span>
          <p className="mt-2 text-sm leading-relaxed text-slate-300">{active.description}</p>
        </div>
      )}
    </>
  );
}

/**
 * Tickets, each expandable to the heartbeats it produced.
 *
 * They were two tabs, but a heartbeat only exists because of a ticket — reading
 * them apart meant holding the ticket id in your head to cross-reference. One
 * table, grouped, with the beat count on the ticket row.
 */
const TICKET_SORTS = [
  // Each carries the direction that is useful for it, so there is no toggle.
  { id: "created", label: "created time", cmp: (a: TicketSummary, b: TicketSummary) => b.created_at.localeCompare(a.created_at) },
  { id: "ticket", label: "ticket id", cmp: (a: TicketSummary, b: TicketSummary) => a.id.localeCompare(b.id) },
] as const;

// One ticket row plus its divider, measured from the rendered table.
const TICKET_ROW_PX = 52;
// "auto" fills the window; the rest are explicit overrides.
const TICKET_PAGE_SIZES = [0, 25, 50, 100];

function TicketsTab({ agentId, onOpenTranscript }: { agentId: string; onOpenTranscript: (id: string) => void }) {
  const { data: rows = [] } = useSWR<TicketSummary[]>(
    `/api/agents/${encodeURIComponent(agentId)}/tickets?limit=100`,
    { refreshInterval: 3000 }
  );
  const { data: beats = [] } = useSWR<Heartbeat[]>(
    `/api/agents/${encodeURIComponent(agentId)}/heartbeats?limit=200`,
    { refreshInterval: 3000 }
  );
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<string>("created");

  const [pageSizePref, setPageSizePref] = useState(TICKET_PAGE_SIZES[0]);
  const [page, setPage] = useState(0);
  const [listEl, setListEl] = useState<HTMLElement | null>(null);
  const fits = useRowsPerPage(listEl, TICKET_ROW_PX, { min: 3, max: 100 });

  const byTicket: Record<string, Heartbeat[]> = {};
  for (const b of beats) (byTicket[b.ticket_id] ??= []).push(b);

  // One agent's tickets are capped at 100, so this filters in the browser —
  // unlike /runs, where the page you hold is a slice of thousands.
  const q = query.trim().toLowerCase();
  const matched = rows
    .filter((r) => !q || r.id.toLowerCase().includes(q) || (r.summary || "").toLowerCase().includes(q)
      || r.run_id.toLowerCase().includes(q))
    .sort(TICKET_SORTS.find((o) => o.id === sort)?.cmp);

  // Same page-fills-the-window rule as Runs and Tasks. A busy agent's 100
  // tickets ran off the bottom in one unbroken scroll, and expanding one to see
  // its heartbeats pushed everything below it further out of reach.
  const total = matched.length;
  const pageSize = pageSizePref || fits;
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const current = Math.min(page, pageCount - 1);
  const shown = matched.slice(current * pageSize, current * pageSize + pageSize);
  const first = total === 0 ? 0 : current * pageSize + 1;
  const last = Math.min(total, (current + 1) * pageSize);

  const beatStatus = (code: number) =>
    code === -1 ? "running" : code === 0 ? "success" : "failed";

  return (
    <div className="flex flex-1 flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-0 flex-1 sm:max-w-md">
          <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="search ticket id, run or summary"
            spellCheck={false}
            className="w-full rounded-md border border-hair bg-canvas py-2 pl-9 pr-8 font-mono text-sm text-slate-200 placeholder:text-slate-600 focus:border-brass-500/50"
          />
          {query && (
            <button
              onClick={() => setQuery("")}
              title="Clear"
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-slate-500 transition hover:text-slate-200"
            >
              <X size={13} />
            </button>
          )}
        </div>

        <span className="ml-auto font-mono text-2xs text-slate-500">sort by</span>
        <div className="flex items-center gap-1 rounded-md border border-hair bg-canvas p-1">
          {TICKET_SORTS.map((o) => (
            <button
              key={o.id}
              onClick={() => setSort(o.id)}
              className={`rounded px-2.5 py-1 font-mono text-2xs transition ${
                sort === o.id ? "bg-brass-500/15 text-brass-300" : "text-slate-500 hover:text-slate-300"
              }`}
            >
              {o.label}
            </button>
          ))}
        </div>

        {query && (
          <span className="font-mono text-2xs text-slate-500">
            {shown.length} match{shown.length === 1 ? "" : "es"}
          </span>
        )}
      </div>

      <Bezel className="overflow-x-auto">
        <table className="w-full min-w-[820px] text-sm">
          <thead>
            <tr className="border-b border-hair text-left">
              {["", "Ticket id", "Run id", "Heartbeats", "Created time", "Summary", "Status"].map((h, i) => (
                <th key={i} className="table-label px-4 py-3">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody ref={setListEl as React.Ref<HTMLTableSectionElement>} className="divide-y divide-hair">
            {shown.map((r) => {
              const hb = byTicket[r.id] ?? [];
              const isOpen = !!open[r.id];
              return (
                <Fragment key={r.id}>
                  <tr className="transition hover:bg-raised">
                    <td className="py-3 pl-4 pr-0">
                      {hb.length > 0 && (
                        <button
                          onClick={() => setOpen((o) => ({ ...o, [r.id]: !o[r.id] }))}
                          title={isOpen ? "Hide heartbeats" : "Show heartbeats"}
                          className="rounded p-0.5 text-slate-500 transition hover:text-slate-200"
                        >
                          <ChevronRight size={14} className={`transition-transform ${isOpen ? "rotate-90" : ""}`} />
                        </button>
                      )}
                    </td>
                    <td className="px-4 py-3 font-mono text-sm">
                      <Link to={`/tickets/${r.id}`} className="text-slate-300 hover:text-brass-300">{r.id}</Link>
                    </td>
                    <td className="px-4 py-3">
                      <Link to={`/runs/${r.run_id}`} className="font-mono text-sm text-slate-400 hover:text-brass-300">
                        {r.run_id.slice(0, 8)}
                      </Link>
                    </td>
                    <td className="px-4 py-3 font-mono text-sm text-slate-300">{hb.length || "—"}</td>
                    <td className="px-4 py-3 text-sm text-slate-300">{fmtDate(r.created_at)}</td>
                    <td className="px-4 py-3 text-sm text-slate-300">
                      <div className="max-w-xl truncate">{r.summary || "—"}</div>
                    </td>
                    <td className="px-4 py-3"><StatusBadge status={r.status} /></td>
                  </tr>

                  {/* The beats get their own table rather than borrowing the
                      ticket's columns — indented, evenly split, own header, so
                      it reads as a nested list instead of a ragged continuation. */}
                  {isOpen && hb.length > 0 && (
                    <tr className="bg-canvas/40">
                      <td />
                      <td colSpan={6} className="py-3 pl-2 pr-4">
                        <div className="border-l-2 border-hair pl-4">
                          <table className="w-full table-fixed text-sm">
                            <thead>
                              <tr className="text-left">
                                {["Heartbeat", "Model", "Started", "Took", "Status"].map((h, i) => (
                                  <th key={i} className="table-label pb-2">{h}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {hb.map((h) => {
                                const d = durationS(h.started_at, h.finished_at);
                                return (
                                  <tr key={h.id}>
                                    <td className="py-1.5 font-mono text-sm">
                                      <button onClick={() => onOpenTranscript(h.id)} className="text-slate-300 hover:text-brass-300">
                                        {h.id.slice(0, 8)}
                                      </button>
                                    </td>
                                    <td className="py-1.5 font-mono text-sm text-slate-400">{h.model}</td>
                                    <td className="py-1.5 text-sm text-slate-400">{fmtDate(h.started_at)}</td>
                                    <td className="py-1.5 font-mono text-sm text-slate-400">
                                      {d ? fmtDuration(d) : "—"}
                                    </td>
                                    <td className="py-1.5">
                                      <span className={[
                                        "inline-flex items-center gap-1.5 font-mono text-sm",
                                        h.exit_code === 0 ? "text-phosphor-300"
                                        : h.exit_code === -1 ? "text-brass-300"
                                        : "text-coral-300",
                                      ].join(" ")}>
                                        {/* Same mapping as the text colors above and the
                                            run-view legend: green = done, amber = running. */}
                                        <span className={[
                                          "lamp",
                                          h.exit_code === 0 ? "lamp-live" : h.exit_code === -1 ? "lamp-brass" : "lamp-coral",
                                        ].join(" ")} />
                                        {beatStatus(h.exit_code)}
                                      </span>
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
            {shown.length === 0 && (
              <tr><td colSpan={7} className="px-4 py-10 text-center text-sm text-slate-500">
                {rows.length === 0 ? `No tickets ever assigned to this agent.` : `No ticket matches "${query}".`}
              </td></tr>
            )}
          </tbody>
        </table>
      </Bezel>

      <div className="mt-auto">
        <Pager
          total={total} first={first} last={last} page={current} pageCount={pageCount}
          pageSizePref={pageSizePref} sizes={TICKET_PAGE_SIZES}
          onSize={(n) => { setPage(Math.floor((current * pageSize) / (n || fits))); setPageSizePref(n); }}
          onPage={setPage}
        />
      </div>
    </div>
  );
}
