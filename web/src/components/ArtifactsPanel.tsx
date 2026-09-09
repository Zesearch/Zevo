import { useEffect, useState } from "react";
import useSWR from "swr";
import { Link } from "react-router-dom";
import {
  Database, ScrollText, Code2, Info, AlertTriangle, ChevronRight, ChevronDown, X,
  type LucideIcon,
} from "lucide-react";
import { assignIterations, iterationOrder, liveIterationKey, type MinimalTicket } from "../lib/iterations";
import { bytes, relPath } from "../lib/format";
import { Kicker } from "./zevo/primitives";

type Artifact = {
  id: string;
  ticket_id: string;
  role: string;
  path: string;
  local_path: string;     // where it lives on the host: runs/<run-id>/<ticket>/…
  exists: boolean;
  availability: "local" | "remote" | "saved_model" | "missing";
  size_bytes: number;
  meta: Record<string, unknown>;
  created_at: string;
};

type ArtifactDetail = Artifact & {
  preview_kind: "text" | "jsonl" | "binary" | "missing";
  preview: string;
};

/** Eight kinds is more distinctions than the column can carry — half of them
 *  differ only in which stage wrote them. Four is what a reader actually sorts
 *  by: the thing that was produced, the record of producing it, the code that
 *  did it, and the small facts about it. The exact kind stays on the row's
 *  tooltip and in the drawer, and the file name says the rest. */
type KindGroup = "data" | "log" | "script" | "info";

function roleGroup(role: string): KindGroup {
  switch (role) {
    case "model":
    case "dataset":
    case "predictions":  return "data";
    case "log":          return "log";
    case "script":       return "script";
    default:             return "info";   // metrics, registry, device_info, …
  }
}

const KIND_STYLE: Record<KindGroup, { icon: LucideIcon; color: string }> = {
  data:   { icon: Database,   color: "text-skyx-300" },
  log:    { icon: ScrollText, color: "text-slate-400" },
  script: { icon: Code2,      color: "text-lilac-300" },
  info:   { icon: Info,       color: "text-phosphor-300" },
};

/**
 * Tab content: list every WorkProduct on the run with size + preview.
 * Click → side drawer with first ~8KB of text artifacts.
 */
export function ArtifactsPanel({ runId, tickets = [] }: { runId: string; tickets?: MinimalTicket[] }) {
  const { data: artifacts = [], error } = useSWR<Artifact[]>(
    runId ? `/api/runs/${encodeURIComponent(runId)}/artifacts` : null,
    { refreshInterval: 5000 },
  );
  const [openId, setOpenId] = useState<string | null>(null);
  // Track which groups are EXPANDED; empty by default → everything collapsed.
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  // Follow the run: the block being worked on opens, and the one before it
  // closes as soon as work moves on. A manual toggle still wins until the run
  // moves to the next block.
  const liveKey = liveIterationKey(tickets);
  useEffect(() => {
    if (liveKey) setExpanded(new Set([liveKey]));
  }, [liveKey]);

  // Tickets that the reader has explicitly closed. Absent = open.
  const [closedTickets, setClosedTickets] = useState<Set<string>>(new Set());
  const toggleTicket = (k: string) =>
    setClosedTickets((prev) => {
      const next = new Set(prev);
      next.has(k) ? next.delete(k) : next.add(k);
      return next;
    });

  const toggle = (k: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(k) ? next.delete(k) : next.add(k);
      return next;
    });
  const { data: detail } = useSWR<ArtifactDetail>(
    openId ? `/api/runs/${encodeURIComponent(runId)}/artifacts/${openId}` : null,
  );

  if (error) {
    return (
      <div className="rounded-bezel border border-coral-500/30 bg-coral-500/10 p-3 text-xs text-coral-300">
        failed to load artifacts: {String((error as Error).message || error)}
      </div>
    );
  }
  if (!artifacts.length) {
    return (
      <div className="rounded-bezel border border-dashed border-hair bg-panel/40 p-8 text-center text-sm text-dim">
        No artifacts yet. Agents emit them onto the manifest as they complete tickets.
      </div>
    );
  }

  // Group artifacts under Baseline / Iteration 1, 2, … via their ticket, so the
  // Artifacts tab lines up with the timeline + Per-ticket tabs. Within a group,
  // order by creation (dataset → model → predictions → metrics → registry).
  const info = assignIterations(tickets);
  const order = iterationOrder(tickets);
  const OTHER = "__other__";
  const buckets = new Map<string, Artifact[]>();
  for (const a of artifacts) {
    const key = info.get(a.ticket_id)?.key ?? OTHER;
    (buckets.get(key) ?? buckets.set(key, []).get(key)!).push(a);
  }
  const sortAsc = (xs: Artifact[]) =>
    [...xs].sort((a, b) => (a.created_at || "").localeCompare(b.created_at || ""));
  // A round is several stages' worth of output, and one flat table mixed them
  // together — you could not see at a glance what train produced versus what
  // eval did. Split each round by the ticket that emitted the artifact, in the
  // order those tickets first produced anything.
  const byTicket = (items: Artifact[]) => {
    const m = new Map<string, Artifact[]>();
    for (const a of sortAsc(items)) {
      const list = m.get(a.ticket_id);
      if (list) list.push(a);
      else m.set(a.ticket_id, [a]);
    }
    return [...m.entries()].map(([ticketId, xs]) => ({ ticketId, items: xs }));
  };
  type Group = {
    key: string; label: string; kind: string;
    count: number; tickets: { ticketId: string; items: Artifact[] }[];
  };
  const grouped: Group[] = [];
  for (const g of order) {
    const items = buckets.get(g.key);
    if (items?.length) {
      grouped.push({ key: g.key, label: g.label, kind: g.kind, count: items.length, tickets: byTicket(items) });
    }
  }
  if (buckets.get(OTHER)?.length) {
    const items = buckets.get(OTHER)!;
    grouped.push({ key: OTHER, label: "Other", kind: "iteration", count: items.length, tickets: byTicket(items) });
  }

  const renderRow = (a: Artifact) => {
    const group = roleGroup(a.role);
    const { icon: Icon, color } = KIND_STYLE[group];
    // Every artifact is ONE line, whatever its kind. A remote model used to get
    // a three-line explanation of where it did and didn't live, which broke the
    // row rhythm of the whole table for the one kind you most want to scan. The
    // path it reports IS its path; the drawer explains the rest.
    const path = relPath(a.local_path || a.path);
    const cut = path.lastIndexOf("/");
    const dir = cut === -1 ? "" : path.slice(0, cut + 1);
    const file = cut === -1 ? path : path.slice(cut + 1);
    const available = a.availability === "local" || a.availability === "saved_model";
    const remote = a.availability === "remote";
    const saved = a.availability === "saved_model";
    return (
      <tr
        key={a.id}
        onClick={() => setOpenId(a.id)}
        className="group/row cursor-pointer transition hover:bg-raised/60"
      >
        <td className="w-[18%] py-2 pl-8 pr-6 align-middle">
          <span
            title={a.role}
            className="inline-flex items-center gap-1.5 rounded-full border border-hair bg-canvas/60 py-0.5 pl-1.5 pr-2.5"
          >
            <Icon
              size={11}
              className={a.availability === "missing" ? "text-coral-400" : color}
            />
            <span className="text-2xs uppercase tracking-[0.1em] text-slate-400">{group}</span>
          </span>
        </td>
        <td className="w-[40.8%] py-2 pr-8 align-middle text-slate-300">
          <div className="truncate">
            {available ? (
              <span className="transition group-hover/row:text-brass-200">
                <span className="text-slate-600">{dir}</span>
                <span className="text-slate-200">{file}</span>
                {saved && (
                  <span className="ml-2 text-phosphor-300">remote · saved model</span>
                )}
              </span>
            ) : remote ? (
              <span className="text-brass-300">
                {path} <span className="text-slate-500">(remote checkpoint)</span>
              </span>
            ) : (
              <span className="text-coral-300">
                <AlertTriangle size={10} className="mr-1 inline" />
                {path} (missing)
              </span>
            )}
          </div>
        </td>
        <td className="w-[19.2%] whitespace-nowrap py-2 pr-8 align-middle text-dim tabular-nums">
          {available ? bytes(a.size_bytes) : "—"}
        </td>
        <td className="w-[22%] whitespace-nowrap py-2 pr-3 align-middle text-slate-600 tabular-nums">
          {a.created_at ? a.created_at.replace("T", " ").slice(0, 16) : ""}
        </td>
      </tr>
    );
  };

  return (
    <>
      {/* Same shape as the Per-ticket tab: a chevron + label + count + rule per
          group, and the column headers live INSIDE each opened group rather
          than as one fixed header above everything. */}
      <div className="space-y-4">
        {grouped.map((g) => {
          const isCollapsed = !expanded.has(g.key);
          return (
            <div key={g.key}>
              <button
                onClick={() => toggle(g.key)}
                className="mb-2 flex w-full items-center gap-2 text-left"
              >
                {isCollapsed ? (
                  <ChevronRight size={14} className="text-dim" />
                ) : (
                  <ChevronDown size={14} className="text-dim" />
                )}
                <span className="font-mono text-xs font-semibold uppercase tracking-[0.14em] text-brass-300">
                  {g.label}
                </span>
                <span className="font-mono text-[13px] text-dim">
                  {g.count} artifact{g.count === 1 ? "" : "s"}
                </span>
                      </button>

              {!isCollapsed && (
                // One frame for the round, with the ticket as a heading row
                // inside it. A frame per ticket boxed every group separately and
                // the round stopped reading as one thing.
                <div className="overflow-x-auto rounded-bezel border border-hair bg-panel/40">
                  <table className="w-full table-fixed text-xs">
                    <thead className="table-label border-b border-hair text-left">
                      <tr>
                        <th className="w-[18%] py-2 pl-8 pr-6 font-normal">role</th>
                        <th className="w-[40.8%] py-2 pr-8 font-normal">path</th>
                        <th className="w-[19.2%] whitespace-nowrap py-2 pr-8 font-normal">size</th>
                        <th className="w-[22%] whitespace-nowrap py-2 pr-3 font-normal">created time</th>
                      </tr>
                    </thead>
                    {g.tickets.map((tg) => (
                      <tbody key={tg.ticketId} className="font-mono">
                        <tr>
                          <td colSpan={4} className="px-0 pt-3">
                            <div className="flex items-center gap-2 border-l-2 border-brass-500/50 bg-brass-500/[0.06] py-1.5 pl-3 pr-3">
                              <button
                                onClick={() => toggleTicket(`${g.key}/${tg.ticketId}`)}
                                aria-label={closedTickets.has(`${g.key}/${tg.ticketId}`) ? "expand" : "collapse"}
                                className="text-dim transition hover:text-brass-300"
                              >
                                {closedTickets.has(`${g.key}/${tg.ticketId}`) ? (
                                  <ChevronRight size={13} />
                                ) : (
                                  <ChevronDown size={13} />
                                )}
                              </button>
                              <Link
                                to={`/tickets/${tg.ticketId}`}
                                className="font-mono text-sm text-brass-300 hover:text-brass-200"
                              >
                                {tg.ticketId}
                              </Link>
                              <span className="font-mono text-2xs text-slate-600">
                                {tg.items.length} artifact{tg.items.length === 1 ? "" : "s"}
                              </span>
                            </div>
                          </td>
                        </tr>
                        {!closedTickets.has(`${g.key}/${tg.ticketId}`) &&
                          tg.items.map((a) => renderRow(a))}
                      </tbody>
                    ))}
                  </table>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Preview drawer */}
      {openId && (
        <div
          className="fixed inset-0 z-50 bg-canvas/70 backdrop-blur-sm"
          onClick={() => setOpenId(null)}
        >
          <div
            className="absolute right-0 top-0 h-full w-[48rem] max-w-full overflow-auto border-l border-hair bg-panel p-6 shadow-bezel"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <div className="min-w-0">
                <Kicker>{detail?.role || "loading…"}</Kicker>
                <div className="mt-1 truncate font-mono text-xs text-dim">
                  {relPath(detail?.local_path || detail?.path || "")}
                </div>
              </div>
              <button
                className="rounded p-1 text-dim hover:text-slate-200"
                onClick={() => setOpenId(null)}
              >
                <X size={16} />
              </button>
            </div>

            {detail && (
              <>
                <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Stat label="ticket" value={detail.ticket_id} />
                  <Stat label="role" value={detail.role} />
                  <Stat
                    label="size"
                    value={detail.availability === "local" || detail.availability === "saved_model"
                      ? bytes(detail.size_bytes) : "—"}
                  />
                  <Stat
                    label="availability"
                    value={detail.availability.replace("_", " ")}
                    tone={detail.availability === "missing" ? "coral" : undefined}
                  />
                </div>

                {Object.keys(detail.meta || {}).length > 0 && (
                  <div className="mt-4">
                    <Kicker>meta</Kicker>
                    <pre className="mt-1.5 overflow-x-auto rounded-bezel border border-hair bg-canvas p-3 font-mono text-[15px] text-slate-300">
{JSON.stringify(detail.meta, null, 2)}
                    </pre>
                  </div>
                )}

                <div className="mt-4">
                  <Kicker>
                    preview · {previewFormat(detail.local_path || detail.path, detail.preview_kind)}
                  </Kicker>
                  <div className="mt-1.5 max-h-[60vh] overflow-auto rounded-bezel border border-hair bg-canvas">
                    <Preview
                      text={detail.preview || ""}
                      path={detail.local_path || detail.path}
                      kind={detail.preview_kind}
                    />
                  </div>
                </div>
              </>
            )}
            {!detail && (
              <div className="mt-6 text-xs text-dim">loading…</div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

/** What to READ the preview as. The backend only distinguishes text from
 *  binary; the extension is what says whether those bytes are a table, a
 *  record, or code, and reading a CSV as one long line is the difference
 *  between seeing your data and seeing a wall. */
function previewFormat(path: string, kind: string): string {
  if (kind === "binary" || kind === "missing") return kind;
  const ext = (path.split(".").pop() || "").toLowerCase();
  if (ext === "csv" || ext === "tsv") return ext;
  if (ext === "json") return "json";
  if (ext === "jsonl") return "jsonl";
  if (["py", "sh", "yaml", "yml", "toml", "js", "ts"].includes(ext)) return ext;
  return "text";
}

/** Parse CSV over the WHOLE text, not line by line. A model's response field
 *  routinely contains both commas and newlines, and splitting on newlines first
 *  tore one record into several — a numbered list inside an answer showed up as
 *  a column of fake keys. Only a newline OUTSIDE quotes ends a record. */
function parseCsv(text: string, delim: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { cur += '"'; i++; } else quoted = false;
      } else cur += c;
      continue;
    }
    if (c === '"') { quoted = true; continue; }
    if (c === delim) { row.push(cur); cur = ""; continue; }
    if (c === "\n") { row.push(cur); rows.push(row); row = []; cur = ""; continue; }
    if (c === "\r") continue;
    cur += c;
  }
  if (cur !== "" || row.length) { row.push(cur); rows.push(row); }
  return rows;
}

function Preview({ text, path, kind }: { text: string; path: string; kind: string }) {
  const fmt = previewFormat(path, kind);
  if (!text) return <div className="p-3 font-mono text-[15px] text-dim">(empty)</div>;

  // ── Table ──
  if (fmt === "csv" || fmt === "tsv") {
    const delim = fmt === "tsv" ? "\t" : ",";
    const rows = parseCsv(text, delim);
    // The preview is a truncated head of the file, so the final record is very
    // likely cut mid-field. Drop it rather than render a broken row.
    if (rows.length > 1) rows.pop();
    const head = rows[0] || [];
    const body = rows.slice(1);
    if (head.length > 1) {
      return (
        <table className="w-full border-collapse font-mono text-[15px]">
          <thead className="table-label sticky top-0 bg-panel">
            <tr>
              {head.map((h, i) => (
                <th key={i} className="whitespace-nowrap border-b border-hair px-3 py-2 text-left font-normal">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {body.map((r, i) => (
              <tr key={i} className="align-top">
                {head.map((_, c) => (
                  <td key={c} className="max-w-[22rem] border-b border-hair/40 px-3 py-1.5 text-slate-300">
                    <div className="max-h-[4.5rem] overflow-hidden whitespace-pre-wrap break-words">
                      {r[c] ?? ""}
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      );
    }
  }

  // ── One record per line, each laid out ──
  if (fmt === "jsonl") {
    const lines = text.split(/\r?\n/).filter((l) => l.trim());
    if (lines.length > 1) lines.pop();   // truncated tail
    return (
      <div className="divide-y divide-hair/40">
        {lines.map((l, i) => {
          let body = l;
          try { body = JSON.stringify(JSON.parse(l), null, 2); } catch { /* leave raw */ }
          return (
            <div key={i} className="flex gap-3 px-3 py-2">
              <span className="shrink-0 select-none font-mono text-[15px] text-slate-600 tabular-nums">{i + 1}</span>
              <pre className="min-w-0 flex-1 whitespace-pre-wrap break-words font-mono text-[15px] text-slate-300">{body}</pre>
            </div>
          );
        })}
      </div>
    );
  }

  // ── A single document, laid out ──
  if (fmt === "json") {
    let body = text;
    try { body = JSON.stringify(JSON.parse(text), null, 2); } catch { /* truncated — show raw */ }
    return (
      <pre className="whitespace-pre-wrap break-words p-3 font-mono text-[15px] text-slate-300">{body}</pre>
    );
  }

  // ── Code: numbered, and NOT wrapped — indentation carries meaning ──
  if (["py", "sh", "yaml", "yml", "toml", "js", "ts"].includes(fmt)) {
    const lines = text.split(/\r?\n/);
    return (
      <pre className="overflow-x-auto p-3 font-mono text-[15px] leading-[1.5]">
        {lines.map((l, i) => (
          <div key={i} className="flex gap-3">
            <span className="w-8 shrink-0 select-none text-right text-slate-600 tabular-nums">{i + 1}</span>
            <span className="text-slate-300">{l || " "}</span>
          </div>
        ))}
      </pre>
    );
  }

  // ── Everything else: as written ──
  return (
    <pre className="whitespace-pre-wrap break-words p-3 font-mono text-[15px] text-slate-300">{text}</pre>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: "coral" }) {
  return (
    <div>
      <Kicker>{label}</Kicker>
      <div className={`mt-1 font-mono ${tone === "coral" ? "text-coral-300" : "text-slate-300"}`}>{value}</div>
    </div>
  );
}
