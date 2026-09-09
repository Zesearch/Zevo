// ZEVO — the four system vitals, as a single full-width instrument strip.
//
// The compact ConsoleBar repeats these essentials on every page; this dashboard
// strip is the detailed view with time windows, links, and cost decomposition.
import { useState, type ReactNode } from "react";
import useSWR from "swr";
import { Link } from "react-router-dom";
import { ArrowUpRight } from "lucide-react";
import type { ModelDTO, RunSummary } from "../../lib/api";
import { fmtCost, fmtDuration } from "../../lib/format";
import { Bezel, Note, Readout } from "./primitives";

// All-time spend, split by what spent it: agent cost scales with how much the
// models talk, GPU cost with how long the hardware was held. A single total
// hides which of the two is running away.
type CostTotal = {
  agent_usd: number;
  gpu_usd: number;
  total_usd: number;
  n_heartbeats: number;
  n_runs: number;
  n_tokens: number;
};

/** A vital tile. With `to` it becomes a link to the page that owns the number;
 *  Cost has no such page, so it stays inert rather than linking somewhere
 *  arbitrary. */
function Tile({ to, children }: { to?: string; children: ReactNode }) {
  const body = (
    <Bezel
      className={`relative flex h-full items-start p-5 ${
        to ? "transition group-hover:border-brass-500/40" : ""
      }`}
    >
      {children}
      {to && (
        <ArrowUpRight
          size={14}
          className="absolute right-4 top-4 text-slate-600 transition group-hover:text-brass-300"
        />
      )}
    </Bezel>
  );
  return to ? (
    <Link to={to} className="group block">
      {body}
    </Link>
  ) : (
    body
  );
}

// Trailing windows for the Time and Cost tiles. All-time aggregates only ever
// go up, so each needs a recent window to remain useful.
const WINDOWS = [
  { id: "1d", label: "1d" },
  { id: "1w", label: "1w" },
  { id: "1m", label: "1m" },
  { id: "all", label: "all" },
] as const;

export function VitalsGrid() {
  const [costWindow, setCostWindow] = useState<string>("all");
  const [timeWindow, setTimeWindow] = useState<string>("all");
  const { data: runs = [] } = useSWR<RunSummary[]>("/api/runs?limit=500", { refreshInterval: 4000 });
  const { data: registry = [] } = useSWR<ModelDTO[]>("/api/models");
  const { data: cost } = useSWR<CostTotal>(
    `/api/cost/total?window=${costWindow}`, { refreshInterval: 15000 });

  const active = runs.filter((r) => r.status === "running").length;
  const succeeded = runs.filter((r) => r.status === "success").length;
  // failed / cancelled / halted all mean "did not deliver"; one count says so.
  const failed = runs.filter((r) =>
    ["failed", "cancelled", "halted"].includes(r.status)).length;

  // Count what the Models page actually shows: models that trace back to a
  // run. Counting the raw table made this tile disagree with the page it links
  // to (8 here, 5 there).
  const runIds = new Set(runs.map((r) => r.id));
  const models = registry.filter((m) => m.run_id && runIds.has(m.run_id)).length;
  const windowMs: Record<string, number> = {
    "1d": 24 * 60 * 60 * 1000,
    "1w": 7 * 24 * 60 * 60 * 1000,
    "1m": 30 * 24 * 60 * 60 * 1000,
  };
  const timeCutoff = timeWindow === "all" ? 0 : Date.now() - windowMs[timeWindow];
  const timedRuns = runs.filter((r) => {
    const finished = Date.parse(r.finished_at ?? "");
    return r.is_terminal
      && r.duration_s != null
      && Number.isFinite(r.duration_s)
      && r.duration_s >= 0
      && (timeWindow === "all" || (Number.isFinite(finished) && finished >= timeCutoff));
  });
  // Sum only terminal runs. Filtering the window by finished_at makes "1d"
  // mean work completed during the last day, including a long run that began
  // before the cutoff.
  const totalTime = timedRuns.reduce((sum, r) => sum + (r.duration_s ?? 0), 0);

  return (
    <div className="grid grid-cols-2 gap-5 lg:grid-cols-4">
      {/* One brass tone across all four — the strip reads as a single
          instrument rather than four independently-coloured statuses. */}
      <Tile to="/runs">
        <Readout
          label="ACTIVE RUNS"
          value={active}
          tone="brass"
          size="lg"
          strongLabel
          // "N on record" alone hid that some of them failed, which is the
          // part worth knowing at a glance.
          hint={
            <span className="text-slate-100">
              {failed > 0
                ? `${runs.length} on record · ${succeeded} succeeded · ${failed} failed`
                : `${runs.length} run${runs.length === 1 ? "" : "s"} on record`}
            </span>
          }
        />
      </Tile>
      <Tile to="/models">
        <Readout label="SAVED MODELS" value={models} tone="brass" size="lg" strongLabel />
      </Tile>
      <Tile>
        <div className="absolute right-4 top-4 flex items-center gap-0.5">
          {WINDOWS.map((w) => (
            <button
              key={w.id}
              onClick={() => setTimeWindow(w.id)}
              className={`rounded px-1.5 py-0.5 font-mono text-2xs transition ${
                timeWindow === w.id
                  ? "bg-brass-500/15 text-brass-300"
                  : "text-slate-600 hover:text-slate-300"
              }`}
            >
              {w.label}
            </button>
          ))}
        </div>
        <Readout
          label="RUN TIME"
          value={fmtDuration(totalTime)}
          tone="brass"
          size="lg"
          strongLabel
        />
      </Tile>
      <Tile>
        {/* The picker sits in the tile's own corner rather than above the
            strip: it changes this number and no other. */}
        <div className="absolute right-4 top-4 flex items-center gap-0.5">
          {WINDOWS.map((w) => (
            <button
              key={w.id}
              onClick={() => setCostWindow(w.id)}
              className={`rounded px-1.5 py-0.5 font-mono text-2xs transition ${
                costWindow === w.id
                  ? "bg-brass-500/15 text-brass-300"
                  : "text-slate-600 hover:text-slate-300"
              }`}
            >
              {w.label}
            </button>
          ))}
        </div>
        <Readout
          label="COST"
          value={fmtCost(cost?.total_usd)}
          tone="brass"
          size="lg"
          strongLabel
          hint={
            <span className="flex flex-wrap gap-x-3 gap-y-0.5 text-slate-100">
              <span>Tokens {fmtCost(cost?.agent_usd)}</span>
              <span className="flex items-center gap-1">
                GPU {fmtCost(cost?.gpu_usd)}
                <Note size={14}>
                  Rented cloud GPUs cost money. Your own cluster or allocation does not.
                </Note>
              </span>
            </span>
          }
        />
      </Tile>
    </div>
  );
}
