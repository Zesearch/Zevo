// ZEVO — the four system vitals, as a single full-width instrument strip.
//
// The compact ConsoleBar repeats these essentials on every page; this dashboard
// strip is the detailed view with time windows, links, and cost decomposition.
import { useState, type ReactNode } from "react";
import useSWR from "swr";
import { Link } from "react-router-dom";
import { ArrowUpRight } from "lucide-react";
import type { ModelDTO, RunStatistics } from "../../lib/api";
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
  const { data: stats, error: runsError, isLoading: runsLoading } = useSWR<RunStatistics>("/api/runs/statistics", { refreshInterval: 4000 });
  const { data: registry = [], error: modelsError } = useSWR<ModelDTO[]>("/api/models");
  const { data: cost } = useSWR<CostTotal>(
    `/api/cost/total?window=${costWindow}`, { refreshInterval: 15000 });

  const active = stats?.active ?? 0;
  const succeeded = stats?.succeeded ?? 0;
  const failed = stats?.failed ?? 0;
  const total = stats?.total ?? 0;

  // Count what the Models page actually shows: models that trace back to a
  // run. Counting the raw table made this tile disagree with the page it links
  // to (8 here, 5 there).
  const models = registry.length;
  const totalTime = stats?.runtime_seconds[timeWindow] ?? 0;

  return (
    <div>
      {(runsError || modelsError) && <p role="alert" className="mb-3 text-coral-300">Some workspace statistics are unavailable. Refresh to retry.</p>}
      <div className="grid grid-cols-2 gap-5 lg:grid-cols-4">
      {/* One brass tone across all four — the strip reads as a single
          instrument rather than four independently-coloured statuses. */}
      <Tile to="/runs">
        <Readout
          label="ACTIVE RUNS"
          value={runsLoading || runsError ? "—" : active}
          tone="brass"
          size="lg"
          strongLabel
          hint={
            <span className="text-slate-100">
              {failed > 0
                ? `${total} runs · ${succeeded} succeeded · ${failed} failed`
                : `${total} run${total === 1 ? "" : "s"}`}
            </span>
          }
        />
      </Tile>
      <Tile to="/models">
        <Readout label="SAVED MODELS" value={modelsError ? "—" : models} tone="brass" size="lg" strongLabel />
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
          value={runsLoading || runsError ? "—" : fmtDuration(totalTime)}
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
    </div>
  );
}
