// ZEVO top console bar — the persistent instrument strip. Wordmark, always-on
// system vitals, and the ⌘K command trigger. Replaces the old sidebar's role as
// the app's anchor.
//
// The vitals are deliberately ALSO on the dashboard as their own tiles: here
// they follow you onto every page, there they are big and clickable. That is
// one duplication we keep on purpose.
//
// The Launch run button is not here — every page that can start a run offers it
// in its own header, and ⌘K reaches it from anywhere.
import useSWR from "swr";
import { Command } from "lucide-react";
import type { ModelDTO, RunSummary } from "../../lib/api";
import { fmtCost, fmtDuration } from "../../lib/format";
import logoUrl from "../../assets/zevo-logo.png";
import { fireCommand } from "../../lib/commands";

type CostTotal = { total_usd: number; gpu_usd: number; agent_usd: number };

function Vital({ label, value, live = false }: { label: string; value: string; live?: boolean }) {
  return (
    <div className="flex flex-col leading-none">
      <div className="flex items-center gap-1.5">
        {live && <span className="lamp lamp-running" />}
        <span className="readout text-lg font-semibold text-brass-300">{value}</span>
      </div>
      <span className="mt-1 font-mono text-[0.6rem] uppercase tracking-[0.18em] text-ink">{label}</span>
    </div>
  );
}

export function ConsoleBar() {
  const { data: runs = [] } = useSWR<RunSummary[]>("/api/runs?limit=500", { refreshInterval: 4000 });
  const { data: registry = [] } = useSWR<ModelDTO[]>("/api/models");
  const { data: cost } = useSWR<CostTotal>("/api/cost/total", { refreshInterval: 15000 });

  const active = runs.filter((r) => r.status === "running").length;
  // Same filter the Models page and the dashboard tile use — count only
  // models that trace back to a run, so all three agree.
  const runIds = new Set(runs.map((r) => r.id));
  const models = registry.filter((m) => m.run_id && runIds.has(m.run_id)).length;
  const completedTime = runs.reduce((sum, r) => (
    r.is_terminal && r.duration_s != null && Number.isFinite(r.duration_s) && r.duration_s >= 0
      ? sum + r.duration_s
      : sum
  ), 0);
  const burn = cost ? fmtCost(cost.total_usd) : "—";

  const mac = typeof navigator !== "undefined" && /Mac/i.test(navigator.platform);

  return (
    <header className="sticky top-0 z-40 flex h-16 items-center border-b border-hair bg-panel/85 pl-5 pr-5 backdrop-blur-md">
      {/* Wordmark: the transparent Z artwork carries the letter, "evo" is text.
          items-baseline, not items-end: box-bottom alignment left "evo" riding
          ~6px high, because the text box reserves descender space that letters
          like e/v/o never use. Baseline alignment puts the image's bottom edge
          on the text baseline, so the glyphs actually sit level with the Z. */}
      {/* Sized so its right edge lands exactly on the rail's border below:
          the rail is w-44 (176px) and the header's own pl-5 eats 20 of it, so
          the wordmark block takes the remaining 156. Without this the vertical
          line in the bar sat 34px inside the vertical line down the page, and
          the two read as two unrelated rules rather than one frame. */}
      <div className="flex w-[calc(11rem-1.25rem)] shrink-0 items-baseline gap-1">
        <img src={logoUrl} alt="Z" className="h-8 w-auto shrink-0" />
        <span className="ml-0.5 font-display text-3xl font-semibold leading-none tracking-tight text-ink">
          evo
        </span>
      </div>

      <div className="hidden items-center gap-6 border-l border-hair pl-6 md:flex">
        <Vital label="Active Runs" value={String(active)} live={active > 0} />
        <Vital label="Saved Models" value={String(models)} />
        <Vital label="Run Time" value={fmtDuration(completedTime)} />
        <Vital label="Cost" value={burn} />
      </div>

      <div className="ml-auto flex items-center gap-2">
        <button
          onClick={() => fireCommand("open-palette")}
          className="btn hidden items-center gap-2 sm:inline-flex"
          title="Command palette"
        >
          <Command size={14} />
          <span className="text-slate-400">Jump to…</span>
          <kbd className="rounded border border-hair bg-canvas px-1.5 py-0.5 font-mono text-2xs text-slate-500">{mac ? "⌘" : "Ctrl"} K</kbd>
        </button>
      </div>
    </header>
  );
}
