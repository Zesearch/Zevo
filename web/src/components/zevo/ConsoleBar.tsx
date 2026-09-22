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
import type { ModelDTO, RunStatistics } from "../../lib/api";
import { fmtCost, fmtDuration } from "../../lib/format";
import logoUrl from "../../assets/zevo-logo.png";
import { fireCommand } from "../../lib/commands";

type CostTotal = { total_usd: number; gpu_usd: number; agent_usd: number };

function Vital({ label, value, live = false }: { label: string; value: string; live?: boolean }) {
  return (
    <div role="group" aria-label={label} className="flex flex-col leading-none">
      <div className="flex items-center gap-1.5">
        {live && <span className="lamp lamp-running" />}
        <span className="readout text-lg font-semibold text-brass-300">{value}</span>
      </div>
      <span className="mt-1 font-mono text-[0.6rem] uppercase tracking-[0.18em] text-ink">{label}</span>
    </div>
  );
}

export function ConsoleBar() {
  const { data: stats, error: runsError } = useSWR<RunStatistics>("/api/runs/statistics", { refreshInterval: 4000 });
  const { data: registry, error: modelsError } = useSWR<ModelDTO[]>("/api/models");
  const { data: cost, error: costError } = useSWR<CostTotal>("/api/cost/total", { refreshInterval: 15000 });

  const runsKnown = !!stats && !runsError;
  const active = runsKnown ? stats.active : null;
  const models = registry && !modelsError ? registry.length : null;
  const completedTime = runsKnown ? stats.runtime_seconds.all : null;
  const burn = cost && !costError ? fmtCost(cost.total_usd) : "—";

  const mac = typeof navigator !== "undefined" && /Mac/i.test(navigator.platform);

  return (
    <header className="sticky top-0 z-40 flex h-16 min-w-0 items-center border-b border-hair bg-panel/85 px-3 sm:px-5 backdrop-blur-md">
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
      <div className="flex w-auto shrink-0 md:w-[calc(11rem-1.25rem)] items-baseline gap-1">
        <img src={logoUrl} alt="Z" className="h-8 w-auto shrink-0" />
        <span className="ml-0.5 font-display text-3xl font-semibold leading-none tracking-tight text-ink">
          evo
        </span>
      </div>

      <div className="hidden items-center gap-5 border-l border-hair pl-5 xl:flex">
        <Vital label="Active runs" value={active === null ? "—" : String(active)} live={active !== null && active > 0} />
        <Vital label="Saved Models" value={models === null ? "—" : String(models)} />
        <Vital label="Run time" value={completedTime === null ? "—" : fmtDuration(completedTime)} />
        <Vital label="Cost" value={burn} />
      </div>

      <div className="ml-auto flex items-center gap-2">
        <div className="hidden sm:block">
          <button
            onClick={() => fireCommand("open-palette")}
            className="btn items-center gap-2"
            title="Command palette"
          >
            <Command size={14} />
            <span className="text-slate-400">Jump to…</span>
            <kbd className="rounded border border-hair bg-canvas px-1.5 py-0.5 font-mono text-2xs text-slate-500">{mac ? "⌘" : "Ctrl"} K</kbd>
          </button>
        </div>
      </div>
    </header>
  );
}
