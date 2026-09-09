import { Fragment, useState } from "react";
import useSWR from "swr";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { ArrowDown, ArrowUp, ChevronsUpDown, Rocket, Search, Trash2, X } from "lucide-react";
import { api } from "../lib/api";
import type { GenerationBackend, GpuProvider, RunSummary } from "../lib/api";
import { StatusBadge } from "../components/StatusBadge";
import { fmtScore, fmtMetric, fmtCost, fmtDate, fmtDuration, shortModel } from "../lib/format";
import { fireCommand } from "../lib/commands";
import { toast } from "../lib/toast";
import { Modal } from "../components/Modal";
import { useRowsPerPage } from "../lib/useRowsPerPage";
import { PAGE_SIZES, Pager } from "../components/zevo/Pager";
import { FileSetView } from "../components/FileSetView";
import { Bezel, PageHead } from "../components/zevo/primitives";

// A run row is one Bezel plus the gap to the next; measured, not guessed.
// Two lines in the first cell now (name over id), so a named run is taller than
// the 70px a single-line row used to be.
const RUN_ROW_PX = 78;
/** GET /runs returns a bare array; the row count rides in X-Total-Count, so
 *  this key needs its own fetcher — the global one only forwards the body. */
async function fetchPage(path: string): Promise<{ items: RunSummary[]; total: number }> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`API ${r.status}: ${await r.text()}`);
  const items: RunSummary[] = await r.json();
  const header = r.headers.get("X-Total-Count");
  return { items, total: header === null ? items.length : Number(header) };
}

// The iteration rail — the Zevo improvement loops as a row of segments, filled
// up to iterations_completed, the current one pulsing when the run is live.
function IterationRail({ r }: { r: RunSummary }) {
  const budget = Math.max(r.iteration_budget || 0, r.iterations_completed || 0, 1);
  const done = r.iterations_completed || 0;
  const running = r.status === "running";
  const segs = Array.from({ length: Math.min(budget, 12) });
  return (
    <div className="flex items-center gap-1" title={`${done} of ${r.iteration_budget || "\u221e"} iterations`}>
      {segs.map((_, i) => {
        const filled = i < done;
        const current = running && i === done;
        return (
          <span
            key={i}
            className={[
              "h-1.5 w-5 rounded-full transition-all",
              filled ? "bg-phosphor-400 shadow-glow-phosphor" : current ? "bg-brass-400 animate-lamp-pulse" : "bg-hair",
            ].join(" ")}
          />
        );
      })}
      <span className="ml-2 font-mono text-2xs text-slate-500">{done}/{r.iteration_budget || "\u221e"}</span>
    </div>
  );
}

function runImprovement(r: RunSummary): number | null {
  const baseline = r.baseline_test_score;
  const champion = r.champion_test_score;
  if (baseline == null || champion == null) return null;
  return r.metric_direction === "min" ? baseline - champion : champion - baseline;
}

function fmtImprovement(r: RunSummary): string {
  const value = runImprovement(r);
  return value == null ? "—" : `${value >= 0 ? "+" : ""}${fmtScore(value, r.metric)}`;
}

function fmtRunDuration(r: RunSummary): string {
  return fmtDuration(r.duration_s);
}

// Sorts the API can do in SQL. Cost and duration are derived per row after the
// query, so they are deliberately not offered — ordering by them would mean
// loading every run to sort a page of 25.
/** What a run was actually given — recorded when it was created, so it is not
 *  the same as what the task says today. */
type RunRequest = {
  task_name: string;
  task_predefined: boolean;
  task_objective: string;
  base_model: string;
  training_method: string;
  method_config: Record<string, unknown>;
  iteration_budget: number;
  stop_threshold: number | null;
  metric: string;
  metric_direction: "max" | "min";
  validation_metric: string;
  validation_metric_direction: "max" | "min";
  max_cost_usd: number;
  max_runtime_hours: number;
  gpu_provider: GpuProvider;
  num_gpus: number;
  generation_backend: GenerationBackend;
  files: {
    role: string;
    kind: "registered" | "huggingface" | "uploaded" | "query" | "path" | "none";
    name: string;
    detail: string;
    remote: boolean;   // fetched at run time, not stored on disk
    url: string;       // where it lives on the web; only set when remote
  }[];
  // What each role actually ran on, read back from this run's heartbeats.
  agents: {
    agent_id: string;
    heartbeats: number;
    variants: { driver: string; model: string; heartbeats: number }[];
  }[];
};

/** Where a file came from — every kind says something, because "no tag" reads
 *  as "unknown" rather than as "a path the user typed". */
const SOURCE_TAG: Record<string, { label: string; cls: string }> = {
  registered: { label: "in files", cls: "border-phosphor-500/40 bg-phosphor-500/10 text-phosphor-300" },
  huggingface: { label: "hf", cls: "border-hair text-slate-400" },
  uploaded: { label: "uploaded", cls: "border-skyx-500/40 bg-skyx-500/10 text-skyx-300" },
  // Described rather than pointed at: the data agent goes and gets it.
  query: { label: "to acquire", cls: "border-brass-500/40 bg-brass-500/10 text-brass-300" },
  path: { label: "path", cls: "border-hair text-slate-500" },
};

function SourceTag({ kind }: { kind: RunRequest["files"][number]["kind"] }) {
  const t = SOURCE_TAG[kind];
  if (!t) return null;
  return (
    <span className={`shrink-0 rounded border px-1.5 py-px font-mono text-[0.58rem] uppercase tracking-[0.1em] ${t.cls}`}>
      {t.label}
    </span>
  );
}

/** The legend, and which of the columns can be ordered by. */
const COLUMNS: { label: string; sort?: string }[] = [
  // The run's own name, with its id beneath — two runs of one task are told
  // apart by the name, and the id is what you paste into a URL or a log.
  { label: "Run" },
  { label: "Task name" },
  { label: "Iterations", sort: "iterations" },
  { label: "Best score", sort: "score" },
  { label: "Improvement", sort: "improvement" },
  { label: "Duration", sort: "time" },
  { label: "Cost", sort: "cost" },
  { label: "Created time" },
  { label: "Status" },
  { label: "" },
];

export function RunsPage() {
  // 0 means "as many as fit"; anything else is an explicit choice.
  const [pageSizePref, setPageSizePref] = useState(PAGE_SIZES[0]);
  // Held in state, not a ref: the hook has to re-measure when the list
  // actually mounts, and a ref object never changes identity.
  const [listEl, setListEl] = useState<HTMLDivElement | null>(null);
  const fits = useRowsPerPage(listEl, RUN_ROW_PX, { min: 3, max: 100 });
  const pageSize = pageSizePref || fits;
  const [page, setPage] = useState(0);
  // Search, sort and grouping all run server-side: filtering the page the
  // client already holds would only ever search 25 of 10,000 rows.
  //
  // The query lives in the URL so /runs?q=med is a link — that is how a task
  // row on Tasks points at its own runs — and so back/forward retrace searches.
  const [params, setParams] = useSearchParams();
  const query = params.get("q") ?? "";
  const setQuery = (v: string) =>
    setParams(v ? { q: v } : {}, { replace: true });
  const [sort, setSort] = useState<string>("started");
  // Direction is state: a column header has to be able to flip it.
  const [order, setOrder] = useState<"asc" | "desc">("desc");
  // Headers are drawn between the groups — the only form of grouping that stays
  // correct once the list is paginated.
  const [group, setGroup] = useState<string>("");
  const grouped = group === "task";

  // The run the user clicked, shown in place rather than by leaving the list.
  // Its inputs come from the RUN, not from the task of the same name: a run
  // keeps task_name as plain text, and the task may have been edited or deleted
  // since — what ran is what the popup has to show.
  const [taskPeek, setTaskPeek] = useState<{ id: string; name: string } | null>(null);
  // One of that task's input files, opened on top of it. Closing this drops
  // you back on the task popup you opened it from.
  const [filePeek, setFilePeek] = useState<{ name: string; file: string } | null>(null);
  const { data: peeked } = useSWR<RunRequest>(
    taskPeek ? `/api/runs/${taskPeek.id}/request` : null,
  );

  const { data, mutate } = useSWR(
    `/api/runs?limit=${pageSize}&offset=${page * pageSize}`
      + `&q=${encodeURIComponent(query.trim())}&sort=${sort}&order=${order}&group=${group}`,
    fetchPage,
    // keepPreviousData holds the old rows while the next page loads, so paging
    // doesn't flash an empty table every three seconds of polling.
    { refreshInterval: 3000, keepPreviousData: true },
  );

  // Only while grouping: the header needs the whole task's count, and the page
  // it is drawn on cannot know it.
  const { data: taskCounts } = useSWR<Record<string, number>>(
    grouped ? `/api/runs/task-counts?q=${encodeURIComponent(query.trim())}` : null,
  );

  const runs = data?.items ?? [];
  const total = data?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const first = total === 0 ? 0 : page * pageSize + 1;
  const last = page * pageSize + runs.length;

  const [deleting, setDeleting] = useState<string>("");
  const nav = useNavigate();

  function resize(n: number) {
    // Keep the first visible run visible instead of jumping to an arbitrary
    // offset when the page size changes.
    setPage(Math.floor((page * pageSize) / (n || fits)));
    setPageSizePref(n);
  }

  async function deleteRun(e: React.MouseEvent, r: RunSummary) {
    e.stopPropagation();
    const active = !r.is_terminal;
    const msg = active
      ? `Run ${r.id.slice(0, 8)} is still ${r.status}. Delete it? In-flight work is cancelled and any rented GPU is destroyed. This cannot be undone.`
      : `Delete run ${r.id.slice(0, 8)}? This permanently removes it and all its stages. This cannot be undone.`;
    if (!window.confirm(msg)) return;
    setDeleting(r.id);
    try {
      await api(`/runs/${encodeURIComponent(r.id)}`, { method: "DELETE" });
      // Deleting the last row of the last page would strand us on an empty one.
      if (runs.length === 1 && page > 0) setPage(page - 1);
      await mutate();
    } catch (err) {
      toast(`Delete failed: ${(err as Error).message || err}`, "error");
    } finally {
      setDeleting("");
    }
  }

  return (
    <div className="flex w-full flex-1 flex-col px-[max(1.5rem,1.5vw)] py-8">
      <PageHead
        title="Runs"
        subtitle="All runs completed by Zevo"
      />

      {/* Search + sort. Any change resets to page 1: staying on page 7 of a
          result set that just shrank to two rows shows an empty table. */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <div className="relative min-w-0 flex-1 sm:max-w-md">
          <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            value={query}
            onChange={(e) => { setQuery(e.target.value); setPage(0); }}
            placeholder="search run name, run id or task name"
            spellCheck={false}
            className="w-full rounded-md border border-hair bg-canvas py-2 pl-9 pr-8 font-mono text-sm text-slate-200 placeholder:text-slate-600 focus:border-brass-500/50"
          />
          {query && (
            <button
              onClick={() => { setQuery(""); setPage(0); }}
              title="Clear"
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-slate-500 transition hover:text-slate-200"
            >
              <X size={13} />
            </button>
          )}
        </div>


        {query && (
          <span className="ml-auto font-mono text-2xs text-slate-500">
            {total} match{total === 1 ? "" : "es"}
          </span>
        )}
        <button
          onClick={() => { setGroup(grouped ? "" : "task"); setPage(0); }}
          className={`${query ? "" : "ml-auto"} rounded-md border px-2.5 py-1.5 font-mono text-2xs transition ${
            grouped
              ? "border-brass-500/40 bg-brass-500/10 text-brass-300"
              : "border-hair bg-canvas text-slate-500 hover:text-slate-300"
          }`}
        >
          Group By Task
        </button>

      </div>

      {/* Column legend. The four measured columns sort: clicking one orders by
          it, clicking it again flips the direction. The arrow states which. */}
      <div className="mb-2 hidden grid-cols-[1fr_0.85fr_1fr_0.7fr_0.7fr_0.6fr_0.6fr_1fr_minmax(6.75rem,0.65fr)_2.75rem] gap-4 border border-transparent px-5 lg:grid">
        {COLUMNS.map((c) => {
          const pad = c.label === "Run" || c.label === "Task name" ? "pl-2" : "";
          if (!c.sort) {
            return <span key={c.label} className={`font-mono text-2xs font-bold uppercase tracking-[0.16em] text-ink ${pad}`}>{c.label}</span>;
          }
          const on = sort === c.sort;
          return (
            <button
              key={c.label}
              onClick={() => {
                // Same column: flip. New column: start on the direction that
                // answers the question — biggest score, longest run, most spent.
                if (on) setOrder(order === "desc" ? "asc" : "desc");
                else { setSort(c.sort!); setOrder("desc"); }
                setPage(0);
              }}
              className={`flex items-center gap-1 text-left transition ${pad} ${on ? "text-brass-300" : "text-slate-500 hover:text-slate-300"}`}
              title={`Sort by ${c.label.toLowerCase()}`}
            >
              <span className={`font-mono text-2xs font-bold uppercase tracking-[0.16em] text-ink ${on ? "!text-brass-300" : ""}`}>{c.label}</span>
              {on
                ? (order === "desc"
                    ? <ArrowDown size={12} className="text-brass-300" />
                    : <ArrowUp size={12} className="text-brass-300" />)
                : <ChevronsUpDown size={12} className="text-ink" />}
            </button>
          );
        })}
      </div>

      {runs.length === 0 ? (
        <Bezel className="p-12 text-center">
          <p className="text-sm text-slate-500">No runs on record yet.</p>
          <button onClick={() => fireCommand("open-new-run")} className="btn btn-brass mx-auto mt-4">
            <Rocket size={14} /> Launch your first run
          </button>
        </Bezel>
      ) : (
        <div ref={setListEl} className="stagger flex flex-col gap-2.5">
          {runs.map((r, i) => (
            <Fragment key={r.id}>
            {/* Sorting by task turns the list into groups; the header appears
                wherever the name changes, which stays correct across pages. */}
            {grouped && (i === 0 || runs[i - 1].task_name !== r.task_name) && (
              <div className="mt-2 flex items-baseline gap-2 px-1 first:mt-0">
                <span className="font-mono text-sm text-brass-300">{r.task_name}</span>
                {/* How many runs this task HAS, not how many of them landed on
                    the page you are looking at — the list is paginated, so the
                    page count answered a question nobody asked. */}
                <span className="font-mono text-2xs text-slate-500">
                  {(() => {
                    const n = taskCounts?.[r.task_name]
                      ?? runs.filter((x) => x.task_name === r.task_name).length;
                    return `${n} run${n === 1 ? "" : "s"}`;
                  })()}
                </span>
              </div>
            )}
            <Bezel
              className="group grid cursor-pointer grid-cols-1 items-center gap-4 px-5 py-4 transition hover:border-brass-500/30 lg:grid-cols-[1fr_0.85fr_1fr_0.7fr_0.7fr_0.6fr_0.6fr_1fr_minmax(6.75rem,0.65fr)_2.75rem]"
            >
              {/* Name over id. A run launched without a name (CLI, API, or one
                  from before the field existed) shows the id alone rather than
                  a placeholder word standing in for a name. */}
              <div onClick={() => nav(`/runs/${r.id}`)} className="min-w-0 pl-2">
                <Link
                  to={`/runs/${r.id}`}
                  onClick={(e) => e.stopPropagation()}
                  title={r.id}
                  className="block min-w-0 truncate font-mono text-sm text-slate-300 group-hover:text-brass-300"
                >
                  {r.run_name || r.id.slice(0, 8)}
                </Link>
                {r.run_name && (
                  <span className="mt-0.5 block font-mono text-2xs text-slate-500" title={r.id}>
                    ({r.id.slice(0, 8)})
                  </span>
                )}
              </div>

              <div onClick={() => nav(`/runs/${r.id}`)} className="flex min-w-0 items-center gap-2 pl-2">
                <button
                  onClick={(e) => { e.stopPropagation(); setTaskPeek({ id: r.id, name: r.task_name }); }}
                  title="What this run was given"
                  className="min-w-0 truncate text-left text-sm text-slate-200 transition hover:text-brass-300 hover:underline"
                >
                  {r.task_name}
                </button>
                {(r.mode === "auto" || r.task_name?.startsWith("objective:")) && (
                  <span className="shrink-0 rounded bg-lilac-500/15 px-1.5 py-0.5 font-mono text-2xs uppercase tracking-wider text-lilac-300">✨ auto</span>
                )}
              </div>

              <div onClick={() => nav(`/runs/${r.id}`)}><IterationRail r={r} /></div>
              <div onClick={() => nav(`/runs/${r.id}`)} className="readout text-sm text-phosphor-300">
                {/* The held-out score, not the validation one the run tuned
                    on: this column ranks runs against each other, and each
                    run's validation set is its own. An Auto run has no
                    metric yet while scoping decides one, so say that rather
                    than show the dash that means "not measured". */}
                {r.scoring_settled === false ? (
                  <span className="font-mono text-2xs uppercase tracking-[0.12em] text-slate-500" title="The scoping stage is still choosing the metric and held-out eval">
                    scoping…
                  </span>
                ) : (
                  fmtScore(r.champion_test_score, r.metric)
                )}
              </div>
              <div
                onClick={() => nav(`/runs/${r.id}`)}
                className={`readout text-sm ${
                  (runImprovement(r) ?? 0) < 0 ? "text-coral-300" : "text-phosphor-300"
                }`}
              >
                {fmtImprovement(r)}
              </div>
              <div onClick={() => nav(`/runs/${r.id}`)} className="readout text-sm text-slate-300">{fmtRunDuration(r)}</div>
              <div onClick={() => nav(`/runs/${r.id}`)} className="readout text-sm text-slate-300">{fmtCost(r.cost_usd)}</div>
              <div onClick={() => nav(`/runs/${r.id}`)} className="text-sm text-slate-300">{fmtDate(r.started_at)}</div>
              <div onClick={() => nav(`/runs/${r.id}`)} className="min-w-0">
                <StatusBadge status={r.status} />
              </div>

              <button
                onClick={(e) => deleteRun(e, r)}
                disabled={deleting === r.id}
                title="Delete run"
                className="-mr-2 justify-self-end rounded-md p-1.5 text-slate-500 transition hover:bg-coral-500/10 hover:text-coral-300 disabled:opacity-40"
              >
                <Trash2 size={14} />
              </button>
            </Bezel>
            </Fragment>
          ))}
        </div>
      )}

      {/* What the run was given, without losing your place in the list.
          Same order the Tasks page uses: what it is, how it was set up, what it
          was fed. */}
      <Modal
        open={!!taskPeek}
        title={taskPeek?.name ?? ""}
        // Wide enough that the long input roles stay on one line, no wider:
        // at 4xl the objective ran a third of a dialog past where the two
        // blocks under it ended, and read as a different column of text.
        width="max-w-3xl"
        onClose={() => setTaskPeek(null)}
        badge={
          !peeked ? null : peeked.task_predefined ? (
            <span className="flex items-center gap-2">
              <span className="rounded border border-phosphor-500/40 bg-phosphor-500/10 px-2 py-0.5 font-mono text-2xs text-phosphor-300">
                pre-defined task
              </span>
              <Link
                to={`/tasks?q=${encodeURIComponent(peeked.task_name)}`}
                className="font-mono text-2xs text-brass-300 hover:underline"
              >
                open in Tasks
              </Link>
            </span>
          ) : (
            <span className="rounded border border-hair px-2 py-0.5 font-mono text-2xs text-slate-500">
              not pre-defined
            </span>
          )
        }
      >
        {!peeked ? (
          <p className="text-sm text-slate-500">loading…</p>
        ) : (
          <div className="space-y-4">
            {peeked.task_objective && (
              <div>
                <div className="font-mono text-2xs uppercase tracking-[0.16em] text-ink">Objective</div>
                <p className="mt-1.5 text-sm leading-relaxed text-slate-300">{peeked.task_objective}</p>
              </div>
            )}

            <div className="border-t border-hair pt-4">
              <div className="font-mono text-2xs uppercase tracking-[0.16em] text-ink">Settings</div>
              {/* Label above value, in a row of even columns. As `label ....
                  value` pairs these were two words at opposite ends of half a
                  dialog, so every row read as a gap with text on both sides and
                  the eye had to travel to pair them up. Stacked, each fact is
                  one small block and the row fills evenly. */}
              <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-4">
                {([
                  ["model", shortModel(peeked.base_model) || "\u2014"],
                  ["method", peeked.training_method || "\u2014"],
                  ...(peeked.training_method === "gkd" ? [[
                    "teacher model", String(peeked.method_config?.teacher_model || "\u2014"),
                  ]] : []),
                  ...(peeked.training_method === "online_dpo" ? [[
                    "reward model", String(peeked.method_config?.reward_model || "\u2014"),
                  ]] : []),
                  [
                    "maximum GPUs",
                    `${peeked.gpu_provider} · ${peeked.num_gpus ? peeked.num_gpus : "∞"}`,
                  ],
                  ["generation backend", peeked.generation_backend],
                  ["iterations", peeked.iteration_budget ? String(peeked.iteration_budget) : "\u221e"],
                  ["budget", peeked.max_cost_usd ? `$${peeked.max_cost_usd}` : "\u221e"],
                  ["time limit", peeked.max_runtime_hours ? `${peeked.max_runtime_hours} h` : "\u221e"],
                  ["test metric / target", `${fmtMetric(peeked.metric)} · ${peeked.metric_direction === "min" ? "Min" : "Max"}`],
                  ["validation metric / target", `${fmtMetric(peeked.validation_metric)} · ${peeked.validation_metric_direction === "min" ? "Min" : "Max"}`],
                  ["stop threshold", peeked.stop_threshold != null ? String(peeked.stop_threshold) : "not set"],
                ] as [string, string][]).map(([k, v]) => (
                  <div key={k} className="min-w-0">
                    <dt className="truncate font-mono text-[0.68rem] uppercase tracking-[0.14em] text-slate-500">
                      {k}
                    </dt>
                    {/* `∞` is a small glyph at this size — it sits around the
                        x-height of the digits beside it and reads as a smudge
                        rather than as "no cap". Only the symbol grows; a number
                        stays the size of every other value in the row. */}
                    <dd
                      title={k === "model" ? peeked.base_model || v : v}
                      className={`mt-1 truncate font-mono text-slate-200 ${
                        v === "\u221e" ? "text-base leading-none" : "text-2xs"
                      }`}
                    >{v}</dd>
                  </div>
                ))}
              </dl>
            </div>

            {peeked.agents?.length > 0 && (
              <div className="border-t border-hair pt-4">
                <div className="font-mono text-2xs uppercase tracking-[0.16em] text-ink">
                  Agent configuration
                </div>
                {/* Read back from the heartbeats, not from the agents table:
                    config is editable between runs, so the table says what an
                    agent would run on today, not what this run used. An agent
                    reconfigured mid-run shows every model it ran on. */}
                <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-3">
                  {peeked.agents.map((a) => (
                    <div key={a.agent_id} className="min-w-0">
                      <dt className="truncate font-mono text-[0.68rem] uppercase tracking-[0.14em] text-slate-500">
                        {a.agent_id}
                      </dt>
                      {a.variants.map((v) => (
                        <dd
                          key={`${v.driver}/${v.model}`}
                          title={`${v.driver} · ${v.model} · ${v.heartbeats} heartbeat${v.heartbeats === 1 ? "" : "s"}`}
                          className="mt-1 truncate font-mono text-2xs text-slate-200"
                        >
                          {shortModel(v.model) || "\u2014"}
                        </dd>
                      ))}
                    </div>
                  ))}
                </dl>
              </div>
            )}

            <div className="border-t border-hair pt-4">
              <div className="font-mono text-2xs uppercase tracking-[0.16em] text-ink">Input files</div>
              <dl className="mt-3 space-y-2">
                {peeked.files.map((f) => (
                  <div key={f.role} className="grid grid-cols-4 items-baseline gap-x-6 font-mono text-2xs">
                    <dt className="col-span-2 whitespace-nowrap text-slate-500">{f.role}</dt>
                    <dd className="col-span-2 flex min-w-0 items-center gap-2">
                      {f.kind === "none" ? (
                        <span className="text-slate-600">
                          not given
                        </span>
                      ) : f.remote && f.url ? (
                        // Fetched from the hub, so there is nothing of it here
                        // to preview. The page it came from is the answer.
                        // Written the same way as the files beside it: a
                        // catalogued input is <dataset>/<what it is>, whether
                        // that is a filename or a hub id.
                        <>
                          <a
                            href={f.url}
                            target="_blank"
                            rel="noreferrer"
                            title={f.url}
                            className="min-w-0 truncate text-slate-300 hover:text-brass-300 hover:underline"
                          >
                            {f.kind === "registered" ? `${f.name}/${f.detail}` : f.name}
                          </a>
                          {f.kind === "registered" && <SourceTag kind="registered" />}
                          <SourceTag kind="huggingface" />
                        </>
                      ) : f.kind === "registered" ? (
                        // In the catalogue, so it can be opened. In place,
                        // and on the file that was clicked: you came here to
                        // read one of this run's inputs, so closing belongs
                        // back on this run, not on the Files page.
                        <>
                          <button
                            onClick={() => setFilePeek({ name: f.name, file: f.detail })}
                            title={f.detail}
                            className="min-w-0 truncate text-left text-slate-300 hover:text-brass-300 hover:underline"
                          >
                            {`${f.name}/${f.detail}`}
                          </button>
                          <SourceTag kind={f.kind} />
                        </>
                      ) : (
                        <>
                          {/* A hub id or a query IS the answer; for a file it is
                              the path that says where it came from. */}
                          <span className="min-w-0 truncate text-slate-300" title={f.detail}>
                            {f.kind === "huggingface" || f.kind === "query" ? f.name : (f.detail || f.name)}
                          </span>
                          <SourceTag kind={f.kind} />
                        </>
                      )}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        )}
      </Modal>

      {/* An input file of that run, opened where it was named. */}
      <Modal
        open={!!filePeek}
        title={filePeek ? (filePeek.file ? `${filePeek.name}/${filePeek.file}` : filePeek.name) : ""}
        width="max-w-3xl"
        onClose={() => setFilePeek(null)}
      >
        {filePeek && (
          <FileSetView
            name={filePeek.name}
            file={filePeek.file}
            onLeave={() => { setFilePeek(null); setTaskPeek(null); }}
          />
        )}
      </Modal>

      <div className="mt-auto">
        <Pager
          total={total} first={first} last={last} page={page} pageCount={pageCount}
          pageSizePref={pageSizePref} sizes={PAGE_SIZES}
          onSize={resize} onPage={setPage}
        />
      </div>
    </div>
  );
}
