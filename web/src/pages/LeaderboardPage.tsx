import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { Link } from "react-router-dom";
import { ArrowDown, ArrowUp, ChevronsUpDown } from "lucide-react";
import { Bezel, Detail, Kicker, PageHead } from "../components/zevo/primitives";
import { Modal } from "../components/Modal";
import { FileSetView } from "../components/FileSetView";
import { TaskSettingHistory } from "../components/TaskSettings";
import { fmtMetric, fmtScore, shortModel } from "../lib/format";

/**
 * Leaderboard — pick a task from the pool, see how every entrant that ran it
 * compares.
 *
 * TWO boards, because a run pairs a base model with a harness and ranking both
 * at once would confound them:
 *   Base model — which model Zevo got the most out of on this task.
 *   Harness    — which model driving the agents ran this task best.
 *
 * The unit is one TASK at a time, because absolute scores mean nothing across
 * tasks: med at 0.60 and capybara at 0.33 measure different things. One
 * task on screen means one honest scale and no cross-task comparison implied.
 *
 * Test score and improvement are paired held-out measurements of the same
 * validation-selected champion. Both stay visible on every row; clicking a
 * column heading orders the rows and drives the otherwise-unlabelled bar.
 *
 * Bars are drawn directly rather than charted: one series, no axes worth the
 * room, and this way the model name, bar and value share a baseline.
 */

type ScoreCell = {
  task_name: string;
  model: string;         // the base model, or the harness model on that board
  // The setting behind the row: its name on the task (`s1`, `s2`, …) and the
  // decisions it made. Empty parts are the ones Zevo decided.
  setting?: string;
  setting_key?: string[];
  base_model?: string;
  level?: string;
  iteration_budget?: number;
  max_cost_usd?: number;
  training_method?: string;
  data?: string;
  version_tag?: string;  // Model board: the model this row IS
  run_name?: string;
  run_id: string;
  baseline_test_score: number | null;
  champion_test_score: number;
  champion_test_score_rank: number;
  cost_usd: number;
  duration_s: number | null;
  improvement: number | null;   // direction-normalized delta; null = no baseline
  improvement_rank: number | null;
  metric: string;
  metric_direction: "max" | "min";
};

/** Just the half of a task this page shows: the problem and its files. */
type TaskDef = {
  name: string;
  task_objective: string;
  test_set: string;
  test_answer_fields: string[];
  metric_type: "builtin" | "custom";
  evaluation_script: string;
  test_sample_submission: string;
  metric: string;
  metric_direction: "max" | "min";
};

type Row = {
  model: string; setting: string; settingKey: string; baseModel: string;
  level: string; iters: number; budget: number;
  method: string; data: string;
  tag: string; runId: string; runName: string;
  value: number;
  testScore: number; baselineTestScore: number | null;
  improvement: number | null;
  color: string; metric: string;
};

const signed = (v: number, metric: string) => `${v >= 0 ? "+" : ""}${fmtScore(v, metric)}`;

/**
 * The measures. `of` pulls the number out of a cell (null = this cell has
 * nothing to say on this measure), `fmt` renders it. Which one leads is the
 * board's call — see BOARDS.
 */
const MEASURES = [
  // "Test score", not "Evaluation score": the board is fed held-out numbers
  // only, and the vague name let a validation score sit under it unnoticed.
  // The task dialog names the same set "Test set", so they now read as one.
  { id: "champion_test_score", label: "Test score", of: (c: ScoreCell) => c.champion_test_score, fmt: (v: number, metric: string) => fmtScore(v, metric) },
  // Direction-normalized change over the run's own baseline — can be negative, so the
  // bars grow from a zero line rather than from the left edge.
  { id: "improvement", label: "Improvement", of: (c: ScoreCell) => c.improvement, fmt: (v: number, metric: string) => signed(v, metric) },
] as const;
type MeasureId = (typeof MEASURES)[number]["id"];

/**
 * The two boards. `entrant` names the column, `unit` counts the pool rows, and
 * `measures` decides what leads — which is the difference that matters:
 *
 * Both boards expose the same two held-out outcomes.
 */
const BOARDS = [
  {
    id: "base",
    label: "Model",
    entrant: "Model",
    subtitle: "Every saved model, with final test score and improvement ranked independently",
    // Nothing is collapsed here, so this counts models.
    unit: (n: number) => `${n} model${n === 1 ? "" : "s"}`,
    measures: ["champion_test_score", "improvement"] as MeasureId[],
  },
  {
    id: "harness",
    label: "Harness",
    entrant: "Harness",
    subtitle: "Every run with a saved model, compared within its setting on both held-out outcomes",
    unit: (n: number) => `${n} run${n === 1 ? "" : "s"}`,
    measures: ["champion_test_score", "improvement"] as MeasureId[],
  },
] as const;
type BoardId = (typeof BOARDS)[number]["id"];

// Distinct hue per entrant, so one keeps its colour whichever task you open.
const SERIES_COLORS = [
  "#4FD1B5", "#E08D38", "#5FB0DD", "#A98FD6", "#E6BB5B",
  "#F26F58", "#7DE8CE", "#8FCDEB", "#C3B0E8",
];

export function LeaderboardPage() {
  const [boardId, setBoardId] = useState<BoardId>("base");
  const board = BOARDS.find((b) => b.id === boardId)!;
  const { data } = useSWR<{ cells: ScoreCell[]; models: string[] }>(
    `/api/leaderboard?by=${boardId}`,
    { refreshInterval: 5000, keepPreviousData: true },
  );
  const [measureId, setMeasureId] = useState<MeasureId>("champion_test_score");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  // What a click on a setting chip opens; and whether the task's own definition
  // is on screen. Both are configuration you want to CHECK, not read on every
  // row — so they live behind the name rather than beside it.
  // The task's settings, as one table. Reading what `s2` means is a question
  // about the TASK, not about the row you happen to be looking at, so it is one
  // button beside the task name rather than a click target on every row.
  const [allSettings, setAllSettings] = useState(false);
  // The model board is a flat ranking by default — every model this task made,
  // best first. Grouping turns it into "best per way of doing it", which is a
  // different question, so it is a switch rather than the only view.
  const [grouped, setGrouped] = useState(false);
  const [taskPeek, setTaskPeek] = useState(false);
  // A file named in the settings table, opened where it was named: you are
  // checking what the training data IS, not going to Files.
  const [peekDataset, setPeekDataset] = useState<
    { name: string; file: string; split: string } | null
  >(null);
  // The task's own definition, for that dialog. The catalogue is small and the
  // launch dialog already holds it, so this is a cache hit, not a round trip.
  const { data: tasks = [] } = useSWR<TaskDef[]>("/api/tasks");
  const [selected, setSelected] = useState<string | null>(null);

  // The active sortable outcome also drives the bar.
  const activeMeasureId = board.measures.includes(measureId) ? measureId : board.measures[0];
  const measure = MEASURES.find((m) => m.id === activeMeasureId)!;
  const cells = data?.cells ?? [];

  // One colour per SETTING, not per model: the bars are there to compare ways
  // of attacking the task, and giving two runs of one setting two colours says
  // they are different things.
  const settingColor = useMemo(() => {
    const keys = [...new Set(cells.map((c) => (c.setting_key ?? []).join("\u0000")))].sort();
    return new Map(keys.map((k, i) => [k, SERIES_COLORS[i % SERIES_COLORS.length]]));
  }, [cells]);

  /** The pool: every task with at least one result, and how many entrants ran it. */
  const pool = useMemo(() => {
    const by = new Map<string, Set<string>>();
    for (const c of cells) {
      if (!by.has(c.task_name)) by.set(c.task_name, new Set());
      // One entry per ENTRANT, and on the Model board an entrant is the whole
      // setting — the same base model trained two ways is two of them.
      // Both boards require a saved model. The harness board counts the Runs
      // that produced them; Settings are the like-for-like grouping boundary.
      by.get(c.task_name)!.add(boardId === "base"
        ? (c.version_tag ?? c.run_id)
        : c.run_id);
    }
    return [...by.entries()]
      .map(([task, entrants]) => ({ task, models: entrants.size }))
      .sort((a, b) => a.task.localeCompare(b.task));
  }, [cells, boardId]);

  // Keep a valid selection as the pool changes (runs finish, tasks appear).
  const active = selected && pool.some((p) => p.task === selected) ? selected : pool[0]?.task ?? null;
  useEffect(() => {
    if (active !== selected) setSelected(active);
  }, [active, selected]);
  const task = tasks.find((t) => t.name === active) ?? null;
  const metricDirection = task?.metric_direction
    ?? cells.find((c) => c.task_name === active)?.metric_direction
    ?? "max";

  // A newly selected task starts in its natural best-first score order. This
  // does not fire when the user merely flips the current column's arrow.
  useEffect(() => {
    if (activeMeasureId === "champion_test_score") {
      setSortOrder(metricDirection === "min" ? "asc" : "desc");
    }
  }, [active, boardId, metricDirection]);

  /** The bars for the selected task, best first, on that task's own scale. */
  const rows = useMemo(() => {
    if (!active) return [];
    return cells
      .filter((c) => c.task_name === active)
      .map((c) => ({
        model: c.model,
        // The setting behind it: its name, and what it decided.
        setting: c.setting ?? "",
        // The key arrives as its parts. Joined here because a Map keyed on the
        // array itself compares by identity, which put every row in a group of
        // its own.
        settingKey: (c.setting_key ?? []).join("\u0000"),
        // The setting's base model — NOT `model`, which on the harness board is
        // the harness that drove the run.
        baseModel: c.base_model ?? "",
        level: c.level ?? "",
        iters: c.iteration_budget ?? 0,
        budget: c.max_cost_usd ?? 0,
        method: c.training_method ?? "",
        data: c.data ?? "",
        tag: c.version_tag ?? "",
        runId: c.run_id,
        runName: c.run_name ?? "",
        value: measure.of(c),
        testScore: c.champion_test_score,
        baselineTestScore: c.baseline_test_score,
        improvement: c.improvement,
        metric: c.metric,
        color: settingColor.get((c.setting_key ?? []).join("\u0000")) ?? SERIES_COLORS[0],
      }))
      .filter((r): r is Row => r.value != null)
      .sort((a, b) => sortOrder === "asc" ? a.value - b.value : b.value - a.value);
  }, [cells, active, measure, settingColor, sortOrder]);

  /** The rows, in the groups the board wants. The model board is one flat list
   *  — every model the task saved, best first. The harness board splits by
   *  setting, because "which harness is better" only means something when the
   *  thing being run is held fixed. */
  const groups = useMemo(() => {
    if (boardId === "base" && !grouped) return [{ key: "all", label: "", rows }];
    const by = new Map<string, Row[]>();
    for (const r of rows) {
      const k = r.settingKey || r.setting || "—";
      if (!by.has(k)) by.set(k, []);
      by.get(k)!.push(r);
    }
    // In setting order — s1, s2, s3 — so the board reads the way the task's own
    // settings table does, rather than in whatever order the rows arrived.
    const seq = (name: string) =>
      /^s\d+$/.test(name) ? Number(name.slice(1)) : Number.MAX_SAFE_INTEGER;
    return [...by.entries()]
      .map(([key, rs]) => ({ key, label: rs[0].setting || "Unsaved configuration", rows: rs }))
      .sort((a, b) => seq(a.label) - seq(b.label) || a.label.localeCompare(b.label));
  }, [rows, boardId, grouped]);

  // Improvement can be negative, so the scale spans [min(0,…), max(0,…)] and
  // bars start at zero. With every value positive this collapses to the old
  // left-edge bars — one code path, no special case.
  const rawLo = Math.min(0, ...rows.map((r) => r.value));
  const rawHi = Math.max(0, ...rows.map((r) => r.value));
  // Once anything is negative the scale goes symmetric, so zero sits mid-track
  // and a loss reads as a loss. A lone -0.6 against a [-0.6, 0] domain would
  // otherwise fill the bar end to end and look catastrophic.
  const m = Math.max(Math.abs(rawLo), Math.abs(rawHi)) || 1;
  const lo = rawLo < 0 ? -m : 0;
  const hi = rawLo < 0 ? m : rawHi;
  const span = hi - lo || 1;
  const zeroPct = ((0 - lo) / span) * 100;

  const chooseMeasure = (next: MeasureId) => {
    if (activeMeasureId === next) {
      setSortOrder((order) => order === "desc" ? "asc" : "desc");
      return;
    }
    setMeasureId(next);
    setSortOrder(next === "champion_test_score" && metricDirection === "min" ? "asc" : "desc");
  };

  return (
    <div className="w-full px-[max(1.5rem,1.5vw)] py-8">
      <PageHead
        title="Leaderboard"
        subtitle={board.subtitle}
        right={
          <div className="flex items-center gap-1 rounded-md border border-hair bg-canvas p-1">
            {BOARDS.map((b) => (
              <button
                key={b.id}
                onClick={() => {
                  setBoardId(b.id);
                  setMeasureId(b.measures[0]);
                  setSortOrder(metricDirection === "min" ? "asc" : "desc");
                }}
                className={`rounded px-3 py-1.5 font-mono text-2xs transition ${
                  boardId === b.id ? "bg-brass-500/15 text-brass-300" : "text-slate-500 hover:text-slate-300"
                }`}
              >
                {b.label}
              </button>
            ))}
          </div>
        }
      />

      <div className="mt-6 grid grid-cols-1 items-start gap-5 lg:grid-cols-[17rem_minmax(0,1fr)]">
        {/* ── The pool ── */}
        <Bezel className="p-4">
          <div className="px-2"><Kicker strong className="!text-sm">Task pool</Kicker></div>
          <div className="mt-3 max-h-[30rem] space-y-0.5 overflow-y-auto pr-1">
            {pool.length === 0 && (
              <p className="px-2 py-6 text-center text-2xs text-slate-500">No task has results yet.</p>
            )}
            {pool.map((p) => {
              const on = p.task === active;
              return (
                <button
                  key={p.task}
                  onClick={() => setSelected(p.task)}
                  className={`flex w-full items-baseline justify-between gap-2 rounded-md px-2 py-2 text-left transition ${
                    on ? "bg-brass-500/12 text-brass-300" : "text-ink hover:bg-raised"
                  }`}
                >
                  <span className="min-w-0 truncate font-mono text-xs">{p.task}</span>
                  <span className="shrink-0 font-mono text-2xs text-slate-400">
                    {board.unit(p.models)}
                  </span>
                </button>
              );
            })}
          </div>
        </Bezel>

        {/* ── The selected task ── */}
        <Bezel className="p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <button
                onClick={() => active && setTaskPeek(true)}
                disabled={!active}
                title={active ? `What ${active} is` : undefined}
                className="font-mono text-base font-semibold text-brass-300 transition hover:underline disabled:no-underline"
              >
                {active ?? "—"}
              </button>
              <button
                onClick={() => active && setAllSettings(true)}
                disabled={!active}
                title="Every setting saved on this task"
                className="rounded-md border border-hair px-2.5 py-1 font-mono text-2xs text-slate-400 transition hover:border-brass-500/40 hover:text-brass-300 disabled:opacity-40"
              >
                all settings
              </button>
            </div>
            {boardId === "base" && (
              <button
                onClick={() => setGrouped((v) => !v)}
                title="Rank every model, or rank them within each setting"
                className={`rounded-md border px-2.5 py-1 font-mono text-2xs transition ${
                  grouped
                    ? "border-brass-500/40 bg-brass-500/10 text-brass-300"
                    : "border-hair text-slate-500 hover:text-slate-300"
                }`}
              >
                Group By Setting
              </button>
            )}
          </div>

          {rows.length === 0 ? (
            <div className="py-20 text-center text-sm text-slate-500">
              {active ? `No ${measure.label.toLowerCase()} recorded for this task yet.` : "No runs have finished yet."}
            </div>
          ) : (
            <div className="mt-5 overflow-x-auto">
              <div className="flex min-w-[86rem] items-center gap-3 border-b border-hair pb-1.5 font-mono text-2xs text-ink">
                <span className={`${boardId === "base" ? "w-44" : "w-60"} shrink-0`}>{board.entrant}</span>
                {/* The harness board's groups already name the setting, so its
                    rows say what they PRODUCED instead of repeating it: harness,
                    then the model it made, then the run it made it in. */}
                {boardId === "base" ? (
                  <>
                    <span className="w-52 shrink-0">Run</span>
                    <span className="w-32 shrink-0 pl-5">Setting</span>
                  </>
                ) : (
                  <>
                    <span className="w-32 shrink-0">Model</span>
                    <span className="w-52 shrink-0">Run</span>
                  </>
                )}
                <span className="mr-4 min-w-[8rem] flex-1" />
                {(["champion_test_score", "improvement"] as MeasureId[]).map((id) => {
                  const label = id === "champion_test_score" ? "Test score" : "Improvement";
                  const on = activeMeasureId === id;
                  return (
                    <button
                      key={id}
                      onClick={() => chooseMeasure(id)}
                      title={`Sort by ${label.toLowerCase()}`}
                      className={`flex w-36 shrink-0 items-center gap-1 text-left transition ${
                        on ? "text-brass-300" : "text-ink hover:text-brass-300"
                      }`}
                    >
                      <span>{label}</span>
                      {on
                        ? sortOrder === "desc" ? <ArrowDown size={12} /> : <ArrowUp size={12} />
                        : <ChevronsUpDown size={12} />}
                    </button>
                  );
                })}
              </div>
              <div className="mt-3 space-y-3">
              {groups.map((g) => (
                <div key={g.key} className="space-y-3">
                  {/* The harness board ranks WITHIN a setting: comparing two
                      harnesses that ran different settings compares the
                      settings. The model board has one group and no heading. */}
                  {g.label && (
                    <div className="pt-1 font-mono text-sm" style={{ color: g.rows[0].color }}>
                      {g.label}
                    </div>
                  )}
                  {g.rows.map((r) => (
                    <div key={r.tag || r.runId + r.model} className="flex min-w-[86rem] items-center gap-3">
                      {/* What this row IS: a saved model, or the harness that
                          drove the run. */}
                      {boardId === "base" ? (
                        // Display and link with the same stable Registry key.
                        <Link
                          to={`/models?q=${encodeURIComponent(r.tag || r.runId.slice(0, 8))}`}
                          className="w-44 shrink-0 truncate font-mono text-sm text-slate-100 transition hover:text-brass-300"
                          title={r.tag ? `Open ${r.tag} in Models` : undefined}
                        >
                          {r.tag || `M-${r.runId.slice(0, 8)}`}
                        </Link>
                      ) : (
                        <span className="w-60 shrink-0 whitespace-nowrap font-mono text-sm text-slate-100" title={r.model}>
                          {shortModel(r.model)}
                        </span>
                      )}
                      {/* Model board: the run it came from, then the setting —
                          named only, in its bar's colour, with the
                          configuration a click away. Harness board: the model
                          it produced, then the run, since the group above
                          already says which setting was held fixed. */}
                      {boardId === "base" ? (
                        <>
                          <Link
                            to={`/runs/${r.runId}`}
                            className="w-52 shrink-0 truncate font-mono text-sm text-slate-300 transition hover:text-brass-300"
                            title={`Open ${r.runName || r.runId.slice(0, 8)}`}
                          >
                            {r.runName || r.runId.slice(0, 8)}
                          </Link>
                          <span className="w-32 shrink-0 truncate pl-5 font-mono text-sm"
                            style={{ color: r.color }}>
                            {r.setting || "Unsaved configuration"}
                          </span>
                        </>
                      ) : (
                        <>
                          <span className="w-32 shrink-0 truncate">
                            {r.tag ? (
                              <Link
                                to={`/models?q=${encodeURIComponent(r.tag)}`}
                                className="font-mono text-sm text-slate-300 transition hover:text-brass-300"
                                title={`Open ${r.tag} in Models`}
                              >
                                {r.tag}
                              </Link>
                            ) : (
                              <span className="font-mono text-sm text-slate-600">no model</span>
                            )}
                          </span>
                          <Link
                            to={`/runs/${r.runId}`}
                            className="w-52 shrink-0 truncate font-mono text-sm text-slate-300 transition hover:text-brass-300"
                            title={`Open ${r.runName || r.runId.slice(0, 8)}`}
                          >
                            {r.runName || r.runId.slice(0, 8)}
                          </Link>
                        </>
                      )}
                      <span className="relative mr-4 h-3 min-w-0 flex-1 rounded-sm bg-hair/60">
                        {lo < 0 && (
                          <span className="absolute inset-y-[-2px] w-px bg-slate-600" style={{ left: `${zeroPct}%` }} />
                        )}
                        <span
                          className="absolute inset-y-0 rounded-sm"
                          style={{
                            // Never zero-width: a bar that vanishes reads as
                            // missing data rather than as a small number.
                            ...(() => {
                              const len = Math.max((Math.abs(r.value) / span) * 100, 1.5);
                              return r.value >= 0
                                ? { left: `${zeroPct}%`, width: `${len}%` }
                                : { left: `${Math.max(zeroPct - len, 0)}%`, width: `${len}%` };
                            })(),
                            background: r.color,
                            opacity: 0.85,
                          }}
                        />
                      </span>
                      <span
                        className="w-36 shrink-0 font-mono text-sm text-slate-100"
                        title={r.baselineTestScore == null
                          ? "No held-out baseline recorded"
                          : `Held-out baseline: ${fmtScore(r.baselineTestScore, r.metric)}`}
                      >
                        {fmtScore(r.testScore, r.metric)}
                      </span>
                      <span className="w-36 shrink-0 font-mono text-sm text-slate-100">
                        {r.improvement == null ? (
                          <span className="text-slate-600">—</span>
                        ) : (
                          signed(r.improvement, r.metric)
                        )}
                      </span>
                    </div>
                  ))}
                </div>
              ))}
              </div>
            </div>
          )}
        </Bezel>
      </div>

      {/* Every setting this task has, as the same table the task's own card
          shows — minus running and deleting, which belong where the settings
          are managed. Here it is a key to the names in the Setting column. */}
      <Modal
        open={allSettings}
        // Titled with just the task, as the task's own dialog is. What the
        // dialog holds is said once, by the Saved settings heading under it.
        title={active ?? ""}
        width="max-w-[95rem]"
        onClose={() => setAllSettings(false)}
      >
        {active && (
          <div className="space-y-5">
            <div>
              <Detail label="Saved settings" section>
                <div className="mt-1">
                  <TaskSettingHistory
                    task={active}
                    readOnly
                    onOpenDataset={(name, file, split) =>
                      setPeekDataset({ name, file: file ?? "", split: split ?? "" })}
                  />
                </div>
              </Detail>
            </div>
          </div>
        )}
      </Modal>

      {/* A dataset, read where it was named. Rendered AFTER the settings dialog
          it opens from: both are z-50, so DOM order is what decides which one
          is on top, and first meant behind. */}
      <Modal
        open={!!peekDataset}
        title={peekDataset
          ? `${peekDataset.name}${peekDataset.file ? `/${peekDataset.file.split("/").pop()}` : ""}`
          : ""}
        width="max-w-3xl"
        onClose={() => setPeekDataset(null)}
      >
        {peekDataset && (
          <FileSetView
            name={peekDataset.name}
            file={peekDataset.file}
            split={peekDataset.split}
            onLeave={() => setPeekDataset(null)}
          />
        )}
      </Modal>

      {/* And what the task itself is: the problem, and the files scoring it. */}
      <Modal
        open={taskPeek}
        title={active ?? ""}
        width="max-w-2xl"
        onClose={() => setTaskPeek(false)}
      >
        {/* A run keeps its task_name as plain text, so a board can show a task
            the catalogue no longer has — deleted, or renamed since. */}
        {!task ? (
          <p className="text-sm text-slate-500">
            No task called “{active}” is in the catalogue any more, so there is nothing
            left to show but the runs it produced.
          </p>
        ) : (
          <div className="space-y-5">
            <div>
              <div className="section-title">Objective</div>
              <p className="mt-1.5 text-sm leading-relaxed text-slate-300">{task.task_objective || "\u2014"}</p>
            </div>
            <div>
              <div className="section-title">Test setup</div>
            <dl className="mt-1.5 grid grid-cols-2 gap-x-6 gap-y-4 border-t border-hair pt-4">
              {([
                ["Metric", fmtMetric(task.metric)],
                ["Metric type", task.metric_type],
                ["Target", task.metric_direction === "min" ? "Min" : "Max"],
                ["Test set", task.test_set],
                ["Test answer fields", (task.test_answer_fields ?? []).join(", ")],
                ["Evaluation script", task.evaluation_script],
                ["Sample submission", task.test_sample_submission],
              ] as [string, string][]).map(([k, v]) => (
                <div key={k} className="min-w-0">
                  <dt className="field-label">{k}</dt>
                  <dd className="mt-1 truncate font-mono text-sm text-slate-100" title={v}>
                    {v ? v.split("/").slice(-2).join("/") : "\u2014"}
                  </dd>
                </div>
              ))}
            </dl>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
