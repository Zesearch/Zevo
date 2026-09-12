import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import useSWR from "swr";
import { Folder, ListChecks, Pencil, Plus, Search, Trash2, X } from "lucide-react";
import { TaskModal } from "../components/TaskModal";
import { TaskSettingHistory } from "../components/TaskSettings";
import { FileSetView } from "../components/FileSetView";
import { Modal } from "../components/Modal";
import { Bezel, Detail, Kicker, PageHead } from "../components/zevo/primitives";
import { PAGE_SIZES, Pager } from "../components/zevo/Pager";
import { fmtDate, fmtMetric, splitDatasetPath } from "../lib/format";
import { useRowsPerPage } from "../lib/useRowsPerPage";
import { api } from "../lib/api";
import type { TaskDTO, TaskSettingDTO } from "../lib/api";
import { fireCommand } from "../lib/commands";

/**
 * Tasks — one card per task: what the problem IS.
 *
 * A task is an objective plus one or more named held-out Test contracts, and
 * that is the whole card. Everything a run DECIDES — training data, model,
 * method, iterations, budget — is a setting, and lives behind the card with
 * the settings this task has actually been run with.
 *
 * Cards rather than rows because the objective is a paragraph: as a table
 * column it was truncated to nothing, and the columns beside it were mostly
 * empty. The name links into Runs filtered to that task, because "what
 * happened when we ran this" is the question a task raises.
 */

// Extends the record the edit dialog prefills from, so a row can be handed to
// it whole — the list already receives every field the form offers.
type TaskSummary = TaskDTO;

// Same sizes the Runs pager offers, so the two lists page the same way. The
// paging is client-side here: the catalogue arrives whole (tens of rows), so a
// page is a slice of what we already hold rather than a server request.
// One task card plus the gap below it; measured from the rendered grid.
const TASK_CARD_PX = 210;
// How many cards a row holds at the widest breakpoint (see the grid classes).
const CARD_COLS = 3;
/**
 * The objective, clamped to three lines with a way past the clamp.
 *
 * Whether it needs one is measured, not guessed from a character count: the
 * card is a grid cell, so how much text three lines hold depends on the
 * viewport. The toggle's row is always present, so a card whose objective fits
 * and one whose objective does not still push the files below them down by the
 * same amount — which is the whole point of clamping in the first place.
 */
function ClampedObjective({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const [clamped, setClamped] = useState(false);
  const ref = useRef<HTMLParagraphElement>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setClamped(el.scrollHeight > el.clientHeight + 1);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
    // `open` re-runs it: an expanded paragraph never overflows, so the state
    // has to be read again when it collapses.
  }, [text, open]);

  return (
    <>
      {/* A FIXED three lines, not a minimum: `leading-relaxed` on 14px text
          makes a line ~22.75px, so three of them overflowed a 60px minimum and
          a two-line objective sat 8px higher than a three-line one. An explicit
          24px line and a 72px box make every card's block the same height. */}
      <p
        ref={ref}
        className={`mt-2 text-sm leading-6 text-slate-400 ${open ? "" : "line-clamp-3 h-[4.5rem]"}`}
      >
        {text}
      </p>
      <div className="h-4">
        {(clamped || open) && (
          <button
            onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}
            className="font-mono text-2xs text-slate-500 transition hover:text-brass-300"
          >
            {open ? "show less" : "show more"}
          </button>
        )}
      </div>
    </>
  );
}

/** A task's test files, the set itself first.
 *
 *  The card used to name only the folder for a catalogued dataset, on the
 *  grounds that the dataset's own card already lists what is in it. But which
 *  file is the TEST SET is a fact about the task, not about the folder, and it
 *  is the one fact this section exists to state. The other entries describe how
 *  that file is read and are the same for every task built from one bundle, so
 *  they fold away.
 */
function TestFiles({
  t, onPeek,
}: { t: TaskSummary; onPeek: (name: string, file: string) => void }) {
  const [open, setOpen] = useState(false);

  /** `ifeval-eval/test.csv`, plus the dataset it belongs to when it has one.
   *
   *  The folder used to sit on its own line above the files, which read fine
   *  until a task drew its files from two datasets and the rows underneath
   *  stopped saying which was which. Qualifying each path says it once, where
   *  it is needed. */
  const fileRow = (path: string) => {
    if (!path) return null;
    const inCatalogue = splitDatasetPath(path);
    if (inCatalogue) {
      // `folder` is the DATASET, and `file` its path inside — which is what the
      // preview is opened with, so a file in a subfolder has to keep the
      // subfolder in it.
      return {
        file: inCatalogue.file, folder: inCatalogue.dataset,
        packaged: true, label: inCatalogue.label,
      };
    }
    const parts = path.replace(/\/+$/, "").split("/");
    const file = parts.pop() || path;
    const folder = parts.pop() || "";
    return { file, folder, packaged: false, label: folder ? `${folder}/${file}` : file };
  };

  const rows = (t.test_sets ?? []).flatMap((item) => [
    { label: item.name, path: item.test_set, value: "" },
    { label: `${item.name} · inference`, path: "", value: item.inference_query },
    { label: `${item.name} · answers`, path: "", value: item.answer_fields.join(", ") },
    {
      label: `${item.name} · metric`,
      path: "",
      value: `${fmtMetric(item.metric)} · ${item.metric_type === "custom" ? "custom" : "built-in"} · ${item.metric_direction}`,
    },
    ...(item.metric_type === "custom" ? [{
      label: `${item.name} · evaluator`, path: item.evaluation_script, value: "",
    }] : []),
    { label: `${item.name} · submission`, path: item.sample_submission, value: "" },
  ]).filter((r) => r.path || r.value);

  if (!rows.length) return <span className="font-mono text-xs text-slate-600">none given</span>;
  const [first, ...rest] = rows;

  const Row = ({ label, path, value }: { label: string; path: string; value: string }) => {
    const f = fileRow(path);
    return (
      <div className="flex min-w-0 items-baseline gap-2">
        <span className="w-[11.5rem] shrink-0 whitespace-nowrap font-mono text-2xs text-slate-100">{label}</span>
        {f ? (
          <span className="flex min-w-0 items-baseline gap-1">
            {/* Only a catalogued file gets the mark: it says "this one you can
                go and open", which is not true of a path typed by hand. */}
            {f.packaged && <Folder size={11} className="shrink-0 translate-y-px text-slate-500" />}
            {f.packaged ? (
              <button
                onClick={(e) => { e.stopPropagation(); onPeek(f.folder, f.file); }}
                title={path}
                className="min-w-0 truncate font-mono text-2xs text-slate-300 transition hover:text-brass-300 hover:underline"
              >
                {f.label}
              </button>
            ) : (
              <span title={path} className="min-w-0 truncate font-mono text-2xs text-slate-300">{f.label}</span>
            )}
          </span>
        ) : (
          <span className="min-w-0 truncate font-mono text-2xs text-slate-300">{value}</span>
        )}
      </div>
    );
  };

  return (
    // A box rather than an indent. The rows had to read as the contents of TEST
    // FILES and not as another section of the card, and indenting them bought
    // that by pushing them out of line with everything else on the card. The
    // box says the same thing while keeping its own left edge where the rest of
    // the card's is, and the darker fill separates it from the card behind it.
    <div className="min-w-0 space-y-1 rounded-md border border-hair bg-canvas p-2.5">
      <Row {...first} />
      <div className={`space-y-1 ${open ? "" : "hidden"}`}>
        {rest.map((r) => <Row key={r.label} {...r} />)}
      </div>
      {rest.length > 0 && (
        <button
          onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}
          className="font-mono text-2xs text-slate-500 transition hover:text-brass-300"
        >
          {open ? "show less" : `show ${rest.length} more`}
        </button>
      )}
    </div>
  );
}

export function TasksPage() {
  const { data: tasks = [], isLoading, mutate } = useSWR<TaskSummary[]>("/api/tasks", { refreshInterval: 5000 });
  const [busy, setBusy] = useState<string>("");
  const [creating, setCreating] = useState(false);
  // The row being edited, or null. Held as the row itself rather than a name so
  // the dialog opens already filled from what the list holds — no second fetch.
  const [editing, setEditing] = useState<TaskSummary | null>(null);
  // The task whose card was clicked, by name rather than by row: the list
  // refreshes every five seconds, and a held object would go stale.
  const [detail, setDetail] = useState<string | null>(null);
  const [addingSetting, setAddingSetting] = useState(false);
  // A dataset shown in place: you are checking what a file IS, and leaving the
  // page to do it loses the task you were reading.
  // Which dataset the card asked to see, and which file in it. Clicking
  // `medqa-tiny-eval/test.csv` means that file: opening on the dataset's
  // default made you find it again in a list you had just pointed at.
  const [peekDataset, setPeekDataset] = useState<
    { name: string; file: string; split?: string } | null
  >(null);
  const [params, setParams] = useSearchParams();
  const query = params.get("q") ?? "";
  const setQuery = (v: string) => setParams(v ? { q: v } : {}, { replace: true });
  // 0 means "as many as fit"; anything else is an explicit choice.
  const [pageSizePref, setPageSizePref] = useState(PAGE_SIZES[0]);
  // Held in state, not a ref: the hook has to re-measure when the list
  // actually mounts, and a ref object never changes identity.
  const [listEl, setListEl] = useState<HTMLDivElement | null>(null);
  // Cards tile three to a row at xl, so what fits is rows × columns. The hook
  // only knows about height; the width is not measured, so this over-counts on
  // a narrow window — harmless, since a short page just scrolls a little.
  const fits = useRowsPerPage(listEl, TASK_CARD_PX, { min: 1, max: 40 }) * CARD_COLS;
  const pageSize = pageSizePref || fits;
  const [page, setPage] = useState(0);

  /** Run this task again the way `s` ran it: the task supplies the problem and
   *  its files, the setting supplies every decision. The backend rewrites the
   *  objective from the two, so what the agents read matches the setting. */
  function runSetting(t: TaskSummary, s: TaskSettingDTO) {
    setDetail(null);
    fireCommand("open-new-run", undefined, {
      taskName: t.name,
      settingId: s.id,
    });
  }

  async function deleteTask(name: string) {
    // Runs keep their task_name as plain text, so history is untouched — say so
    // rather than letting the prompt imply past runs are at stake.
    if (!window.confirm(`Delete the task "${name}"? Runs already started from it are not affected.`)) return;
    setBusy(name);
    try {
      await api(`/tasks/${encodeURIComponent(name)}`, { method: "DELETE" });
      await mutate();
    } finally {
      setBusy("");
    }
  }

  // The catalogue is tens of rows, not thousands, so this filters in the
  // browser — unlike /runs, where the page you hold is a slice of the whole.
  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const out = tasks.filter(
      (t) => !q || t.name.toLowerCase().includes(q) || t.task_objective.toLowerCase().includes(q)
    );
    // Newest first, name as the tiebreak. There is no sort control: the
    // catalogue is short enough to read, and two orderings of six cards is a
    // choice nobody was asking to make.
    return out.sort((a, b) =>
      (b.created_at || "").localeCompare(a.created_at || "") || a.name.localeCompare(b.name));
  }, [tasks, query]);

  // Re-read from the live list rather than holding the row: a task edited in
  // the dialog behind this one should not show its old objective here.
  const opened = detail ? tasks.find((t) => t.name === detail) ?? null : null;

  // Keep the first visible row visible when the page size changes, and never
  // sit on a page that a new filter has emptied.
  const total = rows.length;
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const current = Math.min(page, pageCount - 1);
  const shown = rows.slice(current * pageSize, current * pageSize + pageSize);
  const first = total === 0 ? 0 : current * pageSize + 1;
  const last = Math.min(total, (current + 1) * pageSize);
  const resize = (n: number) => {
    setPage(Math.floor((current * pageSize) / (n || fits)));
    setPageSizePref(n);
  };

  return (
    <div className="flex w-full flex-1 flex-col px-[max(1.5rem,1.5vw)] py-8">
      <PageHead
        title="Tasks"
        subtitle="User-defined, reusable tasks ready for Zevo to run"
        right={
          <>
            <button onClick={() => setCreating(true)} className="btn btn-brass">
              <Plus size={14} /> New task
            </button>
          </>
        }
      />

      <TaskModal open={creating} onClose={() => setCreating(false)} onSaved={() => void mutate()} />
      <TaskModal
        open={!!editing}
        task={editing}
        onClose={() => setEditing(null)}
        onSaved={() => void mutate()}
      />

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <div className="relative min-w-0 flex-1 sm:max-w-md">
          <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            value={query}
            onChange={(e) => { setQuery(e.target.value); setPage(0); }}
            placeholder="search task name or objective"
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
      </div>

      {isLoading ? (
        <div className="text-sm text-slate-500">loading…</div>
      ) : tasks.length === 0 ? (
        <Bezel className="p-12 text-center">
          <p className="text-sm text-slate-500">No tasks yet.</p>
          <button onClick={() => setCreating(true)} className="btn btn-brass mx-auto mt-4">
            <Plus size={14} /> Create your first task
          </button>
        </Bezel>
      ) : (
        <div>

          {/* Cards, not rows, and built like the ones on Files and Models: an
              icon beside the name, then the substance. A task is a paragraph
              of objective plus four filenames — as table columns that was one
              wide row of mostly-empty cells with the objective truncated to
              nothing. */}
          <div ref={setListEl} className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {shown.map((t) => (
              <Bezel
                key={t.name}
                onClick={() => { setAddingSetting(false); setDetail(t.name); }}
                className="group flex cursor-pointer flex-col gap-4 p-5 transition hover:border-brass-500/40 hover:shadow-glow-brass"
              >
                {/* Header: what it is, and what you can do to it. No `run`
                    button: a run needs a SETTING, which is what opening the
                    card is for. */}
                <div className="flex min-w-0 items-center justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-1.5">
                    <ListChecks size={14} className="shrink-0 text-brass-400/70" />
                    {/* The name goes to this task's runs — the question a card
                        raises. The card itself opens its settings. */}
                    <Link
                      to={`/runs?q=${encodeURIComponent(t.name)}`}
                      onClick={(e) => e.stopPropagation()}
                      className="readout min-w-0 truncate text-sm font-semibold text-ink transition hover:text-brass-300 hover:underline"
                      title={`See every run of ${t.name}`}
                    >
                      {t.name}
                    </Link>
                  </div>
                  <span className="flex shrink-0 items-center gap-1">
                    <button
                      onClick={(e) => { e.stopPropagation(); setEditing(t); }}
                      disabled={!!busy}
                      title="Edit task"
                      className="rounded-md p-1 text-slate-600 transition hover:text-brass-300 disabled:opacity-40"
                    >
                      <Pencil size={13} />
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); void deleteTask(t.name); }}
                      disabled={!!busy}
                      title="Delete task"
                      className="rounded-md p-1 text-slate-600 transition hover:text-coral-300 disabled:opacity-40"
                    >
                      <Trash2 size={13} />
                    </button>
                    {/* No chevron: the settings row below already says the card
                        opens, and says it with a count. Two affordances for one
                        action read as two actions. */}
                  </span>
                </div>

                {/* Labelled like the files below it, and held to a fixed
                    height whatever the objective's length — the cards sit in a
                    grid, and a section that starts at a different y on every
                    card is a grid you cannot read across. */}
                <div>
                  <Kicker strong>Objective</Kicker>
                  <ClampedObjective text={t.task_objective || "\u2014"} />
                </div>

                {/* The test set named outright, because it IS the task: what
                    a run is finally judged on. The three that describe how it
                    is READ sit behind one click. */}
                <div>
                  <Kicker strong>Test setup</Kicker>
                  <div className="mt-2 space-y-2">
                    <TestFiles
                      t={t}
                      onPeek={(name, file) => setPeekDataset({ name, file })}
                    />

                  </div>
                </div>

                <div className="mt-auto flex items-center justify-between border-t border-hair pt-3 font-mono text-2xs text-slate-300">
                  <span>created {fmtDate(t.created_at)}</span>
                  <span className="transition group-hover:text-brass-300">settings →</span>
                </div>
              </Bezel>
            ))}

            {total === 0 && (
              <div className="col-span-full px-4 py-10 text-center text-sm text-slate-500">
                No task matches "{query}".
              </div>
            )}
          </div>
        </div>
      )}

      {/* A task opened: what it is, and the settings it has been attacked
          with. A dialog rather than an expanding card — a card that grows to
          five times its height reflows every card beside it. Wide, because a
          setting is six decisions and they belong on one line. */}
      <Modal
        open={!!detail}
        title={detail ?? ""}
        // Wider than the 7xl the other dialogs use: a setting row carries two
        // file columns now, and validation's opens into four lines of its own.
        // At 7xl the extra width came out of the training column, which then
        // truncated the dataset name. Not wider than the columns need, though:
        // the surplus went to the widest one, and validation ended up with a
        // hand's width of nothing between it and the model.
        width="max-w-[95rem]"
        onClose={() => { setAddingSetting(false); setDetail(null); }}
      >
        {opened && (
          <div className="space-y-5">
            {/* The objective is on the card you opened this from, and no
                "default setting" block either: a run is started FROM a setting,
                so the settings are the only thing this dialog is for. A task
                nobody has run yet gets an empty table it can be filled into,
                rather than a dead end. */}
            <div>
              <Detail label="Saved settings" section hideLabel={addingSetting}>
                <div className="mt-1">
                  <TaskSettingHistory
                    task={opened.name}
                    onRun={(s) => runSetting(opened, s)}
                    onOpenDataset={(name, file, split) =>
                      setPeekDataset({ name, file: file ?? "", split: split ?? "" })}
                    busy={!!busy}
                    onAddingChange={setAddingSetting}
                  />
                </div>
              </Detail>
            </div>
          </div>
        )}
      </Modal>

      {/* A dataset, read where it was named. Rendered AFTER the task dialog
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

      <Pager
        total={total} first={first} last={last} page={current} pageCount={pageCount}
        pageSizePref={pageSizePref} onSize={resize} onPage={setPage}
      />
    </div>
  );
}
