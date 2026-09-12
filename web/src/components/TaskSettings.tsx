import { useEffect, useState } from "react";
import useSWR from "swr";
import { ChevronRight, Folder, Pencil, Plus, Trash2, X } from "lucide-react";
import { Note } from "./zevo/primitives";
import { shortModel, splitDatasetPath } from "../lib/format";
import {
  BUILTIN_METRICS,
  FileSlot,
  MODEL_ID_HINT,
  TrainingDataField,
  ValidationSetField,
} from "./RunInputs";
import { api } from "../lib/api";
import type { TaskSettingDTO } from "../lib/api";
import { toast } from "../lib/toast";
import { ThemedSelect } from "./ThemedSelect";

/**
 * A task's settings, read back from the runs it has had.
 *
 * A task is the PROBLEM — an objective and the files that score it. A setting
 * is one way of attacking it: which base model, which training method, whether
 * the training data is handed over. The autonomy level follows the canonical
 * ownership ladder, which is why it belongs here and not on the task.
 *
 * Settings are stored explicitly and run history is joined back onto each
 * matching configuration. A saved setting may therefore have run_count=0.
 */

export const LEVEL_BLURB: Record<string, string> = {
  L1: "Data, model and method are pinned; Zevo optimizes only inside that branch.",
  L2: "Model and method are pinned; Zevo changes Data only after the active Data branch is exhausted.",
  L3: "Model is pinned; Zevo exhausts Data branches before advancing to the next Method.",
  L4: "Zevo exhausts Data, then Method, then Base-model branches in nested order.",
  Custom: "This configuration uses a non-ladder combination of pinned decisions.",
};

export const LEVEL_TONE: Record<string, string> = {
  L1: "border-phosphor-500/40 bg-phosphor-500/10 text-phosphor-300",
  L2: "border-skyx-500/40 bg-skyx-500/10 text-skyx-300",
  L3: "border-brass-500/40 bg-brass-500/10 text-brass-300",
  L4: "border-lilac-500/40 bg-lilac-500/10 text-lilac-300",
  Custom: "border-hair bg-raised text-slate-300",
};

function levelOf(dataset: string, baseModel: string, trainingMethod: string): string {
  const pins = [dataset, baseModel, trainingMethod].map((value) => !!value.trim());
  if (pins[0] && pins[1] && pins[2]) return "L1";
  if (!pins[0] && pins[1] && pins[2]) return "L2";
  if (!pins[0] && pins[1] && !pins[2]) return "L3";
  if (!pins[0] && !pins[1] && !pins[2]) return "L4";
  return "Custom";
}

export function LevelBadge({ level, title }: { level: string; title?: string }) {
  return (
    <span
      title={title ?? LEVEL_BLURB[level]}
      className={`w-fit shrink-0 rounded-full border px-2.5 py-0.5 font-mono text-2xs ${LEVEL_TONE[level] || "border-hair text-slate-400"}`}
    >
      {level}
    </span>
  );
}

/** What a setting trains on and validates on, as one fact or two.
 *
 *  One when they come out of the same dataset folder. Two when they genuinely have
 *  two origins, because then the row is describing two decisions and collapsing
 *  them would hide one.
 */
function dataRows(s: TaskSettingDTO): {
  label: "train" | "val"; value: string; slice: string; split: string;
  folder: string; packaged: boolean; file: string; title: string;
}[] {
  const of = (
    label: "train" | "val", raw: string, split: string, config: string,
    src: TaskSettingDTO["data_source"], empty: string,
  ) => {
    if (!raw) {
      return { label, value: empty, slice: "", split: "", folder: "", packaged: false, file: "", title: empty };
    }
    const named = src && src.kind !== "none" ? src.name : "";
    // Inside the catalogue, the name is the DATASET and the leaf is the whole
    // path under it: `capybara-if` + `train/train.json`. Taking only the last
    // segment dropped the subfolder, which both hid which of three same-named
    // files this is and broke the link — the drawer is opened with this value.
    const inCatalogue = splitDatasetPath(raw);
    const leaf = inCatalogue ? inCatalogue.file : (raw.split("/").pop() || raw);
    // The slice is a field of its own rather than part of the name: it is what
    // tells two entries of one repo apart, so it must not be the part that
    // truncation eats when the path is long.
    //
    // A local file has no slice at all. `remote` is the flag that means "this
    // file is fetched, not on disk" — a catalogued hub entry comes back as kind
    // `registered` with the DATASET's name, so the kind alone does not tell you
    // it is a repo.
    //
    // Blank means different things on the two lanes, so only the training lane
    // fills it in: there, blank IS `train` to the loader, and a row that says
    // nothing reads as unspecified. On the validation lane blank is resolved at
    // launch to whichever slice looks validation-like, which for the repos we
    // see is usually `test` — printing `train` there would name the one slice
    // the loader refuses to pick.
    const isHub = !!src && (src.kind === "huggingface" || src.remote);
    const shown = split || (label === "train" ? "train" : "");
    const slice = isHub ? [config, shown].filter(Boolean).join("/") : "";
    return {
      label,
      value: named ? `${named}/${leaf}` : leaf,
      slice,
      folder: named,
      packaged: !!named,
      // A hub entry names the repo: the drawer cannot preview it, but it can
      // mark which of the bundle's entries you arrived for.
      file: isHub ? raw : leaf,
      split,
      title: raw + (slice ? ` (${slice})` : ""),
    };
  };
  return [
    of("train", s.dataset, s.dataset_split, s.dataset_config, s.data_source,
       "Zevo decides"),
    of("val", s.validation_set, s.validation_split, s.validation_config,
       s.validation_data_source,
       // Kept in step with zevo.engine.method.validation_split constants.
       "Inherited from Test · 20% · min 200 rows"),
  ];
}

/** The Validation binding fields that sit under the set itself.
 *
 *  A scoring set is four things, not one: the rows, which of their columns hold
 *  the answers, the script that grades them, and the template saying what
 *  inference must emit. The test side has said so for a while; validation said
 *  only which file, which made the other three look like they did not exist.
 *
 *  Folded away by default, because the ordinary setting leaves all three empty
 *  and the row would then be three lines of "derived from what you already
 *  see". What each empty one MEANS is the value shown, since that is the fact
 *  someone opening this is actually after.
 */
/** Shared key column for the Validation setup facts. */
const valKeyCls =
  "w-28 shrink-0 whitespace-nowrap font-mono text-2xs text-slate-100";

function ValidationRest({
  s, onOpenDataset,
}: {
  s: TaskSettingDTO;
  onOpenDataset?: (name: string, file?: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const carved = !s.validation_set.trim();
  // A derived Validation lane is one Test-owned contract, not four separately
  // configured values. The table states that contract on its data row above.
  if (carved) return null;
  // The same four keys the task card lists its test files under, so the two
  // blocks can be read against each other. Which set this is comes from the
  // column heading; `data` is the first row there and the set itself here.
  //
  // `file: true` marks the rows that ARE files, and they are drawn the way
  // `data` is: catalogue name, folder mark, and a click that opens it. They
  // used to print a bare leaf — `val_eval.py` with no folder and nothing to
  // click — which both hid which bundle it came from and read as a different
  // kind of thing from the row directly above it.
  const rows = [
    {
      label: "metric",
      file: false,
      value: carved
        ? `inherited from Test · ${s.validation_metric_type} · ${s.validation_metric} · ${s.validation_metric_direction}`
        : `${s.validation_metric_type} · ${s.validation_metric} · ${s.validation_metric_direction}`,
      empty: "built-in · token_f1 · max",
    },
    ...(s.validation_metric_type === "custom" ? [{
      label: "evaluation script",
      file: true,
      value: s.validation_evaluation_script || "",
      empty: "missing",
    }] : []),
    {
      label: "answer fields",
      file: false,
      value: (s.validation_answer_fields ?? []).join(", "),
      empty: carved ? "inherited from the Test contract" : "required for named Validation data",
    },
    {
      label: "sample submission",
      file: true,
      value: s.validation_sample_submission || "",
      empty: carved ? "inherited from the Test contract" : "required for named Validation data",
    },
  ];
  return (
    // `text-left`: the table centres its cells, which reads fine for one value
    // and badly for a label-and-value list, where the eye needs one left edge.
    <span className="block text-left">
      {/* One line per file, key then value. */}
      {open && (
        <span className="mt-0.5 block space-y-0.5">
          {rows.map((r) => {
            const inCatalogue = r.file ? splitDatasetPath(r.value) : null;
            return (
              <span key={r.label} className="flex min-w-0 items-baseline gap-1.5">
                <span className={valKeyCls}>{r.label}</span>
                {inCatalogue ? (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onOpenDataset?.(inCatalogue.dataset, inCatalogue.file);
                    }}
                    title={r.value}
                    className="flex min-w-0 items-baseline gap-1 text-slate-300 transition hover:text-brass-300"
                  >
                    <Folder size={11} className="shrink-0 translate-y-px text-slate-500" />
                    <span className="min-w-0 truncate font-mono text-2xs hover:underline">
                      {inCatalogue.label}
                    </span>
                  </button>
                ) : (
                  <span
                    title={r.value || r.empty}
                    className={`min-w-0 truncate font-mono text-2xs ${
                      r.value ? "text-slate-300" : "text-slate-500"
                    }`}
                  >
                    {r.value
                      ? (r.file ? (r.value.split("/").pop() || r.value) : r.value)
                      : r.empty}
                  </span>
                )}
              </span>
            );
          })}
        </span>
      )}
      {/* Under the rows, not above them: it is what the list ends with on the
          task card, and a control that sits above what it reveals reads as a
          heading for it. */}
      <button
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}
        className="mt-1 font-mono text-2xs text-slate-500 transition hover:text-brass-300"
      >
        {open ? "show less" : `show ${rows.length} more`}
      </button>
    </span>
  );
}

/** One `label value` pair inside a setting row — used where the settings are a
 *  line rather than a table (the launch dialog picks from them). */
function Fact({ label, value, title, tone = "text-slate-100", big = false }: {
  label: string; value: string; title?: string; tone?: string; big?: boolean;
}) {
  // Nothing shrinks and nothing truncates: the row scrolls instead. Every fact
  // on it is one of the decisions the setting IS, and a row that cuts the last
  // two off describes a different setting than the one you are picking.
  return (
    <span className="flex shrink-0 items-baseline gap-1.5 whitespace-nowrap">
      <span className="font-mono text-[0.62rem] uppercase tracking-[0.12em] text-slate-500">
        {label}
      </span>
      <span className={`font-mono ${big ? "text-sm leading-none" : "text-2xs"} ${tone}`}
        title={title ?? value}>
        {value}
      </span>
    </span>
  );
}

type SettingDataRow = ReturnType<typeof dataRows>[number];

function SettingDatasetValue({
  row, onOpenDataset,
}: {
  row: SettingDataRow;
  onOpenDataset?: (name: string, file?: string, split?: string) => void;
}) {
  const value = (
    <>
      <span className="min-w-0 break-words">{row.value}</span>
      {row.slice && (
        <span className="shrink-0 rounded border border-hair px-1 py-px text-[0.58rem] text-slate-400">
          {row.slice}
        </span>
      )}
    </>
  );

  return row.packaged ? (
    <button
      type="button"
      onClick={(event) => {
        event.stopPropagation();
        onOpenDataset?.(row.folder, row.file, row.slice ? row.split : "");
      }}
      title={row.title}
      className="flex min-w-0 max-w-full items-start gap-1.5 text-left font-mono text-xs text-slate-100 transition hover:text-brass-300 hover:underline"
    >
      <Folder size={11} className="mt-0.5 shrink-0 text-slate-500" />
      {value}
    </button>
  ) : (
    <div
      title={row.title}
      className={`flex min-w-0 items-start gap-1.5 font-mono text-xs ${
        row.folder ? "text-slate-100" : "text-brass-300"
      }`}
    >
      {value}
    </div>
  );
}

function SettingChoice({
  label, value, title, query, zevo = false, children,
}: {
  label: string;
  value: string;
  title?: string;
  query?: string;
  zevo?: boolean;
  children?: React.ReactNode;
}) {
  return (
    <div className="min-w-0 rounded-md border border-hair bg-raised/55 p-3">
      <div className="table-label">{label}</div>
      <div
        title={title || value}
        className={`mt-1.5 min-w-0 break-words font-mono text-xs leading-relaxed ${
          zevo ? "text-brass-300" : "text-slate-100"
        }`}
      >
        {children ?? value}
      </div>
      {query !== undefined && (
        <div className="mt-2 border-t border-hair/70 pt-2">
          <span className="font-mono text-[0.58rem] uppercase tracking-[0.12em] text-slate-600">
            query
          </span>
          <p
            title={query || "No query set"}
            className={`mt-0.5 line-clamp-2 break-words font-mono text-[0.64rem] leading-relaxed ${
              query ? "text-slate-400" : "text-slate-600"
            }`}
          >
            {query || "not set"}
          </p>
        </div>
      )}
    </div>
  );
}

/**
 * The settings this task can be attacked with, most-run first.
 *
 * `onPick` turns each row into a button that fills a form with it — the next
 * run is almost always a small edit of an old one, and that is what the launch
 * dialog wants. `onRun` gives the table a launch button per row instead: a run
 * needs a setting, so the setting IS the thing you start.
 */
export function TaskSettingHistory({
  task, onPick, onRun, onOpenDataset, onAddingChange,
  readOnly = false, busy = false, selectedId = "",
}: {
  task: string;
  onPick?: (s: TaskSettingDTO) => void;
  onRun?: (s: TaskSettingDTO) => void;
  /** Given, the training data opens where you are instead of navigating to
   *  Files — you are checking what it is, not going somewhere. */
  /** `file` names which entry inside the dataset to open on: a filename on
   *  disk, or a hub id, with `split` telling two entries of one repo apart. */
  onOpenDataset?: (name: string, file?: string, split?: string) => void;
  /** Lets the containing Task dialog hide its own "Saved settings" heading
   *  while the add form has replaced that saved-settings view. */
  onAddingChange?: (adding: boolean) => void;
  /** A reference copy: no delete, no way to add. Used where the table explains
   *  what the settings ARE rather than being the place you manage them. */
  readOnly?: boolean;
  busy?: boolean;
  /** The row whose values are in the form. Marked, because a form that filled
   *  itself in should say what filled it. */
  selectedId?: string;
}) {
  const { data, isLoading, mutate } = useSWR<TaskSettingDTO[]>(
    task ? `/api/tasks/${encodeURIComponent(task)}/settings` : null,
  );
  // "" = nothing open, "new" = the blank form, an id = that row being edited.
  const [open, setOpen] = useState("");
  const settings = data ?? [];

  useEffect(() => {
    onAddingChange?.(open === "new");
    return () => onAddingChange?.(false);
  }, [open, onAddingChange]);

  async function remove(s: TaskSettingDTO) {
    if (!window.confirm(
      `Remove this ${s.level} setting from ${task}? Runs already made with it are not affected.`,
    )) return;
    try {
      await api(`/tasks/${encodeURIComponent(task)}/settings/${encodeURIComponent(s.id)}`, { method: "DELETE" });
    } catch (e) {
      toast(`Remove failed: ${e instanceof Error ? e.message : e}`, "error");
    }
    await mutate();
  }

  if (!task) return null;
  if (isLoading && !data) {
    return <p className="text-2xs text-slate-500">loading…</p>;
  }

  // The launch dialog: a list you pick from, not a table you edit.
  if (onPick) {
    if (!settings.length) {
      return <p className="text-2xs text-slate-500">No setting recorded for this task yet.</p>;
    }
    return (
      <div className="space-y-1.5">
        {settings.map((s) => (
          <button
            key={s.id}
            type="button"
            onClick={() => onPick(s)}
            title="Fill the form with this setting"
            className={`flex w-full items-center gap-4 overflow-x-auto rounded-lg border px-3 py-2 text-left transition ${
              s.id === selectedId
                ? "border-brass-500/60 bg-brass-500/10 shadow-glow-brass"
                : "border-hair bg-canvas/50 hover:border-brass-500/40 hover:bg-raised"
            }`}
          >
            <span className="shrink-0 font-mono text-2xs text-slate-100">{s.name}</span>
            <LevelBadge level={s.level} />
            {/* The row scrolls rather than truncating: every fact on it is one
                of the decisions the setting IS, and a row that cuts the last
                two off describes a different setting than the one you picked. */}
            {/* One fact when both come out of one folder, two when they do
                not: which rows a setting tunes against is part of what makes it
                a different attempt, and two settings alike but for their
                validation split produced two scores never comparable. */}
            {dataRows(s).map((r) => r.label === "train" ? (
              <span key={r.label} className="shrink-0 space-y-1">
                <Fact label="data"
                  value={r.slice ? `${r.value} · ${r.slice}` : r.value} title={r.title}
                  tone={r.packaged || r.folder ? "text-slate-100" : "text-brass-300"} />
                <Fact label="query" value={s.data_query || "not set"}
                  title={s.data_query || "No data query set"} />
              </span>
            ) : (
              <Fact key={r.label} label="val"
                value={r.slice ? `${r.value} · ${r.slice}` : r.value} title={r.title} />
            ))}
            <Fact label="model" value={shortModel(s.base_model) || "Zevo decides"}
              title={s.base_model || "Zevo decides"}
              tone={s.base_model ? "text-slate-100" : "text-brass-300"} />
            <Fact label="model query" value={s.model_query || "not set"}
              title={s.model_query || "No model query set"} />
            <Fact label="method" value={s.training_method || "Zevo decides"}
              title={
                s.training_method === "gkd"
                  ? `${s.training_method} · teacher ${String(s.method_config?.teacher_model || "missing")}`
                  : s.training_method === "online_dpo"
                  ? `${s.training_method} · reward ${String(s.method_config?.reward_model || "missing")}`
                  : s.training_method || "Zevo decides"
              }
              tone={s.training_method ? "text-slate-100" : "text-brass-300"} />
            <Fact label="method query" value={s.method_query || "not set"}
              title={s.method_query || "No method query set"} />
            <Fact label="iters" value={s.iteration_budget ? String(s.iteration_budget) : "∞"}
              big={!s.iteration_budget} />
            <Fact label="budget" value={s.max_cost_usd ? `$${s.max_cost_usd}` : "∞"}
              big={!s.max_cost_usd} />
            <Fact label="stop at" value={s.stop_threshold != null ? String(s.stop_threshold) : "not set"}
              big={s.stop_threshold == null} />
          </button>
        ))}
      </div>
    );
  }

  // Adding is a separate form state, not another row in the saved-settings
  // table. Showing the table heading and its old rows above it made the form
  // look like an expansion of the list and left two competing page titles.
  if (!readOnly && open === "new") {
    return (
      <div className="min-w-0">
        <SettingForm
          task={task}
          onDone={async () => { setOpen(""); await mutate(); }}
          onCancel={() => setOpen("")}
        />
      </div>
    );
  }

  return (
    <div className="min-w-0 space-y-3">
      {settings.length === 0 && !open && (
        <div className="rounded-lg border border-dashed border-hair px-3 py-4 text-2xs text-slate-400">
          {readOnly
            ? "No setting recorded for this task yet."
            : "No setting recorded for this task yet, because nothing has been run. You can write one down now and run it from here."}
        </div>
      )}

      {settings.map((s) => {
        if (s.id === open) {
          return (
            <SettingForm
              key={s.id}
              task={task}
              existing={s}
              onDone={async () => { setOpen(""); await mutate(); }}
              onCancel={() => setOpen("")}
            />
          );
        }

        const [training, validation] = dataRows(s);
        const methodTitle = s.training_method === "gkd"
          ? `${s.training_method} · teacher ${String(s.method_config?.teacher_model || "missing")}`
          : s.training_method === "online_dpo"
          ? `${s.training_method} · reward ${String(s.method_config?.reward_model || "missing")}`
          : s.training_method || "Zevo decides";

        return (
          <div key={s.id} className="min-w-0 rounded-lg border border-hair bg-canvas/50 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-hair pb-3">
              <div className="flex min-w-0 flex-wrap items-center gap-2.5">
                <span className="font-mono text-sm text-slate-100">{s.name || "—"}</span>
                <LevelBadge level={s.level} />
                <Note to="/levels" size={13}>Autonomy level details.</Note>
              </div>
              <div className="flex shrink-0 items-center gap-1.5">
                {onRun && (
                  <button
                    type="button"
                    onClick={() => onRun(s)}
                    disabled={busy}
                    title="Start a run with this setting"
                    className="rounded-md border border-phosphor-500/40 bg-phosphor-500/10 px-2.5 py-1 font-mono text-2xs text-phosphor-300 transition hover:bg-phosphor-500/20 hover:text-phosphor-200 disabled:opacity-50"
                  >
                    run
                  </button>
                )}
                {!readOnly && (
                  <>
                    <button
                      type="button"
                      onClick={() => setOpen(s.id)}
                      title="Change this setting"
                      className="rounded-md p-1.5 text-slate-500 transition hover:bg-raised hover:text-brass-300"
                    >
                      <Pencil size={13} />
                    </button>
                    <button
                      type="button"
                      onClick={() => void remove(s)}
                      title="Remove this setting"
                      className="rounded-md p-1.5 text-slate-500 transition hover:bg-raised hover:text-coral-300"
                    >
                      <Trash2 size={13} />
                    </button>
                  </>
                )}
              </div>
            </div>

            <div className="mt-3 grid min-w-0 gap-3 md:grid-cols-2 xl:grid-cols-3">
              <SettingChoice label="Data" value={training.value} title={training.title} query={s.data_query}>
                <SettingDatasetValue row={training} onOpenDataset={onOpenDataset} />
              </SettingChoice>
              <SettingChoice
                label="Model"
                value={shortModel(s.base_model) || "Zevo decides"}
                title={s.base_model || "Zevo decides"}
                query={s.model_query}
                zevo={!s.base_model}
              />
              <SettingChoice
                label="Method"
                value={s.training_method || "Zevo decides"}
                title={methodTitle}
                query={s.method_query}
                zevo={!s.training_method}
              />
            </div>

            <div className="mt-3 grid min-w-0 gap-3 lg:grid-cols-[minmax(0,1.6fr)_minmax(18rem,0.9fr)]">
              <div className="min-w-0 rounded-md border border-hair bg-raised/35 p-3">
                <div className="table-label">Validation setup</div>
                <div className="mt-1.5">
                  <SettingDatasetValue row={validation} onOpenDataset={onOpenDataset} />
                  <ValidationRest
                    s={s}
                    onOpenDataset={(name, file) => onOpenDataset?.(name, file, "")}
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2 rounded-md border border-hair bg-raised/35 p-3">
                {[
                  ["iterations", s.iteration_budget ? String(s.iteration_budget) : "∞"],
                  ["budget", s.max_cost_usd ? `$${s.max_cost_usd}` : "∞"],
                  ["stop at", s.stop_threshold != null ? String(s.stop_threshold) : "not set"],
                ].map(([label, value]) => (
                  <div key={label} className="min-w-0 text-center">
                    <div className="font-mono text-2xs uppercase tracking-[0.1em] text-slate-500">{label}</div>
                    <div className="mt-1.5 truncate font-mono text-xs leading-relaxed text-slate-100" title={value}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );
      })}

      {readOnly ? null : open ? null : (
        <button
          type="button"
          onClick={() => setOpen("new")}
          className="flex items-center gap-1.5 rounded-md py-1 font-mono text-2xs text-slate-400 transition hover:text-brass-300"
        >
          <Plus size={12} /> add a setting
        </button>
      )}
    </div>
  );
}
/** Write a setting down before running it — the only way onto the list for a
 *  task nobody has run yet. */
function SettingForm({ task, existing, onDone, onCancel }: {
  task: string;
  /** The row being changed, or undefined to write a new one. Editing keeps the
   *  row's NAME: `s2` is what this task's runs are labelled by, and giving the
   *  same line of attack a new name after a typo fix orphans every reading. */
  existing?: TaskSettingDTO;
  onDone: () => void; onCancel: () => void;
}) {
  const [v, setV] = useState({
    name: existing?.name ?? "",
    dataset: existing?.dataset ?? "",
    dataset_split: existing?.dataset_split ?? "",
    dataset_config: existing?.dataset_config ?? "",
    data_query: existing?.data_query ?? "",
    model_query: existing?.model_query ?? "",
    method_query: existing?.method_query ?? "",
    validation_set: existing?.validation_set ?? "",
    validation_split: existing?.validation_split ?? "",
    validation_config: existing?.validation_config ?? "",
    validation_answer_fields: (existing?.validation_answer_fields ?? []).join(", "),
    validation_sample_submission: existing?.validation_sample_submission ?? "",
    validation_metric_type: existing?.validation_metric_type ?? "builtin",
    validation_metric: existing?.validation_metric ?? "token_f1",
    validation_metric_direction: existing?.validation_metric_direction ?? "max",
    validation_evaluation_script: existing?.validation_evaluation_script ?? "",
    base_model: existing?.base_model ?? "",
    training_method: existing?.training_method ?? "",
    teacher_model: String(existing?.method_config?.teacher_model ?? ""),
    reward_model: String(existing?.method_config?.reward_model ?? ""),
    use_peft: typeof existing?.method_config?.use_peft === "boolean" ? String(existing.method_config.use_peft) : "",
    prompt_framing: existing?.prompt_framing ?? "",
    system_prompt: existing?.system_prompt ?? "",
    loss_objective_config: Object.keys(existing?.loss_objective_config ?? {}).length ? JSON.stringify(existing?.loss_objective_config) : "",
    inference_config: Object.keys(existing?.inference_config ?? {}).length ? JSON.stringify(existing?.inference_config) : "",
    decoding_config: Object.keys(existing?.decoding_config ?? {}).length ? JSON.stringify(existing?.decoding_config) : "",
    iteration_budget: existing?.iteration_budget ? String(existing.iteration_budget) : "",
    max_cost_usd: existing?.max_cost_usd ? String(existing.max_cost_usd) : "",
    stop_threshold: existing?.stop_threshold != null ? String(existing.stop_threshold) : "",
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const put = (k: keyof typeof v) => (value: string) => setV((x) => ({ ...x, [k]: value }));
  const set = (k: keyof typeof v) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setV((x) => ({ ...x, [k]: e.target.value }));
  const cls = "w-full min-w-0 rounded border border-hair bg-canvas px-2 py-1 font-mono text-2xs text-slate-200 placeholder:text-slate-600 placeholder:opacity-100 focus:border-brass-500/50 focus:outline-none";

  // A named validation set must declare its answer fields and submission
  // shape. Its metric contract is independent from the Run's Test scoring
  // values and is validated separately below.
  //
  // Reusing the test side implicitly is unsafe when shapes differ. Point this
  // at a set that does not match and the run can get IFEval's answer fields
  // against conversation records:
  // the data agent is asked to drop `instruction_id_list` and `kwargs` from a
  // file whose keys are id/instruction/response, dropping them is a no-op, and
  // the "questions-only" copy handed to inference still carries every answer.
  // That is a real failure this cost a run, and the form is where it is cheap
  // to prevent.
  const missingValidationFiles = !!v.validation_set.trim() && [
    ["answer fields", v.validation_answer_fields],
    ["sample submission", v.validation_sample_submission],
  ].filter(([, x]) => !String(x).trim()).map(([k]) => k) as string[];
  const validationGaps = Array.isArray(missingValidationFiles) ? missingValidationFiles : [];
  const auxiliaryModelMissing =
    (v.training_method === "gkd" && !v.teacher_model.trim())
    || (v.training_method === "online_dpo" && !v.reward_model.trim());
  const validationMetricMissing = !!v.validation_set.trim() && (
    !v.validation_metric.trim()
    || !v.validation_metric_direction
    || (v.validation_metric_type === "custom" && !v.validation_evaluation_script.trim())
  );

  async function submit() {
    setBusy(true);
    setError("");
    try {
      const threshold = v.stop_threshold.trim() ? Number(v.stop_threshold) : null;
      if (threshold != null && !Number.isFinite(threshold)) {
        throw new Error("Stop threshold must be a finite number.");
      }
      const base = `/tasks/${encodeURIComponent(task)}/settings`;
      const parseObject = (raw: string, label: string) => {
        if (!raw.trim()) return {};
        const parsed: unknown = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
          throw new Error(`${label} must be a JSON object.`);
        }
        return parsed;
      };
      const method_config: Record<string, unknown> = v.training_method === "gkd"
        ? { teacher_model: v.teacher_model.trim() }
        : v.training_method === "online_dpo"
        ? { reward_model: v.reward_model.trim() }
        : {};
      if (v.use_peft) method_config.use_peft = v.use_peft === "true";
      await api(existing ? `${base}/${encodeURIComponent(existing.id)}` : base, {
        method: existing ? "PATCH" : "POST",
        body: JSON.stringify({
          name: v.name.trim(),
          dataset: v.dataset.trim(),
          dataset_split: v.dataset_split.trim(),
          dataset_config: v.dataset_config.trim(),
          data_query: v.data_query.trim(),
          model_query: v.model_query.trim(),
          method_query: v.method_query.trim(),
          validation_set: v.validation_set.trim(),
          validation_split: v.validation_split.trim(),
          validation_config: v.validation_config.trim(),
          validation_answer_fields: v.validation_answer_fields
            .split(",").map((c) => c.trim()).filter(Boolean),
          validation_sample_submission: v.validation_sample_submission.trim(),
          validation_metric_type: v.validation_metric_type,
          validation_metric: v.validation_metric.trim(),
          validation_metric_direction: v.validation_metric_direction,
          validation_evaluation_script: v.validation_metric_type === "custom"
            ? v.validation_evaluation_script.trim()
            : "",
          base_model: v.base_model.trim(),
          training_method: v.training_method.trim(),
          method_config,
          prompt_framing: v.prompt_framing.trim(),
          system_prompt: v.system_prompt.trim(),
          loss_objective_config: parseObject(v.loss_objective_config, "Loss objective config"),
          inference_config: parseObject(v.inference_config, "Inference config"),
          decoding_config: parseObject(v.decoding_config, "Decoding config"),
          iteration_budget: Math.max(0, Number(v.iteration_budget) || 0),
          max_cost_usd: Math.max(0, Number(v.max_cost_usd) || 0),
          stop_threshold: threshold,
        }),
      });
      onDone();
    } catch (e) {
      setError(String((e as Error).message || e));
    } finally {
      setBusy(false);
    }
  }

  // A panel, not a row in the table above. The two data fields are file
  // pickers — the catalogue, the hub, or an upload — and none of those fits in
  // a table cell. Typing the path by hand was the only route the inline row
  // left, which meant a file already sitting in the catalogue had to be
  // transcribed, and a typo in it surfaced hours later as a failed run.
  return (
    <div className="space-y-4 py-1">
      <div className="flex flex-wrap items-center gap-3">
        <span className="field-label shrink-0">
          name
        </span>
        {/* Named, not numbered. `sN` is only the default: it looks like a
            position in a list and is not one — names are never reused, so
            deleting s3 and s4 leaves s1, s2, s5 and the reader counting five.
            What this setting IS cannot be derived either: these three all pin
            the same model and method, and differ only in where the data came
            from. */}
        <input
          value={v.name}
          onChange={set("name")}
          placeholder="what to call this line of attack"
          spellCheck={false}
          className="min-w-0 flex-1 rounded border border-hair bg-canvas px-2 py-1 font-mono text-xs text-slate-100 placeholder:text-slate-600 placeholder:opacity-100 focus:border-brass-500/50 focus:outline-none"
        />
        <LevelBadge level={levelOf(v.dataset, v.base_model, v.training_method)} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {v.validation_set.trim() ? (
          <ValidationSetField
            value={v.validation_set}
            onChange={put("validation_set")}
            split={v.validation_split}
            onSplit={put("validation_split")}
            config={v.validation_config}
            onConfig={put("validation_config")}
            answerFields={v.validation_answer_fields}
            onAnswerFields={put("validation_answer_fields")}
            sampleSubmission={v.validation_sample_submission}
            onSampleSubmission={put("validation_sample_submission")}
            required={validationGaps}
            tag={false}
            note="Independent Validation data and scoring contract."
          />
        ) : (
          <div className="self-start rounded-md border border-brass-500/25 bg-brass-500/[0.04] px-4 py-3">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="field-label !text-brass-200">Inherited from Test</span>
              <span className="font-mono text-xs text-slate-400">20% of Test · min 200 rows</span>
            </div>
            <details className="group mt-3 border-t border-hair pt-2">
              <summary className="flex w-fit cursor-pointer list-none items-center gap-1.5 font-mono text-2xs uppercase tracking-[0.12em] text-slate-500 transition hover:text-slate-300">
                <ChevronRight size={11} className="transition-transform group-open:rotate-90" />
                Use an independent Validation set
              </summary>
              <div className="mt-3">
                <ValidationSetField
                  value={v.validation_set}
                  onChange={put("validation_set")}
                  split={v.validation_split}
                  onSplit={put("validation_split")}
                  config={v.validation_config}
                  onConfig={put("validation_config")}
                  answerFields={v.validation_answer_fields}
                  onAnswerFields={put("validation_answer_fields")}
                  sampleSubmission={v.validation_sample_submission}
                  onSampleSubmission={put("validation_sample_submission")}
                  tag={false}
                  note="Choose a fixed Validation set with its own scoring contract."
                />
              </div>
            </details>
          </div>
        )}
        <div className="space-y-3">
          <TrainingDataField
            value={v.dataset} label="Data" tag={false}
            onChange={put("dataset")}
            split={v.dataset_split}
            onSplit={put("dataset_split")}
            config={v.dataset_config}
            onConfig={put("dataset_config")}
            note="Examples used for training; blank lets Zevo acquire and prepare them"
          />
          <label className="block">
            <span className="field-label mb-1 block">Data query</span>
            <input
              className={cls}
              value={v.data_query}
              onChange={set("data_query")}
              placeholder="optional guidance for data discovery and preparation"
            />
          </label>
        </div>
      </div>

      {v.validation_set.trim() && (<section className="space-y-3">
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
          <label className="block">
            <span className="field-label mb-1 block">Metric type</span>
            <ThemedSelect
              value={v.validation_metric_type}
              onChange={(value) => setV((x) => ({
                ...x,
                validation_metric_type: value as "builtin" | "custom",
                ...(value === "builtin" ? { validation_evaluation_script: "" } : {}),
              }))}
              options={[
                { value: "builtin", label: "Built-in" },
                { value: "custom", label: "Custom" },
              ]}
              ariaLabel="Validation metric type"
              buttonClassName={cls}
            />
          </label>
          <label className="block">
            <span className="field-label mb-1 block">Metric</span>
            {v.validation_metric_type === "builtin" ? (
              <ThemedSelect
                value={v.validation_metric}
                onChange={put("validation_metric")}
                options={BUILTIN_METRICS.map((value) => ({ value, label: value }))}
                placeholder="Choose metric"
                ariaLabel="Built-in Validation metric"
                buttonClassName={cls}
              />
            ) : (
              <input
                className={cls}
                value={v.validation_metric}
                onChange={set("validation_metric")}
                placeholder="metric key written to metrics.json"
              />
            )}
          </label>
          <label className="block">
            <span className="field-label mb-1 block">Target</span>
            <ThemedSelect
              value={v.validation_metric_direction}
              onChange={put("validation_metric_direction")}
              options={[
                { value: "max", label: "Max" },
                { value: "min", label: "Min" },
              ]}
              ariaLabel="Validation metric direction"
              buttonClassName={cls}
            />
          </label>
        </div>
        {v.validation_metric_type === "custom" && (
          <FileSlot
            label="Validation evaluation script"
            tag={false}
            value={v.validation_evaluation_script}
            onChange={put("validation_evaluation_script")}
            hint="Frozen for this setting and run only after sample-submission validation."
          />
        )}
      </section>)}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <label className="block">
          <span className="field-label mb-1 block">Base model</span>
          <input className={cls} value={v.base_model} onChange={set("base_model")} placeholder={MODEL_ID_HINT} />
        </label>
        <label className="block">
          <span className="field-label mb-1 block">Model query</span>
          <input
            className={cls}
            value={v.model_query}
            onChange={set("model_query")}
            placeholder="optional guidance; does not pin an exact model"
          />
        </label>
        <label className="block">
          <span className="field-label mb-1 block">Training method</span>
          <ThemedSelect
            value={v.training_method}
            onChange={(value) => setV((x) => ({ ...x, training_method: value }))}
            options={[
              "lora_sft", "full_sft", "cpo", "dpo", "gkd", "grpo", "kto",
              "online_dpo", "orpo", "rft", "rloo",
            ].map((value) => ({ value, label: value }))}
            placeholder="Optimization method used to train the model; blank lets Zevo choose"
            ariaLabel="Training method"
            buttonClassName={cls}
          />
        </label>
        <label className="block">
          <span className="field-label mb-1 block">Method query</span>
          <input
            className={cls}
            value={v.method_query}
            onChange={set("method_query")}
            placeholder="optional guidance; does not pin an exact method"
          />
        </label>
        {v.training_method === "gkd" && (
          <label className="block lg:col-span-2">
            <span className="field-label mb-1 block">Teacher model</span>
            <input className={cls} value={v.teacher_model} onChange={set("teacher_model")} placeholder="Frozen GKD teacher: Hugging Face owner/model id" />
          </label>
        )}
        {v.training_method === "online_dpo" && (
          <label className="block lg:col-span-2">
            <span className="field-label mb-1 block">Reward model</span>
            <input className={cls} value={v.reward_model} onChange={set("reward_model")} placeholder="Online DPO reward model: Hugging Face owner/model id" />
          </label>
        )}
        {[
          "cpo", "dpo", "gkd", "grpo", "kto", "online_dpo", "orpo", "rft", "rloo",
        ].includes(v.training_method) && (
          <label className="block">
            <span className="field-label mb-1 block">PEFT</span>
            <ThemedSelect
              value={v.use_peft}
              onChange={put("use_peft")}
              options={[
                { value: "true", label: "Use PEFT" },
                { value: "false", label: "Full parameters" },
              ]}
              placeholder="Train Skill default"
              ariaLabel="PEFT"
              buttonClassName={cls}
            />
          </label>
        )}
        <label className="block">
          <span className="field-label mb-1 block">Iterations</span>
          <input className={cls} value={v.iteration_budget} onChange={set("iteration_budget")} placeholder="Maximum optimization rounds; blank is unlimited" />
        </label>
        <label className="block">
          <span className="field-label mb-1 block">Budget</span>
          <input className={cls} value={v.max_cost_usd} onChange={set("max_cost_usd")} placeholder="Maximum cost in USD; blank is unlimited" />
        </label>
        <label className="block">
          <span className="field-label mb-1 block">Stop threshold</span>
          <input className={cls} type="text" inputMode="decimal" value={v.stop_threshold} onChange={set("stop_threshold")} placeholder="Validation score on this Setting metric's own scale; blank disables it" />
        </label>
      </div>

      {/* Detailed prompt, loss, and inference pins are deliberately not a
          generic Setting form. They are entered under their consuming Agent
          in Launch Run → Customized → Agent Configuration. */}

      <div className="flex items-center justify-end gap-1.5 border-t border-hair pt-3">
        {error && <span className="mr-auto text-2xs text-coral-300">{error}</span>}
        <button
          type="button"
          onClick={() => void submit()}
          disabled={busy || !v.name.trim() || validationGaps.length > 0 || auxiliaryModelMissing || validationMetricMissing}
          title={
            !v.name.trim()
              ? "Name it first"
              : validationGaps.length
              ? `A named validation set needs its own ${validationGaps.join(", ")}`
              : auxiliaryModelMissing
              ? `${v.training_method} needs its Hugging Face auxiliary model id`
              : validationMetricMissing
              ? "Complete the Validation metric fields"
              : ""
          }
          className="rounded-md border border-brass-500/40 bg-brass-500/10 px-2.5 py-1 font-mono text-2xs text-brass-300 transition hover:bg-brass-500/20 disabled:opacity-50"
        >
          {busy ? "saving…" : "save"}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="rounded-md p-1 text-slate-600 transition hover:text-slate-300"
        >
          <X size={13} />
        </button>
      </div>
    </div>
  );
}
