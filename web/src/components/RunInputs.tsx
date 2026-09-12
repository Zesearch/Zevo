import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import useSWR from "swr";
import { ChevronRight, Upload, X } from "lucide-react";
import { FILES_ROOT, splitDatasetPath } from "../lib/format";
import { api } from "../lib/api";
import type { FileSetDTO, GenerationBackend, GpuProvider } from "../lib/api";
import { ThemedSelect } from "./ThemedSelect";

/**
 * The named inputs a run needs, as slots rather than a pile of attachments.
 *
 * A single "Attachments (optional)" dropzone only ever fed the FIRST file into
 * the request (as the training set); everything else was uploaded and then
 * ignored. A run needs specific things — data to train on, a test set to be
 * scored against, the script that scores it — and each has to land in its own
 * field, so each gets its own slot and says whether it is required.
 *
 * Everything still arrives via the same upload endpoint; these components only
 * decide which field the returned path goes into.
 */

/** `owner/name` with no extension: the shape of a hub id rather than a path. */
const isHubId = (v: string) => {
  const s = (v || "").trim();
  return /^[^/\s]+\/[^/\s]+$/.test(s) && !s.includes(".");
};

const hubBoxCls =
  "w-full min-w-0 rounded border border-hair bg-canvas px-2 py-1 font-mono text-2xs "
  + "text-slate-200 placeholder:text-slate-600 placeholder:opacity-100 focus:border-brass-500/50 focus:outline-none";
const hubLabelCls = "field-label mb-1 block";

const fieldCls =
  "w-full rounded-md border border-hair bg-canvas p-2.5 text-sm leading-relaxed text-slate-200 placeholder:text-slate-600 placeholder:opacity-100 focus:border-brass-500/50 focus:outline-none";
const completeFieldCls =
  "!border-brass-500/55 !bg-brass-500/[0.07] !text-brass-100";
export const requiredFieldStateCls = (_complete: boolean) =>
  "!border-hair !bg-canvas !text-slate-200 focus:!border-brass-500/50";

async function uploadFile(f: File): Promise<string> {
  const fd = new FormData();
  fd.append("file", f);
  const d = await api<{ path: string }>("/attachments", { method: "POST", body: fd });
  return d.path;
}

/** Strip the container prefix so a path reads the way the repo has it. */
function short(p: string): string {
  if (p.startsWith(FILES_ROOT + "/")) return p.slice(FILES_ROOT.length + 1);
  return p.startsWith("/app/") ? p.slice(5) : p;
}

/** Label + a required/optional tag. Under a "Required"/"Optional" heading the
 *  tag is noise, so `tag={false}` drops it. */
function SlotLabel({ label, required, tag = true }: { label: string; required?: boolean; tag?: boolean }) {
  return (
    <div className="mb-1 flex items-baseline gap-2">
      <span className="field-label">{label}</span>
      {tag && (
        <span className={`font-mono text-[0.62rem] ${required ? "text-brass-300" : "text-slate-600"}`}>
          {required ? "required" : "optional"}
        </span>
      )}
    </div>
  );
}

export type RunSummaryValue = {
  name: string;
  value: string;
  complete?: boolean;
  overridden?: boolean;
};

function SummaryColumn({
  title, values, required,
}: {
  title: string;
  values: RunSummaryValue[];
  required?: boolean;
}) {
  const completeCount = values.filter((value) => value.complete).length;
  const allComplete = completeCount === values.length;
  return (
    <div className="min-w-0 rounded-md border border-hair bg-white/[0.018] p-3">
      <div className="mb-1 flex items-center justify-between gap-3 px-1 pb-2">
        <span className="font-mono text-[0.64rem] uppercase tracking-[0.14em] text-brass-300">
          {title}
        </span>
        <span className={`font-mono text-[0.6rem] ${
          required ? allComplete ? "text-phosphor-300" : "text-coral-300" : "text-slate-600"
        }`}>
          {required ? `${completeCount}/${values.length}` : values.length}
        </span>
      </div>
      <table className="w-full table-fixed border-separate border-spacing-y-0.5 font-mono text-2xs">
        <tbody>
          {values.map(({ name, value, complete, overridden }) => (
            <tr key={name} className="group">
              <th className="w-[42%] rounded-l px-2 py-1.5 text-left font-normal text-slate-500 transition-colors group-hover:bg-white/[0.025]">
                {name}
              </th>
              <td className="rounded-r px-2 py-1.5 text-right transition-colors group-hover:bg-white/[0.025]">
                <span className="inline-flex max-w-full items-center justify-end gap-1.5">
                  {required && (
                    <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                      complete ? "bg-phosphor-400" : "bg-coral-400"
                    }`} />
                  )}
                  <span
                    className={`truncate ${
                      required
                        ? complete ? "text-slate-200" : "italic text-slate-600"
                        : overridden ? "text-brass-300" : "text-slate-400"
                    }`}
                    title={value}
                  >
                    {value}
                  </span>
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function RunSummaryTable({
  requiredValues, optionalValues,
}: {
  requiredValues: RunSummaryValue[];
  optionalValues: RunSummaryValue[];
}) {
  return (
    <div className="mt-3">
      <div className="grid gap-3 md:grid-cols-2">
        <SummaryColumn title="Required Values" values={requiredValues} required />
        <SummaryColumn title="Optional Values" values={optionalValues} />
      </div>
    </div>
  );
}

type PickedFile = { value: string; split: string; config: string };

/** Everything Files holds, as one flat searchable list.
 *
 *  Local files and hub entries together, because "which of my files is this"
 *  is one question: a dataset that fetches its training rows from the hub has
 *  them in Files like any other, and putting the two in different controls
 *  meant knowing which kind a thing was before you could look for it. */
type Option = PickedFile & { dataset: string; name: string; label: string; hint: string };

function catalogueOptions(datasets: FileSetDTO[]): Option[] {
  const out: Option[] = [];
  for (const d of datasets) {
    for (const f of d.files) {
      out.push({
        // The set's own container path from GET /files — never a rebuilt one,
        // so a data-root move cannot strand the picker again.
        value: `${d.path}/${f}`,
        split: "", config: "",
        dataset: d.name, name: f, label: `${d.name}/${f}`, hint: "",
      });
    }
    for (const r of d.source?.remote ?? []) {
      out.push({
        value: r.id,
        split: r.split ?? "",
        config: r.config ?? "",
        dataset: d.name,
        name: r.id.split("/").pop() || r.id,
        label: `${d.name}/${r.id.split("/").pop() || r.id}`,
        hint: `${[r.config, r.split || "train"].filter(Boolean).join("/")} · hf`,
      });
    }
  }
  return out;
}

/** Type to narrow, rather than scroll to find.
 *
 *  This was a `<select>` of every file in the catalogue, which works at a dozen
 *  and stops working at two hundred: a menu you have to read top to bottom is a
 *  worse index than a name you already half know. Matching is per typed word
 *  against the whole `dataset/file` label, so `med test` finds
 *  `medqa-usmle-eval/test.csv` without having to guess the order.
 */
function FilePicker({
  datasets, value, split = "", config = "", onPick, className = "", inputClassName = "",
  placeholder = "from Files",
}: {
  datasets: FileSetDTO[]; value: string; split?: string; config?: string;
  onPick: (p: PickedFile) => void; className?: string; inputClassName?: string;
  placeholder?: string;
}) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const box = useRef<HTMLDivElement>(null);

  const options = useMemo(() => catalogueOptions(datasets), [datasets]);
  const selected = options.find(
    (o) => o.value === value
      && (o.split || "") === (split || "") && (o.config || "") === (config || ""),
  );
  const matches = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    return options
      .filter((o) => words.every((w) => `${o.label} ${o.hint}`.toLowerCase().includes(w)))
      .slice(0, 40);
  }, [options, q]);

  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => {
      if (!box.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, [open]);

  const choose = (o: PickedFile) => { onPick(o); setOpen(false); setQ(""); };

  return (
    <div ref={box} className={`relative ${className}`}>
      <input
        value={open ? q : (selected?.label ?? "")}
        onChange={(e) => { setQ(e.target.value); setActive(0); setOpen(true); }}
        onFocus={() => { setQ(""); setActive(0); setOpen(true); }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") { e.preventDefault(); setActive((i) => Math.min(i + 1, matches.length - 1)); }
          else if (e.key === "ArrowUp") { e.preventDefault(); setActive((i) => Math.max(i - 1, 0)); }
          else if (e.key === "Enter" && matches[active]) { e.preventDefault(); choose(matches[active]); }
          else if (e.key === "Escape") setOpen(false);
        }}
        placeholder={placeholder}
        spellCheck={false}
        className={`${fieldCls} font-mono ${inputClassName}`}
      />
      {open && (
        <div className="absolute z-30 mt-1 max-h-56 w-full overflow-auto rounded-md border border-hair bg-raised p-1 shadow-lg">
          {matches.length === 0 && (
            <div className="px-2 py-1.5 font-mono text-2xs text-slate-500">
              nothing in Files matches that
            </div>
          )}
          {matches.map((o, i) => (
            <Fragment key={`${o.value}:${o.split}:${o.config}`}>
              {/* One heading per dataset, the way the old `<optgroup>` read:
                  the files of one bundle belong together, and repeating the
                  folder on every line spends the width the name needs. The
                  rows stay one flat list underneath so the arrow keys walk
                  across groups without stopping on the headings. */}
              {(i === 0 || matches[i - 1].dataset !== o.dataset) && (
                <div className="px-2 pb-0.5 pt-1.5 font-mono text-[0.58rem] uppercase tracking-[0.1em] text-slate-500">
                  {o.dataset}
                </div>
              )}
              <button
                type="button"
                onMouseDown={(e) => { e.preventDefault(); choose(o); }}
                onMouseEnter={() => setActive(i)}
                className={`flex w-full items-baseline justify-between gap-2 rounded px-2 py-1 pl-3 text-left font-mono text-2xs ${
                  i === active ? "bg-brass-500/15 text-brass-200" : "text-slate-300"
                }`}
              >
                <span className="min-w-0 truncate">{o.name}</span>
                {o.hint && <span className="shrink-0 text-slate-500">{o.hint}</span>}
              </button>
            </Fragment>
          ))}
        </div>
      )}
    </div>
  );
}

/** The hub route: type the id.
 *
 *  No list of ids used before. The ones already in Files are in the Files
 *  picker beside this, with the split they were registered under; offering
 *  them again here would be the same choice twice, and the second copy would
 *  quietly drop the split.
 */
function HubInput({
  value, onChange, className = "",
}: { value: string; onChange: (v: string) => void; className?: string }) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder="or owner/name on HuggingFace"
      spellCheck={false}
      className={`${fieldCls} font-mono ${className}`}
    />
  );
}

/** The upload route, as a box beside the picker rather than a bare button —
 *  the two ways in should look like two ways in. Shows what was uploaded. */
function UploadBox({
  onChange, className = "", buttonClassName = "", label = "Upload files",
}: {
  onChange: (v: string) => void; className?: string; buttonClassName?: string;
  label?: string;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  // Opened, not assumed. The button used to go straight to the system file
  // dialog, which is one way in and hides the other — you cannot discover that
  // dropping works from a control that never mentions it.
  const [open, setOpen] = useState(false);
  const [dragDepth, setDragDepth] = useState(0);

  async function pick(f: File | undefined) {
    if (!f) return;
    setBusy(true);
    setErr("");
    try {
      onChange(await uploadFile(f));
      setOpen(false);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className={className}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={busy}
        className={`btn !text-[13px] disabled:opacity-50 ${buttonClassName}`}
      >
        <Upload size={12} /> {busy ? "Uploading…" : label}
      </button>
      {open && (
        <div
          onDragEnter={(e) => { e.preventDefault(); setDragDepth((d) => d + 1); }}
          onDragOver={(e) => e.preventDefault()}
          onDragLeave={() => setDragDepth((d) => Math.max(0, d - 1))}
          onDrop={(e) => {
            e.preventDefault();
            setDragDepth(0);
            void pick(e.dataTransfer.files?.[0]);
          }}
          className="mt-2 space-y-2 rounded-md border border-hair bg-raised/40 p-3"
        >
          <div
            className={`flex flex-col items-center justify-center gap-1 rounded-md border border-dashed py-5 font-mono text-2xs transition ${
              dragDepth > 0
                ? "border-brass-500/60 bg-brass-500/10 text-brass-300"
                : "border-hair/70 text-slate-500"
            }`}
          >
            <Upload size={16} />
            {dragDepth > 0 ? "Release to upload" : "Drag files here"}
          </div>
          <div className="flex items-center justify-center gap-2">
            <span className="font-mono text-2xs text-slate-600">or</span>
            <button
              type="button"
              onClick={() => input.current?.click()}
              disabled={busy}
              className="btn !text-[13px] disabled:opacity-50"
            >
              Select from your computer
            </button>
          </div>
        </div>
      )}
      <input
        ref={input}
        type="file"
        className="hidden"
        onChange={(e) => { void pick(e.target.files?.[0]); e.target.value = ""; }}
      />
      {err && <div className="mt-1 text-2xs text-coral-300">{err}</div>}
    </div>
  );
}

/** What was chosen, and the way to un-choose it.
 *
 *  Shown INSTEAD of the routes in, not beside them: three empty boxes next to a
 *  filled one reads as three more things to fill, and the whole reason these
 *  fields have three routes is that any one of them is enough. */
function Chosen({
  value, onChange, note = "", className = "",
}: { value: string; onChange: (v: string) => void; note?: string; className?: string }) {
  const inCatalogue = splitDatasetPath(value);
  return (
    <div className={`flex items-center gap-2 rounded-md border border-hair bg-canvas px-2.5 py-2 ${className}`}>
      <span className="min-w-0 flex-1 truncate font-mono text-sm text-slate-100" title={value}>
        {inCatalogue ? inCatalogue.label : short(value)}
      </span>
      {note && <span className="shrink-0 font-mono text-2xs text-slate-500">{note}</span>}
      <button
        type="button"
        onClick={() => onChange("")}
        title="Choose something else"
        className="shrink-0 rounded p-1 text-slate-500 transition hover:text-coral-300"
      >
        <X size={13} />
      </button>
    </div>
  );
}

/** One required-or-optional file: two routes in — the catalogue, or an upload.
 *  Whatever the value ends up being is echoed underneath, so a path that came
 *  from neither (an older task, a hand-edited row) is still visible. */
export function FileSlot({
  label, value, onChange, required = false, hint = "", tag = true,
  controlClassName = "",
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  required?: boolean;
  hint?: string;
  tag?: boolean;
  controlClassName?: string;
}) {
  const { data: datasets = [] } = useSWR<FileSetDTO[]>("/api/files");

  return (
    <div>
      <SlotLabel label={label} required={required} tag={tag} />
      {/* Filled, the routes in go away. Leaving them on screen beside a chosen
          file made the row read as several things still to answer, when any one
          of them was the whole answer. */}
      {value ? (
        <Chosen value={value} onChange={onChange} className={controlClassName} />
      ) : (
        <div className="flex flex-wrap items-start gap-2">
          <FilePicker
            datasets={datasets} value=""
            onPick={(o) => onChange(o.value)}
            className="min-w-0 flex-1"
            inputClassName={controlClassName}
            placeholder={hint || "from Files"}
          />
          <UploadBox
            onChange={onChange}
            className="shrink-0"
            buttonClassName={controlClassName}
          />
        </div>
      )}
    </div>
  );
}

/**
 * Training data, from any of the three places it can come from: the catalogue,
 * HuggingFace, or a file you have right here and nobody has registered yet.
 */
export function TrainingDataField({
  value, onChange, label = "Training data", required = false, note = "", tag = true,
  split, onSplit, config, onConfig, splitPlaceholder = "train",
}: {
  value: string;
  onChange: (v: string) => void;
  label?: string;
  required?: boolean;
  note?: string;
  tag?: boolean;
  /** Pass the pair to offer split/config when the value is a hub id. A repo is
   *  not one table, so an id on its own does not name a set. */
  split?: string;
  onSplit?: (v: string) => void;
  config?: string;
  onConfig?: (v: string) => void;
  splitPlaceholder?: string;
  showHints?: boolean;
}) {
  const { data: datasets = [] } = useSWR<FileSetDTO[]>("/api/files");

  return (
    <div>
      <SlotLabel label={label} required={required} tag={tag} />
      {/* Three routes, three boxes: the catalogue, the hub, this machine. Two
          rows, because three on one line left every box too narrow to read.
          All three go away once one of them has answered — see FileSlot. */}
      <div className="space-y-2">
        {value ? (
          <Chosen
            value={value}
            onChange={(v) => { onChange(v); onSplit?.(""); onConfig?.(""); }}
            note={isHubId(value) ? "hf" : ""}
          />
        ) : (
          <>
            <div className="flex items-start gap-2">
              <FilePicker
                datasets={datasets}
                value=""
                split={split}
                config={config}
                onPick={(o) => { onChange(o.value); onSplit?.(o.split); onConfig?.(o.config); }}
                className="min-w-0 flex-1"
                placeholder={note || "from Files"}
              />
              <HubInput
                value=""
                onChange={(v) => { onChange(v); if (!v) { onSplit?.(""); onConfig?.(""); } }}
                className="min-w-0 flex-1"
              />
            </div>
            {/* An upload has no slice; clearing them stops a stale split
                pointing the run at rows it was never told to use. */}
            <UploadBox
              onChange={(v) => { onChange(v); onSplit?.(""); onConfig?.(""); }}
            />
          </>
        )}
        {onSplit && onConfig && isHubId(value) && (
          <div className="grid grid-cols-2 gap-2">
            <label className="block">
              <span className={hubLabelCls}>Split</span>
              <input
                className={hubBoxCls} value={split ?? ""}
                onChange={(e) => onSplit(e.target.value)}
                list="zevo-hf-splits" placeholder={splitPlaceholder}
              />
            </label>
            <label className="block">
              <span className={hubLabelCls}>Config</span>
              <input
                className={hubBoxCls} value={config ?? ""}
                onChange={(e) => onConfig(e.target.value)}
                placeholder="only if the repo has several"
              />
            </label>
            <datalist id="zevo-hf-splits">
              <option value="train" />
              <option value="validation" />
              <option value="test" />
            </datalist>
          </div>
        )}
      </div>
      {/* No echo of the resolved path/id: the three boxes above already show
          what is selected, so repeating it in grey mono was the same fact
          twice. The hint stays, below the boxes, at the same size as every
          other field's hint — and only while the field is empty, since every
          one of these notes says what happens if you leave it that way. */}
    </div>
  );
}

export type RunInputValues = {
  /** Held-out Test evaluator selected for this Run. */
  metricType: "" | "builtin" | "custom";
  /** Required name of the Test evaluator value. */
  metric: string;
  /** Frozen custom evaluator used only for held-out Test. */
  evaluationScript: string;
  /** Test score direction. A predefined Task may supply the initial value. */
  metricDirection: "" | "max" | "min";
  /** Independent Validation evaluator used only with a supplied Validation set. */
  validationMetricType: "" | "builtin" | "custom";
  validationMetric: string;
  validationEvaluationScript: string;
  validationMetricDirection: "" | "max" | "min";
  dataset: string;
  testSet: string;
  /** Comma-separated ground-truth column names of the test set. */
  answerFields: string;
  validationSet: string;
  /** Same, for the validation set. Required when validationSet is named. */
  validationAnswerFields: string;
  /** Only when validationSet is a hub id: which slice of the repo. */
  validationSplit: string;
  validationConfig: string;
  validationSampleSubmission: string;
  /** Only when dataset is a hub id: which slice of the repo to train on. */
  datasetSplit: string;
  datasetConfig: string;
  /** Acquisition brief used only when no training dataset is pinned. */
  dataQuery: string;
  testSampleSubmission: string;
  baseModel: string;
  /** Natural-language guidance for an unpinned base-model search. */
  modelQuery: string;
  /** The training method, e.g. lora_sft. Empty = Zevo picks — which is one of
   *  the three decisions the autonomy level counts. */
  trainingMethod: string;
  /** Natural-language guidance for an unpinned method search. */
  methodQuery: string;
  /** Auxiliary model ids required by method-specific Train Skills. This
   * release accepts Hugging Face owner/model ids only. */
  teacherModel: string;
  rewardModel: string;
  usePeft: "" | "true" | "false";
  promptFraming: string;
  systemPrompt: string;
  lossObjectiveConfig: string;
  inferenceConfig: string;
  decodingStrategy: "" | "greedy" | "sampling";
  maxNewTokens: string;
  temperature: string;
  topP: string;
  topK: string;
  repetitionPenalty: string;
  seed: string;
  gpuProvider: GpuProvider | "";
  /** When gpuProvider === "cloud", which cloud to rent on. Empty = deployment default. */
  cloudBackend: "" | "vastai" | "lambda";
  /** Verified SSH profile selected for a cluster/instance run. */
  sshHostId: string;
  /** Maximum GPUs the Run may use at once. Empty means no upper bound;
   *  Infrastructure selects a concrete positive count. */
  numGpus: string;
  generation_backend: GenerationBackend | "";
  iterations: string;
  budget: string;
  timeLimitHours: string;
  queueWaitHours: string;
  stopThreshold: string;
};

export const EMPTY_RUN_INPUTS: RunInputValues = {
  metricType: "", metric: "", evaluationScript: "",
  metricDirection: "",
  validationMetricType: "", validationMetric: "",
  validationEvaluationScript: "", validationMetricDirection: "",
  dataset: "", testSet: "", answerFields: "", validationSet: "", validationAnswerFields: "",
  validationSplit: "", validationConfig: "",
  validationSampleSubmission: "",
  datasetSplit: "", datasetConfig: "", dataQuery: "",
  testSampleSubmission: "",
  baseModel: "", modelQuery: "", trainingMethod: "", methodQuery: "",
  teacherModel: "", rewardModel: "", usePeft: "",
  promptFraming: "", systemPrompt: "",
  lossObjectiveConfig: "", inferenceConfig: "",
  decodingStrategy: "",
  maxNewTokens: "", temperature: "", topP: "", topK: "",
  repetitionPenalty: "", seed: "",
  // Blank means the Run-owned server defaults shown in the summary below.
  gpuProvider: "",
  cloudBackend: "",
  sshHostId: "",
  generation_backend: "",
  // Blank resolves once to one GPU on the Run.
  numGpus: "",
  // Empty IS the value here: both are uncapped unless a number is typed, which
  // is what the placeholder says.
  iterations: "", budget: "", timeLimitHours: "", queueWaitHours: "", stopThreshold: "",
};

/** The effective Validation scorer shown and sent by both launch modes. */
export function validationContractFromInputs(inputs: RunInputValues) {
  const independent = !!inputs.validationSet.trim();
  const metricType = independent ? inputs.validationMetricType : "";
  return {
    independent,
    metricType,
    metric: independent ? inputs.validationMetric.trim() : "",
    metricDirection: independent ? inputs.validationMetricDirection : "",
    evaluationScript: metricType === "custom"
      ? inputs.validationEvaluationScript.trim()
      : "",
    answerFields: independent ? inputs.validationAnswerFields : "",
    sampleSubmission: independent ? inputs.validationSampleSubmission.trim() : "",
  };
}

/** A plain optional text field, labelled like the slots around it. */
export function TextField({
  label, value, onChange, placeholder = "", mono = true, hint = "", required = false,
}: {
  label: string; value: string; onChange: (v: string) => void;
  placeholder?: string; mono?: boolean; hint?: string; required?: boolean;
}) {
  return (
    <div>
      <SlotLabel label={label} tag={false} />
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder || hint}
        spellCheck={false}
        className={`${fieldCls} ${mono ? "font-mono" : ""} ${
          required ? requiredFieldStateCls(Boolean(value.trim())) : value.trim() ? completeFieldCls : ""
        }`}
      />
    </div>
  );
}

/** Said in two places, so written once. The owner is not decoration: the model
 *  is downloaded by this exact string, and `Qwen3-0.6B` without it resolves to
 *  nothing — a failure that lands on the GPU box, minutes into a run. */
export const MODEL_ID_HINT =
  "the full HuggingFace id, owner included: Qwen/Qwen3-0.6B, not Qwen3-0.6B";

export const BUILTIN_METRICS = [
  "accuracy", "exact_match", "f1", "token_f1", "bleu", "rouge_l",
];

export function methodConfigFromInputs(inputs: RunInputValues): Record<string, unknown> {
  const config: Record<string, unknown> = {};
  if (inputs.trainingMethod === "gkd" && inputs.teacherModel.trim()) {
    config.teacher_model = inputs.teacherModel.trim();
  }
  if (inputs.trainingMethod === "online_dpo" && inputs.rewardModel.trim()) {
    config.reward_model = inputs.rewardModel.trim();
  }
  if (inputs.usePeft) config.use_peft = inputs.usePeft === "true";
  return config;
}

function parseObject(value: string, label: string): Record<string, unknown> {
  if (!value.trim()) return {};
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    throw new Error(`${label} must be a JSON object.`);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON object.`);
  }
  return parsed as Record<string, unknown>;
}

export function contractPreferencesFromInputs(inputs: RunInputValues) {
  const decoding_config: Record<string, number | string> = {};
  if (inputs.decodingStrategy) {
    decoding_config.decoding_strategy = inputs.decodingStrategy;
  }
  const numeric: Array<[string, string]> = [
    ["max_new_tokens", inputs.maxNewTokens],
    ["temperature", inputs.temperature],
    ["top_p", inputs.topP],
    ["top_k", inputs.topK],
    ["repetition_penalty", inputs.repetitionPenalty],
    ["seed", inputs.seed],
  ];
  for (const [key, raw] of numeric) {
    if (!raw.trim()) continue;
    const value = Number(raw);
    if (!Number.isFinite(value)) throw new Error(`${key.replaceAll("_", " ")} must be a finite number.`);
    decoding_config[key] = value;
  }
  return {
    prompt_framing: inputs.promptFraming.trim(),
    system_prompt: inputs.systemPrompt.trim(),
    loss_objective_config: parseObject(inputs.lossObjectiveConfig, "Loss objective config"),
    inference_config: parseObject(inputs.inferenceConfig, "Inference config"),
    decoding_config,
  };
}


/**
 * The validation set — the one field that can be a path, a hub id, or an
 * upload, and that needs a follow-up question when it is a hub id.
 *
 * One component rather than one per form: Launch run, a task's settings and a
 * Customized Pipeline all ask for the same thing, and three copies of a control this
 * fiddly drift apart on the first change. Which slice of a repo you tune
 * against is not a detail that should be sayable in one form and not another.
 */
export function ValidationSetField({
  value, onChange, split, onSplit, config, onConfig,
  answerFields, onAnswerFields,
  sampleSubmission, onSampleSubmission, required = [], note = "", tag = true,
  showHints = true,
}: {
  value: string;
  onChange: (v: string) => void;
  split: string;
  onSplit: (v: string) => void;
  config: string;
  onConfig: (v: string) => void;
  /** Comma-separated ground-truth columns. Omit the pair to hide the input. */
  answerFields?: string;
  onAnswerFields?: (v: string) => void;
  /** Columns inference must emit. Required when a validation set is named. */
  sampleSubmission?: string;
  onSampleSubmission?: (v: string) => void;
  /** Which of the three are still blank while a set IS named — the caller
   *  decides whether that is allowed; this only marks them. */
  required?: string[];
  note?: string;
  /** False where the surrounding group already says the field is optional. */
  tag?: boolean;
  showHints?: boolean;
}) {
  return (
    <div>
      <TrainingDataField
        label="Validation set"
        value={value}
        onChange={onChange}
        tag={tag}
        split={split}
        onSplit={onSplit}
        config={config}
        onConfig={onConfig}
        // Blank derives Validation from Test before the run starts.
        splitPlaceholder="validation"
        note={note || "empty = 20% of Test (at least 200 rows); otherwise upload Validation"}
        showHints={showHints}
      />
      {/* The other three sit beside the set they describe rather than at the
          far end of the form: they are facts ABOUT that file. Which is why
          they wait for one to be named — with the field still blank there is
          no file for them to be about, and what the run does instead (move
          20% of Test to Validation and hold out the remaining 80%) is one
              sentence, said once, above.

          One hint under the three rather than one each: they fall back the
          same way, and repeating it three times says nothing the first line
          did not. */}
      {value.trim() && (
        <>
          {/* The same TextField the TEST answer fields use. It was a bare input
              on `hubBoxCls` — the compact style meant for the split/config
              boxes that sit inline beside a HuggingFace id — so this one field
              rendered a size smaller than the two FileSlots under it and than
              its opposite number on the test side, for no reason a reader could
              infer. It asks for the same kind of thing; it should look it. */}
          {onAnswerFields && (
            <div className="mt-2">
              <TextField
                label="Answer fields"
                value={answerFields ?? ""}
                onChange={onAnswerFields}
                required
                placeholder=""
                hint="Columns containing the validation ground truth."
              />
            </div>
          )}
          {onSampleSubmission && (
            <div className="mt-2">
              <FileSlot
                label="Sample submission"
                value={sampleSubmission ?? ""}
                onChange={onSampleSubmission}
                required
                tag={false}
                hint="Defines the required validation prediction format."
              />
            </div>
          )}
          {showHints && required.length > 0 && (
            // A named set with the others blank is the pairing that silently
            // ships the answers to inference; say so where it is being made.
            <p className="mt-1 text-2xs text-coral-300">
              a validation set of your own needs its own {required.join(", ")};
              Validation never falls back to the held-out Test contract
            </p>
          )}
        </>
      )}
    </div>
  );
}

/** An optional choice with a named default. */
export function ChoiceField({
  label, value, onChange, options, disabled = false, hint = "", required = false,
}: {
  label: string; value: string; onChange: (v: string) => void;
  options: [string, string][]; disabled?: boolean; hint?: string; required?: boolean;
}) {
  // The light hint stays in the closed control. The shared themed list receives
  // only real choices, so no browser can expose the hint as a menu item.
  const choices = options.filter(([v]) => v !== "");
  const known = choices.some(([v]) => v === value);
  const selected = known ? value : "";
  return (
    <div>
      <SlotLabel label={label} tag={false} />
      <ThemedSelect
        value={selected}
        onChange={onChange}
        options={choices.map(([v, t]) => ({ value: v, label: t }))}
        placeholder={hint}
        ariaLabel={label}
        disabled={disabled}
        buttonClassName={`${fieldCls} h-11 font-mono ${
          required ? requiredFieldStateCls(Boolean(selected)) : selected ? completeFieldCls : "!text-slate-600"
        }`}
      />
    </div>
  );
}

/** A number field where blank means "no limit", said out loud. */
export function NumberField({
  label, value, onChange, placeholder = "", min, max, step, hint = "",
  alignLabel = false,
}: {
  label: string; value: string; onChange: (v: string) => void; placeholder?: string;
  min?: number; max?: number; step?: number | "any"; hint?: string;
  alignLabel?: boolean;
}) {
  const acceptsDecimal = step === "any";
  return (
    <div className="min-w-0">
      {alignLabel ? (
        <div className="mb-1 min-h-[1.05rem]">
          <span className="field-label whitespace-nowrap">{label}</span>
        </div>
      ) : (
        <SlotLabel label={label} tag={false} />
      )}
      <input
        type={acceptsDecimal ? "text" : "number"}
        inputMode={acceptsDecimal ? "decimal" : undefined}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder || hint}
        className={`${fieldCls} h-11 font-mono ${value.trim() ? completeFieldCls : ""}`}
      />
    </div>
  );
}

export function LimitField({
  label, value, onChange, prefix = "", hint = "",
}: { label: string; value: string; onChange: (v: string) => void; prefix?: string; hint?: string }) {
  return (
    <div className="min-w-0">
      <div className="mb-1 min-h-[1.05rem]">
        <span className="field-label whitespace-nowrap">{label}</span>
      </div>
      <div className="relative">
        {prefix && (
          <span className="pointer-events-none absolute inset-y-0 left-2.5 flex items-center font-mono text-sm text-slate-500">
            {prefix}
          </span>
        )}
        <input
          type="number"
          min={0}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={hint}
          className={`${fieldCls} h-11 font-mono ${prefix ? "!pl-7" : ""} ${value.trim() ? completeFieldCls : ""}`}
        />
      </div>
    </div>
  );
}

/**
 * Several files, added the same three ways the single slots offer. The old
 * dropzone could only take an upload, so a file already in the catalogue had to
 * be uploaded a second time to be attached.
 */
export function MultiFileSlot({
  label, values, onChange, hint = "",
}: { label: string; values: string[]; onChange: (v: string[]) => void; hint?: string }) {
  const { data: datasets = [] } = useSWR<FileSetDTO[]>("/api/files");
  const [draft, setDraft] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [showUpload, setShowUpload] = useState(false);
  const [dragDepth, setDragDepth] = useState(0);

  const add = (v: string) => {
    const p = v.trim();
    if (!p || values.includes(p)) return;
    onChange([...values, p]);
    setDraft("");
  };

  async function upload(f: File | undefined) {
    if (!f) return;
    setBusy(true);
    setErr("");
    try {
      add(await uploadFile(f));
      setShowUpload(false);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <SlotLabel label={label} tag={false} />
      <div className="flex flex-wrap items-center gap-2">
        <FilePicker datasets={datasets} value="" onPick={(o) => add(o.value)} className="max-w-[14rem]" />
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); add(draft); } }}
          placeholder={hint}
          spellCheck={false}
          className={`${fieldCls} min-w-0 flex-1 font-mono`}
        />
        <button
          type="button"
          onClick={() => setShowUpload((v) => !v)}
          disabled={busy}
          className="btn shrink-0 !text-[13px] disabled:opacity-50"
        >
          <Upload size={12} /> {busy ? "Uploading…" : "Upload files"}
        </button>
        <input
          ref={input}
          type="file"
          className="hidden"
          onChange={(e) => { void upload(e.target.files?.[0]); e.target.value = ""; }}
        />
      </div>
      {showUpload && (
        <div
          onDragEnter={(e) => { e.preventDefault(); setDragDepth((d) => d + 1); }}
          onDragOver={(e) => e.preventDefault()}
          onDragLeave={() => setDragDepth((d) => Math.max(0, d - 1))}
          onDrop={(e) => {
            e.preventDefault();
            setDragDepth(0);
            void upload(e.dataTransfer.files?.[0]);
          }}
          className="mt-2 space-y-2 rounded-md border border-hair bg-raised/40 p-3"
        >
          <div className={`flex flex-col items-center justify-center gap-1 rounded-md border border-dashed py-5 font-mono text-2xs transition ${
            dragDepth > 0
              ? "border-brass-500/60 bg-brass-500/10 text-brass-300"
              : "border-hair/70 text-slate-500"
          }`}>
            <Upload size={16} />
            {dragDepth > 0 ? "Release to upload" : "Drag files here"}
          </div>
          <div className="flex items-center justify-center gap-2">
            <span className="font-mono text-2xs text-slate-600">or</span>
            <button type="button" onClick={() => input.current?.click()} className="btn !text-[13px]">
              Select from your computer
            </button>
          </div>
        </div>
      )}
      {values.length > 0 && (
        <div className="mt-1.5 space-y-1">
          {values.map((v) => (
            <div key={v} className="flex items-center justify-between gap-2">
              <span className="min-w-0 truncate font-mono text-[0.62rem] text-slate-400">{short(v)}</span>
              <button
                type="button"
                onClick={() => onChange(values.filter((x) => x !== v))}
                title="Remove"
                className="shrink-0 rounded p-1 text-slate-500 transition hover:text-coral-300"
              >
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
      )}
      {err && <div className="mt-1 text-2xs text-coral-300">{err}</div>}
    </div>
  );
}

export type ComputeTargetOption = {
  value: string;
  label: string;
  gpuProvider: GpuProvider;
  cloudBackend: "" | "vastai" | "lambda";
  sshHostId: string;
};

export function computeTargetValue({
  gpuProvider, cloudBackend, sshHostId,
}: Pick<RunInputValues, "gpuProvider" | "cloudBackend" | "sshHostId">): string {
  if (gpuProvider === "cloud") return cloudBackend ? `cloud:${cloudBackend}` : "";
  if (sshHostId) return `connection:${sshHostId}`;
  return gpuProvider ? `environment:${gpuProvider}` : "";
}

/** Available picker values plus the concrete installation default. */
export function useComputeTargets() {
  const { data: hosts } = useSWR<Array<{
    id: string;
    name: string;
    category: "cluster" | "instance";
    host: string;
    status: string;
  }>>("/api/hardware/ssh");
  const { data: settings } = useSWR<{
    entries: Array<{ name: string; present: boolean; preview: string }>;
    ssh_connections: Array<{
      id: "cluster" | "instance";
      label: string;
      configured: boolean;
      status: string;
    }>;
  }>("/api/settings");

  const targets = useMemo<ComputeTargetOption[]>(() => {
    const keyPresent = (key: string) =>
      settings?.entries.some((entry) => entry.name === key && entry.present) ?? false;
    return [
      ...(keyPresent("VASTAI_API_KEY") ? [{
        value: "cloud:vastai", label: "Vast.ai", gpuProvider: "cloud" as const,
        cloudBackend: "vastai" as const, sshHostId: "",
      }] : []),
      ...(keyPresent("LAMBDA_API_KEY") ? [{
        value: "cloud:lambda", label: "Lambda.ai", gpuProvider: "cloud" as const,
        cloudBackend: "lambda" as const, sshHostId: "",
      }] : []),
      ...(settings?.ssh_connections ?? [])
        .filter((connection) => connection.configured)
        .map((connection) => ({
          value: `environment:${connection.id}`,
          label: connection.label,
          gpuProvider: connection.id,
          cloudBackend: "" as const,
          sshHostId: "",
        })),
      ...(hosts ?? [])
        .filter((host) => host.status === "verified")
        .map((host) => ({
          value: `connection:${host.id}`,
          label: host.name || host.host,
          gpuProvider: host.category,
          cloudBackend: "" as const,
          sshHostId: host.id,
        })),
    ];
  }, [hosts, settings]);

  const configuredDefault = settings?.entries.find(
    (entry) => entry.name === "ZEVO_DEFAULT_COMPUTE" && entry.present,
  )?.preview ?? "";
  const defaultTarget = targets.find((target) => target.value === configuredDefault);
  return {
    targets,
    defaultTarget,
    loading: settings === undefined || hosts === undefined,
  };
}

/** Unified GPU picker: cloud API backends plus verified Cluster/Instance SSH
 *  profiles. A blank Run selection displays the concrete Settings default. */
export function BackendPicker({
  gpuProvider, cloudBackend, sshHostId, onChange,
}: {
  gpuProvider: GpuProvider | "";
  cloudBackend: "" | "vastai" | "lambda";
  sshHostId: string;
  onChange: (patch: Partial<RunInputValues>) => void;
}) {
  const compute = useComputeTargets();
  const explicitValue = computeTargetValue({ gpuProvider, cloudBackend, sshHostId });
  const effectiveValue = explicitValue || compute.defaultTarget?.value || "";

  return (
    <div>
      <ChoiceField
        label="GPU backend"
        value={effectiveValue}
        options={compute.targets.map((target) => [target.value, target.label])}
        required
        hint={compute.loading ? "Loading available compute…" : "Choose a GPU backend or set a default in Settings."}
        onChange={(value) => {
          const target = compute.targets.find((item) => item.value === value);
          onChange(target ? {
            gpuProvider: target.gpuProvider,
            cloudBackend: target.cloudBackend,
            sshHostId: target.sshHostId,
          } : { gpuProvider: "", cloudBackend: "", sshHostId: "" });
        }}
      />
      {!explicitValue && compute.defaultTarget && (
        <p className="mt-1.5 inline-flex items-center gap-1.5 rounded-full border border-hair bg-white/[0.025] px-2 py-1 font-mono text-2xs text-slate-400">
          <span>Default from Settings: <span className="text-slate-200">{compute.defaultTarget.label}</span></span>
        </p>
      )}
      {!compute.loading && !effectiveValue && (
        <p className="mt-1.5 inline-flex items-center gap-1.5 rounded-full border border-hair bg-white/[0.025] px-2 py-1 font-mono text-2xs text-slate-400">
          <span>No default is configured. Choose a backend here or set one in Settings.</span>
        </p>
      )}
    </div>
  );
}

/** The three training-side decisions shared by Standard and Auto.
 *
 * Auto changes who supplies the Test contract; it does not create a second
 * ownership model for optimization. Keeping these controls in one component
 * makes the L1–L4 ladder identical in both launch modes.
 */
export function TrainingSetupFields({
  inputs,
  onChange,
}: {
  inputs: Pick<
    RunInputValues,
    | "dataset" | "datasetSplit" | "datasetConfig" | "dataQuery"
    | "baseModel" | "modelQuery" | "trainingMethod" | "methodQuery"
    | "teacherModel" | "rewardModel" | "usePeft"
  >;
  onChange: (patch: Partial<RunInputValues>) => void;
}) {
  const {
    dataset, datasetSplit, datasetConfig, dataQuery,
    baseModel, modelQuery, trainingMethod, methodQuery,
    teacherModel, rewardModel, usePeft,
  } = inputs;

  return (
    <div className="space-y-3">
      <TrainingDataField
        value={dataset} tag={false}
        onChange={(value) => onChange({ dataset: value })}
        split={datasetSplit}
        onSplit={(value) => onChange({ datasetSplit: value })}
        config={datasetConfig}
        onConfig={(value) => onChange({ datasetConfig: value })}
        note="Examples used for training; blank lets Zevo acquire and prepare them."
        showHints={false}
      />
      <TextField
        label="Data query" value={dataQuery}
        onChange={(value) => onChange({ dataQuery: value })}
        placeholder=""
        hint="Guides data acquisition when no training data is selected."
      />
      <TextField
        label="Base model" value={baseModel}
        onChange={(value) => onChange({ baseModel: value })}
        placeholder=""
        hint="Starting model to improve; blank lets Zevo choose."
      />
      <TextField
        label="Model query" value={modelQuery}
        onChange={(value) => onChange({ modelQuery: value })}
        placeholder=""
        hint="Optional guidance for Zevo's model selection; it does not pin an exact model."
      />
      <ChoiceField
        label="Training method" value={trainingMethod}
        onChange={(value) => onChange({ trainingMethod: value })}
        options={[
          ["lora_sft", "lora_sft"],
          ["full_sft", "full_sft"], ["cpo", "cpo"], ["dpo", "dpo"],
          ["gkd", "gkd"], ["grpo", "grpo"], ["kto", "kto"],
          ["online_dpo", "online_dpo"], ["orpo", "orpo"],
          ["rft", "rft"], ["rloo", "rloo"],
        ]}
        hint="Optimization method used to train the model; blank lets Zevo choose."
      />
      <TextField
        label="Method query" value={methodQuery}
        onChange={(value) => onChange({ methodQuery: value })}
        placeholder=""
        hint="Optional guidance for Zevo's method selection and branch order; it does not pin an exact method."
      />
      {trainingMethod === "gkd" && (
        <TextField
          label="Teacher model"
          value={teacherModel}
          onChange={(value) => onChange({ teacherModel: value })}
          required
          hint="Frozen teacher used by GKD; enter its Hugging Face owner/model id."
        />
      )}
      {trainingMethod === "online_dpo" && (
        <TextField
          label="Reward model"
          value={rewardModel}
          onChange={(value) => onChange({ rewardModel: value })}
          required
          hint="Reward model used to rank Online DPO responses; enter its Hugging Face owner/model id."
        />
      )}
      {[
        "cpo", "dpo", "gkd", "grpo", "kto", "online_dpo", "orpo", "rft", "rloo",
      ].includes(trainingMethod) && (
        <ChoiceField
          label="PEFT"
          value={usePeft}
          onChange={(value) => onChange({ usePeft: value as "" | "true" | "false" })}
          options={[["true", "Use PEFT"], ["false", "Full parameters"]]}
          hint="Blank uses the Train Skill default."
        />
      )}
    </div>
  );
}

/** The block of named inputs a full-pipeline run is scored on. */
export function RunInputs({
  metricType, metric, evaluationScript, metricDirection,
  validationMetricType, validationMetric, validationEvaluationScript,
  validationMetricDirection,
  dataset, testSet, answerFields, validationSet, validationAnswerFields,
  validationSplit, validationConfig, validationSampleSubmission,
  datasetSplit, datasetConfig, dataQuery,
  testSampleSubmission,
  baseModel, modelQuery, trainingMethod, methodQuery,
  teacherModel, rewardModel, usePeft,
  gpuProvider, cloudBackend, sshHostId, numGpus, generation_backend,
  iterations, budget, timeLimitHours, queueWaitHours, stopThreshold,
  onChange, extra, requiredPrefix, optionalPrefix, beforeChecklist,
  requiredMissing = [], requiredPrefixValues = [],
}: RunInputValues & {
  onChange: (patch: Partial<RunInputValues>) => void;
  /** Slot for anything the named fields don't cover — rendered under Optional,
   *  because a loose attachment is optional by definition. */
  extra?: React.ReactNode;
  requiredPrefix?: React.ReactNode;
  optionalPrefix?: React.ReactNode;
  /** Mode-specific controls that should remain above the shared checklist. */
  beforeChecklist?: React.ReactNode;
  requiredMissing?: string[];
  requiredPrefixValues?: RunSummaryValue[];
}) {
  // Collapsed by default so the form leads with the actual inputs, not a wall
  // of status. The header still shows the "N missing / ready" badge, so the
  // at-a-glance state is never hidden — the user expands only to see details.
  const [showChecklist, setShowChecklist] = useState(false);
  const [showTestSetup, setShowTestSetup] = useState(true);
  const [showTraining, setShowTraining] = useState(false);
  const [showValidationSetup, setShowValidationSetup] = useState(false);
  const [showOthers, setShowOthers] = useState(false);
  const compute = useComputeTargets();
  const explicitComputeValue = computeTargetValue({ gpuProvider, cloudBackend, sshHostId });
  const effectiveComputeTarget = (
    compute.targets.find((target) => target.value === explicitComputeValue)
    ?? (!explicitComputeValue ? compute.defaultTarget : undefined)
  );
  const validationIsIndependent = !!validationSet.trim();
  const updateValidationSet = (value: string) => onChange({
    validationSet: value,
    ...(!value.trim() ? {
      validationMetricType: "",
      validationMetric: "",
      validationMetricDirection: "",
      validationEvaluationScript: "",
      validationAnswerFields: "",
      validationSampleSubmission: "",
    } : {}),
  });
  const requiredValues: RunSummaryValue[] = [
    ...requiredPrefixValues,
    {
      name: "GPU backend",
      value: effectiveComputeTarget?.label || "Not set",
      complete: Boolean(effectiveComputeTarget),
    },
    { name: "Test metric type", value: metricType || "Not set", complete: Boolean(metricType) },
    { name: "Test metric", value: metric.trim() || "Not set", complete: Boolean(metric.trim()) },
    ...(metricType === "custom" ? [{
      name: "Test evaluation script",
      value: evaluationScript.trim() ? short(evaluationScript) : "Not set",
      complete: Boolean(evaluationScript.trim()),
    }] : []),
    { name: "Test target", value: metricDirection || "Not set", complete: Boolean(metricDirection) },
    { name: "Test set", value: testSet.trim() ? short(testSet) : "Not set", complete: Boolean(testSet.trim()) },
    { name: "Test answer fields", value: answerFields.trim() || "Not set", complete: Boolean(answerFields.trim()) },
    {
      name: "Test sample submission",
      value: testSampleSubmission.trim() ? short(testSampleSubmission) : "Not set",
      complete: Boolean(testSampleSubmission.trim()),
    },
    ...(validationIsIndependent ? [
      {
        name: "Validation metric type",
        value: validationMetricType || "Not set",
        complete: Boolean(validationMetricType),
      },
      {
        name: "Validation metric",
        value: validationMetric.trim() || "Not set",
        complete: Boolean(validationMetric.trim()),
      },
      ...(validationMetricType === "custom" ? [{
        name: "Validation evaluation script",
        value: validationEvaluationScript.trim() ? short(validationEvaluationScript) : "Not set",
        complete: Boolean(validationEvaluationScript.trim()),
      }] : []),
      {
        name: "Validation target",
        value: validationMetricDirection || "Not set",
        complete: Boolean(validationMetricDirection),
      },
      {
        name: "Validation answer fields",
        value: validationAnswerFields.trim() || "Not set",
        complete: Boolean(validationAnswerFields.trim()),
      },
      {
        name: "Validation sample submission",
        value: validationSampleSubmission.trim() ? short(validationSampleSubmission) : "Not set",
        complete: Boolean(validationSampleSubmission.trim()),
      },
    ] : []),
    ...(trainingMethod === "gkd" ? [{
      name: "Teacher model", value: teacherModel.trim() || "Not set", complete: Boolean(teacherModel.trim()),
    }] : []),
    ...(trainingMethod === "online_dpo" ? [{
      name: "Reward model", value: rewardModel.trim() || "Not set", complete: Boolean(rewardModel.trim()),
    }] : []),
  ];
  const optionalValues: RunSummaryValue[] = [
    ...(validationIsIndependent ? [
      { name: "Validation set", value: short(validationSet), overridden: true },
    ] : [
      {
        name: "Validation",
        value: "Inherited from Test · 20% · min 200 rows",
        overridden: false,
      },
    ]),
    { name: "Training data", value: dataset.trim() ? short(dataset) : "Prepared by Zevo", overridden: !!dataset.trim() },
    { name: "Data query", value: dataQuery.trim() || "not set", overridden: !!dataQuery.trim() },
    { name: "Base model", value: baseModel.trim() || "Selected by Zevo", overridden: !!baseModel.trim() },
    { name: "Model query", value: modelQuery.trim() || "not set", overridden: !!modelQuery.trim() },
    { name: "Training method", value: trainingMethod.trim() || "Decided by Zevo", overridden: !!trainingMethod.trim() },
    { name: "Method query", value: methodQuery.trim() || "not set", overridden: !!methodQuery.trim() },
    ...(["cpo", "dpo", "gkd", "grpo", "kto", "online_dpo", "orpo", "rft", "rloo"].includes(trainingMethod)
      ? [{ name: "PEFT", value: usePeft === "true" ? "Use PEFT" : usePeft === "false" ? "Full parameters" : "Skill default", overridden: Boolean(usePeft) }]
      : []),
    { name: "Maximum GPUs", value: numGpus.trim() || "unlimited", overridden: !!numGpus.trim() },
    { name: "Generation backend", value: (generation_backend || "vllm").toUpperCase(), overridden: !!generation_backend },
    { name: "Iterations", value: iterations.trim() || "unlimited", overridden: !!iterations.trim() },
    { name: "Budget", value: budget.trim() ? `$${budget.trim()}` : "unlimited", overridden: !!budget.trim() },
    { name: "Time limit", value: timeLimitHours.trim() ? `${timeLimitHours.trim()} h` : "unlimited", overridden: !!timeLimitHours.trim() },
    { name: "Queue limit", value: queueWaitHours.trim() ? `${queueWaitHours.trim()} h` : "24 h", overridden: !!queueWaitHours.trim() },
    { name: "Stop threshold", value: stopThreshold.trim() || "not set", overridden: !!stopThreshold.trim() },
  ];
  return (
    <div className="space-y-6">
      {/* Two groups, and they are the two halves of the concept: what a run
          cannot start without — the files that define and score the problem —
          and the setting it attacks the problem with, every part of which Zevo
          will decide for you if you leave it alone. */}
      <section className="space-y-3">
        <div>
          <h3 className="section-title !text-brass-300">Required</h3>
        </div>
        {requiredPrefix}
        {/* Compute backend sits here — right under the objective and always
            visible — because which GPU/provider a Run lands on is a first-class
            decision, not an "Others" afterthought. The finer knobs (GPU count,
            generation backend) stay tucked away below. */}
        <div className="pt-1">
          <BackendPicker
            gpuProvider={gpuProvider}
            cloudBackend={cloudBackend}
            sshHostId={sshHostId}
            onChange={onChange}
          />
        </div>
        <div className="space-y-3 pt-1">
          <button
            type="button"
            onClick={() => setShowTestSetup((v) => !v)}
            className="flex w-full items-center gap-1.5 text-left"
          >
            <ChevronRight
              size={12}
              className={`shrink-0 text-slate-400 transition-transform ${showTestSetup ? "rotate-90" : ""}`}
            />
            <span className="field-label !text-slate-100">Test Setup</span>
          </button>
          {showTestSetup && (
            <div className="space-y-3">
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <ChoiceField
                  label="Test metric type"
                  value={metricType}
                  required
                  onChange={(v) => onChange({
                    metricType: v as "" | "builtin" | "custom",
                    ...(v === "builtin" ? { evaluationScript: "" } : {}),
                  })}
                  options={[["builtin", "Built-in"], ["custom", "Custom"]]}
                  hint="Choose"
                />
                <div>
                  <SlotLabel label="Test metric" tag={false} />
                  {metricType === "builtin" ? (
                    <ThemedSelect
                      value={metric}
                      onChange={(value) => onChange({ metric: value })}
                      options={BUILTIN_METRICS.map((value) => ({ value, label: value }))}
                      placeholder="Choose metric"
                      ariaLabel="Built-in Test metric"
                      buttonClassName={`h-10 w-full rounded-md border bg-canvas px-2.5 font-mono text-sm ${requiredFieldStateCls(Boolean(metric.trim()))}`}
                    />
                  ) : metricType === "custom" ? (
                    <input
                      value={metric}
                      onChange={(e) => onChange({ metric: e.target.value })}
                      placeholder="e.g. benchmark_average"
                      className={`h-10 w-full rounded-md border bg-canvas px-2.5 font-mono text-sm placeholder:text-slate-600 focus:outline-none ${requiredFieldStateCls(Boolean(metric.trim()))}`}
                    />
                  ) : (
                    <ThemedSelect
                      value=""
                      onChange={() => undefined}
                      options={[]}
                      placeholder="Choose metric type first"
                      ariaLabel="Test metric"
                      disabled
                      buttonClassName={`h-10 w-full rounded-md border bg-canvas px-2.5 font-mono text-sm ${requiredFieldStateCls(false)}`}
                    />
                  )}
                </div>
                <ChoiceField
                  label="Test target"
                  value={metricDirection}
                  required
                  onChange={(v) => onChange({ metricDirection: v as "" | "max" | "min" })}
                  options={[["max", "Max"], ["min", "Min"]]}
                  hint="Choose"
                />
              </div>
              {metricType === "custom" && (
                <FileSlot
                  label="Test evaluation script" tag={false} value={evaluationScript}
                  onChange={(v) => onChange({ evaluationScript: v })}
                  required
                  hint="Frozen for Test and run only after predictions match the Test sample submission."
                />
              )}
              {/* Two halves of one split, and they must not be the same file: the
                  evaluator needs the answers, while Inference must never see
                  them. */}
              <FileSlot
                label="Test set" tag={false} value={testSet}
                onChange={(v) => onChange({ testSet: v })}
                required
                hint="Held-out data used only for the final test score."
              />
              {/* Not a second file: the fields. The data agent drops exactly these
                  to build the questions-only copy inference is given, so the pair can
                  never drift out of sync the way two hand-maintained files did. */}
              <TextField
                label="Test answer fields"
                value={answerFields}
                onChange={(v) => onChange({ answerFields: v })}
                required
                placeholder="Ground-truth columns, e.g. answer, gold"
                hint="Columns or keys containing the test ground truth."
              />
              <FileSlot
                label="Test sample submission" tag={false} value={testSampleSubmission}
                onChange={(v) => onChange({ testSampleSubmission: v })}
                required
                hint="Defines prediction columns, order, and example formatting; its row count need not match Test."
              />
            </div>
          )}
        </div>
      </section>

      {/* Optional controls stay compact as three independent accordions. */}
      <section className="space-y-3">
        <h3 className="section-title !text-brass-300">Optional</h3>
        <div className="flex flex-col gap-3">
        {optionalPrefix && <div>{optionalPrefix}</div>}
        <div className="order-2 space-y-3">
          <button
            type="button"
            onClick={() => setShowValidationSetup((v) => !v)}
            className="flex w-full items-center gap-1.5 text-left"
          >
            <ChevronRight
              size={12}
              className={`shrink-0 text-slate-400 transition-transform ${showValidationSetup ? "rotate-90" : ""}`}
            />
            <span className="field-label !text-slate-100">Validation Setup</span>
          </button>
          {showValidationSetup && (
            /* Empty delegates one deterministic 20% split of Test to the engine
               before optimization begins and inherits Test's scoring contract. */
            <div className="space-y-3">
              {!validationIsIndependent ? (
                <div className="rounded-md border border-brass-500/25 bg-brass-500/[0.04] px-4 py-3">
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
                        value={validationSet}
                        onChange={updateValidationSet}
                        split={validationSplit}
                        onSplit={(v) => onChange({ validationSplit: v })}
                        config={validationConfig}
                        onConfig={(v) => onChange({ validationConfig: v })}
                        answerFields={validationAnswerFields}
                        onAnswerFields={(v) => onChange({ validationAnswerFields: v })}
                        sampleSubmission={validationSampleSubmission}
                        onSampleSubmission={(v) => onChange({ validationSampleSubmission: v })}
                        tag={false}
                        note="Choose a fixed Validation set with its own scoring contract."
                        showHints={false}
                      />
                    </div>
                  </details>
                </div>
              ) : (<>
              <ValidationSetField
                value={validationSet}
                onChange={updateValidationSet}
                split={validationSplit}
                onSplit={(v) => onChange({ validationSplit: v })}
                config={validationConfig}
                onConfig={(v) => onChange({ validationConfig: v })}
                answerFields={validationAnswerFields}
                onAnswerFields={(v) => onChange({ validationAnswerFields: v })}
                sampleSubmission={validationSampleSubmission}
                onSampleSubmission={(v) => onChange({ validationSampleSubmission: v })}
                tag={false}
                note="Independent Validation data and scoring contract."
                showHints={false}
              />
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <ChoiceField
                  label="Validation metric type"
                  value={validationMetricType}
                  required
                  onChange={(v) => onChange({
                    validationMetricType: v as "" | "builtin" | "custom",
                    ...(v === "builtin" ? { validationEvaluationScript: "" } : {}),
                  })}
                  options={[["builtin", "Built-in"], ["custom", "Custom"]]}
                  hint="Choose"
                />
                <div>
                  <SlotLabel label="Validation metric" tag={false} />
                  {validationMetricType === "builtin" ? (
                    <ThemedSelect
                      value={validationMetric}
                      onChange={(value) => onChange({ validationMetric: value })}
                      options={BUILTIN_METRICS.map((value) => ({ value, label: value }))}
                      placeholder="Choose metric"
                      ariaLabel="Built-in Validation metric"
                      buttonClassName={`h-10 w-full rounded-md border bg-canvas px-2.5 font-mono text-sm ${requiredFieldStateCls(Boolean(validationMetric.trim()))}`}
                    />
                  ) : validationMetricType === "custom" ? (
                    <input
                      value={validationMetric}
                      onChange={(e) => onChange({ validationMetric: e.target.value })}
                      placeholder="e.g. token_f1"
                      className={`h-10 w-full rounded-md border bg-canvas px-2.5 font-mono text-sm placeholder:text-slate-600 focus:outline-none ${requiredFieldStateCls(Boolean(validationMetric.trim()))}`}
                    />
                  ) : (
                    <ThemedSelect
                      value=""
                      onChange={() => undefined}
                      options={[]}
                      placeholder="Choose metric type first"
                      ariaLabel="Validation metric"
                      disabled
                      buttonClassName={`h-10 w-full rounded-md border bg-canvas px-2.5 font-mono text-sm ${requiredFieldStateCls(false)}`}
                    />
                  )}
                </div>
                <ChoiceField
                  label="Validation target"
                  value={validationMetricDirection}
                  required
                  onChange={(v) => onChange({ validationMetricDirection: v as "" | "max" | "min" })}
                  options={[["max", "Max"], ["min", "Min"]]}
                  hint="Choose"
                />
              </div>
              {validationMetricType === "custom" && (
                <FileSlot
                  label="Validation evaluation script" tag={false}
                  value={validationEvaluationScript}
                  onChange={(v) => onChange({ validationEvaluationScript: v })}
                  required
                  hint="Frozen for Validation and run only after predictions match its sample submission."
                />
              )}
              </>)}
            </div>
          )}
        </div>
        <div className="order-1 space-y-3">
          <button
            type="button"
            onClick={() => setShowTraining((v) => !v)}
            className="flex w-full items-center gap-1.5 text-left"
          >
            <ChevronRight
              size={12}
              className={`shrink-0 text-slate-400 transition-transform ${showTraining ? "rotate-90" : ""}`}
            />
            <span className="field-label !text-slate-100">Training</span>
          </button>
          {showTraining && (
            <TrainingSetupFields
              inputs={{
                dataset, datasetSplit, datasetConfig, dataQuery,
                baseModel, modelQuery, trainingMethod, methodQuery,
                teacherModel, rewardModel, usePeft,
              }}
              onChange={onChange}
            />
          )}
        </div>
        <div className="order-3 space-y-3">
          <button
            type="button"
            onClick={() => setShowOthers((v) => !v)}
            className="flex w-full items-center gap-1.5 text-left"
          >
            <ChevronRight
              size={12}
              className={`shrink-0 text-slate-400 transition-transform ${showOthers ? "rotate-90" : ""}`}
            />
            <span className="field-label !text-slate-100">Others</span>
          </button>
          {showOthers && (
            <div className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          {/* A uniform Run envelope: Infrastructure chooses an actual positive
              count at or below this maximum for every provider. Empty means the
              user imposes no GPU-count limit. The backend/provider itself is
              chosen up top, under the objective. */}
          <NumberField
            label="Maximum GPUs" value={numGpus}
            onChange={(v) => onChange({ numGpus: v })}
            min={1}
            placeholder=""
            hint="Maximum GPUs Zevo may use at once; the actual plan may use fewer. Blank means unlimited."
          />
          <ChoiceField
            label="Generation backend" value={generation_backend}
            onChange={(v) => onChange({ generation_backend: v as GenerationBackend | "" })}
            options={[["vllm", "vllm"], ["hf", "hf"]]}
            hint="Backend used to generate predictions; blank defaults to vLLM."
          />
        </div>
        {/* Limits stay blank by default. The checklist resolves those blanks
            to "unlimited" (or "not set" for the score threshold). */}
        <div className="grid grid-cols-1 items-start gap-3 md:grid-cols-2 lg:grid-cols-[0.75fr_0.85fr_1.25fr_1.65fr_1.1fr]">
          <LimitField label="Iterations" value={iterations} onChange={(v) => onChange({ iterations: v })}
            hint="Maximum optimization rounds; blank means no iteration cap." />
          <LimitField label="Budget" value={budget} onChange={(v) => onChange({ budget: v })} prefix="$"
            hint="Maximum run cost in USD; blank means no cost cap." />
          <LimitField label="Time limit (hours)" value={timeLimitHours}
            onChange={(v) => onChange({ timeLimitHours: v })}
            hint="Active experiment time only; Slurm queue wait is excluded." />
          <LimitField label="Max queue wait (hours)" value={queueWaitHours}
            onChange={(v) => onChange({ queueWaitHours: v })}
            hint="Maximum Slurm PENDING time; blank defaults to 24 hours (max 168)." />
          <NumberField
            label="Stop threshold" value={stopThreshold}
            onChange={(v) => onChange({ stopThreshold: v })}
            step="any"
            placeholder=""
            hint="Validation score that ends the Run, using this metric's own scale; blank disables it."
            alignLabel
          />
        </div>
        {extra}
            </div>
          )}
        </div>
      </div>
      </section>
      {beforeChecklist}
      <section>
        <button
          type="button"
          onClick={() => setShowChecklist((v) => !v)}
          className="flex w-full items-center justify-between gap-3 text-left"
        >
          <span className="flex items-center gap-1.5">
            <ChevronRight
              size={13}
              className={`shrink-0 text-slate-400 transition-transform ${showChecklist ? "rotate-90" : ""}`}
            />
            <h3 className="section-title !text-brass-300">Checklist</h3>
          </span>
          <span className={`font-mono text-[0.62rem] uppercase tracking-[0.12em] ${
            requiredMissing.length ? "text-coral-300" : "text-phosphor-300"
          }`}>
            {requiredMissing.length ? `${requiredMissing.length} missing` : "ready"}
          </span>
        </button>
        {showChecklist && (
          <RunSummaryTable requiredValues={requiredValues} optionalValues={optionalValues} />
        )}
      </section>
    </div>
  );
}
