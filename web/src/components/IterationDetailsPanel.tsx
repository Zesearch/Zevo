import { useMemo, useState } from "react";
import useSWR from "swr";
import { ChevronRight, X } from "lucide-react";
import type { RunDetail } from "../lib/api";
import { assignIterations, iterationOrder } from "../lib/iterations";
import { fmtScore } from "../lib/format";
import { Kicker } from "./zevo/primitives";

type Artifact = {
  id: string; ticket_id: string; role: string; path: string; local_path: string;
  exists: boolean; availability: string; size_bytes: number;
  meta: Record<string, unknown>; created_at: string;
};

type ArtifactDetail = Artifact & {
  preview_kind: string;
  preview: string;
  preview_columns: string[];
  preview_rows: Array<Record<string, unknown>>;
};
type DetailKind = "data" | "train" | "inference";
type Selection = {
  artifactId?: string;
  testArtifactId?: string;
  title: string;
  kind: DetailKind;
  configuration?: unknown;
  trainingExample?: unknown;
  dataRows?: unknown;
  validationRows?: unknown;
  testRows?: unknown;
};

const TRAIN_PARAMETERS = [
  "num_epochs", "batch_size", "effective_batch_size", "gradient_accumulation_steps",
  "learning_rate", "max_seq_len", "optimizer", "lr_scheduler_type", "warmup_ratio",
  "weight_decay", "precision", "world_size", "lora_r", "lora_alpha", "lora_dropout",
];

const INFERENCE_PARAMETERS = [
  "generation_backend", "batch_size", "max_new_tokens", "temperature", "top_p",
  "top_k", "repetition_penalty", "seed", "prompt_framing", "model_reasoning_type",
  "stop_token_ids", "max_model_len", "gpu_memory_utilization", "tensor_parallel_size",
];

function text(value: unknown): string {
  if (value == null || value === "") return "—";
  if (Array.isArray(value)) return value.map(String).join(", ") || "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function displayKey(key: string): string {
  return key.replaceAll("_", " ");
}

function configValue(value: unknown, key: string): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const queue: Record<string, unknown>[] = [value as Record<string, unknown>];
  const seen = new Set<Record<string, unknown>>();
  while (queue.length) {
    const current = queue.shift()!;
    if (seen.has(current)) continue;
    seen.add(current);
    if (current[key] != null && current[key] !== "") return current[key];
    for (const nested of Object.values(current)) {
      if (nested && typeof nested === "object" && !Array.isArray(nested)) {
        queue.push(nested as Record<string, unknown>);
      }
    }
  }
  return undefined;
}

function importantParameters(value: unknown, kind: "train" | "inference") {
  const keys = kind === "train" ? TRAIN_PARAMETERS : INFERENCE_PARAMETERS;
  return keys.flatMap((key) => {
    const found = configValue(value, key);
    return found == null || found === "" ? [] : [{ key, value: text(found) }];
  });
}

function datasetName(meta: Record<string, unknown>): string {
  const recipe = (
    meta.data_recipe && typeof meta.data_recipe === "object" && !Array.isArray(meta.data_recipe)
      ? meta.data_recipe as Record<string, unknown>
      : {}
  );
  const source = text(meta.dataset_name || recipe.dataset_name);
  if (source === "—") return source;
  const clean = source.split("@")[0].split(/[?#]/)[0].replace(/\/+$/, "");
  const parts = clean.split("/").filter(Boolean);
  let name = parts.at(-1) || clean;
  if (/\.(csv|jsonl?|ndjson|parquet|pq)$/i.test(name) && parts.length > 1) name = parts.at(-2)!;
  return name.replace(/\.(csv|jsonl?|ndjson|parquet|pq)$/i, "") || "—";
}

function numberText(value: unknown): string {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric >= 0 ? numeric.toLocaleString() : "—";
}

export function IterationDetailsPanel({ run }: { run: RunDetail }) {
  const { data: artifacts = [], error } = useSWR<Artifact[]>(
    `/api/runs/${encodeURIComponent(run.id)}/artifacts`,
    { refreshInterval: 5000 },
  );
  const [selection, setSelection] = useState<Selection | null>(null);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [resultLane, setResultLane] = useState<"validation" | "test">("validation");
  const [previewPage, setPreviewPage] = useState(1);
  const detailArtifactId = selection?.kind === "inference"
    ? (resultLane === "test" ? selection.testArtifactId : selection.artifactId)
    : selection?.artifactId;
  const { data: detail } = useSWR<ArtifactDetail>(
    detailArtifactId
      ? `/api/runs/${encodeURIComponent(run.id)}/artifacts/${detailArtifactId}?preview_page=${previewPage}&preview_page_size=10`
      : null,
  );

  const groups = useMemo(() => {
    const ticketInfo = assignIterations(run.tickets);
    const ticketById = new Map(run.tickets.map((ticket) => [ticket.id, ticket]));
    const byKey = new Map<string, Artifact[]>();
    for (const artifact of artifacts) {
      const key = ticketInfo.get(artifact.ticket_id)?.key;
      if (key) byKey.set(key, [...(byKey.get(key) || []), artifact]);
    }
    let activeData: Artifact[] = [];
    return iterationOrder(run.tickets).map((entry) => {
      const artifactsForRound = byKey.get(entry.key) || [];
      const hasNewData = artifactsForRound.some((artifact) => {
        const ticket = ticketById.get(artifact.ticket_id);
        return ticket?.agent_id === "data" && ticket.lane === "optimization"
          && artifact.role === "training_dataset";
      });
      if (hasNewData) {
        activeData = artifactsForRound.filter((artifact) => {
          const ticket = ticketById.get(artifact.ticket_id);
          return ticket?.agent_id === "data" && ticket.lane === "optimization";
        });
      }
      return { ...entry, artifacts: artifactsForRound, activeData: [...activeData] };
    }).filter((entry) => entry.artifacts.some((artifact) =>
      artifact.role === "training_dataset"
      || artifact.role === "checkpoint"
      || artifact.role === "predictions"
    ));
  }, [artifacts, run.tickets]);

  if (error) return <div className="rounded-bezel border border-coral-500/30 p-4 text-sm text-coral-300">Failed to load iteration details.</div>;
  if (!groups.length) return <div className="rounded-bezel border border-dashed border-hair p-8 text-center text-sm text-dim">No iteration evidence yet.</div>;

  const ticketById = new Map(run.tickets.map((ticket) => [ticket.id, ticket]));
  const selectedParameters = selection?.kind === "train" || selection?.kind === "inference"
    ? importantParameters(selection.configuration, selection.kind)
    : [];
  const selectedTotalRows = selection?.kind === "data"
    ? selection.dataRows
    : resultLane === "test" ? selection?.testRows : selection?.validationRows;
  const selectedTotal = Number(selectedTotalRows);
  const previewPageCount = Number.isFinite(selectedTotal) && selectedTotal > 0
    ? Math.min(10, Math.ceil(selectedTotal / 10))
    : 1;

  return (
    <div className="space-y-5">
      {groups.map((group) => {
        const isCollapsed = collapsed.has(group.key);
        const data = group.activeData.find((item) => item.role === "training_dataset");
        const train = group.artifacts.find((item) => {
          const ticket = ticketById.get(item.ticket_id);
          return ticket?.agent_id === "train" && item.role === "checkpoint"
            && item.meta.checkpoint_kind !== "intermediate";
        }) || group.artifacts.find((item) => ticketById.get(item.ticket_id)?.agent_id === "train" && item.role === "checkpoint");
        const infer = group.artifacts.find((item) => {
          const ticket = ticketById.get(item.ticket_id);
          return ticket?.agent_id === "inference" && ticket.lane === "optimization"
            && item.role === "predictions";
        });
        const testPrediction = group.artifacts.find((item) => {
          const ticket = ticketById.get(item.ticket_id);
          return ticket?.lane === "held_out_test" && item.role === "predictions";
        });
        const testEvidence = testPrediction || group.artifacts.find((item) => {
          const ticket = ticketById.get(item.ticket_id);
          return ticket?.lane === "held_out_test" && item.role === "scoring_questions";
        });
        const journal = run.history.find((item) => (
          group.num === 0
            ? item.source === "baseline"
            : Number(item.iteration) === group.num && item.source !== "baseline"
        )) || run.history.find((item) => Number(item.iteration) === group.num);
        const dataMeta = data?.meta || {};
        const trainMeta = train?.meta || {};
        const inferMeta = infer?.meta || {};
        const testMeta = testEvidence?.meta || {};
        const model = text(trainMeta.base_model || inferMeta.base_model || journal?.base_model);
        const method = text(trainMeta.training_method || journal?.training_method);
        const trainParameters = importantParameters(trainMeta.configuration, "train");
        const inferenceParameters = importantParameters(inferMeta.configuration, "inference");
        const testRows = testMeta.n_rows ?? testMeta.question_rows ?? testMeta.n_rows_out;

        return (
          <section key={group.key} className="overflow-hidden rounded-bezel border border-hair bg-panel/60">
            <button
              type="button"
              aria-expanded={!isCollapsed}
              onClick={() => setCollapsed((current) => {
                const next = new Set(current);
                next.has(group.key) ? next.delete(group.key) : next.add(group.key);
                return next;
              })}
              className={`flex w-full flex-wrap items-baseline gap-x-6 gap-y-2 border-l-2 border-l-brass-400/70 px-4 py-3 text-left transition hover:bg-brass-500/[0.035] ${
                isCollapsed ? "" : "border-b border-hair"
              }`}
            >
              <ChevronRight
                size={13}
                className={`self-center text-brass-400 transition-transform ${isCollapsed ? "" : "rotate-90"}`}
              />
              <span className="font-mono text-xs font-semibold uppercase tracking-[0.14em] text-brass-300">
                {group.label}
              </span>
              <span className="readout text-sm text-slate-100">
                <span className="mr-1.5 font-mono text-2xs uppercase tracking-[0.12em] text-slate-100">Val</span>
                {fmtScore(journal?.score, run.validation_metric)}
              </span>
              <span className="readout text-sm text-phosphor-300">
                <span className="mr-1.5 font-mono text-2xs uppercase tracking-[0.12em] text-phosphor-300">Test</span>
                {fmtScore(journal?.test_score, run.metric)}
              </span>
            </button>
            {!isCollapsed && (
            <div className="grid grid-cols-1 divide-y divide-hair md:grid-cols-2 xl:grid-cols-4 xl:divide-x xl:divide-y-0">
              <Fact
                label="Data"
                value={group.num === 0 ? "—" : `Train ${numberText(dataMeta.training_rows)} · ${datasetName(dataMeta)}`}
                onClick={group.num > 0 && data ? () => {
                  setPreviewPage(1);
                  setSelection({
                    artifactId: data.id,
                    title: "Dataset preview",
                    kind: "data",
                    trainingExample: configValue(trainMeta.configuration, "training_data_example"),
                    dataRows: dataMeta.training_rows,
                  });
                } : undefined}
              />
              <Fact
                label="Model"
                value={model}
              />
              <Fact
                label="Train method"
                value={method}
                onClick={trainParameters.length ? () => setSelection({
                  title: "Train method",
                  kind: "train",
                  configuration: trainMeta.configuration,
                }) : undefined}
              />
              <Fact
                label="Inference / Prediction"
                value={`Val ${numberText(inferMeta.n_rows)} · Test ${numberText(testRows)}`}
                onClick={inferenceParameters.length || infer || testPrediction ? () => {
                  setPreviewPage(1);
                  setResultLane(infer ? "validation" : "test");
                  setSelection({
                    artifactId: infer?.id,
                    testArtifactId: testPrediction?.id,
                    title: "Inference / Prediction",
                    kind: "inference",
                    configuration: inferMeta.configuration,
                    validationRows: inferMeta.n_rows,
                    testRows,
                  });
                } : undefined}
              />
            </div>
            )}
          </section>
        );
      })}

      {selection && (
        <div className="fixed inset-0 z-50 flex justify-end bg-black/45" onClick={() => setSelection(null)}>
          <aside className="h-full w-[92vw] max-w-7xl overflow-auto border-l border-hair bg-panel p-5 shadow-2xl" onClick={(event) => event.stopPropagation()}>
            <div className="mb-4 flex items-center justify-between gap-3">
              <Kicker strong>{selection.title}</Kicker>
              <button type="button" className="btn" onClick={() => setSelection(null)}><X size={14} /></button>
            </div>

            {selectedParameters.length > 0 && (
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
                {selectedParameters.map(({ key, value }) => (
                  <div key={key}
                    className="min-w-0 rounded-md border border-brass-500/25 bg-canvas/70 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.025)]">
                    <Kicker className="!text-brass-300">{displayKey(key)}</Kicker>
                    <p className="mt-2 break-words font-mono text-sm text-slate-200">{value}</p>
                  </div>
                ))}
              </div>
            )}
            {selection.kind === "inference" && (selection.artifactId || selection.testArtifactId) && (
              <div className="mt-4 flex items-center gap-2">
                <button type="button" disabled={!selection.artifactId}
                  onClick={() => { setResultLane("validation"); setPreviewPage(1); }}
                  className={`btn ${resultLane === "validation" ? "btn-brass" : ""} disabled:cursor-not-allowed disabled:opacity-40`}>
                  Validation · {numberText(selection.validationRows)}
                </button>
                <button type="button" disabled={!selection.testArtifactId}
                  onClick={() => { setResultLane("test"); setPreviewPage(1); }}
                  className={`btn ${resultLane === "test" ? "btn-brass" : ""} disabled:cursor-not-allowed disabled:opacity-40`}>
                  Test · {numberText(selection.testRows)}
                </button>
              </div>
            )}
            {(selection.kind === "data" || (selection.kind === "inference" && detailArtifactId)) && (
              detail?.preview_rows?.length
                ? <div className={selection.kind === "inference" ? "mt-3" : ""}>
                    <TablePreview columns={detail.preview_columns} rows={detail.preview_rows} page={previewPage} pageSize={10} />
                    <Pagination page={previewPage} pageCount={previewPageCount} onPage={setPreviewPage} />
                  </div>
                : <div className={`${selection.kind === "inference" ? "mt-3" : ""} rounded-md border border-dashed border-hair p-8 text-center text-sm text-dim`}>
                    {detail ? "No tabular preview is available." : "Loading preview…"}
                  </div>
            )}
            {selection.kind === "data" && (
              <TrainingExamplePanel
                example={selection.trainingExample}
                rows={detail?.preview_rows || []}
                page={previewPage}
              />
            )}
          </aside>
        </div>
      )}
    </div>
  );
}

function objectValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function placeholderText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value == null) return "";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

const PLACEHOLDER_PATTERN = /^<[A-Z][A-Za-z0-9_:.-]*>$/;

function isPlaceholder(value: unknown): value is string {
  return typeof value === "string" && PLACEHOLDER_PATTERN.test(value);
}

function collectPlaceholderValues(
  template: unknown,
  actual: unknown,
  values: Map<string, string>,
) {
  if (isPlaceholder(template)) {
    values.set(template, placeholderText(actual));
    return;
  }
  if (Array.isArray(template) && Array.isArray(actual)) {
    // Method-shaped records are not guaranteed to have identical array
    // offsets. A normalizer may inject a constant system message into the
    // synthetic contract while the stored row begins with `user`, and a
    // conversation may contain one, two, or many turns. Match object items by
    // their constant scalar discriminators (role/type/kind/...) first, then
    // fall back to the next unused position for arrays without discriminators.
    // Nothing here knows a dataset key or a chat role in advance.
    const unused = new Set(actual.map((_, index) => index));
    const templateRecords = template.map(objectValue);
    const discriminatorKeys = templateRecords.every(Boolean) && templateRecords.length
      ? Object.keys(templateRecords[0]!).filter((key) => templateRecords.every((record) => {
          const value = record?.[key];
          return value == null
            || typeof value === "number"
            || typeof value === "boolean"
            || (typeof value === "string" && !isPlaceholder(value));
        }))
      : [];
    template.forEach((item, templateIndex) => {
      const itemRecord = objectValue(item);
      const discriminators = itemRecord
        ? discriminatorKeys.map((key) => [key, itemRecord[key]] as const)
        : [];
      let actualIndex = -1;
      if (discriminators.length) {
        actualIndex = [...unused].find((index) => {
          const candidate = objectValue(actual[index]);
          return !!candidate && discriminators.every(([key, value]) => candidate[key] === value);
        }) ?? -1;
      }
      if (actualIndex < 0 && !discriminators.length) {
        actualIndex = unused.has(templateIndex) ? templateIndex : ([...unused][0] ?? -1);
      }
      if (actualIndex < 0) return;
      unused.delete(actualIndex);
      collectPlaceholderValues(item, actual[actualIndex], values);
    });
    return;
  }
  const templateRecord = objectValue(template);
  const actualRecord = objectValue(actual);
  if (!templateRecord || !actualRecord) return;
  for (const [key, item] of Object.entries(templateRecord)) {
    const fallback = key === "record_id" && actualRecord.id != null ? actualRecord.id : actualRecord[key];
    collectPlaceholderValues(item, fallback, values);
  }
}

function realizeSequence(sequence: string, values: Map<string, string>): string {
  let realized = sequence;
  for (const [placeholder, value] of [...values.entries()].sort((a, b) => b[0].length - a[0].length)) {
    realized = realized.split(placeholder).join(value);
  }
  return realized;
}

function sequencePlaceholders(sequence: string): string[] {
  return [...sequence.matchAll(/<[A-Z][A-Za-z0-9_:.-]*>/g)]
    .map((match) => match[0])
    .filter((value, index, all) => all.indexOf(value) === index);
}

type RealizedTrainingView = { name: string; input: string; label: string };

function realizedTrainingViews(
  row: Record<string, unknown>,
  sourceRecord: unknown,
  sequences: Array<[string, string]>,
  targets: string[],
  sequenceTargets: Record<string, unknown> | null,
): RealizedTrainingView[] {
  const values = new Map<string, string>();
  collectPlaceholderValues(sourceRecord, row, values);

  const views = sequences.flatMap(([name, sequence]) => {
    const referenced = sequencePlaceholders(sequence);
    if (referenced.some((placeholder) => !values.has(placeholder))) return [];

    const explicitTargets = stringList(sequenceTargets?.[name]);
    const presentTargets = (explicitTargets.length ? explicitTargets : targets)
      .filter((target) => sequence.includes(target) && values.has(target));
    if (!presentTargets.length) return [];

    // Older contracts declare the union of targets across all rendered
    // sequences. In an autoregressive sequence the last declared target is the
    // active one; earlier assistant responses are context for a later turn.
    // Newer/method-specific contracts can remove that ambiguity with
    // sequence_loss_targets and may name several jointly-supervised spans.
    const activeTargets = explicitTargets.length
      ? presentTargets
      : [presentTargets.reduce((latest, target) => (
          sequence.lastIndexOf(target) > sequence.lastIndexOf(latest) ? target : latest
        ))];
    const targetStart = Math.min(...activeTargets.map((target) => sequence.indexOf(target)));
    if (targetStart < 0) return [];
    return [{
      name,
      input: realizeSequence(sequence.slice(0, targetStart), values),
      // Show the exact supervised suffix, including template terminators. This
      // is more truthful than displaying only a guessed raw "label" field.
      label: realizeSequence(sequence.slice(targetStart), values),
    }];
  });
  if (views.length) return views;

  // A malformed/older display contract should never turn an existing target
  // into a blank panel. Use only target placeholders structurally bound above;
  // the raw record is context. This is deliberately not a key-name heuristic.
  const fallbackTargets = targets.filter((target) => values.has(target));
  return fallbackTargets.length ? [{
    name: "record",
    input: JSON.stringify(row, null, 2),
    label: fallbackTargets.map((target) => values.get(target) || "").join("\n"),
  }] : [];
}

function TrainingExamplePanel({
  example, rows, page,
}: {
  example: unknown;
  rows: Array<Record<string, unknown>>;
  page: number;
}) {
  const contract = objectValue(example);
  const sourceRecord = contract?.source_record;
  const rendered = objectValue(contract?.rendered_sequences);
  const sequences = rendered
    ? Object.entries(rendered).filter((entry): entry is [string, string] => typeof entry[1] === "string")
    : [];
  const targets = stringList(contract?.loss_target_placeholders);
  const sequenceTargets = objectValue(contract?.sequence_loss_targets);

  if (!sequences.length || !rows.length) return null;

  return (
    <section className="mt-6 border-t border-hair pt-5">
      <Kicker strong className="!text-brass-300">Rendered training example</Kicker>
      <div className="mt-4 space-y-3">
        {rows.slice(0, 3).map((row, index) => {
          const views = realizedTrainingViews(
            row, sourceRecord, sequences, targets, sequenceTargets,
          );
          return (
            <div key={index} className="overflow-hidden rounded-md border border-hair bg-canvas/55">
              <div className="border-b border-hair bg-panel/70 px-4 py-2.5 font-mono text-xs text-slate-400">
                Example {(page - 1) * 10 + index + 1}
              </div>
              {views.length ? views.map((view, viewIndex) => (
                <div key={`${view.name}-${viewIndex}`}>
                  {views.length > 1 && (
                    <div className="border-b border-hair bg-panel/40 px-4 py-2 font-mono text-2xs uppercase tracking-wider text-dim">
                      {displayKey(view.name)}
                    </div>
                  )}
                  <div className="grid gap-px bg-hair lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
                    <div className="min-w-0 bg-canvas p-4">
                      <Kicker>Input</Kicker>
                      <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-slate-300">{view.input}</pre>
                    </div>
                    <div className="min-w-0 bg-canvas p-4">
                      <Kicker>Label</Kicker>
                      <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-brass-100">{view.label}</pre>
                    </div>
                  </div>
                </div>
              )) : (
                <div className="bg-canvas p-4 text-sm text-dim">
                  This record does not match the Train artifact's declared display structure.
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function Fact({
  label, value, hint, onClick,
}: {
  label: string; value: string; hint?: string; onClick?: () => void;
}) {
  const content = (
    <>
      <Kicker className="!text-slate-400">{label}</Kicker>
      <p className="mt-2 break-words font-mono text-sm text-slate-200">{value}</p>
      {hint && <p className="mt-1 break-words font-mono text-2xs leading-relaxed text-slate-500">{hint}</p>}
    </>
  );
  return onClick ? (
    <button type="button" onClick={onClick}
      className="min-w-0 p-4 text-left transition hover:bg-brass-500/[0.04] focus:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-brass-400">
      {content}
    </button>
  ) : <div className="min-w-0 p-4">{content}</div>;
}

function TablePreview({
  columns, rows, page, pageSize,
}: {
  columns: string[];
  rows: Array<Record<string, unknown>>;
  page: number;
  pageSize: number;
}) {
  return (
    <div className="overflow-hidden rounded-md border border-hair bg-canvas">
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse text-left font-mono text-xs">
          <thead className="sticky top-0 z-10 bg-panel">
            <tr>
              <th className="border-b border-r border-hair px-3 py-2 text-2xs font-medium uppercase tracking-[0.12em] text-slate-500">#</th>
              {columns.map((column) => (
                <th key={column} className={`${tableColumnWidth(column)} border-b border-r border-hair px-3 py-2 text-2xs font-medium uppercase tracking-[0.12em] text-slate-400 last:border-r-0`}>
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex} className="align-top even:bg-white/[0.015]">
                <td className="border-b border-r border-hair px-3 py-3 text-slate-600">{(page - 1) * pageSize + rowIndex + 1}</td>
                {columns.map((column) => (
                  <td key={column} className={`${tableColumnWidth(column)} border-b border-r border-hair px-3 py-3 text-slate-300 last:border-r-0`}>
                    <TableCell column={column} value={row[column]} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Pagination({
  page, pageCount, onPage,
}: {
  page: number;
  pageCount: number;
  onPage: (page: number) => void;
}) {
  if (pageCount <= 1) return null;
  return (
    <div className="mt-3 flex flex-wrap items-center justify-center gap-1.5">
      <button type="button" className="btn" disabled={page <= 1}
        onClick={() => onPage(Math.max(1, page - 1))}>Previous</button>
      {Array.from({ length: pageCount }, (_, index) => index + 1).map((item) => (
        <button key={item} type="button" onClick={() => onPage(item)}
          className={`btn min-w-9 ${page === item ? "btn-brass" : ""}`}>
          {item}
        </button>
      ))}
      <button type="button" className="btn" disabled={page >= pageCount}
        onClick={() => onPage(Math.min(pageCount, page + 1))}>Next</button>
    </div>
  );
}

function tableColumnWidth(column: string): string {
  const normalized = column.trim().toLowerCase();
  if (normalized === "id" || normalized.endsWith("_id")) return "w-32 min-w-32 max-w-32";
  if (normalized === "prediction" || normalized.startsWith("prediction ·")) {
    return "w-[40rem] min-w-[32rem] max-w-[48rem]";
  }
  if (normalized === "ground_truth" || normalized.startsWith("ground_truth ·")) {
    return "w-[28rem] min-w-[24rem] max-w-[36rem]";
  }
  return "min-w-32 max-w-[34rem]";
}

function TableCell({ column, value }: { column: string; value: unknown }) {
  if (value == null || value === "") return <span className="text-slate-600">—</span>;
  const normalized = column.trim().toLowerCase();
  if ((normalized === "id" || normalized.endsWith("_id")) && typeof value !== "object") {
    return <div title={String(value)} className="max-w-28 truncate">{String(value)}</div>;
  }
  if (typeof value === "object") {
    return <pre className="max-h-64 min-w-64 overflow-auto whitespace-pre-wrap break-words leading-relaxed">{JSON.stringify(value, null, 2)}</pre>;
  }
  return <div className="max-h-64 min-w-24 overflow-auto whitespace-pre-wrap break-words leading-relaxed">{String(value)}</div>;
}
