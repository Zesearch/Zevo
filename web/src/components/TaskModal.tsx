import { BUILTIN_METRICS } from "../lib/scoringMetrics";
import { useEffect, useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import { CreationModal } from "./CreationModal";
import { SettingNavigation } from "./SettingNavigation";
import { FileSlot, TrainingDataField, RunSummaryTable } from "./RunInputs";
import { PredictionColumnField } from "./PredictionColumnField";
import { ThemedSelect } from "./ThemedSelect";
import { api } from "../lib/api";


export type TaskTestSetRecord = {
  name: string;
  test_set: string;
  inference_query: string;
  sample_submission: string;
  metric_type: "builtin" | "custom";
  metric: string;
  answer_fields: string[];
  prediction_column?: string;
  metric_direction: "max" | "min";
  evaluation_script: string;
  evaluator_sha256: string;
};

export type TaskRecord = {
  name: string;
  task_objective: string;
  test_sets: TaskTestSetRecord[];
};

type TestSetDraft = Omit<TaskTestSetRecord, "answer_fields"> & {
  answer_fields: string;
};



const field =
  "w-full rounded-md border border-hair bg-canvas p-2.5 text-sm leading-relaxed text-slate-200 placeholder:text-slate-600 placeholder:opacity-100 focus:border-brass-500/50 focus:outline-none";
const head = "field-label mb-2 block";

const emptyTestSet = (): TestSetDraft => ({
  name: "",
  test_set: "",
  inference_query: "",
  sample_submission: "",
  metric_type: "builtin",
  metric: "",
  answer_fields: "",
  metric_direction: "max",
  evaluation_script: "",
  evaluator_sha256: "",
});

const answerFields = (value: string) =>
  value.split(",").map((item) => item.trim()).filter(Boolean);

export function TaskModal({
  open, onClose, onSaved, task = null,
}: {
  open: boolean;
  onClose: () => void;
  onSaved: () => void;
  task?: TaskRecord | null;
}) {
  const editing = !!task;
  const [attempted, setAttempted] = useState(false);
  const [step, setStep] = useState("basics");
  const [name, setName] = useState("");
  const [objective, setObjective] = useState("");
  const [testSets, setTestSets] = useState<TestSetDraft[]>([]);
  const [expandedTestSet, setExpandedTestSet] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setStep("basics");
    setAttempted(false);
    setExpandedTestSet(null);
    setName(task?.name ?? "");
    setObjective(task?.task_objective ?? "");
    setTestSets(
      task?.test_sets?.length
        ? task.test_sets.map((item) => ({
            ...item,
            metric_type: item.metric_type ?? "builtin",
            evaluation_script: item.evaluation_script ?? "",
            evaluator_sha256: item.evaluator_sha256 ?? "",
            answer_fields: item.answer_fields.join(", "),
          }))
        : [],
    );
    setError(null);
  }, [open, task?.name]);

  const updateTestSet = <K extends keyof TestSetDraft>(
    index: number, key: K, value: TestSetDraft[K],
  ) => {
    setTestSets((current) => current.map(
      (item, position) => position === index ? { ...item, [key]: value } : item,
    ));
  };

  const updateTestSetFields = (index: number, values: Partial<TestSetDraft>) => {
    setTestSets((current) => current.map(
      (item, position) => position === index ? { ...item, ...values } : item,
    ));
  };

  const missingFields = (item: TestSetDraft) => [
    !item.name.trim() && "Name", !item.test_set.trim() && "Test set",
    !item.inference_query.trim() && "Inference query",
    !item.sample_submission.trim() && "Sample submission",
    !answerFields(item.answer_fields).length && "Test answer fields",
    !(item.metric_type === "builtin" ? BUILTIN_METRICS.includes(item.metric) : item.metric.trim()) && "Metric",
    item.metric_type === "custom" && !item.evaluation_script.trim() && "Evaluation script",
  ].filter(Boolean);
  const checkTests = () => {
    setAttempted(true);
    if (!testSets.length) { setStep("tests"); return false; }
    const index = testSets.findIndex(item => missingFields(item).length);
    if (index >= 0) { setExpandedTestSet(index); setStep("tests"); return false; }
    return true;
  };
  const go = (next: string) => {
    if (next === "checklist") {
      setAttempted(true);
      if (!name.trim() || !objective.trim()) { setStep("basics"); return; }
      if (!checkTests()) return;
    }
    setStep(next);
  };
  const mark = (value: string) => attempted && !value.trim() ? " launch-field-missing" : "";

  const complete = !!(
    name.trim()
    && objective.trim()
    && testSets.length
    && testSets.every((item) =>
      item.name.trim()
      && item.test_set.trim()
      && item.inference_query.trim()
      && item.sample_submission.trim()
      && (
        (item.metric_type === "builtin" && BUILTIN_METRICS.includes(item.metric))
        || (item.metric_type === "custom"
          && item.metric.trim()
          && item.evaluation_script.trim())
      )
      && answerFields(item.answer_fields).length,
    )
  );

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      if (!complete) throw new Error("Complete every Test set contract.");
      const normalized = testSets.map((item) => ({
        ...item,
        name: item.name.trim(),
        test_set: item.test_set.trim(),
        inference_query: item.inference_query.trim(),
        sample_submission: item.sample_submission.trim(),
        metric_type: item.metric_type,
        metric: item.metric.trim().toLowerCase(),
        answer_fields: answerFields(item.answer_fields),
        prediction_column: item.metric_type === "builtin" ? item.prediction_column?.trim() || "" : "",
        metric_direction: item.metric_direction,
        evaluation_script: item.metric_type === "custom"
          ? item.evaluation_script.trim() : "",
        evaluator_sha256: item.metric_type === "custom"
          ? item.evaluator_sha256 : "",
      }));
      const names = normalized.map((item) => item.name.toLowerCase());
      if (new Set(names).size !== names.length) {
        throw new Error("Test set names must be unique.");
      }
      await api(editing ? `/tasks/${encodeURIComponent(task!.name)}` : "/tasks", {
        method: editing ? "PATCH" : "POST",
        body: JSON.stringify({
          ...(editing ? {} : { name: name.trim() }),
          task_objective: objective.trim(),
          test_sets: normalized,
        }),
      });
      onSaved();
      onClose();
    } catch (reason) {
      setError(String((reason as Error).message || reason));
    } finally {
      setBusy(false);
    }
  }

  return (
    <CreationModal wide
      open={open}
      title={editing ? "Edit Task" : "Create Task"}
      heading={editing ? task!.name : "Define your task"}
      description="Set the objective and the Test sets that will measure success."
      onClose={onClose}
    >
      <SettingNavigation label="Task setup" active={step} onSelect={go} items={[
        { id: "basics", label: "Basic information", group: "Required", complete: !!name.trim() && !!objective.trim() },
        { id: "tests", label: "Test sets", group: "Required", complete: testSets.length > 0 && testSets.every(item => !missingFields(item).length) },
        { id: "checklist", label: "Checklist" },
      ]}>
      <div className="space-y-8">
        <section hidden={step !== "basics"} className="space-y-8">
        <div>
          <label className={head}>Task name</label>
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
            disabled={editing}
            placeholder="e.g. olmo-instruct-evaluation"
            spellCheck={false}
            className={`${field} font-mono ${editing ? "cursor-not-allowed text-slate-500" : ""}${mark(name)}`}
          />
        </div>

        <div>
          <label className={head}>Objective</label>
          <textarea
            value={objective}
            onChange={(event) => setObjective(event.target.value)}
            rows={3}
            placeholder="What should Zevo improve?"
            className={`${field} font-mono${mark(objective)}`}
          />
        </div>

        </section>
        <section hidden={step !== "tests"} className="space-y-6">
        <div className="flex flex-wrap items-end justify-end gap-4">
          <button
            type="button"
            onClick={() => { if (testSets.length && !checkTests()) return; setAttempted(false); setExpandedTestSet(testSets.length); setTestSets((current) => [...current, emptyTestSet()]); }}
            className="launch-add-set"
          >
            <Plus size={13} /> Add Test set
          </button>
        </div>

        {attempted && !testSets.length && <p role="alert" className="text-sm text-coral-300">Add at least one Test set before continuing.</p>}
        <div className="divide-y divide-hair">
          {testSets.map((item, index) => (
            <section
              key={index}
              className="py-5"
            >
              <div className="flex items-center justify-between gap-3">
                <span className="flex min-w-0 flex-1 items-center gap-3">
                  <span className={`shrink-0 tabular-nums ${expandedTestSet === index ? "suite-entry-title" : "font-mono text-2xs text-slate-600"}`}>{String(index + 1).padStart(2, "0")}</span>
                  <span className={`min-w-0 break-words ${expandedTestSet === index ? "suite-entry-title" : "font-mono text-xs text-slate-200"}`}>{item.name || "Test set"}</span>
                </span>
                <button type="button" className="launch-next launch-next-compact" disabled={expandedTestSet === index && missingFields(item).length > 0}
                  aria-expanded={expandedTestSet === index}
                  onClick={() => setExpandedTestSet(expandedTestSet === index ? null : index)}>
                  {expandedTestSet === index ? "Done" : "Edit"}
                </button>
                {testSets.length > 1 && (
                  <button
                    type="button"
                    title="Remove Test set"
                    onClick={() => {
                      setTestSets((current) => current.filter((_, position) => position !== index));
                      setExpandedTestSet(current => current === null ? null : current === index ? Math.max(0, index - 1) : current > index ? current - 1 : current);
                    }}
                    className="rounded p-1.5 text-slate-500 transition hover:bg-coral-500/10 hover:text-coral-300"
                  >
                    <Trash2 size={14} />
                  </button>
                )}
              </div>

              <div hidden={expandedTestSet !== index} className="mt-6 space-y-6">
                {attempted && missingFields(item).length > 0 && <p role="alert" className="text-sm text-coral-300">Complete: {missingFields(item).join(", ")}.</p>}
                <div>
                  <div className="field-label mb-1">Name</div>
                  <input
                    value={item.name}
                    onChange={(event) => updateTestSet(index, "name", event.target.value)}
                    placeholder="e.g. MATH or HumanEvalPlus"
                    className={`${field} font-mono${mark(item.name)}`}
                  />
                </div>

                <div className={attempted && !item.test_set.trim() ? "rounded-md outline outline-1 outline-coral-400/60 outline-offset-4" : ""}>
                <TrainingDataField
                  label="Test set"
                  required
                  tag={false}
                  value={item.test_set}
                  onChange={(value) => updateTestSet(index, "test_set", value)}
                  note="from Files"
                />
                </div>

                <div>
                  <div className="field-label mb-1">Inference query</div>
                  <textarea
                    value={item.inference_query}
                    onChange={(event) => updateTestSet(index, "inference_query", event.target.value)}
                    rows={3}
                    placeholder="Exact instruction. Use {input} or {question}, or Zevo appends the row."
                    className={`${field} font-mono${mark(item.inference_query)}`}
                  />
                </div>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                  <div>
                    <div className="field-label mb-1">Metric</div>
                    <ThemedSelect
                      value={item.metric_type === "custom" ? "__other__" : item.metric}
                      onChange={(value) => updateTestSetFields(index, value === "__other__"
                        ? {
                            metric_type: "custom", metric: "",
                            evaluation_script: "", evaluator_sha256: "",
                          }
                        : {
                            metric_type: "builtin", metric: value,
                            evaluation_script: "", evaluator_sha256: "",
                          })}
                      options={[
                        ...BUILTIN_METRICS.map((value) => ({
                          value,
                          label: value,
                        })),
                        { value: "__other__", label: "Other (custom script)" },
                      ]}
                      placeholder="Choose metric"
                      ariaLabel={`Metric for Test set ${index + 1}`}
                      buttonClassName={`${field} h-10 font-mono${mark(item.metric)}`}
                    />
                  </div>
                  <div>
                    <div className="field-label mb-1">Target</div>
                    <ThemedSelect
                      value={item.metric_direction}
                      onChange={(value) => updateTestSet(
                        index, "metric_direction", value as "max" | "min",
                      )}
                      options={[
                        { value: "max", label: "Maximize" },
                        { value: "min", label: "Minimize" },
                      ]}
                      ariaLabel={`Metric target for Test set ${index + 1}`}
                      buttonClassName={`${field} h-10 font-mono`}
                    />
                  </div>
                  <div>
                    <div className="field-label mb-1">Test answer fields</div>
                    <input
                      value={item.answer_fields}
                      onChange={(event) => updateTestSet(index, "answer_fields", event.target.value)}
                      placeholder="answer, or comma-separated fields"
                      className={`${field} h-10 font-mono${mark(item.answer_fields)}`}
                    />
                  </div>
                </div>

                {item.metric_type === "builtin" && (
                  <PredictionColumnField
                    sample={item.sample_submission} dataset={item.test_set}
                    answerFields={answerFields(item.answer_fields)} value={item.prediction_column || ""}
                    onChange={(value) => updateTestSet(index, "prediction_column", value)}
                  />
                )}

                {item.metric_type === "custom" && (
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <div>
                      <div className="field-label mb-1">Custom metric name</div>
                      <input
                        value={item.metric}
                        onChange={(event) => updateTestSet(index, "metric", event.target.value)}
                        placeholder="e.g. pass@1, loss, reward"
                        className={`${field} h-10 font-mono${mark(item.metric)}`}
                      />
                    </div>
                    <FileSlot
                      label="Evaluation script"
                      required
                      tag={false}
                      controlClassName={`h-10${mark(item.evaluation_script)}`}
                      value={item.evaluation_script}
                      onChange={(value) => updateTestSetFields(index, {
                        evaluation_script: value, evaluator_sha256: "",
                      })}
                      hint="Python scorer (.py)"
                    />
                  </div>
                )}

                <FileSlot
                  label="Sample submission"
                  controlClassName={mark(item.sample_submission)}
                  required
                  tag={false}
                  value={item.sample_submission}
                  onChange={(value) => updateTestSet(index, "sample_submission", value)}
                  hint="Prediction columns and example output format"
                />
              </div>
            </section>
          ))}
        </div>

        </section>
        <section hidden={step !== "checklist"} className="space-y-6">
          <RunSummaryTable optionalValues={[]} requiredValues={[
            { name: "Task name", value: name },
            { name: "Objective", value: objective },
            ...testSets.flatMap((item, index) => [
              { name: `Test set ${index + 1}`, value: item.name },
              { name: "Data", value: item.test_set },
              { name: "Inference query", value: item.inference_query },
              { name: "Metric", value: `${item.metric} · ${item.metric_direction === "max" ? "Maximize" : "Minimize"}` },
              { name: "Answer fields", value: item.answer_fields },
              { name: "Sample submission", value: item.sample_submission },
              ...(item.evaluation_script ? [{ name: "Evaluation script", value: item.evaluation_script }] : []),
            ]),
          ]} />
        </section>
        {error && (
          <div className="rounded-md border border-coral-500/30 bg-coral-500/10 p-2.5 text-2xs text-coral-300">
            {error}
          </div>
        )}
      </div>

      <div className="creation-actions">
        {step !== "basics" ? <button onClick={() => setStep(step === "checklist" ? "tests" : "basics")} disabled={busy} className="launch-back">← Back</button> : <span />}
        {step !== "checklist" ? <button type="button" className="launch-next" onClick={() => go(step === "basics" ? "tests" : "checklist")}>Continue →</button> : (
        <button
          onClick={() => void submit()}
          disabled={busy || !complete}
          className="launch-next"
        >
          {busy ? (editing ? "Saving…" : "Creating…") : editing ? "Save changes" : "Create task"}
        </button>)}
      </div>
      </SettingNavigation>
    </CreationModal>
  );
}
