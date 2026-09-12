import { useEffect, useState } from "react";
import { Modal } from "./Modal";
import { FileSlot } from "./RunInputs";
import { api } from "../lib/api";
import { ThemedSelect } from "./ThemedSelect";


/**
 * Create or edit a task: the PROBLEM. A name, what the model should be able to
 * do, and the held-out scoring specification — nothing about how to attack it.
 *
 * Training data, base model, method and budgets are a SETTING, and a task has
 * many; they are added from the task's own card, where the ones it has already
 * been run with are listed beside them. So this form no longer asks for a
 * "default" one nobody chose.
 */

/** The row as the list holds it — what edit mode prefills from. */
export type TaskRecord = {
  name: string;
  task_objective: string;
  test_set: string;
  test_answer_fields: string[];
  test_sample_submission: string;
  metric_type: "builtin" | "custom";
  evaluation_script: string;
  evaluator_sha256: string;
  metric: string;
  metric_direction: "max" | "min";
};

const cols = (s: string) => s.split(",").map((c) => c.trim()).filter(Boolean);
const BUILTIN_TASK_METRICS = ["accuracy", "exact_match", "f1", "token_f1", "bleu", "rouge_l"];

/** A comma-separated column list, laid out like the file slots beside it. */
function TextRow({
  label, value, onChange, placeholder, hint,
}: {
  label: string; value: string; onChange: (v: string) => void;
  placeholder?: string; hint?: string;
}) {
  return (
    <div>
      <div className="field-label mb-1">{label}</div>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder || hint}
        spellCheck={false}
        className="w-full rounded-md border border-hair bg-canvas p-2 font-mono text-sm text-slate-200 placeholder:text-slate-600 placeholder:opacity-100 focus:border-brass-500/50 focus:outline-none"
      />
    </div>
  );
}

const field =
  "w-full rounded-md border border-hair bg-canvas p-2.5 text-sm leading-relaxed text-slate-200 placeholder:text-slate-600 placeholder:opacity-100 focus:border-brass-500/50 focus:outline-none";
// These name the top-level blocks of the Task definition. Field labels inside
// Test setup fields use the smaller shared `field-label` scale.
const head = "section-title mb-2 block !text-brass-300";

export function TaskModal({
  open, onClose, onSaved, task = null,
}: {
  open: boolean;
  onClose: () => void;
  onSaved: () => void;
  /** Null = create a new task; a row = edit that one in place. */
  task?: TaskRecord | null;
}) {
  const editing = !!task;
  const [name, setName] = useState("");
  const [objective, setObjective] = useState("");
  // 0 means unlimited on both, and that is what an empty box submits.
  const [testSet, setTestSet] = useState("");
  // Comma-separated in the form, a list on the wire. The data agent drops
  // these to build the questions-only copy inference is given.
  const [answerFields, setAnswerFields] = useState("");
  const [testSampleSubmission, setTestSampleSubmission] = useState("");
  const [metricType, setMetricType] = useState<"" | "builtin" | "custom">("");
  const [evaluationScript, setEvaluationScript] = useState("");
  const [metric, setMetric] = useState("");
  const [metricDirection, setMetricDirection] = useState<"" | "max" | "min">("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const normalizedMetric = metric.trim().toLowerCase();
  const hasEvaluator = metricType === "builtin"
    ? BUILTIN_TASK_METRICS.includes(normalizedMetric)
    : metricType === "custom" && !!evaluationScript.trim();
  const formComplete = !!(
    name.trim()
    && objective.trim()
    && metricType
    && normalizedMetric
    && metricDirection
    && testSet.trim()
    && cols(answerFields).length
    && testSampleSubmission.trim()
    && hasEvaluator
  );

  // Reopening always re-seeds: from the row in edit mode, from nothing in
  // create mode. `task.name` is in the deps so switching rows without closing
  // the dialog reloads the form rather than showing the previous task's values.
  useEffect(() => {
    if (!open) return;
    const t = task;
    setName(t?.name ?? "");
    setObjective(t?.task_objective ?? "");
    // 0 is "unlimited", and the box shows that as empty rather than as "0".
    setTestSet(t?.test_set ?? "");
    setAnswerFields((t?.test_answer_fields ?? []).join(", "));
    setTestSampleSubmission(t?.test_sample_submission ?? "");
    setMetricType(t?.metric_type ?? "");
    setEvaluationScript(t?.evaluation_script ?? "");
    setMetric(t?.metric ?? "");
    setMetricDirection(t?.metric_direction ?? "");
    setError(null);
  }, [open, task?.name]);


  async function submit() {
    setBusy(true);
    setError(null);
    try {
      if (!name.trim()) throw new Error("Give the task a name.");
      if (!objective.trim()) throw new Error("Describe what the task should achieve.");
      if (!metricType) throw new Error("Choose the Test metric type.");
      if (!metric.trim()) throw new Error("Name the metric produced by the evaluator.");
      if (!metricDirection) throw new Error("Choose whether the evaluation target is Max or Min.");
      if (!testSet.trim()) throw new Error("Choose the held-out test set.");
      if (!cols(answerFields).length) throw new Error("Name at least one test answer field.");
      if (!testSampleSubmission.trim()) throw new Error("Choose the test sample submission.");
      if (!hasEvaluator) throw new Error(
        metricType === "custom"
          ? "Choose the custom evaluation script."
          : "Choose one of Zevo's built-in metrics.",
      );
      // `name` is never sent in edit mode: it is the key runs reference.
      const url = editing ? `/tasks/${encodeURIComponent(task!.name)}` : "/tasks";
      await api(url, {
        method: editing ? "PATCH" : "POST",
        body: JSON.stringify({
          ...(editing ? {} : { name: name.trim() }),
          task_objective: objective.trim(),
          test_set: testSet.trim(),
          test_answer_fields: cols(answerFields),
          test_sample_submission: testSampleSubmission.trim(),
          metric_type: metricType,
          evaluation_script: metricType === "custom" ? evaluationScript.trim() : "",
          metric: metric.trim(),
          metric_direction: metricDirection,
        }),
      });
      onSaved();
      onClose();
    } catch (e) {
      setError(String((e as Error).message || e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal open={open} title={editing ? `Edit ${task!.name}` : "New task"} onClose={onClose} width="max-w-2xl">
      <div className="max-h-[70vh] space-y-4 overflow-y-auto pr-1">
        <div>
          <label className={head}>Task name</label>
          {/* Locked while editing: runs record the name as plain text, so
              renaming here would detach this task from its own run history
              rather than taking it along. Delete and recreate to rename. */}
          <input value={name} onChange={(e) => setName(e.target.value)} spellCheck={false}
            disabled={editing}
            title={editing ? "A task cannot be renamed. Its runs are recorded under this name." : undefined}
            placeholder="Stable name used by Runs, e.g. med"
            className={`${field} font-mono ${editing ? "cursor-not-allowed text-slate-500" : ""}`} />
        </div>

        <div>
          <label className={head}>Objective</label>
          <textarea value={objective} onChange={(e) => setObjective(e.target.value)} rows={4}
            placeholder="What should the fine-tuned model be able to do, and how is it scored?"
            className={`${field} font-mono`} />
        </div>

        <div>
          <label className={head}>Test metric contract</label>
          <div className="space-y-3">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div>
                <div className="field-label mb-1">Type</div>
                <ThemedSelect
                  value={metricType}
                  onChange={(value) => {
                    const next = value as "" | "builtin" | "custom";
                    setMetricType(next);
                    setMetric(next === "builtin" && !BUILTIN_TASK_METRICS.includes(normalizedMetric) ? "" : metric);
                    if (next === "builtin") setEvaluationScript("");
                  }}
                  options={[
                    { value: "builtin", label: "Built-in" },
                    { value: "custom", label: "Custom" },
                  ]}
                  placeholder="Choose"
                  ariaLabel="Test metric type"
                  buttonClassName={`${field} h-10 font-mono`}
                />
              </div>
              <div>
                <div className="field-label mb-1">Metric</div>
                {metricType === "builtin" ? (
                  <ThemedSelect
                    value={normalizedMetric}
                    onChange={setMetric}
                    options={BUILTIN_TASK_METRICS.map((value) => ({ value, label: value }))}
                    placeholder="Choose metric"
                    ariaLabel="Built-in Test metric"
                    buttonClassName={`${field} h-10 font-mono`}
                  />
                ) : metricType === "custom" ? (
                  <input value={metric} onChange={(e) => setMetric(e.target.value)}
                    placeholder="e.g. pass@1 or reward"
                    className={`${field} h-10 font-mono`} />
                ) : (
                  <ThemedSelect
                    value=""
                    onChange={() => undefined}
                    options={[]}
                    placeholder="Choose metric type first"
                    ariaLabel="Test metric"
                    disabled
                    buttonClassName={`${field} h-10 font-mono`}
                  />
                )}
              </div>
              <div>
                <div className="field-label mb-1">Target</div>
                <ThemedSelect value={metricDirection}
                  onChange={(value) => setMetricDirection(value as "" | "max" | "min")}
                  options={[{ value: "max", label: "Max" }, { value: "min", label: "Min" }]}
                  placeholder="Max or Min"
                  ariaLabel="Test target"
                  buttonClassName={`${field} h-10 font-mono`} />
              </div>
            </div>
            {metricType === "custom" && (
              <FileSlot label="Test evaluation script" required tag={false}
                value={evaluationScript} onChange={setEvaluationScript}
                hint="Frozen for held-out Test and run only after predictions match the Test sample submission" />
            )}
          </div>
        </div>

        <div>
          {/* The held-out specification, and nothing else: how the task is
              ATTACKED (training data, model, method, budgets) is a setting. */}
          <label className={head}>Test setup</label>
          <div className="space-y-3">
            <FileSlot label="Test set" required tag={false} value={testSet} onChange={setTestSet}
              hint="scored by the harness alone, never by the loop" />
            <TextRow label="Test answer fields" value={answerFields} onChange={setAnswerFields}
              placeholder="answer, gold"
              hint="where the ground truth lives in the test set: a column of a CSV, a key of a JSON record. Dropped to make the copy inference sees." />
            <FileSlot label="Test sample submission" required tag={false} value={testSampleSubmission} onChange={setTestSampleSubmission}
              hint="prediction columns, order, and example formatting; the example row count need not match Test" />
          </div>
        </div>

        {error && (
          <div className="rounded-md border border-coral-500/30 bg-coral-500/10 p-2.5 text-2xs text-coral-300">
            {error}
          </div>
        )}
      </div>

      <div className="mt-4 flex justify-end gap-2 border-t border-hair pt-4">
        <button onClick={onClose} className="btn">cancel</button>
        <button onClick={() => void submit()} disabled={busy || !formComplete}
          className="btn btn-brass uppercase tracking-[0.14em] disabled:cursor-not-allowed disabled:opacity-30">
          {busy ? (editing ? "saving…" : "creating…") : editing ? "save changes" : "create task"}
        </button>
      </div>
    </Modal>
  );
}
