import { useEffect, useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import { Modal } from "./Modal";
import { FileSlot, TrainingDataField } from "./RunInputs";
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

const BUILTIN_TASK_METRICS = [
  "accuracy", "exact_match", "f1", "token_f1", "bleu", "rouge_l",
  "mc_loglikelihood", "accuracy_norm",
];

const field =
  "w-full rounded-md border border-hair bg-canvas p-2.5 text-sm leading-relaxed text-slate-200 placeholder:text-slate-600 placeholder:opacity-100 focus:border-brass-500/50 focus:outline-none";
const head = "section-title mb-2 block !text-brass-300";

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
  const [name, setName] = useState("");
  const [objective, setObjective] = useState("");
  const [testSets, setTestSets] = useState<TestSetDraft[]>([emptyTestSet()]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
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
        : [emptyTestSet()],
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
        (item.metric_type === "builtin" && BUILTIN_TASK_METRICS.includes(item.metric))
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
        name: item.name.trim(),
        test_set: item.test_set.trim(),
        inference_query: item.inference_query.trim(),
        sample_submission: item.sample_submission.trim(),
        metric_type: item.metric_type,
        metric: item.metric.trim().toLowerCase(),
        answer_fields: answerFields(item.answer_fields),
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
    <Modal
      open={open}
      title={editing ? `Edit ${task!.name}` : "New task"}
      onClose={onClose}
      width="max-w-3xl"
    >
      <div className="max-h-[72vh] space-y-5 overflow-y-auto pr-1">
        <div>
          <label className={head}>Task name</label>
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
            disabled={editing}
            placeholder="e.g. olmo-instruct-evaluation"
            spellCheck={false}
            className={`${field} font-mono ${editing ? "cursor-not-allowed text-slate-500" : ""}`}
          />
        </div>

        <div>
          <label className={head}>Objective</label>
          <textarea
            value={objective}
            onChange={(event) => setObjective(event.target.value)}
            rows={3}
            placeholder="What should Zevo improve?"
            className={`${field} font-mono`}
          />
        </div>

        <div className="flex items-end justify-between gap-4">
          <div>
            <label className={`${head} !mb-1`}>Test sets</label>
            <p className="text-2xs leading-relaxed text-slate-500">
              Each Test set carries its own inference query, output schema, answer fields, and metric.
              Zevo reports every score and their unweighted average.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setTestSets((current) => [...current, emptyTestSet()])}
            className="btn shrink-0"
          >
            <Plus size={13} /> add Test set
          </button>
        </div>

        <div className="space-y-4">
          {testSets.map((item, index) => (
            <section
              key={index}
              className="rounded-lg border border-hair bg-white/[0.018] p-4"
            >
              <div className="mb-4 flex items-center justify-between">
                <span className="font-mono text-xs uppercase tracking-[0.14em] text-brass-300">
                  Test set {index + 1}
                </span>
                {testSets.length > 1 && (
                  <button
                    type="button"
                    title="Remove Test set"
                    onClick={() => setTestSets((current) =>
                      current.filter((_, position) => position !== index))}
                    className="rounded p-1.5 text-slate-500 transition hover:bg-coral-500/10 hover:text-coral-300"
                  >
                    <Trash2 size={14} />
                  </button>
                )}
              </div>

              <div className="space-y-3">
                <div>
                  <div className="field-label mb-1">Name</div>
                  <input
                    value={item.name}
                    onChange={(event) => updateTestSet(index, "name", event.target.value)}
                    placeholder="e.g. MATH or HumanEvalPlus"
                    className={`${field} font-mono`}
                  />
                </div>

                <TrainingDataField
                  label="Test set"
                  required
                  tag={false}
                  value={item.test_set}
                  onChange={(value) => updateTestSet(index, "test_set", value)}
                  note="from Files"
                />

                <div>
                  <div className="field-label mb-1">Inference query</div>
                  <textarea
                    value={item.inference_query}
                    onChange={(event) => updateTestSet(index, "inference_query", event.target.value)}
                    rows={3}
                    placeholder="Exact instruction. Use {input} or {question}, or Zevo appends the row."
                    className={`${field} font-mono`}
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
                        ...BUILTIN_TASK_METRICS.map((value) => ({ value, label: value })),
                        { value: "__other__", label: "Other (custom script)" },
                      ]}
                      placeholder="Choose metric"
                      ariaLabel={`Metric for Test set ${index + 1}`}
                      buttonClassName={`${field} h-10 font-mono`}
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
                      className={`${field} h-10 font-mono`}
                    />
                  </div>
                </div>

                {item.metric_type === "custom" && (
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <div>
                      <div className="field-label mb-1">Custom metric name</div>
                      <input
                        value={item.metric}
                        onChange={(event) => updateTestSet(index, "metric", event.target.value)}
                        placeholder="e.g. pass@1, loss, reward"
                        className={`${field} h-10 font-mono`}
                      />
                    </div>
                    <FileSlot
                      label="Evaluation script"
                      required
                      tag={false}
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

        {error && (
          <div className="rounded-md border border-coral-500/30 bg-coral-500/10 p-2.5 text-2xs text-coral-300">
            {error}
          </div>
        )}
      </div>

      <div className="mt-4 flex justify-end gap-2 border-t border-hair pt-4">
        <button onClick={onClose} className="btn">cancel</button>
        <button
          onClick={() => void submit()}
          disabled={busy || !complete}
          className="btn btn-brass uppercase tracking-[0.14em] disabled:cursor-not-allowed disabled:opacity-30"
        >
          {busy ? (editing ? "saving…" : "creating…") : editing ? "save changes" : "create task"}
        </button>
      </div>
    </Modal>
  );
}
