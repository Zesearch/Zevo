import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { useNavigate } from "react-router-dom";
import { ChevronRight, RotateCw } from "lucide-react";
import { AttachmentDropzone, type Attachment } from "./AttachmentDropzone";
import {
  ChoiceField,
  NumberField,
  RunInputs,
  TextField,
  EMPTY_RUN_INPUTS,
  contractPreferencesFromInputs,
  methodConfigFromInputs,
  validationContractFromInputs,
  type RunInputValues,
} from "./RunInputs";
import { TaskNameInput } from "./TaskNameInput";
import { api } from "../lib/api";
import { ThemedSelect } from "./ThemedSelect";
import { TaskSettingHistory } from "./TaskSettings";
import type { AgentCustomization, TaskDTO, TaskSettingDTO, UserRequest } from "../lib/api";

const columns = (s: string) =>
  s.split(",").map((c) => c.trim()).filter(Boolean);

/**
 * Customized Pipeline — a guided pipeline where the user fills in each agent's marching
 * orders up front (instructions, input files, output location, hyperparameters).
 * Strict blocks reject deviations; advisory blocks allow only documented fallbacks.
 *
 * Posts POST /runs { mode:"customized_pipeline", user_request, customizations, ... }.
 */

type ParamType = "text" | "number" | "select" | "list";
type ParamSpec = {
  k: string;
  label: string;
  type: ParamType;
  opts?: string[];
  step?: string;
  placeholder?: string;
  min?: number;
  integer?: boolean;
};

// Per-agent knobs surfaced in the form. Keys match the payload fields the
// orchestrator merges into each ticket (see zevo/contracts + orchestrator platform.md).
const PIPELINE: { id: string; label: string; parameters: ParamSpec[] }[] = [
  // Provider and GPU maximum are Run inputs. Infrastructure derives every other
  // resource value from run context, so its customization block is guidance
  // and file/output policy only; it has no duplicate sizing form.
  { id: "infrastructure", label: "Infrastructure", parameters: [] },
  { id: "data", label: "Data", parameters: [
    {
      k: "method_ids", label: "Data method IDs", type: "list",
      placeholder: "e.g. acquire_hf, reformat_jsonl, inline_transform",
    },
    { k: "target_size", label: "Target rows", type: "number", min: 1, integer: true, placeholder: "Approximate prepared training-record count" },
  ]},
  { id: "train", label: "Train", parameters: [
    { k: "num_epochs", label: "Epochs", type: "number", min: 1, integer: true, placeholder: "Number of passes over the training data" },
    { k: "batch_size", label: "Batch size", type: "number", min: 1, integer: true, placeholder: "Training examples per device per step" },
    { k: "learning_rate", label: "Learning rate", type: "number", min: 0, step: "any", placeholder: "Optimizer step size (> 0)" },
    { k: "lora_r", label: "LoRA r", type: "number", min: 0, integer: true, placeholder: "LoRA adapter rank; ignored by non-LoRA methods" },
    { k: "lora_alpha", label: "LoRA alpha", type: "number", min: 0, integer: true, placeholder: "LoRA adapter scaling factor" },
    { k: "max_seq_len", label: "Max seq len", type: "number", min: 1, integer: true, placeholder: "Maximum tokens in each training example" },
  ]},
  { id: "inference", label: "Inference", parameters: [] },
  { id: "registry", label: "Registry", parameters: [] },
];

type AgentState = {
  instructions: string;
  inputs: Attachment[];
  output_dir: string;
  parameters: Record<string, string>;
  enforcement: "" | "strict" | "advisory";
  open: boolean;
};

const emptyAgent = (open = false): AgentState => ({ instructions: "", inputs: [], output_dir: "", parameters: {}, enforcement: "", open });

// Same input and label treatment as the other two modes in the launch dialog —
// they sit behind one set of tabs, so a different field size and a different
// label tracking read as two different products.
const fieldCls =
  "w-full rounded-md border border-hair bg-canvas p-2.5 text-sm leading-relaxed text-slate-200 placeholder:text-slate-600 placeholder:opacity-100 focus:border-brass-500/50 focus:outline-none";
const completeFieldCls =
  "!border-brass-500/55 !bg-brass-500/[0.07] !text-brass-100";
const labelCls = "field-label mb-1.5 block";
// Block headings match the other modes exactly — the same `section-title` the
// full-pipeline form uses, not a kicker a size down. The dimmer labelCls stays
// for the fields inside an agent's own panel.
// The same `section-title` the other two modes use — nothing extra, so every
// mode's first field starts level with the mode cards beside it.
/**
 * The customized form on its own, without a dialog around it — the Launch-run
 * modal renders this as the Single Stage run mode. `onDone` fires after a successful
 * start so the host can close itself.
 */
export function CustomizedRunForm({
  onDone,
  onDirtyChange,
}: {
  onDone: () => void;
  onDirtyChange?: (dirty: boolean) => void;
}) {
  const nav = useNavigate();
  const { data: tasks = [] } = useSWR<TaskDTO[]>("/api/tasks");
  // Run-level fields
  const [taskName, setTaskName] = useState("");
  // What to call this particular execution, as opposed to the task it runs.
  const [runName, setRunName] = useState("");
  const [objective, setObjective] = useState("");
  const [inputs, setInputs] = useState<RunInputValues>(EMPTY_RUN_INPUTS);
  const [pickedSetting, setPickedSetting] = useState("");
  // Matches Full Pipeline: the save-reuse prompt appears only after the user
  // has chosen or changed a Setting-owned value, not on an untouched form.
  const [touched, setTouched] = useState(false);
  const [saveSetting, setSaveSetting] = useState(false);
  const [settingName, setSettingName] = useState("");
  // Per-Agent customization state
  const [agents, setAgents] = useState<Record<string, AgentState>>(
    () => Object.fromEntries(PIPELINE.map((a) => [a.id, emptyAgent(false)])),
  );
  const [showAgentConfiguration, setShowAgentConfiguration] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const dirtyForm = !!(
    taskName.trim() || runName.trim() || objective.trim()
    || Object.values(inputs).some((value) => String(value || "").trim())
    || saveSetting || settingName.trim()
    || Object.values(agents).some((agent) => (
      agent.instructions.trim()
      || agent.inputs.length
      || agent.output_dir.trim()
      || Object.values(agent.parameters).some((value) => String(value || "").trim())
    ))
  );

  useEffect(() => {
    onDirtyChange?.(dirtyForm);
  }, [dirtyForm, onDirtyChange]);

  const predefined = useMemo(
    () => tasks.find((task) => task.name === taskName.trim()) ?? null,
    [tasks, taskName],
  );
  const { data: savedSettings = [] } = useSWR<TaskSettingDTO[]>(
    predefined ? `/api/tasks/${encodeURIComponent(predefined.name)}/settings` : null,
  );
  const picked = savedSettings.find((setting) => setting.id === pickedSetting) ?? null;
  const nameTaken = !!settingName.trim() && savedSettings.some(
    (setting) => setting.name.trim().toLowerCase() === settingName.trim().toLowerCase(),
  );

  useEffect(() => {
    if (!predefined) return;
    setObjective(predefined.task_objective || "");
    setPickedSetting("");
    setInputs((v) => ({
      ...v,
      testSet: predefined.test_set || "",
      answerFields: (predefined.test_answer_fields ?? []).join(", "),
      metricType: predefined.metric_type,
      evaluationScript: predefined.evaluation_script || "",
      testSampleSubmission: predefined.test_sample_submission || "",
      metric: predefined.metric,
      metricDirection: predefined.metric_direction,
    }));
  }, [predefined?.name]); // eslint-disable-line react-hooks/exhaustive-deps

  function applySetting(s: TaskSettingDTO) {
    setPickedSetting(s.id);
    setTouched(true);
    setInputs((v) => ({
      ...v,
      dataset: s.dataset || "",
      datasetSplit: s.dataset_split || "",
      datasetConfig: s.dataset_config || "",
      dataQuery: s.data_query || "",
      modelQuery: s.model_query || "",
      methodQuery: s.method_query || "",
      validationSet: s.validation_set || "",
      validationSplit: s.validation_split || "",
      validationConfig: s.validation_config || "",
      validationAnswerFields: (s.validation_answer_fields ?? []).join(", "),
      validationSampleSubmission: s.validation_sample_submission || "",
      validationMetricType: s.validation_metric_type,
      validationMetric: s.validation_metric,
      validationMetricDirection: s.validation_metric_direction,
      validationEvaluationScript: s.validation_evaluation_script || "",
      baseModel: s.base_model || "",
      trainingMethod: s.training_method || "",
      teacherModel: String(s.method_config?.teacher_model || ""),
      rewardModel: String(s.method_config?.reward_model || ""),
      usePeft: typeof s.method_config?.use_peft === "boolean" ? String(s.method_config.use_peft) as "true" | "false" : "",
      promptFraming: s.prompt_framing || "",
      systemPrompt: s.system_prompt || "",
      lossObjectiveConfig: Object.keys(s.loss_objective_config || {}).length ? JSON.stringify(s.loss_objective_config) : "",
      inferenceConfig: Object.keys(s.inference_config || {}).length ? JSON.stringify(s.inference_config) : "",
      decodingStrategy: (s.decoding_config?.decoding_strategy || "") as "" | "greedy" | "sampling",
      maxNewTokens: s.decoding_config?.max_new_tokens != null ? String(s.decoding_config.max_new_tokens) : "",
      temperature: s.decoding_config?.temperature != null ? String(s.decoding_config.temperature) : "",
      topP: s.decoding_config?.top_p != null ? String(s.decoding_config.top_p) : "",
      topK: s.decoding_config?.top_k != null ? String(s.decoding_config.top_k) : "",
      repetitionPenalty: s.decoding_config?.repetition_penalty != null ? String(s.decoding_config.repetition_penalty) : "",
      seed: s.decoding_config?.seed != null ? String(s.decoding_config.seed) : "",
      iterations: s.iteration_budget ? String(s.iteration_budget) : "",
      budget: s.max_cost_usd ? String(s.max_cost_usd) : "",
      stopThreshold: s.stop_threshold != null ? String(s.stop_threshold) : "",
    }));
  }

  function changeTaskName(next: string) {
    if (next.trim() !== taskName.trim()) {
      setPickedSetting("");
      setTouched(false);
      setSaveSetting(false);
      setSettingName("");
      if (!tasks.some((task) => task.name === next.trim())) {
        setObjective("");
        setInputs((v) => ({
          ...v,
          testSet: "",
          answerFields: "",
          metricType: "",
          evaluationScript: "",
          testSampleSubmission: "",
          metric: "",
          metricDirection: "",
        }));
      }
    }
    setTaskName(next);
  }

  function setDetailedInput(patch: Partial<RunInputValues>) {
    const runOnly = Object.keys(patch).every(
      (key) => key === "timeLimitHours" || key === "queueWaitHours",
    );
    if (!runOnly) {
      setPickedSetting("");
      setTouched(true);
    }
    setInputs((current) => ({ ...current, ...patch }));
  }

  const requiredMissing = [
    !runName.trim() && "Run name",
    !taskName.trim() && "Task name",
    !objective.trim() && "Objective",
    !inputs.metricType && "Test metric type",
    !inputs.metric.trim() && "Metric",
    inputs.metricType === "custom" && !inputs.evaluationScript.trim() && "Evaluation script",
    !inputs.metricDirection && "Target",
    !!inputs.validationSet.trim()
      && !inputs.validationMetricType
      && "Validation metric type",
    !!inputs.validationSet.trim()
      && !inputs.validationMetric.trim()
      && "Validation metric",
    !!inputs.validationSet.trim()
      && inputs.validationMetricType === "custom"
      && !inputs.validationEvaluationScript.trim()
      && "Validation evaluation script",
    !!inputs.validationSet.trim()
      && !inputs.validationMetricDirection
      && "Validation target",
    !inputs.testSet.trim() && "Test set",
    columns(inputs.answerFields).length === 0 && "Test answer fields",
    !inputs.testSampleSubmission.trim() && "Sample submission",
    !!inputs.validationSet.trim()
      && columns(inputs.validationAnswerFields).length === 0
      && "Validation answer fields",
    !!inputs.validationSet.trim()
      && !inputs.validationSampleSubmission.trim()
      && "Validation sample submission",
    inputs.trainingMethod === "gkd" && !inputs.teacherModel.trim() && "Teacher model",
    inputs.trainingMethod === "online_dpo" && !inputs.rewardModel.trim() && "Reward model",
  ].filter(Boolean) as string[];

  const patch = (id: string, next: Partial<AgentState>) =>
    setAgents((s) => ({ ...s, [id]: { ...s[id], ...next } }));
  const setParam = (id: string, k: string, v: string) =>
    setAgents((s) => ({ ...s, [id]: { ...s[id], parameters: { ...s[id].parameters, [k]: v } } }));

  function buildCustomizations(): { agents: Record<string, AgentCustomization> } {
    const out: Record<string, AgentCustomization> = {};
    for (const spec of PIPELINE) {
      const a = agents[spec.id];
      // Coerce parameters: numbers -> number, drop empties.
      const parameters: Record<string, string | number | boolean | string[]> = {};
      for (const p of spec.parameters) {
        const raw = (a.parameters[p.k] ?? "").trim();
        if (!raw) continue;
        if (p.type === "number") {
          const value = Number(raw);
          if (!Number.isFinite(value) || value < (p.min ?? 0) || (p.k === "learning_rate" && value <= 0)) {
            throw new Error(`${spec.label} · ${p.label} is outside its allowed range.`);
          }
          if (p.integer && !Number.isInteger(value)) {
            throw new Error(`${spec.label} · ${p.label} must be a whole number.`);
          }
          parameters[p.k] = value;
        } else if (p.type === "list") {
          parameters[p.k] = raw.split(",").map((v) => v.trim()).filter(Boolean);
        } else {
          parameters[p.k] = raw;
        }
      }
      const block = {
        instructions: a.instructions.trim(),
        input_paths: a.inputs.map((x) => x.path),
        output_dir: a.output_dir.trim(),
        parameters,
        enforcement: a.enforcement || "strict",
      };
      // Only include a block that carries something.
      if (block.instructions || block.input_paths.length || block.output_dir || Object.keys(parameters).length) {
        out[spec.id] = block;
      }
    }
    return { agents: out };
  }

  const previewContractPreferences = useMemo(() => {
    try {
      return contractPreferencesFromInputs(inputs);
    } catch {
      return {
        prompt_framing: inputs.promptFraming.trim(),
        system_prompt: inputs.systemPrompt.trim(), loss_objective_config: {},
        inference_config: {}, decoding_config: {},
      };
    }
  }, [inputs]);

  const validationContract = validationContractFromInputs(inputs);

  const userRequest: UserRequest = {
    task_objective: objective.trim(),
    metric: inputs.metric.trim(),
    metric_direction: inputs.metricDirection as "max" | "min",
    metric_type: inputs.metricType as "builtin" | "custom",
    evaluation_script: inputs.metricType === "custom"
      ? inputs.evaluationScript.trim()
      : "",
    evaluator_sha256: "",
    validation_metric: validationContract.metric,
    validation_metric_direction: validationContract.metricDirection,
    validation_metric_type: validationContract.metricType,
    validation_evaluation_script: validationContract.evaluationScript,
    validation_evaluator_sha256: "",
    training_method: inputs.trainingMethod.trim(),
    method_config: methodConfigFromInputs(inputs),
    ...previewContractPreferences,
    dataset: inputs.dataset.trim() || agents["data"]?.inputs[0]?.path || "",
    dataset_split: inputs.datasetSplit.trim(),
    dataset_config: inputs.datasetConfig.trim(),
    data_query: inputs.dataQuery.trim(),
    model_query: inputs.modelQuery.trim(),
    method_query: inputs.methodQuery.trim(),
    base_model: inputs.baseModel.trim(),
    test_set: inputs.testSet.trim(),
    test_answer_fields: columns(inputs.answerFields),
    validation_set: inputs.validationSet.trim(),
    validation_split: inputs.validationSplit.trim(),
    validation_config: inputs.validationConfig.trim(),
    validation_answer_fields: columns(validationContract.answerFields),
    validation_sample_submission: validationContract.sampleSubmission,
    test_sample_submission: inputs.testSampleSubmission.trim(),
    constraints: [],
  };

  async function start() {
    setBusy(true);
    setError(null);
    try {
      if (!runName.trim()) throw new Error("Give the run a name.");
      if (!taskName.trim()) throw new Error("Give the run a task name.");
      if (!objective.trim()) throw new Error("Describe the objective.");
      if (!inputs.metricType) throw new Error("Choose the Test metric type.");
      if (!inputs.metric.trim()) throw new Error("Name the evaluation metric.");
      if (inputs.metricType === "custom" && !inputs.evaluationScript.trim()) {
        throw new Error("Choose the custom evaluation script.");
      }
      if (!inputs.metricDirection) throw new Error("Choose whether the evaluation target is Max or Min.");
      if (validationContract.independent) {
        if (!validationContract.metricType) throw new Error("Choose the Validation metric type.");
        if (!validationContract.metric) throw new Error("Choose the Validation metric.");
        if (validationContract.metricType === "custom" && !validationContract.evaluationScript) {
          throw new Error("Choose the custom Validation evaluation script.");
        }
        if (!validationContract.metricDirection) throw new Error("Choose the Validation target.");
      }
      if (!inputs.testSet.trim()) throw new Error("Choose the held-out test set.");
      if (!columns(inputs.answerFields).length) throw new Error("Name at least one test answer field.");
      if (!inputs.testSampleSubmission.trim()) throw new Error("Choose the test sample submission.");
      if (inputs.validationSet.trim()) {
        if (!columns(inputs.validationAnswerFields).length) throw new Error("A named validation set requires validation answer fields.");
        if (!inputs.validationSampleSubmission.trim()) throw new Error("A named validation set requires a validation sample submission.");
      }
      if (!pickedSetting && saveSetting && !settingName.trim()) throw new Error("Name the setting you want to save.");
      if (!pickedSetting && saveSetting && nameTaken) throw new Error("That setting name is already taken.");
      const user_request = {
        ...userRequest,
        ...contractPreferencesFromInputs(inputs),
        dataset: userRequest.dataset || agents["data"]?.inputs[0]?.path || "",
      };
      const threshold = inputs.stopThreshold.trim()
        ? Number(inputs.stopThreshold)
        : null;
      if (threshold != null && !Number.isFinite(threshold)) {
        throw new Error("Stop threshold must be a finite number.");
      }
      const runtimeHours = inputs.timeLimitHours.trim()
        ? Number(inputs.timeLimitHours)
        : 0;
      if (!Number.isFinite(runtimeHours) || runtimeHours < 0) {
        throw new Error("Time limit must be a non-negative number of hours.");
      }
      const queueWaitHours = inputs.queueWaitHours.trim()
        ? Number(inputs.queueWaitHours)
        : 24;
      if (!Number.isFinite(queueWaitHours) || queueWaitHours <= 0 || queueWaitHours > 168) {
        throw new Error("Max queue wait must be greater than 0 and at most 168 hours.");
      }
      const body = {
        mode: "customized_pipeline",
        task_name: taskName.trim(),
        run_name: runName.trim(),
        user_request,
        customizations: buildCustomizations(),
        // Blank = unlimited, which the backend spells 0.
        iteration_budget: inputs.iterations.trim() ? Math.max(0, Number(inputs.iterations)) : 0,
        max_cost_usd: inputs.budget.trim() ? Math.max(0, Number(inputs.budget)) : 0,
        max_runtime_hours: runtimeHours,
        max_queue_wait_hours: queueWaitHours,
        ...(threshold != null ? { stop_threshold: threshold } : {}),
        ...(inputs.gpuProvider ? { gpu_provider: inputs.gpuProvider } : {}),
        ...(inputs.gpuProvider === "cloud" && inputs.cloudBackend ? { cloud_backend: inputs.cloudBackend } : {}),
        ...(["cluster", "instance"].includes(inputs.gpuProvider) && inputs.sshHostId ? { ssh_host_id: inputs.sshHostId } : {}),
        num_gpus: Number(inputs.numGpus) || 0,
        ...(inputs.generation_backend ? { generation_backend: inputs.generation_backend } : {}),
        ...(pickedSetting ? { setting_id: pickedSetting } : {}),
        ...(!pickedSetting && saveSetting
          ? { save_setting: true, setting_name: settingName.trim() }
          : {}),
      };
      const d = await api<{ run_id?: string }>("/runs", {
        method: "POST",
        body: JSON.stringify(body),
      });
      onDone();
      if (d.run_id) nav(`/runs/${d.run_id}?tab=timeline`);
    } catch (e) {
      setError(String((e as Error).message || e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="max-h-[60vh] space-y-4 overflow-y-auto pr-1">
        <RunInputs {...inputs} requiredMissing={requiredMissing}
          requiredPrefix={(
            <>
              <div>
                <label className={labelCls}>Run name</label>
                <input value={runName} onChange={(e) => setRunName(e.target.value)}
                  placeholder="what to call this run"
                  className={`${fieldCls} font-mono ${runName.trim() ? completeFieldCls : ""}`} />
              </div>
              <div>
                <label className={labelCls}>Task name</label>
                <TaskNameInput value={taskName} onChange={changeTaskName}
                  placeholder="a predefined task, or name your own"
                  className={`${fieldCls} font-mono ${taskName.trim() ? completeFieldCls : ""}`} />
              </div>
              <div>
                <label className={labelCls}>Objective</label>
                {predefined ? (
                  <p className="rounded-md border border-brass-500/55 bg-brass-500/[0.07] p-2.5 font-mono text-sm leading-relaxed text-brass-100">
                    {predefined.task_objective}
                  </p>
                ) : (
                  <textarea value={objective} onChange={(e) => setObjective(e.target.value)} rows={4}
                    placeholder="e.g. Fine-tune to answer MedQA multiple-choice questions."
                    className={`${fieldCls} font-mono ${objective.trim() ? completeFieldCls : ""}`} />
                )}
              </div>
            </>
          )}
          optionalPrefix={predefined ? (
            <section className="rounded-md border border-hair bg-canvas/40 p-3">
              <h3 className="section-title mb-3">Saved Settings</h3>
              <TaskSettingHistory
                task={predefined.name}
                onPick={applySetting}
                selectedId={pickedSetting}
                readOnly
              />
            </section>
          ) : null}
          onChange={(patch) => {
            const runOnly = Object.keys(patch).every(
              (key) => key === "timeLimitHours" || key === "queueWaitHours",
            );
            if (!runOnly) {
              setPickedSetting("");
              setTouched(true);
            }
            setInputs((v) => ({ ...v, ...patch }));
          }}
          beforeChecklist={(
        <section>
          <button
            type="button"
            onClick={() => setShowAgentConfiguration((v) => !v)}
            className="flex w-full items-center justify-between text-left"
          >
            <span className="flex items-center gap-1.5">
              <ChevronRight
                size={13}
                className={`shrink-0 text-slate-400 transition-transform ${showAgentConfiguration ? "rotate-90" : ""}`}
              />
              <h3 className="section-title !text-brass-300">Agent Configuration</h3>
            </span>
          </button>
          {showAgentConfiguration && (
          <div className="mt-4 space-y-4">
          {/* Each Agent is a subtitle at the same left edge. Its fields form a
              separate panel only after that subtitle is opened. */}
          {PIPELINE.map((spec) => {
          const a = agents[spec.id];
          return (
            <section key={spec.id}>
              <button type="button" onClick={() => patch(spec.id, { open: !a.open })}
                className="flex w-full items-center justify-between text-left">
                <span className="flex items-center gap-1.5">
                  <ChevronRight size={13} className={`shrink-0 text-slate-500 transition-transform ${a.open ? "rotate-90" : ""}`} />
                  <span className="font-mono text-xs uppercase tracking-[0.14em] text-slate-300">{spec.label}</span>
                </span>
              </button>
              {a.open && (
                <div className="mt-2 space-y-3 rounded-md border border-hair bg-canvas/50 p-3.5">
                  <div>
                    <label className={labelCls}>Instructions to this agent</label>
                    <textarea value={a.instructions} onChange={(e) => patch(spec.id, { instructions: e.target.value })} rows={2}
                      placeholder="Additional guidance applied only to this agent" className={fieldCls} />
                  </div>
                  <div>
                    <label className={labelCls}>Enforcement</label>
                    <ThemedSelect value={a.enforcement}
                      onChange={(value) => patch(spec.id, { enforcement: value as "strict" | "advisory" })}
                      options={[
                        { value: "strict", label: "Strict" },
                        { value: "advisory", label: "Advisory" },
                      ]}
                      placeholder="Strict by default: fail rather than deviate"
                      ariaLabel="Enforcement"
                      buttonClassName={fieldCls} />
                  </div>
                  {spec.id === "train" && (
                    <div>
                      <TextField
                        label="Loss objective config"
                        value={inputs.lossObjectiveConfig}
                        onChange={(value) => setDetailedInput({ lossObjectiveConfig: value })}
                        hint='Method-specific JSON object, e.g. {"beta":0.1}'
                      />
                    </div>
                  )}
                  {spec.id === "inference" && (
                    <div className="space-y-3">
                      <TextField
                        label="Prompt framing"
                        value={inputs.promptFraming}
                        onChange={(value) => setDetailedInput({ promptFraming: value })}
                        hint="chat, chat:owner/model, completion, or text"
                      />
                      <TextField
                        label="System prompt"
                        value={inputs.systemPrompt}
                        onChange={(value) => setDetailedInput({ systemPrompt: value })}
                        hint="Exact system turn for chat framing"
                        mono={false}
                      />
                      <TextField
                        label="Inference config"
                        value={inputs.inferenceConfig}
                        onChange={(value) => setDetailedInput({ inferenceConfig: value })}
                        hint='Task mapping and parsing JSON, e.g. {"input_fields":["question"]}'
                      />
                      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                        <ChoiceField
                          label="Decoding strategy"
                          value={inputs.decodingStrategy}
                          onChange={(value) => setDetailedInput({ decodingStrategy: value as "" | "greedy" | "sampling" })}
                          options={[["greedy", "Greedy"], ["sampling", "Sampling"]]}
                          hint="Blank lets baseline Inference select"
                        />
                        <NumberField label="Max new tokens" value={inputs.maxNewTokens}
                          onChange={(value) => setDetailedInput({ maxNewTokens: value })}
                          min={1} step={1} hint="Maximum generated tokens" />
                        <NumberField label="Temperature" value={inputs.temperature}
                          onChange={(value) => setDetailedInput({ temperature: value })}
                          min={0} step="any" hint="Sampling temperature" />
                        <NumberField label="Top P" value={inputs.topP}
                          onChange={(value) => setDetailedInput({ topP: value })}
                          min={0} max={1} step="any" hint="Nucleus sampling cutoff" />
                        <NumberField label="Top K" value={inputs.topK}
                          onChange={(value) => setDetailedInput({ topK: value })}
                          min={0} step={1} hint="Top-k sampling cutoff" />
                        <NumberField label="Repetition penalty" value={inputs.repetitionPenalty}
                          onChange={(value) => setDetailedInput({ repetitionPenalty: value })}
                          min={0} step="any" hint="Token repetition penalty" />
                        <NumberField label="Seed" value={inputs.seed}
                          onChange={(value) => setDetailedInput({ seed: value })}
                          step={1} hint="Generation random seed" />
                      </div>
                    </div>
                  )}
                  {spec.parameters.length > 0 && (
                    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                      {spec.parameters.map((p) => (
                        <div key={p.k}>
                          <label className={labelCls}>{p.label}</label>
                          {p.type === "select" ? (
                            <ThemedSelect value={a.parameters[p.k] ?? ""}
                              onChange={(value) => setParam(spec.id, p.k, value)}
                              options={(p.opts || []).map((o) => ({ value: o, label: o }))}
                              placeholder={p.placeholder || `${p.label} used by this agent`}
                              ariaLabel={p.label}
                              buttonClassName={fieldCls} />
                          ) : (
                            <input type={p.type === "number" ? "number" : "text"} step={p.step ?? (p.integer ? "1" : undefined)}
                              min={p.type === "number" ? (p.min ?? 0) : undefined}
                              value={a.parameters[p.k] ?? ""} onChange={(e) => setParam(spec.id, p.k, e.target.value)}
                              placeholder={p.placeholder || `${p.label} used by this agent`} className={fieldCls} />
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                  <div>
                    <label className={labelCls}>Anything else this agent needs</label>
                    <AttachmentDropzone attachments={a.inputs} onChange={(next) => patch(spec.id, { inputs: next })} />
                  </div>
                  <div>
                    <label className={labelCls}>Output location</label>
                    <input value={a.output_dir} onChange={(e) => patch(spec.id, { output_dir: e.target.value })}
                      placeholder="Directory for this agent's outputs; blank uses its ticket workspace" className={fieldCls} />
                  </div>
                </div>
              )}
            </section>
          );
          })}
          </div>
          )}
        </section>
          )}
        />

        {error && <div className="rounded-md border border-coral-500/30 bg-coral-500/10 p-2.5 text-2xs text-coral-300">{error}</div>}
      </div>

      <div className="mt-4 flex items-center justify-end gap-2 border-t border-hair pt-4">
        {taskName.trim() && touched && (
          <div className="mr-auto min-w-0 flex-1">
            {picked ? (
              <p className="font-mono text-2xs text-slate-500">
                reusing <span className="text-brass-300">{picked.name || picked.id.slice(0, 8)}</span>, so nothing new is saved
              </p>
            ) : (
              <div className="flex flex-wrap items-center gap-2">
                <label className="flex shrink-0 items-center gap-2 font-mono text-2xs text-slate-400">
                  <input
                    type="checkbox"
                    checked={saveSetting}
                    onChange={(event) => setSaveSetting(event.target.checked)}
                    className="accent-brass-500"
                  />
                  {predefined
                    ? "save this setting for later reuse"
                    : "save this task and setting for later reuse"}
                </label>
                {saveSetting && (
                  <input
                    value={settingName}
                    onChange={(event) => setSettingName(event.target.value)}
                    placeholder="name it"
                    spellCheck={false}
                    className={`min-w-0 flex-1 rounded border bg-canvas px-2 py-1 font-mono text-2xs text-slate-100 placeholder:text-slate-600 placeholder:opacity-100 ${
                      nameTaken ? "border-coral-500/50" : "border-hair focus:border-brass-500/50"
                    }`}
                  />
                )}
                {saveSetting && nameTaken && (
                  <span className="shrink-0 font-mono text-2xs text-coral-300">
                    that name is taken
                  </span>
                )}
              </div>
            )}
          </div>
        )}
        <button onClick={onDone} className="btn">cancel</button>
        <button onClick={start} disabled={
          busy
          || requiredMissing.length > 0
          || (!pickedSetting && saveSetting && (nameTaken || !settingName.trim()))
        }
          title={requiredMissing.length
            ? `Complete: ${requiredMissing.join(", ")}`
            : ""}
          className="btn btn-brass min-w-[9rem] justify-center uppercase tracking-[0.14em] disabled:opacity-40">
          {busy ? <><RotateCw size={13} className="animate-spin" /> starting…</> : "start run"}
        </button>
      </div>
    </>
  );
}
