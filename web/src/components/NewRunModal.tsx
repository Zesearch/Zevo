import { useEffect, useMemo, useState } from "react";
import { ChevronRight, Eraser } from "lucide-react";
import useSWR from "swr";
import { useNavigate } from "react-router-dom";
import { Modal } from "./Modal";
import { AttachmentDropzone, type Attachment } from "./AttachmentDropzone";
import { CustomizedRunForm } from "./CustomizedRunModal";
import { ModeInfo } from "./RunModeInfo";
import {
  RunInputs,
  RunSummaryTable,
  MultiFileSlot,
  BackendPicker,
  ChoiceField,
  LimitField,
  NumberField,
  TextField,
  TrainingSetupFields,
  EMPTY_RUN_INPUTS,
  methodConfigFromInputs,
  computeTargetValue,
  requiredFieldStateCls,
  useComputeTargets,
  validationContractFromInputs,
  type RunInputValues,
} from "./RunInputs";
import { TaskNameInput } from "./TaskNameInput";
import { TaskSettingHistory } from "./TaskSettings";
import { api } from "../lib/api";
import type {
  CreateRunRequest,
  GenerationBackend,
  TaskDTO,
  TaskSettingDTO,
  UserRequest,
} from "../lib/api";
import { ThemedSelect } from "./ThemedSelect";

export type RunLaunchMode = "auto" | "full_pipeline" | "customized_pipeline" | "single_stage";
type Mode = RunLaunchMode;
type Complexity = "simple" | "advanced";

type TaskSummary = TaskDTO;

type AgentLite = {
  id: string;
  name: string;
  title: string;
};

const SYSTEM_OWNED_EXPERIMENT_DETAILS = {
  prompt_framing: "",
  system_prompt: "",
  loss_objective_config: {},
  inference_config: {},
  decoding_config: {},
} satisfies Pick<
  UserRequest,
  | "prompt_framing"
  | "system_prompt"
  | "loss_objective_config"
  | "inference_config"
  | "decoding_config"
>;

/**
 * Build a UserRequest dict (per schemas/orchestrator.py) from the
 * freeform inputs. Strict mode requires every field; we pass "" / [] for
 * anything the user did not specify.
 */
function buildUserRequest(
  nl: string,
  inputs: RunInputValues,
): UserRequest {
  if (!inputs.metricType) throw new Error("Choose the Test metric type.");
  const validation = validationContractFromInputs(inputs);
  if (validation.independent && !validation.metricType) {
    throw new Error("Choose the Validation metric type.");
  }
  const columns = (v: string) => v.split(",").map((c) => c.trim()).filter(Boolean);
  return {
    task_objective: nl,
    metric: inputs.metric.trim(),
    metric_direction: inputs.metricDirection as "max" | "min",
    metric_type: inputs.metricType,
    evaluation_script: inputs.metricType === "custom" ? inputs.evaluationScript.trim() : "",
    evaluator_sha256: "",
    validation_metric: validation.metric,
    validation_metric_direction: validation.metricDirection,
    validation_metric_type: validation.metricType,
    validation_evaluation_script: validation.evaluationScript,
    validation_evaluator_sha256: "",
    training_method: inputs.trainingMethod,
    method_config: methodConfigFromInputs(inputs),
    // Full Pipeline owns the detailed prompt, loss, and inference contract.
    // Empty values are intentional: Inference and Train declare their owned
    // values in durable Specialist YAML files.
    ...SYSTEM_OWNED_EXPERIMENT_DETAILS,
    dataset: inputs.dataset,
    data_query: inputs.dataQuery.trim(),
    model_query: inputs.modelQuery.trim(),
    method_query: inputs.methodQuery.trim(),
    base_model: inputs.baseModel,
    dataset_split: inputs.datasetSplit.trim(),
    dataset_config: inputs.datasetConfig.trim(),
    test_set: inputs.testSet,
    test_answer_fields: columns(inputs.answerFields),
    validation_set: inputs.validationSet,
    validation_split: inputs.validationSplit.trim(),
    validation_config: inputs.validationConfig.trim(),
    validation_answer_fields: columns(validation.answerFields),
    validation_sample_submission: validation.sampleSubmission,
    test_sample_submission: inputs.testSampleSubmission,
    constraints: [],
  };
}

/** The run-envelope limits, read the same way for every mode. Blank stays
 *  blank: omitted limits resolve to 0 (no hard cap) on the server. */
function limitsFromInputs(inputs: RunInputValues): Record<string, number> {
  const limits: Record<string, number> = {};
  if (inputs.iterations.trim()) limits.iteration_budget = Math.max(0, Number(inputs.iterations));
  if (inputs.budget.trim()) limits.max_cost_usd = Math.max(0, Number(inputs.budget));
  if (inputs.timeLimitHours.trim()) {
    const hours = Number(inputs.timeLimitHours);
    if (!Number.isFinite(hours) || hours < 0) {
      throw new Error("Time limit must be a non-negative number of hours.");
    }
    limits.max_runtime_hours = hours;
  }
  if (inputs.queueWaitHours.trim()) {
    const hours = Number(inputs.queueWaitHours);
    if (!Number.isFinite(hours) || hours <= 0 || hours > 168) {
      throw new Error("Max queue wait must be greater than 0 and at most 168 hours.");
    }
    limits.max_queue_wait_hours = hours;
  }
  if (inputs.stopThreshold.trim()) {
    const threshold = Number(inputs.stopThreshold);
    if (!Number.isFinite(threshold)) {
      throw new Error("Stop threshold must be a finite number.");
    }
    limits.stop_threshold = threshold;
  }
  return limits;
}

/** Where the work runs: provider, cloud/SSH target, GPU cap and generation
 *  backend. Only what the user set is sent; blank means the server default. */
function backendFromInputs(inputs: RunInputValues) {
  return {
    ...(inputs.gpuProvider ? { gpu_provider: inputs.gpuProvider } : {}),
    ...(inputs.gpuProvider === "cloud" && inputs.cloudBackend ? { cloud_backend: inputs.cloudBackend } : {}),
    ...(["cluster", "instance"].includes(inputs.gpuProvider) && inputs.sshHostId ? { ssh_host_id: inputs.sshHostId } : {}),
    num_gpus: Number(inputs.numGpus) || 0,
    ...(inputs.generation_backend ? { generation_backend: inputs.generation_backend } : {}),
  };
}

const fieldCls =
  "w-full rounded-md border border-hair bg-canvas p-2.5 text-sm leading-relaxed text-slate-200 placeholder:text-slate-600 placeholder:opacity-100 focus:border-brass-500/50 focus:outline-none";

export function NewRunModal({
  open,
  onClose,
  initialMode = "auto",
  initialAgent,
  initialTaskName,
  initialSettingId,
  initialInputs,
}: {
  open: boolean;
  onClose: () => void;
  initialMode?: Mode;
  initialAgent?: string;
  initialTaskName?: string;
  initialSettingId?: string;
  initialInputs?: Record<string, string>;
}) {
  const nav = useNavigate();
  const { data: tasks = [] } = useSWR<TaskSummary[]>(open ? "/api/tasks" : null);
  const { data: agents = [] } = useSWR<AgentLite[]>(open ? "/api/agents" : null);

  const [mode, setMode] = useState<Mode>(initialMode);
  // Always the full (advanced) view — the simple/advanced toggle was removed.
  const complexity: Complexity = "advanced";

  // Full-pipeline form
  const [nl, setNl] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  // The named inputs. They used to be the first attachment (dataset) plus a
  // JSON hints blob; every other uploaded file was silently unused.
  const [inputs, setInputs] = useState<RunInputValues>(EMPTY_RUN_INPUTS);
  const compute = useComputeTargets();
  // Every run is named. A name that matches a predefined task runs that task
  // as-is; any other name is a custom task and needs an objective.
  const [taskName, setTaskName] = useState("");
  // What to call THIS execution. The task name says what is being run; many
  // runs share one, so this is what tells them apart in the list afterwards.
  const [runName, setRunName] = useState("");
  // Which past setting the Optional fields came from, so the form can say where
  // its own contents came from. Cleared when the task changes.
  const [pickedSetting, setPickedSetting] = useState("");
  // Whether this run's configuration is written down as a reusable setting,
  // and under what name. Every run used to record one automatically, which is
  // how a task collects settings nobody chose to keep — and why they end up
  // called s1..s5 rather than anything.
  // Whether the inputs have been touched at all — by picking a setting or by
  // typing. Only used to decide when the footer starts talking: naming a task
  // decides nothing, so there is nothing yet to offer to save.
  const [touched, setTouched] = useState(false);
  const [saveSetting, setSaveSetting] = useState(false);
  const [settingName, setSettingName] = useState("");

  // Single-agent form
  const [agentId, setAgentId] = useState<string>(
    initialAgent && initialAgent !== "orchestrator" ? initialAgent : "",
  );
  const [singleNl, setSingleNl] = useState("");
  const [singleAttachments, setSingleAttachments] = useState<Attachment[]>([]);
  const [showSingleChecklist, setShowSingleChecklist] = useState(false);
  // Auto form shares runName / taskName / nl and the envelope half of `inputs`
  // (GPU backend, limits) with Full Pipeline; these only fold its two panels.
  const [showAutoChecklist, setShowAutoChecklist] = useState(false);
  const [showAutoTest, setShowAutoTest] = useState(false);
  const [showAutoTraining, setShowAutoTraining] = useState(false);
  const [showAutoOthers, setShowAutoOthers] = useState(false);
  const [autoTestQuery, setAutoTestQuery] = useState("");
  // One shared owner for all four help popovers. Independent local state let
  // two portaled panels remain open and overlap when their icons were clicked
  // in succession.
  const [openModeInfo, setOpenModeInfo] = useState<Mode | null>(null);
  // Customized owns its form state internally. A key remount gives the shared
  // clear-all control the same semantics in all three modes, while this flag
  // lets the button use the same disabled state as the other forms.
  const [customizedResetKey, setCustomizedResetKey] = useState(0);
  const [customizedDirty, setCustomizedDirty] = useState(false);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /** Put every field back to empty. Filling this dialog in is fiddly and
   *  a wrong predefined task leaves nine fields populated from it, so
   *  starting over should not mean closing and reopening. */
  function clearForm() {
    setNl("");
    setAttachments([]);
    setInputs(EMPTY_RUN_INPUTS);
    setTaskName("");
    setRunName("");
    setPickedSetting("");
    setTouched(false);
    setSaveSetting(false);
    setSettingName("");
    setSingleNl("");
    setSingleAttachments([]);
    setAgentId(initialAgent && initialAgent !== "orchestrator" ? initialAgent : "");
    setShowSingleChecklist(false);
    setShowAutoChecklist(false);
    setShowAutoTest(false);
    setShowAutoTraining(false);
    setShowAutoOthers(false);
    setAutoTestQuery("");
    setOpenModeInfo(null);
    setError(null);
  }

  function clearCurrentForm() {
    if (mode === "customized_pipeline") {
      setCustomizedResetKey((value) => value + 1);
      setCustomizedDirty(false);
      return;
    }
    clearForm();
  }

  // The form itself needs no resetting: App mounts this component only while
  // the dialog is open, so every field is already at its initial value. This
  // seeds the two things that come from props instead.
  useEffect(() => {
    if (!open) return;
    setMode(initialMode);
    setOpenModeInfo(null);
    if (initialTaskName) setTaskName(initialTaskName);
    if (initialAgent && initialAgent !== "orchestrator") setAgentId(initialAgent);
  }, [open, initialMode, initialAgent, initialTaskName]);

  /** Take a setting this task has been run with. Only the setting: the files
   *  belong to the task and are already filled in, and re-applying an old run's
   *  paths would undo a file the user has just swapped for this run. */
  function applySetting(s: TaskSettingDTO) {
    const customizedContract = [
      s.prompt_framing,
      s.system_prompt,
      Object.keys(s.loss_objective_config || {}).length,
      Object.keys(s.inference_config || {}).length,
      Object.keys(s.decoding_config || {}).length,
    ].some(Boolean);
    if (customizedContract) {
      setError(
        "This Setting contains prompt, loss, or inference pins. Launch it in Customized Pipeline.",
      );
      return;
    }
    setPickedSetting(s.id);
    setError(null);
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
      // These fields were verified empty above. Keep the explicit clearing so
      // switching from Customized state cannot leave hidden pins behind.
      promptFraming: "",
      systemPrompt: "",
      lossObjectiveConfig: "",
      inferenceConfig: "",
      decodingStrategy: "",
      maxNewTokens: "",
      temperature: "",
      topP: "",
      topK: "",
      repetitionPenalty: "",
      seed: "",
      // NOT the GPU provider or the generation_backend: a setting no longer carries
      // them, because where the work runs is a choice about today rather than
      // about the experiment. Overwriting them here is what put a blank into a
      // dropdown with no blank option. The Run-level choice or the concrete
      // Settings default remains authoritative when a saved Setting is picked.
      iterations: s.iteration_budget ? String(s.iteration_budget) : "",
      budget: s.max_cost_usd ? String(s.max_cost_usd) : "",
      stopThreshold: s.stop_threshold != null ? String(s.stop_threshold) : "",
    }));
  }

  const trimmedTask = taskName.trim();
  // The orchestrator hands work to the others, while Evaluation is a fixed
  // system runner rather than a callable Agent. Neither belongs in Single Stage.
  const callableAgents = useMemo(
    () => agents.filter((a) => !["orchestrator", "evaluation"].includes(a.id)),
    [agents],
  );

  const predefined = useMemo(
    () => tasks.find((c) => c.name === trimmedTask),
    [tasks, trimmedTask],
  );

  const requiredColumns = (v: string) =>
    v.split(",").map((c) => c.trim()).filter(Boolean);
  const explicitComputeValue = computeTargetValue(inputs);
  const effectiveComputeTarget = (
    compute.targets.find((target) => target.value === explicitComputeValue)
    ?? (!explicitComputeValue ? compute.defaultTarget : undefined)
  );
  const validationContract = validationContractFromInputs(inputs);
  const fullMissing = [
    !runName.trim() && "Run name",
    !trimmedTask && "Task name",
    !(predefined?.task_objective || nl).trim() && "Objective",
    !effectiveComputeTarget && "GPU backend",
    !inputs.metricType && "Test metric type",
    !inputs.metric.trim() && "Metric",
    inputs.metricType === "custom" && !inputs.evaluationScript.trim() && "Evaluation script",
    !inputs.metricDirection && "Target",
    validationContract.independent
      && !validationContract.metricType
      && "Validation metric type",
    validationContract.independent
      && !validationContract.metric
      && "Validation metric",
    validationContract.independent
      && validationContract.metricType === "custom"
      && !validationContract.evaluationScript
      && "Validation evaluation script",
    validationContract.independent
      && !validationContract.metricDirection
      && "Validation target",
    !inputs.testSet.trim() && "Test set",
    requiredColumns(inputs.answerFields).length === 0 && "Test answer fields",
    !inputs.testSampleSubmission.trim() && "Sample submission",
    !!inputs.validationSet.trim()
      && requiredColumns(inputs.validationAnswerFields).length === 0
      && "Validation answer fields",
    !!inputs.validationSet.trim()
      && !inputs.validationSampleSubmission.trim()
      && "Validation sample submission",
    inputs.trainingMethod === "gkd" && !inputs.teacherModel.trim() && "Teacher model",
    inputs.trainingMethod === "online_dpo" && !inputs.rewardModel.trim() && "Reward model",
  ].filter(Boolean) as string[];
  const singleMissing = [
    !runName.trim() && "Run name",
    !trimmedTask && "Task name",
    !singleNl.trim() && "Objective",
    !agentId && "Agent",
  ].filter(Boolean) as string[];
  // Auto always starts a fresh custom task — the server 400s a predefined
  // name, because the scoping stage exists to decide what that task already
  // says. Caught here so the button, not the launch, is what refuses it.
  const autoNameClash = mode === "auto" && !!predefined;
  const autoMissing = [
    !runName.trim() && "Run name",
    !trimmedTask && "Task name",
    autoNameClash && "Task name that is not a predefined task",
    !nl.trim() && "Objective",
    !effectiveComputeTarget && "GPU backend",
    inputs.trainingMethod === "gkd" && !inputs.teacherModel.trim() && "Teacher model",
    inputs.trainingMethod === "online_dpo" && !inputs.rewardModel.trim() && "Reward model",
  ].filter(Boolean) as string[];
  const autoLevel = ({
    "true,true,true": "L1",
    "false,true,true": "L2",
    "false,true,false": "L3",
    "false,false,false": "L4",
  } as Record<string, string>)[[
    !!inputs.dataset.trim(), !!inputs.baseModel.trim(), !!inputs.trainingMethod.trim(),
  ].join(",")] ?? "Custom";
  const autoRequiredValues = [
    { name: "Run name", value: runName.trim() || "Not set", complete: Boolean(runName.trim()) },
    {
      name: "Task name",
      value: trimmedTask || "Not set",
      complete: Boolean(trimmedTask) && !autoNameClash,
    },
    { name: "Objective", value: nl.trim() || "Not set", complete: Boolean(nl.trim()) },
    {
      name: "GPU backend",
      value: effectiveComputeTarget?.label || "Not set",
      complete: Boolean(effectiveComputeTarget),
    },
    ...(inputs.trainingMethod === "gkd" ? [{
      name: "Teacher model",
      value: inputs.teacherModel.trim() || "Not set",
      complete: Boolean(inputs.teacherModel.trim()),
    }] : []),
    ...(inputs.trainingMethod === "online_dpo" ? [{
      name: "Reward model",
      value: inputs.rewardModel.trim() || "Not set",
      complete: Boolean(inputs.rewardModel.trim()),
    }] : []),
  ];
  const autoOptionalValues: Array<{ name: string; value: string; overridden: boolean }> = [
    { name: "Test query", value: autoTestQuery.trim() || "not set", overridden: !!autoTestQuery.trim() },
    { name: "Training data", value: inputs.dataset.trim() || "Prepared by Zevo", overridden: !!inputs.dataset.trim() },
    { name: "Data query", value: inputs.dataQuery.trim() || "not set", overridden: !!inputs.dataQuery.trim() },
    { name: "Base model", value: inputs.baseModel.trim() || "Selected by Zevo", overridden: !!inputs.baseModel.trim() },
    { name: "Model query", value: inputs.modelQuery.trim() || "not set", overridden: !!inputs.modelQuery.trim() },
    { name: "Training method", value: inputs.trainingMethod.trim() || "Decided by Zevo", overridden: !!inputs.trainingMethod.trim() },
    { name: "Method query", value: inputs.methodQuery.trim() || "not set", overridden: !!inputs.methodQuery.trim() },
    { name: "Maximum GPUs", value: inputs.numGpus.trim() || "unlimited", overridden: !!inputs.numGpus.trim() },
    { name: "Generation backend", value: (inputs.generation_backend || "vllm").toUpperCase(), overridden: !!inputs.generation_backend },
    { name: "Iterations", value: inputs.iterations.trim() || "unlimited", overridden: !!inputs.iterations.trim() },
    { name: "Budget", value: inputs.budget.trim() ? `$${inputs.budget.trim()}` : "unlimited", overridden: !!inputs.budget.trim() },
    { name: "Time limit", value: inputs.timeLimitHours.trim() ? `${inputs.timeLimitHours.trim()} h` : "unlimited", overridden: !!inputs.timeLimitHours.trim() },
    { name: "Queue limit", value: inputs.queueWaitHours.trim() ? `${inputs.queueWaitHours.trim()} h` : "24 h", overridden: !!inputs.queueWaitHours.trim() },
    { name: "Stop threshold", value: inputs.stopThreshold.trim() || "not set", overridden: !!inputs.stopThreshold.trim() },
  ];
  const selectedAgent = callableAgents.find((agent) => agent.id === agentId);
  const singleRequiredValues = [
    { name: "Run name", value: runName.trim() || "Not set", complete: Boolean(runName.trim()) },
    { name: "Task name", value: trimmedTask || "Not set", complete: Boolean(trimmedTask) },
    {
      name: "Agent",
      value: selectedAgent ? selectedAgent.title || selectedAgent.id : "Not set",
      complete: Boolean(selectedAgent),
    },
    { name: "Objective", value: singleNl.trim() || "Not set", complete: Boolean(singleNl.trim()) },
  ];
  const singleOptionalValues = [{
    name: "Attachments",
    value: singleAttachments.length ? `${singleAttachments.length} selected` : "none",
    overridden: singleAttachments.length > 0,
  }];
  const launchMissing =
    mode === "auto" ? autoMissing
    : mode === "full_pipeline" ? fullMissing
    : singleMissing;

  function changeFullTaskName(next: string) {
    // Test assets belong to the task. When the user leaves a catalogued task,
    // do not let its held-out files become the invisible starting point of a
    // newly named task.
    if (predefined && next.trim() !== predefined.name) {
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
      setPickedSetting("");
      setSaveSetting(false);
      setSettingName("");
      setTouched(false);
    }
    setTaskName(next);
  }

  // The settings already on this task, for the name-clash check.
  const { data: savedSettings = [] } = useSWR<TaskSettingDTO[]>(
    predefined ? `/api/tasks/${encodeURIComponent(trimmedTask)}/settings` : null,
  );

  // Which setting this configuration already IS, answered by the SERVER so the
  // rule lives in one place.
  //

  const picked = savedSettings.find((x) => x.id === pickedSetting) ?? null;
  // No task-value fallback: a blank field is sent blank, so the configuration
  // this run will use is exactly what the form shows.
  const effective = (formValue: string) => (formValue || "").trim();
  // Every field a setting stores, because every one of them tells two settings
  // apart (server: `setting_identity`). Sending only the four compared fields
  // is what made changing the Budget answer "same as the saved setting": the
  // cap never reached the question.
  const matchQuery = new URLSearchParams({
    base_model: effective(inputs.baseModel),
    training_method: effective(inputs.trainingMethod),
    method_config: JSON.stringify(methodConfigFromInputs(inputs)),
    prompt_framing: "",
    system_prompt: "",
    loss_objective_config: "{}",
    inference_config: "{}",
    decoding_config: "{}",
    dataset: effective(inputs.dataset),
    dataset_split: inputs.datasetSplit.trim(),
    dataset_config: inputs.datasetConfig.trim(),
    data_query: inputs.dataQuery.trim(),
    model_query: inputs.modelQuery.trim(),
    method_query: inputs.methodQuery.trim(),
    validation_set: inputs.validationSet.trim(),
    validation_split: inputs.validationSplit.trim(),
    validation_config: inputs.validationConfig.trim(),
    validation_answer_fields: validationContract.answerFields.trim(),
    validation_sample_submission: validationContract.sampleSubmission,
    validation_metric_type: validationContract.metricType,
    validation_metric: validationContract.metric,
    validation_metric_direction: validationContract.metricDirection,
    validation_evaluation_script: validationContract.evaluationScript,
    iteration_budget: inputs.iterations.trim(),
    max_cost_usd: inputs.budget.trim(),
    stop_threshold: inputs.stopThreshold.trim(),
  }).toString();
  const { data: matched } = useSWR<{ match: TaskSettingDTO | null }>(
    predefined
      ? `/api/tasks/${encodeURIComponent(trimmedTask)}/settings/match?${matchQuery}`
      : null,
  );
  const alreadySaved = matched?.match ?? null;
  const nameTaken = !!settingName.trim() && savedSettings.some(
    (x) => (x.name || "").trim().toLowerCase() === settingName.trim().toLowerCase(),
  );

  // Naming a predefined task fills in what the TASK is: its objective and the
  // held-out scoring specification. Not the setting — that is what the saved
  // settings above are for, and pre-filling one would put a decision in the
  // form that nobody made. Left empty, Zevo decides each of them, which is now
  // literally true: the request is sent as the form stands (see submit), so a
  // blank Training data box reaches the run as blank.
  // Keyed on the task NAME, not the object: tasks are re-fetched on window
  // focus, and a fresh array means fresh object identities — depending on the
  // object made every refocus re-run this reset and silently wipe whatever
  // the user had edited.
  const predefinedName = predefined?.name ?? "";
  useEffect(() => {
    // Not in Auto: a predefined name there is a mistake the checklist reports,
    // and prefilling the task's scoring fields — or clearing the objective the
    // user is mid-way through typing — would be acting on it.
    if (!open || !predefined || mode === "auto") return;
    // A different task is a different question; nothing chosen for the last one
    // carries over, and neither does the fact that something was.
    setTouched(false);
    setPickedSetting("");
    setNl("");
    setInputs((v) => ({
      ...v,
      testSet: predefined.test_set || "",
      answerFields: (predefined.test_answer_fields ?? []).join(", "),
      metricType: predefined.metric_type,
      evaluationScript: predefined.evaluation_script || "",
      testSampleSubmission: predefined.test_sample_submission || "",
      // A page may open this canonical launch form with suggested experiment
      // already selected. These values are still editable; Run name remains
      // blank and required, so the shortcut cannot silently start work.
      ...(initialTaskName === predefined.name ? initialInputs : {}),
      // A predefined Task supplies Launch defaults. The user may adjust this
      // Run's Test metric without changing the saved Task.
      metric: predefined.metric,
      metricDirection: predefined.metric_direction,
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, predefinedName, mode]);

  // A saved Setting arrives after its SWR request, later than the Task itself.
  // Apply it once here rather than launching from the Tasks page through a
  // second, less strict API path.
  useEffect(() => {
    if (!open || !initialSettingId) return;
    const setting = savedSettings.find((s) => s.id === initialSettingId);
    if (setting) applySetting(setting);
    // The modal is mounted fresh for every opening, so one successful match is
    // sufficient and cannot leak into the next launch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, initialSettingId, savedSettings.length]);

  const dirtyForm = !!(
    taskName.trim() || runName.trim() || nl.trim() || singleNl.trim() || autoTestQuery.trim()
    || attachments.length || singleAttachments.length
    || Object.values(inputs).some((v) => String(v || "").trim())
  );

  async function submitFull() {
    // A predefined name executes that package; anything else ships the task the
    // user described, under the name they gave it.
    // Blank stays blank: omitted limits resolve to 0 (no hard cap).
    const columns = (v: string) => v.split(",").map((c) => c.trim()).filter(Boolean);
    const validation = validationContractFromInputs(inputs);
    if (!inputs.metricType) throw new Error("Choose the Test metric type.");
    if (!inputs.metric.trim()) throw new Error("Name the evaluation metric.");
    if (inputs.metricType === "custom" && !inputs.evaluationScript.trim()) {
      throw new Error("Choose the custom evaluation script.");
    }
    if (!inputs.testSet.trim()) throw new Error("Choose the held-out test set.");
    if (!inputs.metricDirection) throw new Error("Choose whether the evaluation target is Max or Min.");
    if (validation.independent) {
      if (!validation.metricType) throw new Error("Choose the Validation metric type.");
      if (!validation.metric) throw new Error("Choose the Validation metric.");
      if (validation.metricType === "custom" && !validation.evaluationScript) {
        throw new Error("Choose the custom Validation evaluation script.");
      }
      if (!validation.metricDirection) throw new Error("Choose the Validation target.");
    }
    if (!columns(inputs.answerFields).length) throw new Error("Name at least one test answer field.");
    if (!inputs.testSampleSubmission.trim()) throw new Error("Choose the test sample submission.");
    if (inputs.validationSet.trim()) {
      if (!columns(inputs.validationAnswerFields).length) {
        throw new Error("A named validation set requires validation answer fields.");
      }
      if (!inputs.validationSampleSubmission.trim()) {
        throw new Error("A named validation set requires a validation sample submission.");
      }
    }
    // Explicit now. Unticked means this configuration is not written down;
    // already saved means there is nothing to write, the row exists.
    const saveFields = alreadySaved
      ? { save_setting: false, setting_id: alreadySaved.id }
      : { save_setting: saveSetting, setting_name: settingName.trim() };
    const limits = limitsFromInputs(inputs);
    let body: CreateRunRequest;
    if (predefined) {
      // Build the complete canonical request from what the form shows.
      const edited = {
        dataset: inputs.dataset,
        dataset_split: inputs.datasetSplit.trim(),
        dataset_config: inputs.datasetConfig.trim(),
        test_set: inputs.testSet,
        test_answer_fields: columns(inputs.answerFields),
        validation_set: inputs.validationSet,
        validation_split: inputs.validationSplit.trim(),
        validation_config: inputs.validationConfig.trim(),
        validation_answer_fields: columns(validation.answerFields),
        validation_sample_submission: validation.sampleSubmission,
        test_sample_submission: inputs.testSampleSubmission,
        base_model: inputs.baseModel,
        model_query: inputs.modelQuery.trim(),
        // Part of the setting, so an edit to it has to count as a change.
        training_method: inputs.trainingMethod,
        method_query: inputs.methodQuery.trim(),
        method_config: methodConfigFromInputs(inputs),
        ...SYSTEM_OWNED_EXPERIMENT_DETAILS,
      };
      // ALWAYS the full request, never the bare task name.
      //
      // Sending only `task_name` let the server fall back to the task row,
      // which still carries a dataset, a base model and a method from before
      // those became the SETTING's business. So an empty Training data box did
      // not mean "Zevo decides" — it meant `trl-lib/Capybara`, invisibly, and
      // the launch quietly ran an L1 configuration the form was showing as
      // blank. What the form says is what the run gets: blank is blank, and
      // blank is Zevo's to decide.
      //
      // The task still supplies what a task IS — its objective and four test
      // fields (prefilled above, so they travel in `edited`).
      body = {
        task_name: trimmedTask,
        run_name: runName.trim(),
        user_request: {
          task_objective: predefined.task_objective,
          metric: inputs.metric.trim(),
          metric_direction: inputs.metricDirection as "max" | "min",
          metric_type: inputs.metricType as "builtin" | "custom",
          evaluation_script: inputs.metricType === "custom"
            ? inputs.evaluationScript.trim()
            : "",
          evaluator_sha256: "",
          validation_metric: validation.metric,
          validation_metric_direction: validation.metricDirection,
          validation_metric_type: validation.metricType,
          validation_evaluation_script: validation.evaluationScript,
          validation_evaluator_sha256: "",
          data_query: inputs.dataQuery.trim(),
          constraints: [],
          ...edited,
        },
        ...limits,
        mode: "full_pipeline",
        ...backendFromInputs(inputs),
        ...saveFields,
      };
    } else {
      body = {
        task_name: trimmedTask,
        run_name: runName.trim(),
        user_request: buildUserRequest(nl, inputs),
        ...limits,
        mode: "full_pipeline",
        ...backendFromInputs(inputs),
        ...saveFields,
      };
    }
    const out = await api<{ run_id: string }>("/runs", {
      method: "POST",
      body: JSON.stringify(body),
    });
    onClose();
    nav(`/runs/${out.run_id}?tab=timeline`);
  }

  /** Auto replaces only the Test setup. Training ownership is expressed by
   *  the same three optional decisions as Standard, including its L1–L4
   *  ladder; scoring fields remain absent and are derived before the loop. */
  async function submitAuto() {
    const body: CreateRunRequest = {
      mode: "auto",
      task_name: trimmedTask,
      run_name: runName.trim(),
      user_request: {
        task_objective: nl.trim(),
        test_query: autoTestQuery.trim(),
        dataset: inputs.dataset,
        dataset_split: inputs.datasetSplit.trim(),
        dataset_config: inputs.datasetConfig.trim(),
        data_query: inputs.dataQuery.trim(),
        base_model: inputs.baseModel.trim(),
        model_query: inputs.modelQuery.trim(),
        training_method: inputs.trainingMethod,
        method_query: inputs.methodQuery.trim(),
        method_config: methodConfigFromInputs(inputs),
        constraints: [],
      },
      ...limitsFromInputs(inputs),
      ...backendFromInputs(inputs),
    };
    const out = await api<{ run_id: string; status: string }>("/runs", {
      method: "POST",
      body: JSON.stringify(body),
    });
    onClose();
    nav(`/runs/${out.run_id}?tab=timeline`);
  }

  async function submitSingle() {
    const body = {
      agent_id: agentId,
      input_format: "freeform",
      task_name: trimmedTask,
      run_name: runName.trim(),
      payload: {
        request: singleNl,
        attachments: singleAttachments.map((a) => a.path),
      },
    };
    const out = await api<{ id: string }>("/tickets", {
      method: "POST",
      body: JSON.stringify(body),
    });
    onClose();
    nav(`/tickets/${out.id}`);
  }

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      if (mode === "auto") {
        if (!runName.trim()) {
          throw new Error("Give the run a name.");
        }
        if (!trimmedTask) {
          throw new Error("Give the run a task name.");
        }
        if (predefined) {
          throw new Error(
            `"${trimmedTask}" is a predefined task. Auto starts a fresh task, so give it a new name — or run the predefined task in Standard.`,
          );
        }
        if (!nl.trim()) {
          throw new Error("Describe the goal — that is the one thing Auto needs from you.");
        }
        await submitAuto();
      } else if (mode === "full_pipeline") {
        if (!runName.trim()) {
          throw new Error("Give the run a name.");
        }
        if (!trimmedTask) {
          throw new Error("Give the run a task name.");
        }
        if (!predefined && !nl.trim()) {
          throw new Error(
            `"${trimmedTask}" is not a predefined task, so describe what you want to achieve.`,
          );
        }
        await submitFull();
      } else {
        if (!runName.trim()) {
          throw new Error("Give the run a name.");
        }
        if (!trimmedTask) {
          throw new Error("Give the run a task name.");
        }
        if (!singleNl.trim()) {
          throw new Error("Describe what you want this agent to do.");
        }
        if (!agentId) throw new Error("Pick an agent.");
        await submitSingle();
      }
    } catch (e) {
      setError(String((e as Error).message || e));
    } finally {
      setBusy(false);
    }
  }

  // One shared control for all three modes. Customized reports its dirty state
  // from the child form and is reset by remounting that form.
  const clearBtn = (
    <button
      type="button"
      onClick={clearCurrentForm}
      disabled={mode === "customized_pipeline" ? !customizedDirty : !dirtyForm}
      title="Empty every field on this form"
      // Short enough to sit on the heading line without reaching the input.
      className="btn !py-0.5 !text-2xs disabled:opacity-30"
    >
      <Eraser size={12} /> clear all
    </button>
  );

  // Rendered by the dialog under its scroller, so "start run" stays in view
  // however long the form is. Customized carries its own footer.
  const footer = mode !== "customized_pipeline" ? (
        <div className="flex justify-end gap-2">
          {/* Cancel stays small and quiet; starting a run is the action this
              dialog exists for, so it carries the weight — same proportion as
              the task dialog's create/cancel pair. */}
          {/* What happens to this configuration after the run — asked only once
              there is a configuration to ask about. Naming the task alone
              decides nothing: the fields are still empty and the Optional
              block is still folded, so announcing "same as s2" there answers a
              question nobody asked, about values the form is not showing.
              (It was not wrong — this task row carries `trl-lib/Capybara` and
              the rest, which IS s2 — but a true statement about invisible
              values reads as a bug.) */}
          {mode === "full_pipeline" && trimmedTask && touched && (
            <div className="mr-auto min-w-0 flex-1">
              {alreadySaved ? (
                <p className="font-mono text-2xs text-slate-500">
                  {picked ? "reusing " : "same as the saved setting "}
                  <span className="text-brass-300">
                    {alreadySaved.name || alreadySaved.id.slice(0, 8)}
                  </span>
                  , so nothing new is saved
                </p>
              ) : (
                <div className="flex flex-wrap items-center gap-2">
                  <label className="flex shrink-0 items-center gap-2 font-mono text-2xs text-slate-400">
                    <input
                      type="checkbox"
                      checked={saveSetting}
                      onChange={(e) => setSaveSetting(e.target.checked)}
                      className="accent-brass-500"
                    />
                    {predefined
                      ? "save this setting for later reuse"
                      : "save this task and setting for later reuse"}
                  </label>
                  {saveSetting && (
                    <input
                      value={settingName}
                      onChange={(e) => setSettingName(e.target.value)}
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
          <button onClick={onClose} className="btn">
            cancel
          </button>
          <button
            onClick={submit}
            disabled={
              busy
              || launchMissing.length > 0
              // A clash would 409 at the end of a launch, which is the worst
              // moment to find out.
              || (saveSetting && !alreadySaved && (nameTaken || !settingName.trim()))
            }
            title={
              launchMissing.length
                ? `Complete: ${launchMissing.join(", ")}`
                : ""
            }
            // min-w holds the width steady across "start run" / "starting…".
            className="btn btn-brass min-w-[9rem] justify-center uppercase tracking-[0.14em] disabled:opacity-30"
          >
            {busy ? "starting…" : "start run"}
          </button>
        </div>
  ) : undefined;

  return (
    <Modal open={open} title="Start a new run" onClose={onClose} width="max-w-[78rem]" footer={footer}>
      <div className="space-y-5 pr-2">
        {/* Two columns: WHICH kind of run on the left, WHAT it is on the right.
            Stacked across the top, the three mode cards took a third of the
            dialog's height to answer a question asked once, and pushed the form
            they configure below the fold. */}
        <div className="grid gap-6 lg:grid-cols-[15rem_minmax(0,1fr)]">
        <div className="flex flex-col gap-2 lg:sticky lg:top-0 lg:self-start">
          {/* First and default: the mode that asks the least. Everything the
              other cards make you decide, this one hands to the scoping
              agents. */}
          <div className="relative">
          <button
            onClick={() => { setMode("auto"); setOpenModeInfo(null); }}
            className={`w-full rounded-bezel border p-3 pr-9 text-left transition ${
              mode === "auto"
                ? "border-brass-500/50 bg-brass-500/10 text-brass-200 shadow-glow-brass"
                : "bezel-flat text-slate-300 hover:border-brass-500/30"
            }`}
          >
            <div className="font-display text-sm font-semibold">Auto</div>
            <p className="mt-1 text-xs leading-snug text-slate-400">
              Zevo defines the Test contract; choose or delegate data, model, and method.
            </p>
          </button>
          <ModeInfo
            mode="auto" open={openModeInfo === "auto"}
            onToggle={() => setOpenModeInfo((value) => value === "auto" ? null : "auto")}
            onClose={() => setOpenModeInfo(null)}
          />
          </div>
          <div className="relative">
          <button
            onClick={() => { setMode("full_pipeline"); setOpenModeInfo(null); }}
            className={`w-full rounded-bezel border p-3 pr-9 text-left transition ${
              mode === "full_pipeline"
                ? "border-brass-500/50 bg-brass-500/10 text-brass-200 shadow-glow-brass"
                : "bezel-flat text-slate-300 hover:border-brass-500/30"
            }`}
          >
            <div className="font-display text-sm font-semibold">Standard</div>
            <p className="mt-1 text-xs leading-snug text-slate-400">
              Let Zevo configure and run the complete agent workflow.
            </p>
          </button>
          <ModeInfo
            mode="full_pipeline" open={openModeInfo === "full_pipeline"}
            onToggle={() => setOpenModeInfo((value) => value === "full_pipeline" ? null : "full_pipeline")}
            onClose={() => setOpenModeInfo(null)}
          />
          </div>
          <div className="relative">
          <button
            onClick={() => { setMode("customized_pipeline"); setOpenModeInfo(null); }}
            className={`w-full rounded-bezel border p-3 pr-9 text-left transition ${
              mode === "customized_pipeline"
                ? "border-brass-500/50 bg-brass-500/10 text-brass-200 shadow-glow-brass"
                : "bezel-flat text-slate-300 hover:border-brass-500/30"
            }`}
          >
            <div className="font-display text-sm font-semibold">Customized</div>
            <p className="mt-1 text-xs leading-snug text-slate-400">
              Run the full workflow with configuration for each agent.
            </p>
          </button>
          <ModeInfo
            mode="customized_pipeline" open={openModeInfo === "customized_pipeline"}
            onToggle={() => setOpenModeInfo((value) => value === "customized_pipeline" ? null : "customized_pipeline")}
            onClose={() => setOpenModeInfo(null)}
          />
          </div>
          <div className="relative">
          <button
            onClick={() => { setMode("single_stage"); setOpenModeInfo(null); }}
            className={`w-full rounded-bezel border p-3 pr-9 text-left transition ${
              mode === "single_stage"
                ? "border-brass-500/50 bg-brass-500/10 text-brass-200 shadow-glow-brass"
                : "bezel-flat text-slate-300 hover:border-brass-500/30"
            }`}
          >
            <div className="font-display text-sm font-semibold">Single Stage</div>
            <p className="mt-1 text-xs leading-snug text-slate-400">
              Send one focused task directly to one agent.
            </p>
          </button>
          <ModeInfo
            mode="single_stage" open={openModeInfo === "single_stage"}
            onToggle={() => setOpenModeInfo((value) => value === "single_stage" ? null : "single_stage")}
            onClose={() => setOpenModeInfo(null)}
          />
          </div>
          <div className="mt-1">{clearBtn}</div>
        </div>

        <div className="relative min-w-0 space-y-5">
        {mode === "auto" ? (
          <div className="space-y-6">
            <section className="space-y-3">
              <div>
                <h3 className="section-title !text-brass-300">Required</h3>
              </div>
              <Field label="Run name">
                <input
                  value={runName}
                  onChange={(e) => setRunName(e.target.value)}
                  placeholder="what to call this run, e.g. bar-exam-take1"
                  className={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(runName.trim()))}`}
                  spellCheck={false}
                />
              </Field>
              {/* A plain input, not TaskNameInput: that one offers the
                  predefined tasks, and Auto cannot take one. */}
              <Field label="Task name">
                <input
                  value={taskName}
                  onChange={(e) => setTaskName(e.target.value)}
                  placeholder="name the task, e.g. bar-exam-reasoning"
                  className={`${fieldCls} font-mono ${
                    requiredFieldStateCls(Boolean(trimmedTask) && !autoNameClash)
                  }`}
                  spellCheck={false}
                />
                {autoNameClash && (
                  <p className="mt-1.5 font-mono text-2xs text-coral-300">
                    "{trimmedTask}" is a predefined task. Auto starts a fresh one — pick another
                    name, or run it in Full Pipeline.
                  </p>
                )}
              </Field>
              <Field label="Objective">
                <textarea
                  value={nl}
                  onChange={(e) => setNl(e.target.value)}
                  placeholder="e.g. Improve a small open model's accuracy on US bar-exam style MCQs."
                  rows={4}
                  className={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(nl.trim()))}`}
                  spellCheck
                />
              </Field>
              <BackendPicker
                gpuProvider={inputs.gpuProvider}
                cloudBackend={inputs.cloudBackend}
                sshHostId={inputs.sshHostId}
                onChange={(patch) => setInputs((value) => ({ ...value, ...patch }))}
              />
            </section>

            <section className="space-y-3">
              <h3 className="section-title !text-brass-300">Optional</h3>
              <div className="space-y-3">
            <section className="space-y-3">
              <button
                type="button"
                onClick={() => setShowAutoTest((value) => !value)}
                className="flex w-full items-center gap-1.5 text-left"
              >
                <ChevronRight
                  size={12}
                  className={`shrink-0 text-slate-400 transition-transform ${showAutoTest ? "rotate-90" : ""}`}
                />
                <span className="field-label !text-slate-100">Test</span>
              </button>
              {showAutoTest && (
                <TextField
                  label="Test query"
                  value={autoTestQuery}
                  onChange={setAutoTestQuery}
                  placeholder=""
                  hint="Describe requirements for the Test set Zevo should select or create."
                />
              )}
            </section>
            <section className="space-y-3">
              <button
                type="button"
                onClick={() => setShowAutoTraining((value) => !value)}
                className="flex w-full items-center justify-between gap-3 text-left"
              >
                <span className="flex items-center gap-1.5">
                  <ChevronRight
                    size={12}
                    className={`shrink-0 text-slate-400 transition-transform ${showAutoTraining ? "rotate-90" : ""}`}
                  />
                  <span className="field-label !text-slate-100">Training</span>
                </span>
                <span className="font-mono text-2xs text-slate-500">{autoLevel}</span>
              </button>
              {showAutoTraining && (
                <TrainingSetupFields
                  inputs={inputs}
                  onChange={(patch) => setInputs((value) => ({ ...value, ...patch }))}
                />
              )}
            </section>

            {/* The run envelope — where it runs and how much it may spend. */}
            <section className="space-y-3">
              <button
                type="button"
                onClick={() => setShowAutoOthers((v) => !v)}
                className="flex w-full items-center gap-1.5 text-left"
              >
                <ChevronRight
                  size={12}
                  className={`shrink-0 text-slate-400 transition-transform ${showAutoOthers ? "rotate-90" : ""}`}
                />
                <span className="field-label !text-slate-100">Others</span>
              </button>
              {showAutoOthers && (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <NumberField
                      label="Maximum GPUs" value={inputs.numGpus}
                      onChange={(v) => setInputs((s) => ({ ...s, numGpus: v }))}
                      min={1}
                      placeholder=""
                      hint="Maximum GPUs Zevo may use at once; the actual plan may use fewer. Blank means unlimited."
                    />
                    <ChoiceField
                      label="Generation backend" value={inputs.generation_backend}
                      onChange={(v) => setInputs((s) => ({ ...s, generation_backend: v as GenerationBackend | "" }))}
                      options={[["vllm", "vllm"], ["hf", "hf"]]}
                      hint="Backend used to generate predictions; blank defaults to vLLM."
                    />
                  </div>
                  <div className="grid grid-cols-1 items-start gap-3 md:grid-cols-2 lg:grid-cols-[0.75fr_0.85fr_1.25fr_1.65fr_1.1fr]">
                    <LimitField label="Iterations" value={inputs.iterations}
                      onChange={(v) => setInputs((s) => ({ ...s, iterations: v }))}
                      hint="Maximum optimization rounds; blank means no iteration cap." />
                    <LimitField label="Budget" value={inputs.budget} prefix="$"
                      onChange={(v) => setInputs((s) => ({ ...s, budget: v }))}
                      hint="Maximum run cost in USD; blank means no cost cap." />
                    <LimitField label="Time limit (hours)" value={inputs.timeLimitHours}
                      onChange={(v) => setInputs((s) => ({ ...s, timeLimitHours: v }))}
                      hint="Active experiment time only; Slurm queue wait is excluded." />
                    <LimitField label="Max queue wait (hours)" value={inputs.queueWaitHours}
                      onChange={(v) => setInputs((s) => ({ ...s, queueWaitHours: v }))}
                      hint="Maximum Slurm PENDING time; blank defaults to 24 hours (max 168)." />
                    <NumberField
                      label="Stop threshold" value={inputs.stopThreshold}
                      onChange={(v) => setInputs((s) => ({ ...s, stopThreshold: v }))}
                      step="any"
                      placeholder=""
                      hint="Validation score that ends the Run, on the metric scoping picks; blank disables it."
                      alignLabel
                    />
                  </div>
                </div>
              )}
            </section>
              </div>
            </section>
            <section>
              <button type="button" onClick={() => setShowAutoChecklist((v) => !v)}
                className="flex w-full items-center justify-between gap-3 text-left">
                <span className="flex items-center gap-1.5">
                  <ChevronRight size={13}
                    className={`shrink-0 text-slate-500 transition-transform ${showAutoChecklist ? "rotate-90" : ""}`} />
                  <span className="section-title !text-brass-300">Checklist</span>
                </span>
                <span className={`font-mono text-[0.62rem] uppercase tracking-[0.12em] ${
                  autoMissing.length ? "text-coral-300" : "text-phosphor-300"
                }`}>
                  {autoMissing.length ? `${autoMissing.length} missing` : "ready"}
                </span>
              </button>
              {showAutoChecklist && (
                <RunSummaryTable
                  requiredValues={autoRequiredValues}
                  optionalValues={autoOptionalValues}
                />
              )}
            </section>

          </div>
        ) : mode === "customized_pipeline" ? (
          <CustomizedRunForm
            key={customizedResetKey}
            onDone={onClose}
            onDirtyChange={setCustomizedDirty}
          />
        ) : mode === "full_pipeline" ? (
          <RunInputs
            {...inputs}
            requiredMissing={fullMissing}
            requiredPrefixValues={[
              { name: "Run name", value: runName.trim() || "Not set", complete: Boolean(runName.trim()) },
              { name: "Task name", value: trimmedTask || "Not set", complete: Boolean(trimmedTask) },
              {
                name: "Objective",
                value: (predefined?.task_objective || nl).trim() || "Not set",
                complete: Boolean((predefined?.task_objective || nl).trim()),
              },
            ]}
            requiredPrefix={(
              <>
                <Field label="Run name">
                  <input value={runName} onChange={(e) => setRunName(e.target.value)}
                    placeholder="what to call this run, e.g. capy-lora-r16-take2"
                    className={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(runName.trim()))}`} spellCheck={false} />
                </Field>
                <Field label="Task name">
                  <TaskNameInput value={taskName} onChange={changeFullTaskName}
                    placeholder="e.g. med for a predefined task, or name your own"
                    className={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(trimmedTask))}`} />
                </Field>
                {predefined ? (
                  <Field label="Objective">
                    <p className="rounded-md border border-hair bg-canvas p-2.5 font-mono text-sm leading-relaxed text-slate-100">
                      {predefined.task_objective}
                    </p>
                  </Field>
                ) : (
                  <Field label="Objective">
                    <textarea value={nl} onChange={(e) => setNl(e.target.value)}
                      placeholder="e.g. Train a model that answers football-rules questions from this PDF, target 80% accuracy."
                      rows={4} className={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(nl.trim()))}`} spellCheck />
                  </Field>
                )}
              </>
            )}
            optionalPrefix={predefined ? (
              <Field label="Saved settings">
                <TaskSettingHistory task={trimmedTask} onPick={applySetting} selectedId={pickedSetting} />
              </Field>
            ) : null}
            onChange={(patch) => {
              // This deadline belongs only to the current Run, so changing it
              // neither detaches nor dirties an otherwise exact Saved Setting.
              const runOnly = Object.keys(patch).every(
                (key) => key === "timeLimitHours" || key === "queueWaitHours",
              );
              if (!runOnly) {
                setPickedSetting("");
                setTouched(true);
              }
              setInputs((v) => ({ ...v, ...patch }));
            }}
            extra={complexity === "advanced" ? (
              <MultiFileSlot label="Other files" values={attachments.map((a) => a.path)}
                hint="Additional reference files available to the pipeline."
                onChange={(paths) => {
                  setAttachments(paths.map((path) => ({
                    path, name: path.split("/").pop() || path, size_bytes: 0, mime: "",
                  })));
                  setInputs((v) => v.dataset || !paths[0] ? v : { ...v, dataset: paths[0] });
                }} />
            ) : null}
          />
        ) : (
          <div className="space-y-6">
            <section className="space-y-3">
              <div>
                <h3 className="section-title !text-brass-300">Required</h3>
              </div>
              <Field label="Run name">
                <input
                  value={runName}
                  onChange={(e) => setRunName(e.target.value)}
                  placeholder="what to call this run, e.g. capy-lora-r16-take2"
                  className={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(runName.trim()))}`}
                  spellCheck={false}
                />
              </Field>

              <Field label="Task name">
                <input
                  value={taskName}
                  onChange={(e) => setTaskName(e.target.value)}
                  placeholder="what this work is"
                  className={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(trimmedTask))}`}
                  spellCheck={false}
                />
              </Field>

              <Field label="Agent">
                <ThemedSelect
                  value={agentId}
                  onChange={setAgentId}
                  options={callableAgents.map((a) => ({ value: a.id, label: `${a.id} · ${a.title}` }))}
                  placeholder="Select the agent that will run this stage"
                  ariaLabel="Agent"
                  buttonClassName={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(agentId))}`}
                />
              </Field>
              <Field label="Objective">
                <textarea
                  value={singleNl}
                  onChange={(e) => setSingleNl(e.target.value)}
                  placeholder="e.g. Make a 200-row chat-JSONL dataset from this PDF, USMLE-style MCQs."
                  rows={4}
                  className={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(singleNl.trim()))}`}
                  spellCheck
                />
              </Field>
            </section>

            <section className="space-y-3">
              <div>
                <h3 className="section-title !text-brass-300">Optional</h3>
              </div>
              <Field label="Attachments">
                <AttachmentDropzone
                  attachments={singleAttachments}
                  onChange={setSingleAttachments}
                />
              </Field>
            </section>
            <section>
              <button type="button" onClick={() => setShowSingleChecklist((v) => !v)}
                className="flex w-full items-center justify-between gap-3 text-left">
                <span className="flex items-center gap-1.5">
                  <ChevronRight size={13}
                    className={`shrink-0 text-slate-500 transition-transform ${showSingleChecklist ? "rotate-90" : ""}`} />
                  <span className="section-title !text-brass-300">Checklist</span>
                </span>
                <span className={`font-mono text-[0.62rem] uppercase tracking-[0.12em] ${
                  singleMissing.length ? "text-coral-300" : "text-phosphor-300"
                }`}>
                  {singleMissing.length ? `${singleMissing.length} missing` : "ready"}
                </span>
              </button>
              {showSingleChecklist && (
                <RunSummaryTable
                  requiredValues={singleRequiredValues}
                  optionalValues={singleOptionalValues}
                />
              )}
            </section>

          </div>
        )}
        </div>
        </div>

        {error && mode !== "customized_pipeline" && (
          <div className="rounded-md border border-coral-500/30 bg-coral-500/10 p-2.5 text-2xs text-coral-300">
            {error}
          </div>
        )}

      </div>
    </Modal>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-3">
        <div className="field-label">{label}</div>
      </div>
      {children}
    </div>
  );
}
