import { SaveRunPrompt } from "./SaveRunPrompt";
import { ClearableInput, ClearableTextarea, ClearFieldsContext } from "./ClearableField";
import "./launchForm.css";
import { trainingStepComplete, runtimeStepComplete, trainingStepReady, runtimeStepReady } from "../lib/launchStepStatus";
import { resourceSummaryValues } from "../lib/runSummary";
import { ModelSaveFields, defaultModelSavePolicy, modelSaveReady, modelSaveSummaryValues } from "./ModelSaveFields";
import { useLaunchSetting } from "../lib/useLaunchSetting";
import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { useNavigate } from "react-router-dom";
import { LaunchStepFlow } from "./LaunchStepFlow";
import { useLaunchPreflight } from "./LaunchPreflight";
import { Modal } from "./Modal";
import { AttachmentDropzone, type Attachment } from "./AttachmentDropzone";
import { CustomizedRunForm } from "./CustomizedRunModal";
import { ModeInfo } from "./RunModeInfo";
import {
  RunInputs,
  RunSummaryTable,
  BackendPicker,
  ChoiceField,
  LimitField,
  NumberField,
  TextField,
  TrainingSetupFields,
  EMPTY_RUN_INPUTS,
  methodConfigFromInputs,
  trainingMethodSupportsLora,
  useLoraLabel,
  normalizeScoringSuite,
  scoringSuiteMissing,
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
import { RunSetupProgressView, useRunSetupProgress } from "./RunSetupProgress";

export type RunLaunchMode = "auto" | "full_pipeline" | "customized_pipeline" | "single_stage";
type Mode = RunLaunchMode;

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
  const testSets = normalizeScoringSuite(inputs.testSets);
  const testMissing = scoringSuiteMissing(testSets, "Test");
  if (testMissing.length) throw new Error(`Complete the Test suite: ${testMissing.join(", ")}.`);
  const primaryTest = testSets[0];
  const validationSets = normalizeScoringSuite(inputs.validationSets);
  const validation = validationContractFromInputs(inputs);
  return {
    task_objective: nl,
    test_sets: testSets,
    validation_sets: validationSets,
    metric: primaryTest.metric,
    metric_direction: primaryTest.metric_direction,
    metric_type: primaryTest.metric_type,
    evaluation_script: primaryTest.metric_type === "custom" ? primaryTest.evaluation_script : "",
    evaluator_sha256: primaryTest.evaluator_sha256,
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
    test_set: primaryTest.test_set,
    test_answer_fields: primaryTest.answer_fields,
    test_sample_submission: primaryTest.sample_submission,
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
    if (!Number.isFinite(hours) || hours < 0 || hours > 168) {
      throw new Error("Max queue wait must be between 0 and 168 hours; blank or 0 means unlimited.");
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
        gpu_allocation_mode: inputs.gpuAllocationMode,
        adaptive_mid_training_evaluation: inputs.adaptiveMidTrainingEvaluation,
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
  const preflight = useLaunchPreflight();
  const [modelSave, setModelSave] = useState({ ...defaultModelSavePolicy });
  const { data: tasks = [] } = useSWR<TaskSummary[]>(open ? "/api/tasks" : null);
  const { data: agents = [] } = useSWR<AgentLite[]>(open ? "/api/agents" : null);

  const [mode, setMode] = useState<Mode>(initialMode);
  const [screen, setScreen] = useState<"mode" | "source" | "form" | "save">("mode");
  const [customContext, setCustomContext] = useState({ taskName: "", settingName: "" });
  const [sourceSkipped, setSourceSkipped] = useState(false);
  const [newTaskName, setNewTaskName] = useState("");
  // Always the full (advanced) view — the simple/advanced toggle was removed.

  // Full-pipeline form
  const [nl, setNl] = useState("");
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
  const [automaticTaskId] = useState(() => crypto.randomUUID());
  const [flowReset, setFlowReset] = useState(0);
  // Which past setting the Optional fields came from, so the form can say where
  // its own contents came from. Cleared when the task changes.
  const { sourceSetting, pickedSetting, setPickedSetting, editSetting } = useLaunchSetting();
  // Whether this run's configuration is written down as a reusable setting,
  // and under what name. Every run used to record one automatically, which is
  // how a task collects settings nobody chose to keep — and why they end up
  // called s1..s5 rather than anything.
  // Whether the inputs have been touched at all — by picking a setting or by
  // typing. Only used to decide when the footer starts talking: naming a task
  // decides nothing, so there is nothing yet to offer to save.
  const [, setTouched] = useState(false);
  const [saveSetting, setSaveSetting] = useState(false);
  const [settingName, setSettingName] = useState("");

  // Single-agent form
  const [agentId, setAgentId] = useState<string>(
    initialAgent && initialAgent !== "orchestrator" ? initialAgent : "",
  );
  const [singleNl, setSingleNl] = useState("");
  const [singleAttachments, setSingleAttachments] = useState<Attachment[]>([]);
  // Auto shares runName, objective and runtime controls with Standard;
  // its task identity is generated independently of the saved Task selector.
  const [autoTestQuery, setAutoTestQuery] = useState("");
  // One shared owner for all four help popovers. Independent local state let
  // two portaled panels remain open and overlap when their icons were clicked
  // in succession.
  const [openModeInfo, setOpenModeInfo] = useState<Mode | null>(null);
  // Customized owns its form state and is seeded when a configuration is chosen.
  const [customizedResetKey, setCustomizedResetKey] = useState(0);
  const [customizedSeed, setCustomizedSeed] = useState<{
    taskName: string;
    settingId: string;
    runName: string;
    inputs?: Partial<RunInputValues>;
  }>({
    taskName: initialTaskName || "",
    settingId: initialSettingId || "",
    runName: "",
    inputs: initialInputs as Partial<RunInputValues> | undefined,
  });

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const runSetup = useRunSetupProgress();

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
      validationSets: s.validation_sets,
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

  const automaticTaskStem = runName.trim().replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 42) || "run";
  const trimmedTask = mode === "auto" || mode === "single_stage"
    ? `${mode === "auto" ? "auto" : "stage"}-${automaticTaskStem}-${automaticTaskId.slice(0, 8)}`
    : taskName.trim();
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
    !modelSaveReady(modelSave) && "Final model destination",
    ...(!predefined?.test_sets?.length ? scoringSuiteMissing(inputs.testSets, "Test") : []),
    ...(inputs.validationSets.length
      ? scoringSuiteMissing(inputs.validationSets, "Validation") : []),
    inputs.trainingMethod === "gkd" && !inputs.teacherModel.trim() && "Teacher model",
    inputs.trainingMethod === "online_dpo" && !inputs.rewardModel.trim() && "Reward model",
  ].filter(Boolean) as string[];
  const singleMissing = [
    !runName.trim() && "Run name",
    !singleNl.trim() && "Objective",
    !agentId && "Agent",
  ].filter(Boolean) as string[];
  // Auto scopes a fresh task; only the visible required fields block launch.
  const autoMissing = [
    !runName.trim() && "Run name",
    !nl.trim() && "Objective",
    !effectiveComputeTarget && "GPU backend",
    !modelSaveReady(modelSave) && "Final model destination",
    inputs.trainingMethod === "gkd" && !inputs.teacherModel.trim() && "Teacher model",
    inputs.trainingMethod === "online_dpo" && !inputs.rewardModel.trim() && "Reward model",
  ].filter(Boolean) as string[];
  const autoRequiredValues = [
    ...modelSaveSummaryValues(modelSave),
    { name: "Run name", value: runName.trim() || "Not set", complete: Boolean(runName.trim()) },
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
  const autoOptionalValues = [
    { name: "Evaluation guidance", value: autoTestQuery.trim() || "not set", overridden: !!autoTestQuery.trim() },
    { name: "Training data", value: inputs.dataset.trim() || "Prepared by Zevo", overridden: !!inputs.dataset.trim() },
    { name: "Training split", value: inputs.datasetSplit || "Default", overridden: !!inputs.datasetSplit },
    { name: "Training config", value: inputs.datasetConfig || "Default", overridden: !!inputs.datasetConfig },
    { name: "Data query", value: inputs.dataQuery.trim() || "not set", overridden: !!inputs.dataQuery.trim() },
    { name: "Base model", value: inputs.baseModel.trim() || "Selected by Zevo", overridden: !!inputs.baseModel.trim() },
    { name: "Model query", value: inputs.modelQuery.trim() || "not set", overridden: !!inputs.modelQuery.trim() },
    { name: "Training method", value: inputs.trainingMethod.trim() || "Decided by Zevo", overridden: !!inputs.trainingMethod.trim() },
    ...(trainingMethodSupportsLora(inputs.trainingMethod)
      ? [{ name: "Use LoRA", value: useLoraLabel(inputs.usePeft), overridden: Boolean(inputs.usePeft) }]
      : []),
    { name: "Method query", value: inputs.methodQuery.trim() || "not set", overridden: !!inputs.methodQuery.trim() },
    ...resourceSummaryValues(inputs),
  ];
  const selectedAgent = callableAgents.find((agent) => agent.id === agentId);
  const singleRequiredValues = [
    { name: "Run name", value: runName.trim() || "Not set", complete: Boolean(runName.trim()) },
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
    setSourceSkipped(false);
    // Test assets belong to the task. When the user leaves a catalogued task,
    // do not let its held-out files become the invisible starting point of a
    // newly named task.
    if (predefined && next.trim() !== predefined.name) {
      setInputs((v) => ({
        ...v,
        testSets: [],
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
    validation_sets: JSON.stringify(inputs.validationSets),
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
  // A setting explicitly picked by the user is authoritative until a
  // Setting-owned field changes (the shared onChange path clears
  // `pickedSetting`).  Do not flash "save for reuse" while the exact-match
  // request catches up, and preserve its id when the Run is submitted.
  const reusableSetting = picked ?? alreadySaved;
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
    const primaryTest = predefined.test_sets?.[0];
    setInputs((v) => ({
      ...v,
      testSets: [],
      testSet: primaryTest?.test_set || predefined.test_set || "",
      answerFields: (primaryTest?.answer_fields ?? predefined.test_answer_fields ?? []).join(", "),
      metricType: primaryTest?.metric_type ?? predefined.metric_type,
      evaluationScript: primaryTest?.evaluation_script || predefined.evaluation_script || "",
      testSampleSubmission: primaryTest?.sample_submission || predefined.test_sample_submission || "",
      // A page may open this canonical launch form with suggested experiment
      // already selected. These values are still editable; Run name remains
      // blank and required, so the shortcut cannot silently start work.
      ...(initialTaskName === predefined.name ? initialInputs : {}),
      // Scalar fields remain the primary projection required by UserRequest.
      // The visible, immutable launch contract is the full Task suite.
      metric: primaryTest?.metric ?? predefined.metric,
      metricDirection: primaryTest?.metric_direction ?? predefined.metric_direction,
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

  async function submitFull() {
    if (sourceSkipped && saveSetting && !newTaskName.trim()) throw new Error("Name the Task you want to save.");
    // A predefined name executes that package; anything else ships the task the
    // user described, under the name they gave it.
    // Blank stays blank: omitted limits resolve to 0 (no hard cap).
    const validation = validationContractFromInputs(inputs);
    if (!predefined?.test_sets?.length) {
      const missing = scoringSuiteMissing(inputs.testSets, "Test");
      if (missing.length) throw new Error(`Complete the Test suite: ${missing.join(", ")}.`);
    }
    if (inputs.validationSets.length) {
      const missing = scoringSuiteMissing(inputs.validationSets, "Validation");
      if (missing.length) throw new Error(`Complete the Validation suite: ${missing.join(", ")}.`);
    }
    // Explicit now. Unticked means this configuration is not written down;
    // already saved means there is nothing to write, the row exists.
    const saveFields = reusableSetting
      ? { save_setting: false, setting_id: reusableSetting.id }
      : { save_setting: saveSetting, setting_name: settingName.trim() };
    const limits = limitsFromInputs(inputs);
    let body: CreateRunRequest;
    if (predefined) {
      const primaryTest = predefined.test_sets[0];
      if (!primaryTest) throw new Error("The selected Task has no Test suite.");
      const validationSets = normalizeScoringSuite(inputs.validationSets);
      // Build the complete canonical request from what the form shows.
      const edited = {
        dataset: inputs.dataset,
        dataset_split: inputs.datasetSplit.trim(),
        dataset_config: inputs.datasetConfig.trim(),
        test_sets: predefined.test_sets,
        test_set: primaryTest.test_set,
        test_answer_fields: primaryTest.answer_fields,
        validation_sets: validationSets,
        test_sample_submission: primaryTest.sample_submission,
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
      // The task still supplies what a task IS — its objective and complete
      // Test suite. The scalar values are only the suite's primary projection.
      body = {
        task_name: sourceSkipped && saveSetting ? newTaskName.trim() : trimmedTask,
        run_name: runName.trim(),
        user_request: {
          task_objective: predefined.task_objective,
          metric: primaryTest.metric,
          metric_direction: primaryTest.metric_direction,
          metric_type: primaryTest.metric_type,
          evaluation_script: primaryTest.metric_type === "custom"
            ? primaryTest.evaluation_script
            : "",
          evaluator_sha256: primaryTest.evaluator_sha256,
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
        task_name: sourceSkipped && saveSetting ? newTaskName.trim() : trimmedTask,
        run_name: runName.trim(),
        user_request: buildUserRequest(nl, inputs),
        ...limits,
        mode: "full_pipeline",
        ...backendFromInputs(inputs),
        ...saveFields,
      };
    }
    body = { ...body, model_save_policy: modelSave };
    if (!(await preflight.check(body))) return;
    const setupId = runSetup.begin();
    body = { ...body, setup_id: setupId };
    const out = await api<{ run_id?: string; setup_id?: string; status: string }>("/runs", {
      method: "POST",
      body: JSON.stringify(body),
    });
    const runId = out.run_id || (await runSetup.wait(out.setup_id || setupId)).run_id;
    if (!runId) throw new Error("Run setup completed without a Run id.");
    onClose();
    nav(`/runs/${runId}?tab=timeline`);
  }

  /** Auto replaces only the Test setup. Training ownership is expressed by
   *  the same three optional decisions as Standard, including its L1–L4
   *  ladder; scoring fields remain absent and are derived before the loop. */
  async function submitAuto() {
    const body: CreateRunRequest = {
      mode: "auto",
      model_save_policy: modelSave,
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
    if (!(await preflight.check(body))) return;
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
        if (!singleNl.trim()) {
          throw new Error("Describe what you want this agent to do.");
        }
        if (!agentId) throw new Error("Pick an agent.");
        await submitSingle();
      }
    } catch (e) {
      setError(String((e as Error).message || e));
    } finally {
      runSetup.stop();
      setBusy(false);
    }
  }

  const launchFeedback = <div className="space-y-3">
    {(mode === "auto" || mode === "full_pipeline") && preflight.panel}
    {error && <div role="alert" className="whitespace-pre-line text-sm text-coral-300">{error}</div>}
    {busy && runSetup.progress && <RunSetupProgressView progress={runSetup.progress} />}
  </div>;
  const reviewAction = <div className="space-y-4">
    {screen === "form" && launchFeedback}
    <button type="button" className="launch-next" disabled={busy || launchMissing.length > 0 || (["auto", "full_pipeline"].includes(mode) && !modelSaveReady(modelSave))} onClick={() => {
      if (mode === "full_pipeline" && !reusableSetting) setScreen("save");
      else void submit();
    }}>{busy ? "Starting…" : "Start Run →"}</button>
  </div>;
  const headerTask = mode === "customized_pipeline" ? customContext.taskName : sourceSkipped ? (saveSetting ? newTaskName : "") : taskName;
  const headerSetting = mode === "customized_pipeline" ? customContext.settingName : picked?.name || "";

  return (
    <ClearFieldsContext.Provider value={true}><Modal fullscreen open={open} title="Launch Run" onClose={onClose} width="max-w-[78rem]">
      <div className="launch-experience mx-auto flex h-full min-h-0 w-full max-w-[1400px] flex-col md:w-[94%]">
        {screen !== "mode" && <div className="mb-8 grid shrink-0 grid-cols-[auto_minmax(0,1fr)] items-center gap-5 md:grid-cols-[1fr_auto_1fr]">
          <button type="button" className="launch-back justify-self-start" onClick={() => setScreen("mode")}>← Change mode</button>
          <div className="flex min-w-0 flex-wrap items-center justify-center gap-x-5 gap-y-2 text-sm text-slate-300">
            <span><span className="mr-2 text-xs text-slate-500">Mode</span>{mode === "auto" ? "Auto" : mode === "full_pipeline" ? "Standard" : mode === "customized_pipeline" ? "Customized" : "Single Stage"}</span>
            {!["auto", "single_stage"].includes(mode) && headerTask && <span className="inline-flex min-w-0 items-center gap-2"><span className="text-xs text-slate-500">Task</span><span className="max-w-[22rem] truncate" title={headerTask}>{headerTask}</span></span>}
            {!["auto", "single_stage"].includes(mode) && headerSetting && <span className="inline-flex min-w-0 items-center gap-2"><span className="text-xs text-slate-500">Setting</span><span className="max-w-[12rem] truncate" title={headerSetting}>{headerSetting}</span></span>}
          </div>
        </div>}
        {screen === "save" && <SaveRunPrompt newTask={!predefined} askTaskName={sourceSkipped} save={saveSetting} onSave={setSaveSetting} taskName={newTaskName} onTaskName={setNewTaskName} settingName={settingName} onSettingName={setSettingName} nameTaken={nameTaken} busy={busy} onBack={() => setScreen("form")} onLaunch={() => void submit()} feedback={launchFeedback} />}
        <div hidden={screen !== "mode"} className="my-auto py-10">
          <p className="mb-4 text-xs uppercase tracking-[0.24em] text-brass-300">A new starting point</p>
          <h3 className="font-display text-4xl font-semibold tracking-tight text-ink md:text-5xl">What would you like to run?</h3>
          <p className="mb-10 mt-4 text-sm text-slate-400">Choose how much you want to define. We’ll take it one step at a time.</p>
          <div className="grid gap-4 sm:grid-cols-2">
          {/* First and default: the mode that asks the least. Everything the
              other cards make you decide, this one hands to the scoping
              agents. */}
          <div className="relative">
          <button
            onClick={() => { setMode("auto"); setFlowReset(value => value + 1); setScreen("form"); setOpenModeInfo(null); }}
            className={`h-full min-h-[160px] w-full rounded-2xl border border-transparent p-6 pr-10 text-left transition duration-200 hover:-translate-y-1 ${
              mode === "auto"
                ? "bg-brass-500/[0.06] text-brass-200"
                : "bg-white/[0.015] text-slate-300 hover:bg-white/[0.04]"
            }`}
          >
            <div className="font-display text-xl font-semibold">Auto</div>
            <p className="mt-1 text-xs leading-snug text-slate-400">
              Describe the goal; Zevo chooses the Test suite, then runs Standard.
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
            onClick={() => { setMode("full_pipeline"); setFlowReset(value => value + 1); setScreen("source"); setOpenModeInfo(null); }}
            className={`h-full min-h-[160px] w-full rounded-2xl border border-transparent p-6 pr-10 text-left transition duration-200 hover:-translate-y-1 ${
              mode === "full_pipeline"
                ? "bg-brass-500/[0.06] text-brass-200"
                : "bg-white/[0.015] text-slate-300 hover:bg-white/[0.04]"
            }`}
          >
            <div className="font-display text-xl font-semibold">Standard</div>
            <p className="mt-1 text-xs leading-snug text-slate-400">
              Choose the Task and basic Setting; Zevo owns execution details.
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
            onClick={() => {
              setCustomizedSeed({
                taskName,
                settingId: pickedSetting,
                runName,
                inputs,
              });
              setCustomizedResetKey((value) => value + 1);
              setMode("customized_pipeline");
              setFlowReset(value => value + 1); setScreen("source");
              setOpenModeInfo(null);
            }}
            className={`h-full min-h-[160px] w-full rounded-2xl border border-transparent p-6 pr-10 text-left transition duration-200 hover:-translate-y-1 ${
              mode === "customized_pipeline"
                ? "bg-brass-500/[0.06] text-brass-200"
                : "bg-white/[0.015] text-slate-300 hover:bg-white/[0.04]"
            }`}
          >
            <div className="font-display text-xl font-semibold">Customized</div>
            <p className="mt-1 text-xs leading-snug text-slate-400">
              Standard plus hyperparameter, inference, or per-agent overrides.
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
            onClick={() => { setMode("single_stage"); setFlowReset(value => value + 1); setScreen("form"); setOpenModeInfo(null); }}
            className={`h-full min-h-[160px] w-full rounded-2xl border border-transparent p-6 pr-10 text-left transition duration-200 hover:-translate-y-1 ${
              mode === "single_stage"
                ? "bg-brass-500/[0.06] text-brass-200"
                : "bg-white/[0.015] text-slate-300 hover:bg-white/[0.04]"
            }`}
          >
            <div className="font-display text-xl font-semibold">Single Stage</div>
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
          </div>
        </div>
        <section hidden={screen !== "source"} aria-label="Starting configuration" className="my-auto w-full max-w-4xl self-center py-8 md:px-8">
          <h3 className="font-display text-4xl font-semibold tracking-tight text-ink">Start with something familiar?</h3>
          <p className="mb-8 mt-4 text-sm leading-relaxed text-slate-400">Reuse a Task and saved Setting, or skip to start fresh. You can choose to save your new Task and Setting at the end.</p>
          <div className="space-y-6">
                <Field label="Task name">
                  <TaskNameInput value={sourceSkipped ? "" : taskName} onChange={changeFullTaskName}
                    placeholder="Choose a Task"
                    className={fieldCls} />
                </Field>
            {!predefined && !sourceSkipped && taskName.trim() && <Field label="Saved settings"><p className="text-sm text-slate-500">No saved settings for this Task.</p></Field>}
            {predefined && <>
              <Field label="Saved settings">
                <TaskSettingHistory compact onClear={() => { setPickedSetting(""); setInputs({ ...EMPTY_RUN_INPUTS }); setTouched(true); setSaveSetting(false); }}
                  task={trimmedTask}
                  onPick={applySetting}
                  selectedId={sourceSetting}
                  modified={Boolean(sourceSetting && reusableSetting?.id !== sourceSetting)}
                />
              </Field>
            </>}
          </div>
          <div className="mt-10 flex items-center justify-end pt-5">
            <button type="button" className="launch-next" onClick={() => {
              const skip = sourceSkipped || !taskName.trim();
              const nextTask = skip ? `task-${crypto.randomUUID().slice(0, 8)}` : taskName;
              setSourceSkipped(skip);
              if (skip) { setTaskName(nextTask); setPickedSetting(""); setSaveSetting(false); }
              if (mode === "customized_pipeline") {
                setCustomizedSeed({ taskName: nextTask, settingId: skip ? "" : pickedSetting, runName, inputs });
                setCustomizedResetKey(value => value + 1);
              }
              setFlowReset(value => value + 1); setScreen("form");
            }}>{sourceSkipped || !taskName.trim() ? "Skip →" : "Continue →"}</button>
          </div>
        </section>
        <div hidden={screen !== "form"} className={screen === "form" ? "relative flex min-h-0 flex-1 flex-col gap-4" : "hidden"}>
        {mode === "auto" ? (
          <LaunchStepFlow reviewAction={reviewAction} key={`auto-${flowReset}`} steps={[
            { label: "Basic info", group: "Required", complete: autoRequiredValues.filter(value => !["Teacher model", "Reward model"].includes(value.name)).every(value => value.complete), content: <><Field label="Run name">
                <ClearableInput
                  value={runName}
                  onChange={(e) => setRunName(e.target.value)}
                  placeholder="what to call this run, e.g. bar-exam-take1"
                  className={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(runName.trim()))}`}
                  spellCheck={false}
                />
              </Field><Field label="Objective">
                <ClearableTextarea
                  value={nl}
                  onChange={(e) => setNl(e.target.value)}
                  placeholder="e.g. Improve a small open model's accuracy on US bar-exam style MCQs."
                  rows={4}
                  className={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(nl.trim()))}`}
                  spellCheck
                />
              </Field><BackendPicker
                gpuProvider={inputs.gpuProvider}
                cloudBackend={inputs.cloudBackend}
                sshHostId={inputs.sshHostId}
                onChange={(patch) => setInputs((value) => ({ ...value, ...patch }))}
              /><ModelSaveFields value={modelSave} onChange={setModelSave} /></> },
            { label: "Training", group: "Optional", ready: trainingStepReady(inputs), complete: trainingStepComplete(inputs), content: <TrainingSetupFields inputs={inputs} onChange={patch => setInputs(value => ({ ...value, ...patch }))} /> },
            { label: "Evaluation", group: "Optional", complete: Boolean(autoTestQuery.trim()), content: <TextField label="Evaluation guidance" value={autoTestQuery} onChange={setAutoTestQuery} placeholder="" hint="Optional coverage or format guidance; Zevo decides the Test suite and each scoring contract." /> },
            { label: "Others", group: "Optional", ready: runtimeStepReady(inputs), complete: runtimeStepComplete(inputs), content:                 <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <NumberField
                      label="Maximum GPUs" value={inputs.numGpus}
                      onChange={(v) => setInputs((s) => ({ ...s, numGpus: v }))}
                      min={1}
                      placeholder=""
                      hint="Maximum GPUs Zevo may use at once; the actual plan may use fewer. Blank means unlimited."
                    />
                    <ChoiceField
            label="GPU allocation" value={inputs.gpuAllocationMode}
            onChange={(v) => setInputs((s) => ({ ...s, gpuAllocationMode: v as "per_stage" | "per_run" | "per_submission" }))}
            options={[["per_run", "Entire run"], ["per_stage", "Per stage"], ["per_submission", "Per submission (Slurm)"]]}
            hint="Entire run retains GPUs across stages. Per stage retains GPUs within a stage, including repair. Per submission (Slurm only) releases GPUs after each workload and requests a new allocation for retries."
          />
          <ChoiceField
            label="Mid-training evaluation" value={inputs.adaptiveMidTrainingEvaluation ? "enabled" : "disabled"}
            onChange={(v) => setInputs((s) => ({ ...s, adaptiveMidTrainingEvaluation: v === "enabled" }))}
            options={[["disabled", "Off"], ["enabled", "Agent decides"]]}
            hint="Let the agent evaluate Validation checkpoints during training and decide whether to stop early."
          />
          <ChoiceField
                      label="Generation backend" value={inputs.generation_backend || "vllm"}
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
                      hint="Maximum Slurm PENDING time; blank means unlimited (optional cap: max 168 hours)." />
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
 },
            { label: "Checklist", group: "Review", complete: autoMissing.length === 0, content: <><RunSummaryTable requiredValues={autoRequiredValues} optionalValues={autoOptionalValues} /></> },
          ]} />
        ) : mode === "customized_pipeline" ? (
          <CustomizedRunForm onContextChange={setCustomContext} guided askTaskNameOnSave={sourceSkipped}
            key={customizedResetKey}
            onDone={onClose}
            initialTaskName={customizedSeed.taskName || initialTaskName}
            initialSettingId={customizedSeed.settingId || initialSettingId}
            initialRunName={customizedSeed.runName}
            initialInputs={customizedSeed.inputs}
          />
        ) : mode === "full_pipeline" ? (
          <RunInputs reviewAction={reviewAction} key={`standard-${flowReset}`} hideSource
            extra={<ModelSaveFields value={modelSave} onChange={setModelSave} />}
            extraRequiredValues={modelSaveSummaryValues(modelSave)}
            {...inputs}
            taskTestSuite={predefined?.test_sets} validationSettingId={pickedSetting}
            requiredMissing={fullMissing}
            requiredPrefixValues={[
              { name: "Run name", value: runName.trim() || "Not set", complete: Boolean(runName.trim()) },
              { name: "Task name", value: sourceSkipped ? (saveSetting && newTaskName.trim() ? newTaskName.trim() : "New task") : trimmedTask || "Not set", complete: Boolean(trimmedTask) },
              {
                name: "Objective",
                value: (predefined?.task_objective || nl).trim() || "Not set",
                complete: Boolean((predefined?.task_objective || nl).trim()),
              },
            ]}
            taskSelector={(
                <Field label="Task name">
                  <TaskNameInput value={sourceSkipped ? "" : taskName} onChange={changeFullTaskName}
                    placeholder="Choose a Task"
                    className={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(trimmedTask))}`} />
                </Field>
            )}
            requiredPrefix={(
              <>
                <Field label="Run name">
                  <ClearableInput value={runName} onChange={(e) => setRunName(e.target.value)}
                    placeholder="what to call this run, e.g. capy-lora-r16-take2"
                    className={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(runName.trim()))}`} spellCheck={false} />
                </Field>

                {predefined ? (
                  <Field label="Objective">
                    <p className={`launch-readonly-value rounded-md border bg-canvas p-2.5 font-mono text-sm leading-relaxed ${requiredFieldStateCls(true)}`}>
                      {predefined.task_objective}
                    </p>
                  </Field>
                ) : (
                  <Field label="Objective">
                    <ClearableTextarea value={nl} onChange={(e) => setNl(e.target.value)}
                      placeholder="e.g. Train a model that answers football-rules questions from this PDF, target 80% accuracy."
                      rows={4} className={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(nl.trim()))}`} spellCheck />
                  </Field>
                )}
              </>
            )}
            optionalPrefix={predefined ? (
              <Field label="Saved settings">
                <TaskSettingHistory compact
                  task={trimmedTask}
                  onPick={applySetting}
                  selectedId={sourceSetting}
                  modified={Boolean(sourceSetting && reusableSetting?.id !== sourceSetting)}
                />
              </Field>
            ) : null}
            onChange={(patch) => {
              if (editSetting(patch, inputs)) setTouched(true);
              setInputs((v) => ({ ...v, ...patch }));
            }}

          />
        ) : (
          <LaunchStepFlow reviewAction={reviewAction} key={`single-${flowReset}`} steps={[
            { label: "Basic info", group: "Required", complete: Boolean(runName.trim() && singleNl.trim()), content: <><Field label="Run name">
                <ClearableInput
                  value={runName}
                  onChange={(e) => setRunName(e.target.value)}
                  placeholder="what to call this run, e.g. capy-lora-r16-take2"
                  className={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(runName.trim()))}`}
                  spellCheck={false}
                />
              </Field><Field label="Objective">
                <ClearableTextarea
                  value={singleNl}
                  onChange={(e) => setSingleNl(e.target.value)}
                  placeholder="e.g. Make a 200-row chat-JSONL dataset from this PDF, USMLE-style MCQs."
                  rows={4}
                  className={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(singleNl.trim()))}`}
                  spellCheck
                />
              </Field></> },
            { label: "Agent", group: "Required", complete: Boolean(selectedAgent), content: <Field label="Agent">
                <ThemedSelect
                  value={agentId}
                  onChange={setAgentId}
                  options={callableAgents.map((a) => ({ value: a.id, label: `${a.id} · ${a.title}` }))}
                  placeholder="Select the agent that will run this stage"
                  ariaLabel="Agent"
                  buttonClassName={`${fieldCls} font-mono ${requiredFieldStateCls(Boolean(agentId))}`}
                />
              </Field> },
            { label: "Attachments", group: "Optional", complete: singleAttachments.length > 0, content: <Field label="Attachments">
                <AttachmentDropzone
                  attachments={singleAttachments}
                  onChange={setSingleAttachments}
                />
              </Field> },
            { label: "Checklist", group: "Review", complete: singleMissing.length === 0, content: <><RunSummaryTable requiredValues={singleRequiredValues} optionalValues={singleOptionalValues} /></> },
          ]} />
        )}
        </div>
      </div>
    </Modal></ClearFieldsContext.Provider>
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
