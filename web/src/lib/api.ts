/** Thin fetch wrapper — the one error-shaping implementation for every
 *  mutating call (POST/PATCH/PUT/DELETE). All paths are relative to /api:
 *  in dev they go through the Vite proxy (see vite.config.ts), in prod through
 *  nginx -> backend:8000. SWR GET calls keep using the fetcher in main.tsx.
 *
 *  A JSON body gets the JSON content type; a FormData body must NOT set one,
 *  or the browser cannot append the multipart boundary. Failures throw an
 *  Error whose message prefers FastAPI's `{"detail": ...}` over the raw body,
 *  so every caller surfaces the same shape of message. */
export const API_BASE = "/api";

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const isForm = init?.body instanceof FormData;
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(isForm ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const body = await res.text();
    let msg = body;
    try {
      const d = JSON.parse(body);
      if (d?.detail) msg = typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail);
    } catch {
      /* not JSON — keep the raw body */
    }
    throw new Error(`API ${res.status}: ${msg || res.statusText}`);
  }
  if (res.status === 204) return undefined as T;
  const body = await res.text();
  return (body ? JSON.parse(body) : undefined) as T;
}

// ----- Types (mirror backend Pydantic DTOs; keep in sync manually for now) -----

export type RunStatus = "planning" | "running" | "success" | "degraded" | "failed" | "halted" | "cancelled";
export type RunMode = "auto" | "full_pipeline" | "customized_pipeline" | "single_stage";
/** Where an Auto run's held-out evaluation came from once scoping settled it.
 *  Empty while scoping is still running (and for every non-Auto run). */
export type EvalSource = "" | "public_benchmark" | "synthesized";
export type TicketStatus = "queued" | "running" | "repairing" | "awaiting_input" | "waiting_external" | "succeeded" | "degraded" | "failed" | "skipped" | "cancelled";
export type TicketInputFormat = "typed" | "freeform";
export type TicketLane = "optimization" | "held_out_test";
export type GenerationBackend = "hf" | "vllm";
export type GpuProvider = "cluster" | "cloud" | "instance";
export type MetricDirection = "max" | "min";
export type MetricType = "builtin" | "custom";

export type TaskTestSet = {
  name: string;
  test_set: string;
  inference_query: string;
  sample_submission: string;
  metric_type: MetricType;
  metric: string;
  answer_fields: string[];
  metric_direction: MetricDirection;
  evaluation_script: string;
  evaluator_sha256: string;
};

export type TaskDTO = {
  name: string;
  task_objective: string;
  test_sets: TaskTestSet[];
  test_set: string;
  test_answer_fields: string[];
  test_sample_submission: string;
  metric_type: MetricType;
  evaluation_script: string;
  evaluator_sha256: string;
  metric: string;
  metric_direction: MetricDirection;
  run_count: number;
  last_run_at: string;
  best_test_score: number | null;
  created_at: string;
};

export type UserRequest = {
  task_objective: string;
  test_sets?: TaskTestSet[];
  /** Effective held-out Test scoring values for this Run. */
  metric: string;
  metric_direction: MetricDirection;
  metric_type: MetricType;
  evaluation_script: string;
  evaluator_sha256?: string;
  /** Validation contract; inherited from Test when validation_set is blank. */
  validation_metric: string;
  validation_metric_direction: MetricDirection | "";
  validation_metric_type: MetricType | "";
  validation_evaluation_script: string;
  validation_evaluator_sha256?: string;
  training_method: string;
  method_config: Record<string, unknown>;
  prompt_framing?: string;
  system_prompt?: string;
  loss_objective_config?: Record<string, unknown>;
  inference_config?: Record<string, unknown>;
  decoding_config?: Record<string, unknown>;
  dataset: string;
  dataset_split?: string;
  dataset_config?: string;
  data_query: string;
  model_query?: string;
  method_query?: string;
  base_model: string;
  test_set: string;
  test_answer_fields: string[];
  validation_set?: string;
  validation_split?: string;
  validation_config?: string;
  validation_answer_fields?: string[];
  validation_sample_submission?: string;
  test_sample_submission: string;
  constraints: string[];
};

/** Auto derives the Test/evaluation contract. Optimization keeps the same
 *  optional data/model/method pins and queries as Standard, so the ordinary
 *  L1–L4 ownership ladder still applies. */
export type AutoUserRequest = {
  task_objective: string;
  test_query?: string;
  dataset?: string;
  dataset_split?: string;
  dataset_config?: string;
  data_query?: string;
  base_model?: string;
  model_query?: string;
  training_method?: string;
  method_query?: string;
  method_config?: Record<string, unknown>;
  constraints?: string[];
};

export type AgentCustomization = {
  instructions: string;
  input_paths: string[];
  output_dir: string;
  parameters: Record<string, string | number | boolean | string[]>;
  enforcement: "strict" | "advisory";
};

/** The run envelope every launch mode shares: identity plus limits and where
 *  the work runs. Mode-specific payloads are added in CreateRunRequest. */
type RunEnvelope = {
  task_name: string;
  run_name: string;
  iteration_budget?: number;
  stop_threshold?: number;
  max_cost_usd?: number;
  max_runtime_hours?: number;
  max_queue_wait_hours?: number;
  generation_backend?: GenerationBackend;
  /** Maximum GPUs the Run may use at once; 0/omitted means unlimited. */
  num_gpus?: number;
  gpu_provider?: GpuProvider;
  // Which cloud to rent on when gpu_provider === "cloud" (empty = deployment
  // default). Sent by the run modals; declared here so the body is fully typed.
  cloud_backend?: "" | "vastai" | "lambda";
  // Verified SSH profile id for a cluster/instance run (empty = env fallback).
  ssh_host_id?: string;
};

export type CreateRunRequest = RunEnvelope & (
  | {
      mode?: "full_pipeline" | "customized_pipeline";
      user_request?: UserRequest;
      customizations?: { agents: Record<string, AgentCustomization>; note?: string };
      save_setting?: boolean;
      setting_name?: string;
      setting_id?: string;
    }
  // Auto owns the held-out contract; settings and per-agent customizations are
  // still not part of this arm.
  | {
      mode: "auto";
      user_request: AutoUserRequest;
    }
);

export type TaskSettingDTO = {
  id: string;
  name: string;
  created_at: string;
  base_model: string;
  training_method: string;
  method_config: Record<string, unknown>;
  prompt_framing: string;
  system_prompt: string;
  loss_objective_config: Record<string, unknown>;
  inference_config: Record<string, unknown>;
  decoding_config: Record<string, unknown>;
  dataset: string;
  dataset_split: string;
  dataset_config: string;
  validation_set: string;
  validation_split: string;
  validation_config: string;
  validation_answer_fields: string[];
  validation_sample_submission: string;
  validation_metric_type: MetricType;
  validation_metric: string;
  validation_metric_direction: MetricDirection;
  validation_evaluation_script: string;
  validation_evaluator_sha256: string;
  data_source: { kind: string; name: string; detail: string; remote: boolean; url: string };
  validation_data_source: { kind: string; name: string; detail: string; remote: boolean; url: string };
  level: string;
  data_query: string;
  model_query: string;
  method_query: string;
  iteration_budget: number;
  max_cost_usd: number;
  stop_threshold: number | null;
  last_run_id: string;
  last_run_name: string;
  last_run_at: string;
  run_count: number;
  best_test_score: number | null;
};

/** One method in an agent's pool: the slug, the payload id, and the one-liner
 *  from the card's own frontmatter. */
export type SkillCardDTO = {
  name: string;
  method: string;
  description: string;
};

export type AgentDTO = {
  id: string;
  name: string;
  title: string;
  reports_to: string;
  default_driver: string;
  default_model: string;
  sandbox: "none" | "openshell";
  output_schema: string;
  skills: string[];
  /** The same pool with each card's own description, so the console can say
   *  what a skill IS. */
  skill_cards: SkillCardDTO[];
  identity_path: string;
};

/** One heartbeat row, as GET /heartbeats serves it
 *  (src/zevo/api/routers/ui/heartbeats.py). The single source for every
 *  heartbeat shape in the frontend; narrower endpoints use Pick<> of this. */
export type HeartbeatDTO = {
  id: string;
  ticket_id: string;
  agent_id: string;
  driver: string;
  model: string;
  started_at: string;
  finished_at: string;
  exit_code: number;
  error_message: string;
  stdout_path: string;
  stdout_size_bytes: number;
  /** finished_at is null. */
  is_live: boolean;
  // Per-heartbeat usage + cost.
  input_tokens: number;
  output_tokens: number;
  cached_input_tokens: number;
  reasoning_output_tokens: number;
  estimated_cost_usd: number;
  /** Immutable operation performed by this activation. */
  operation: string;
  /** submit | collect | repair | continue for multi-activation work. */
  activation_phase: string;
  /** Compact Orchestrator result used to place sequential actions on
   * the Run Timeline. Empty for Specialist heartbeats. */
  action: string;
  action_summary: string;
  child_ticket_id: string;
};

/** A file of a file set that is NOT on disk — fetched at run time. */
export type RemoteFileDTO = {
  role: string;
  kind: "huggingface" | "url";
  id: string;
  url: string;
  /** Which slice of the hub repo. "" = the loader's default, `train`. */
  split: string;
  /** The named subset, for repos that ship several. "" = the default config. */
  config: string;
};

export type FileSetSourceDTO = {
  note: string;
  remote: RemoteFileDTO[];
};

/** One file set, as GET /files serves it (src/zevo/api/routers/ui/files.py). */
export type FileSetDTO = {
  name: string;
  path: string;
  size_bytes: number;
  files: string[];
  created_at: string;
  source: FileSetSourceDTO | null;
};

/** One managed compute allocation. Cluster rows are created immediately after
 * `sbatch`, so `created_at` is also the start of a Slurm queue wait. */
export type InfraInstanceDTO = {
  id: string;
  instance_id: string;
  provider: GpuProvider;
  status: string;
  run_id: string | null;
  ticket_id: string | null;
  gpu_name: string;
  gpu_count: number;
  vram_gb: number;
  dph: number;
  ssh_host: string;
  ssh_port: number;
  ssh_user: string;
  meta: Record<string, unknown>;
  created_at: string;
  ready_at: string | null;
  released_at: string | null;
  release_reason: string;
  uptime_seconds: number;
  estimated_cost_usd: number;
};

/** What the cancel-time rescue did with the weights. */
export type CancelOutcome = {
  weights?: "download" | "hf" | "discard";
  model_path?: string;
  hf_url?: string;
  registry_version_tag?: string;
  note?: string;
  error?: string;
  finished_at?: string;
};

export type RunSummary = {
  id: string;
  task_name: string;
  // What the user called this execution; required by every current launch.
  run_name: string;
  setting_id: string;
  /** Immutable display name retained if the reusable Setting is deleted. */
  setting_name: string;
  task_objective: string;
  /** Composed objective, including pinned Setting clauses, given to Agents. */
  agent_objective: string;
  status: RunStatus;
  /** Computed by the backend from its canonical terminal-status definition. */
  is_terminal: boolean;
  summary: string;
  registry_version_tag: string;
  halted_reason: string;
  /** Cancel asked for; the champion checkpoint is still being copied off the
   *  box. The status stays "running" until that finishes. */
  cancelling: boolean;
  cancel_policy: { weights?: "download" | "hf" | "discard"; local_dir?: string; hf_repo_id?: string };
  cancel_outcome: CancelOutcome;
  started_at: string;
  finished_at: string | null;
  // Zevo model-improvement iteration fields.
  iteration_budget: number;
  iterations_completed: number;
  stop_threshold: number | null;
  metric: string;
  metric_direction: "max" | "min";
  /** False only while an Auto run's scoping stage is still deciding the
   *  metric and held-out eval — `metric` is "" and `metric_direction` a
   *  placeholder until then. Every other mode is settled from the start. */
  scoring_settled: boolean;
  eval_source: EvalSource;
  /** Validation contract drives optimization, stop decisions, and champion selection. */
  validation_metric: string;
  validation_metric_direction: "max" | "min";
  /** Hard $ cap for the run. 0 = uncapped, rendered as the infinity glyph the
   *  way iteration_budget is. The server has always sent it; nothing on this
   *  side had asked for it until the Cost card started reading against it. */
  max_cost_usd: number;
  /** Run-only wall-clock cap in hours. 0 = unlimited. */
  max_runtime_hours: number;
  max_queue_wait_hours: number;
  queue_wait_seconds: number;
  /** True only while a managed cluster allocation is still pending. */
  queue_waiting: boolean;
  best_validation_score: number | null;
  cost_usd: number;
  duration_s: number | null;
  // Lifted out of the run's history by the API so cross-run charts need only
  // this list endpoint. null = absent. The trained score excludes the
  // baseline probe, unlike best_validation_score which includes it.
  baseline_validation_score: number | null;
  best_trained_validation_score: number | null;
  /** The held-out baseline is measured on the same test contract as the
   *  validation-selected champion. */
  baseline_test_score: number | null;
  // The HELD-OUT test set: what the run is actually judged by, measured by the
  // harness so nothing in the loop could tune against it. The trusted
  // dashboard can observe it live; Agent requests receive null until terminal.
  champion_test_score: number | null;
  /** 'user' | 'train' | 'test' — where this run's validation set came from. */
  validation_source: string;
  /** Run-level maximum GPU count; zero means unlimited. */
  num_gpus: number;
  mode: RunMode;
  customizations: Record<string, unknown>;
  decision_pins: Record<string, unknown>;
  model_lineages: Record<string, Record<string, unknown>>;
};

export type TicketDTO = {
  id: string;
  run_id: string;
  agent_id: string;
  status: TicketStatus;
  input_format: TicketInputFormat;
  lane: TicketLane;
  iteration: number;
  payload: Record<string, unknown>;
  customization: Record<string, unknown>;
  inputs: Record<string, unknown>;
  summary: string;
  error_message: string;
  repair_attempts: number;
  repair_route: "" | "self" | "orchestrator" | "terminal";
  created_at: string;
  updated_at: string;
};

export type MessageDTO = {
  id: string;
  author: string;
  body: string;
  created_at: string;
  triggered_wakeup_id: string;
};

export type NoticeDTO = {
  id: string;
  code: string;
  severity: string;
  body: string;
  created_at: string;
};

export type HeartbeatResultDTO = {
  id: string;
  heartbeat_id: string;
  agent_id: string;
  status: "succeeded" | "degraded" | "failed";
  output: Record<string, unknown>;
  created_at: string;
};

export type WorkProductDTO = {
  id: string;
  role: string;
  path: string;
  local_path: string;     // host path: runs/<run-id>/<ticket>/…
  meta: Record<string, unknown>;
  created_at: string;
};

export type ExecutionEventDTO = {
  /** Agent activation that received the event. */
  heartbeat_id: string;
  /** Concrete process launch within that activation. A restarted Trainer has
   *  a new attempt, so equal step numbers remain separate histories. */
  attempt_id: string;
  event_type: "attempt" | "phase" | "progress";
  phase: string;
  current_step: number;
  total_steps: number;
  loss: number;
  extras: Record<string, unknown>;
  ts: string;
};

export type TicketDetail = TicketDTO & {
  messages: MessageDTO[];
  notices: NoticeDTO[];
  results: HeartbeatResultDTO[];
  work_products: WorkProductDTO[];
  execution_events: ExecutionEventDTO[];
  resolved_configs: Record<string, Record<string, unknown>>;
};

export type IterationHistoryEntry = {
  iteration: number;
  /** Data preparation operations and the training method used this round. */
  method_ids: string[];
  training_method: string;
  base_model: string;
  /** VALIDATION score — what the orchestrator saw and steered by. */
  score: number;
  /**
   * HELD-OUT test score for the same iteration, written by the harness after
   * the fact. Absent while the held-out measurement is still running, and
   * stripped entirely from the copy the orchestrator is handed.
   */
  test_score?: number;
  /** Named component scores for a multi-Test Task. Kept private from Agents. */
  test_scores?: Record<string, number>;
  test_metrics?: Record<string, string>;
  test_metric_directions?: Record<string, MetricDirection>;
  source: "baseline" | "trained";
  // The round's record, one fact per field. `analysis` is what was LEARNED, as
  // distinct from `result` which is what the score did.
  action: string;
  result: string;
  analysis: string;
  next: string;
};

export type RunDetail = RunSummary & {
  // The total split by what spent it: agent tokens vs RENTED GPU time. GPU is
  // legitimately 0 on a run that used the user's own hardware.
  agent_cost_usd: number;
  gpu_cost_usd: number;
  /** The model that DROVE the agents, not the one they trained. Empty until the
   *  first heartbeat lands. */
  harness_model: string;
  tickets: Array<{
    id: string;
    agent_id: string;
    input_format: TicketInputFormat;
    lane: TicketLane;
    iteration: number;
    operation: string;
    model_source: "" | "base_model" | "checkpoint";
    test_set_name: string;
    status: TicketStatus;
    summary: string;
    error_message: string;
    created_at: string;
  }>;
  history: IterationHistoryEntry[];
};

/** Read-only per-agent cost split for a run. The agent rows are the LLM/agent
 *  cost grouped by role; GPU and totals mirror the run's budget snapshot, so
 *  nothing here re-prices work. `agent_cost_usd` equals the sum of the rows. */
export type CostBreakdown = {
  run_id: string;
  agents: Array<{ agent_id: string; cost_usd: number; heartbeats: number }>;
  agent_cost_usd: number;
  gpu_cost_usd: number;
  total_cost_usd: number;
};

export type ModelDTO = {
  /** Stable `M-<run8>` identity owned by the producing Run. The one identifier for
   *  this model: the database/manifest tag, directory on disk, and label shown
   *  here and on the leaderboard. */
  version_tag: string;
  iteration: number;
  run_id: string | null;
  base_model: string;
  training_method: string;
  dataset_source: string;
  model_path: string;
  task_objective: string;
  metric: string;
  metric_direction: MetricDirection;
  eval: Record<string, unknown>;
  registered_at: string;
  // The producing run's task, when it had one. Empty for custom-objective runs.
  task_name?: string;
  /** HELD-OUT score of this Run's validation-selected champion. `eval.score`
   *  is the VALIDATION number the registry agent was handed. null = never measured. */
  champion_test_score: number | null;
  /** Paired held-out baseline and direction-normalized change for this saved
   *  champion. Positive improvement is always better. */
  baseline_test_score: number | null;
  improvement: number | null;
  /** `model_path`, absolute on the host. Empty when the deployment did not say
   *  where the repo lives, or when this version kept no model. */
  model_path_abs: string;
};
