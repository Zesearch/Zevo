"""
Input + output contracts surfaced from the Orchestrator Agent.

The orchestrator is driven by the docs in `playbook/agents/orchestrator/` and
runs through one of the agent drivers (claude_cli / codex_cli / bedrock /
openrouter). Three Pydantic types remain here:

  - `UserRequest`        -- what the user submits when creating a run.
                            Imported by the backend routers.
  - `OrchestratePayload` -- the strictly validated JSONB shape of an
                            `orchestrate-...-NNN` ticket's payload.
  - `SupervisorAction`   -- the structured output the orchestrator
                            emits at the end of each heartbeat (read by
                            the runner). Referenced by
                            playbook/agents/orchestrator/identity.md frontmatter.

Child Ticket envelopes and per-Specialist payloads are supplied as exact JSON
Schemas on every `OrchestratorTaskInput`; prose examples are explanatory only.
"""

from __future__ import annotations

import re
from typing import Any, List, Literal
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from zevo.contracts._base import StrictResult
from zevo.contracts.customizations import RunCustomizations
from zevo.contracts.inference import GenerationTerminationSummary
from zevo.contracts.memory import AgentMemoryContext
from zevo.contracts.prompting import (
    canonical_prompt_contract,
    normalize_prompt_framing,
    validate_decoding_config,
    validate_inference_config,
    validate_loss_objective_config,
)
from zevo.contracts.task_protocol import TaskInferenceProtocol
from zevo.contracts.tickets import RunMode, RunStatus, TicketStatus
from zevo.contracts.train import TrainingDiagnostics


BUILTIN_METRICS = frozenset({
    "accuracy", "exact_match", "f1", "token_f1", "bleu", "rouge_l",
    # Opt-in length-normalized log-likelihood option scoring for multiple-choice
    # QA. Inference emits per-option log-likelihoods; Evaluation argmaxes them
    # deterministically. `accuracy_norm` is an alias for the same scorer.
    "mc_loglikelihood", "accuracy_norm",
})

# The built-in metrics whose scoring contract implies a deterministic +1/0
# correctness reward: each checks an exactly-correct discrete answer (a
# multiple-choice option or an exact string), so a GRPO/RFT verifier can derive
# the reward from the same gold answer the metric already compares against.
# Overlap/graded metrics (f1, token_f1, bleu, rouge_l) are deliberately excluded
# — they score partial similarity, not correctness, so a +1/0 verifier would
# misrepresent them — and a `custom` evaluator is opaque, so it never implies a
# verifiable reward. This is the signal that promotes the SFT -> RFT -> GRPO
# progression; see `verifiable_reward_available`.
VERIFIABLE_REWARD_METRICS = frozenset({
    "accuracy", "exact_match", "mc_loglikelihood", "accuracy_norm",
})


def verifiable_reward_available(metric: str, metric_type: str) -> bool:
    """Whether the scoring contract implies a deterministic correctness reward.

    True only for a built-in metric in `VERIFIABLE_REWARD_METRICS` (an
    exactly-correct discrete answer). Such tasks can derive a +1/0 GRPO/RFT
    reward from the same gold answer the metric uses, which is what makes the
    verifiable-reward progression (SFT -> RFT -> GRPO) a legitimate deliberate
    lever rather than a last resort. Pure and side-effect free.
    """
    return (metric_type or "").strip().lower() == "builtin" and (
        (metric or "").strip().lower() in VERIFIABLE_REWARD_METRICS
    )
_CAUSAL_DIAGNOSIS = re.compile(
    r"(?i)\b(overfit(?:ting)?|under-?produc\w*|format error\w*|"
    r"verbosity error\w*|hallucinat\w*|memorization)\b"
)
_UNCERTAINTY_LANGUAGE = re.compile(
    r"(?i)\b(possible|possibly|may|might|could|suggests?|consistent with|uncertain)\b"
)


def _require_evidence_language(text: str, *, field: str) -> str:
    value = str(text or "").strip()
    if _CAUSAL_DIAGNOSIS.search(value) and not _UNCERTAINTY_LANGUAGE.search(value):
        raise ValueError(
            f"{field} makes a causal/model-error diagnosis without deterministic "
            "error-analysis evidence; report the observed metric trend or mark "
            "the interpretation explicitly uncertain"
        )
    return value


class SupervisorTrigger(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["run_created", "ticket_completed"]
    ticket_id: str = ""
    agent_id: str = ""
    status: TicketStatus | Literal[""] = ""


class RuntimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gpu_provider: Literal["cluster", "cloud", "instance"]
    num_gpus: int = Field(
        ge=0,
        description=(
            "Maximum GPUs the Run may use at once; zero means unlimited and "
            "actual plans always select at least one."
        ),
    )
    generation_backend: Literal["hf", "vllm"]
    max_queue_wait_hours: float = Field(24, gt=0, le=168)


class BudgetSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_cost_usd: float = Field(ge=0)
    spent_usd: float = Field(ge=0)
    remaining_usd: float = Field(ge=0)
    projected_next_iteration_usd: float = Field(0, ge=0)
    can_afford_next_iteration: bool | None = None
    projection_basis: str = ""
    max_runtime_hours: float = Field(0, ge=0)
    max_queue_wait_hours: float = Field(24, gt=0, le=168)
    queue_wait_hours: float = Field(0, ge=0)
    elapsed_runtime_hours: float = Field(0, ge=0)
    remaining_runtime_hours: float = 0
    over_time_limit: bool = False
    projected_next_iteration_hours: float = Field(0, ge=0)
    can_finish_next_iteration_in_time: bool | None = None
    runtime_projection_basis: str = ""


class OrchestratorApiRoutes(BaseModel):
    """Exact read/mutation routes available to every supervisor heartbeat."""

    model_config = ConfigDict(extra="forbid")
    openapi: Literal["/api/openapi.json"] = "/api/openapi.json"
    run_detail: Literal["/api/runs/{run_id}"] = "/api/runs/{run_id}"
    tickets: Literal["/api/tickets?run_id={run_id}"] = "/api/tickets?run_id={run_id}"
    budget: Literal["/api/runs/{run_id}/budget"] = "/api/runs/{run_id}/budget"
    should_stop: Literal["/api/runs/{run_id}/should-stop"] = "/api/runs/{run_id}/should-stop"
    create_ticket: Literal["/api/tickets"] = "/api/tickets"
    patch_run: Literal["/api/runs/{run_id}"] = "/api/runs/{run_id}"
    ticket_detail: Literal["/api/tickets/{ticket_id}"] = "/api/tickets/{ticket_id}"
    retry_status: Literal[
        "/api/tickets/{ticket_id}/retry-status"
    ] = "/api/tickets/{ticket_id}/retry-status"


class IterationHistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    iteration: int = Field(ge=0)
    source: Literal["baseline", "trained"]
    method_ids: list[str] = Field(default_factory=list)
    training_method: str = ""
    base_model: str = ""
    score: float | None = None
    training_diagnostics: TrainingDiagnostics | None = Field(
        None,
        description=(
            "Engine-derived training and Validation loss summary for a trained "
            "candidate. Diagnostic evidence only; score remains the model-selection authority."
        ),
    )
    generation_termination: GenerationTerminationSummary | None = Field(
        None,
        description=(
            "Engine-derived Validation inference termination summary. It tells "
            "the Orchestrator whether responses stopped normally or exhausted "
            "the generation limit without exposing Validation text."
        ),
    )
    action: str = ""
    result: str = ""
    analysis: str = ""
    next: str = ""


# =====================================================================
# Input: UserRequest
# =====================================================================

class UserRequest(BaseModel):
    """The user's request. Test and Validation metric contracts are explicit;
    configurable inputs use empty-string / 0 / [] as 'not specified'.
    """

    model_config = ConfigDict(extra="forbid")

    task_objective: str = Field(..., description="The user's high-level goal, in natural language.")
    metric: str = Field(
        min_length=1,
        description="Name of the held-out Test metric, e.g. benchmark_average.",
    )
    metric_direction: Literal["max", "min"] = Field(
        ...,
        description=(
            "Direction of this Run's held-out Test metric: max when higher "
            "scores are better, min when lower scores are better."
        ),
    )
    metric_type: Literal["builtin", "custom"] = Field(
        "builtin",
        description=(
            "This Run's Test evaluator kind. Built-in runs Zevo's deterministic "
            "implementation; custom runs the one frozen evaluation_script."
        ),
    )
    evaluation_script: str = Field(
        "",
        description=(
            "This Run's custom Test evaluator. Empty for a built-in Test metric."
        ),
    )
    evaluator_sha256: str = Field(
        "",
        pattern=r"^(?:|[0-9a-f]{64})$",
        description="SHA-256 of the frozen custom Test evaluator; empty for built-ins.",
    )
    validation_metric_type: Literal["", "builtin", "custom"] = Field(
        "",
        description=(
            "Validation evaluator kind. Leave blank with no validation_set; "
            "Run creation inherits metric_type from Test."
        ),
    )
    validation_metric: str = Field(
        "",
        description=(
            "Metric used to select iterations. Leave blank with no "
            "validation_set; Test-derived Validation inherits Test's metric."
        ),
    )
    validation_metric_direction: Literal["", "max", "min"] = Field(
        "",
        description=(
            "Direction used for champion selection and stop thresholds. Leave "
            "blank with no validation_set to inherit Test's direction."
        ),
    )
    validation_evaluation_script: str = Field(
        "",
        description=(
            "Frozen custom Validation evaluator; empty for a built-in metric. "
            "Test-derived Validation reuses Test's frozen evaluator."
        ),
    )
    validation_evaluator_sha256: str = Field(
        "",
        pattern=r"^(?:|[0-9a-f]{64})$",
        description="SHA-256 of the custom Validation evaluator.",
    )

    @field_validator("metric")
    @classmethod
    def normalize_test_metric(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("metric must not be blank")
        return normalized

    @field_validator("validation_metric")
    @classmethod
    def normalize_validation_metric(cls, value: str) -> str:
        return value.strip()
    training_method: str = Field(
        ...,
        description=(
            "Pinned Train method, e.g. 'lora_sft'. Use '' to let the Train "
            "Agent record Zevo's active Method branch in each train_config.yaml; "
            "the branch changes only after its nested search is exhausted."
        ),
    )
    method_query: str = Field(
        "",
        description=(
            "Optional natural-language guidance for Zevo's unpinned training-"
            "method selection and branch order. This is a preference, not an "
            "exact training_method pin."
        ),
    )
    method_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Pinned method-specific configuration. Auxiliary models support "
            "Hugging Face owner/model ids only: gkd requires teacher_model and "
            "online_dpo requires reward_model. Supplied keys remain fixed; "
            "when this mapping is empty, Zevo may choose compatible values in "
            "each executed iteration."
        ),
    )
    # Customized-Pipeline decision pins. Full Pipeline requires all six fields
    # to be empty and lets the responsible Specialist record them in YAML.
    # In Customized Pipeline, a supplied value is
    # binding and becomes part of the Setting identity.
    prompt_framing: str = Field(
        "",
        description=(
            "Customized Pipeline pin for the whole Run: chat, "
            "chat:<template-model>, completion, or text. Full Pipeline requires "
            "an empty value and lets each Base-model branch's Baseline Inference "
            "select and record it for that lineage."
        ),
    )
    system_prompt: str = Field(
        "",
        description="Customized Pipeline system-turn pin. Full Pipeline lets baseline Inference select it.",
    )
    loss_objective_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Customized Pipeline method-specific semantic loss pins. Full "
            "Pipeline requires an empty mapping and lets Train select and record the loss settings."
        ),
    )
    inference_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Customized Pipeline pin for task mapping and parsing. Full Pipeline lets baseline Inference select it.",
    )
    inference_protocol: TaskInferenceProtocol | None = Field(
        default=None,
        description=(
            "Task-owned semantic instruction/output protocol. The API resolves "
            "an automatic protocol for predefined dataset Tasks; callers only "
            "supply this field when using the advanced contract directly."
        ),
    )
    decoding_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Customized Pipeline generation pins. Full Pipeline lets baseline Inference select them.",
    )
    dataset: str = Field(..., description="User-uploaded training data path. Use '' for none (will acquire).")
    dataset_split: str = Field(
        default="",
        description=(
            "When `dataset` is a HuggingFace id: which slice of the repo to "
            "train on. '' falls back to what the catalogue declares for that "
            "id, and then to `train`."
        ),
    )
    dataset_config: str = Field(
        default="",
        description="Named subset for hub repos that ship several. '' = the default config.",
    )
    data_query: str = Field(
        "",
        description=(
            "Optional natural-language guidance for data discovery and "
            "preparation when dataset is unpinned. This is not an exact "
            "dataset pin; use '' to derive guidance from the task objective."
        ),
    )
    base_model: str = Field(
        ...,
        description=(
            "Exact model id/path when user-pinned. Use '' to let Inference "
            "select one from task fit, access/license, and resource evidence."
        ),
    )
    model_query: str = Field(
        "",
        description=(
            "Optional natural-language guidance for Zevo's unpinned base-model "
            "selection and branch order. This is a preference, not an exact "
            "base_model pin."
        ),
    )
    test_set: str = Field(
        ...,
        description=(
            "Path to the FULL, HELD-OUT test set — WITH the ground-truth columns. "
            "This is the goal the run is judged by, and the orchestrator never "
            "sees a score from it: the engine runs it on its own after each "
            "iteration. Optimize against `validation_set` instead."
        ),
    )
    test_answer_fields: List[str] = Field(
        default_factory=list,
        description=(
            "Where the ground truth lives in `test_set` — the field(s) the eval "
            "script grades against. A COLUMN when the set is a CSV, a KEY when "
            "it is JSON, which is why these are fields and not columns. The data "
            "agent drops exactly these to build the questions-only copy "
            "inference is given, so the model never sees an answer. Naming them "
            "is how a run stops needing a hand-prepared second file for every "
            "set it scores. Required whenever `test_set` is supplied."
        ),
    )
    validation_set: str = Field(
        default="",
        description=(
            "Path to the FULL validation set — WITH the ground-truth columns. "
            "This is what every iteration is scored on. It is kept in the "
            "engine scoring boundary and is not exposed to Orchestrator or Data; "
            "they receive only measured scores after evaluation. Leave '' and Zevo deterministically "
            "takes 20% of Test before the Run begins. The derived Validation set "
            "must contain at least 200 rows; otherwise upload Validation. Those "
            "rows are removed from the final held-out Test population."
        ),
    )
    validation_split: str = Field(
        default="",
        description=(
            "When `validation_set` is a HuggingFace id: which slice of the repo "
            "to use as the validation set. Run creation fetches it to a local "
            "file before anything starts, so the loop still sees an ordinary "
            "path. '' picks the repo's validation-like split and fails if it "
            "has none — defaulting to `train` would tune on training rows."
        ),
    )
    validation_config: str = Field(
        default="",
        description="Named subset for hub repos that ship several. '' = the default config.",
    )
    validation_answer_fields: List[str] = Field(
        default_factory=list,
        description=(
            "Where the ground truth lives in `validation_set` — a column of a "
            "CSV or a key of a JSON record. Required when validation_set is "
            "supplied; Test-derived Validation inherits the Test answer fields."
        ),
    )
    validation_sample_submission: str = Field(
        default="",
        description=(
            "Submission template for the validation set: the columns inference "
            "must emit for the Validation metric contract to read. Required when a "
            "validation_set is supplied; Test-derived Validation receives the "
            "same schema/example contract as Test. Its example-row count does "
            "not need to match the Validation population."
        ),
    )
    test_sample_submission: str = Field(
        ...,
        description=(
            "Submission schema/example for the held-out test set. It defines "
            "prediction columns, order, and representative formatting; its "
            "example-row count does not need to match the Test population."
        ),
    )
    constraints: List[str] = Field(..., description="e.g. ['no external APIs']. Use [] for no constraints.")
    # Runtime fields deliberately do not live in UserRequest.
    # They configure one execution rather than define the task, and therefore
    # have a single wire location: top-level CreateRunRequest. The runner stamps
    # the persisted Run values onto worker inputs.

    @model_validator(mode="after")
    def validate_experiment_preferences(self) -> "UserRequest":
        self.training_method = self.training_method.strip().lower()
        if self.prompt_framing:
            self.prompt_framing = normalize_prompt_framing(self.prompt_framing)
            framing, _reasoning_type, system = canonical_prompt_contract(
                prompt_framing=self.prompt_framing,
                model_reasoning_type="non_thinking",
                system_prompt=self.system_prompt,
            )
            self.prompt_framing = framing
            self.system_prompt = system
        else:
            if self.system_prompt.strip():
                raise ValueError("system_prompt requires prompt_framing")
        if self.loss_objective_config and not self.training_method:
            raise ValueError("loss_objective_config requires training_method")
        self.loss_objective_config = validate_loss_objective_config(
            self.training_method, self.loss_objective_config,
        )
        self.inference_config = validate_inference_config(self.inference_config)
        self.decoding_config = validate_decoding_config(self.decoding_config)
        return self


class AutoUserRequest(BaseModel):
    """An Auto Run's objective plus its optional optimization choices.

    Every scoring field of :class:`UserRequest` (metric, direction, Test set,
    answer fields, sample submission, Validation contract) is deliberately
    ABSENT: the Data agent's ``scope_problem`` work order derives that contract.
    Data, base model, and training method use the same optional pins and hints
    as Full Pipeline, preserving the ordinary L1–L4 ownership ladder.
    """

    model_config = ConfigDict(extra="forbid")

    task_objective: str = Field(
        min_length=1, description="The user's high-level goal, in natural language.",
    )
    test_query: str = Field(
        "",
        description=(
            "Optional guidance for selecting or creating the held-out Test "
            "contract in Auto mode."
        ),
    )
    dataset: str = Field(
        "", description="Exact training source pin; empty lets Zevo acquire data.",
    )
    dataset_split: str = ""
    dataset_config: str = ""
    data_query: str = Field(
        "", description="Optional guidance for training-data discovery (advisory).",
    )
    base_model: str = Field(
        "", description="Exact base-model pin; empty lets Zevo select a model.",
    )
    model_query: str = Field(
        "", description="Optional guidance for base-model selection (advisory).",
    )
    training_method: str = Field(
        "", description="Exact Train method pin; empty lets Zevo select a method.",
    )
    method_query: str = Field(
        "", description="Optional guidance for training-method selection (advisory).",
    )
    method_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific pins; valid only with training_method.",
    )
    constraints: List[str] = Field(
        default_factory=list, description="e.g. ['no external APIs'].",
    )

    @model_validator(mode="after")
    def normalize(self) -> "AutoUserRequest":
        self.task_objective = self.task_objective.strip()
        if not self.task_objective:
            raise ValueError("task_objective must not be blank")
        self.test_query = self.test_query.strip()
        self.dataset = self.dataset.strip()
        self.dataset_split = self.dataset_split.strip()
        self.dataset_config = self.dataset_config.strip()
        self.data_query = self.data_query.strip()
        self.base_model = self.base_model.strip()
        self.model_query = self.model_query.strip()
        self.training_method = self.training_method.strip().lower()
        self.method_query = self.method_query.strip()
        self.constraints = [str(c).strip() for c in self.constraints]
        if any(not c for c in self.constraints):
            raise ValueError("constraints must contain non-empty strings")
        return self


def inherit_test_validation_contract(request: UserRequest) -> UserRequest:
    """Resolve the only valid contract for Test-derived Validation.

    When Validation is carved from Test, it is another population from the
    same scoring problem. It therefore inherits every scoring choice from Test
    instead of accepting a second, potentially contradictory metric contract.
    A separately supplied Validation set remains fully independent.
    """
    if request.validation_set.strip():
        return request
    return request.model_copy(update={
        "validation_metric_type": request.metric_type,
        "validation_metric": request.metric,
        "validation_metric_direction": request.metric_direction,
        "validation_evaluation_script": request.evaluation_script,
        "validation_evaluator_sha256": request.evaluator_sha256,
        # Empty remains the request/Setting sentinel for "derived". Run split
        # settlement materializes the inherited answer fields and a lane-local
        # copy of Test's submission template before any Agent receives them.
        "validation_answer_fields": [],
        "validation_sample_submission": "",
    })



def scoring_asset_errors(request: UserRequest) -> list[str]:
    """Return invalid assets or implementations for either scoring lane.

    Kept outside the Pydantic model so `/preflight` can accept an incomplete
    draft and explain it. Run creation treats the same list as a hard error.
    Built-ins are implemented by the engine; each custom metric requires that
    scoring lane's authoritative evaluator. Answer fields and an
    effective submission shape are always required for a named scoring lane.
    """
    request = inherit_test_validation_contract(request)
    errors: list[str] = []
    test_metric = request.metric.strip().lower()
    validation_metric = request.validation_metric.strip().lower()
    if not request.test_set.strip():
        errors.append("test_set is required")
    if not request.test_answer_fields:
        errors.append("test_answer_fields are required")
    if not request.test_sample_submission.strip():
        errors.append("test_sample_submission is required")
    if request.metric_type == "builtin":
        if test_metric not in BUILTIN_METRICS:
            errors.append(
                f"unknown built-in Test metric {request.metric!r}; built-ins are "
                + ", ".join(sorted(BUILTIN_METRICS))
            )
        if request.evaluation_script.strip() or request.evaluator_sha256:
            errors.append("built-in metrics must not carry a custom evaluator")
    else:
        if not request.evaluation_script.strip():
            errors.append("custom Test metrics require evaluation_script")
    if not request.validation_metric_type:
        errors.append("validation_set requires validation_metric_type")
    elif not validation_metric:
        errors.append("validation_set requires validation_metric")
    elif request.validation_metric_type == "builtin":
        if validation_metric not in BUILTIN_METRICS:
            errors.append(
                "unknown built-in Validation metric "
                f"{request.validation_metric!r}; built-ins are "
                + ", ".join(sorted(BUILTIN_METRICS))
            )
        if (
            request.validation_evaluation_script.strip()
            or request.validation_evaluator_sha256
        ):
            errors.append(
                "built-in Validation metrics must not carry a custom evaluator"
            )
    elif not request.validation_evaluation_script.strip():
        errors.append(
            "custom Validation metrics require validation_evaluation_script"
        )
    if not request.validation_metric_direction:
        errors.append("validation_set requires validation_metric_direction")
    if request.validation_set:
        if not request.validation_answer_fields:
            errors.append("validation_set requires validation_answer_fields")
        if not request.validation_sample_submission.strip():
            errors.append("validation_set requires validation_sample_submission")
    return errors


# =====================================================================
# Supervisor ticket payload (documentation of the JSONB wire format).
# =====================================================================

class OrchestratePayload(BaseModel):
    """JSONB payload of the run's one `orchestrate-<run8>-001` ticket.

    The same ticket is re-woken after relevant child completions for the whole
    run. Its trigger and history snapshots are refreshed;
    the ticket suffix is neither a wake counter nor an iteration counter.
    """
    model_config = ConfigDict(extra="forbid")

    task_objective: str = Field(
        min_length=1,
        description=(
            "The task's original problem statement. It is also stored unchanged "
            "in user_request.task_objective; keep both distinct from the composed "
            "agent_objective. Registry copies this field."
        ),
    )
    agent_objective: str = Field(
        min_length=1,
        description=(
            "Composed execution objective: the original task plus the Setting "
            "clauses this Run pins. This is the goal the supervisor executes."
        ),
    )
    user_request: UserRequest
    mode: RunMode = Field(default="full_pipeline")
    customizations: RunCustomizations = Field(
        default_factory=RunCustomizations,
        description="Per-agent overrides used only by customized_pipeline runs.",
    )
    trigger: SupervisorTrigger
    history: list[IterationHistoryEntry] = Field(
        default_factory=list,
        description=(
            "Validation-only journal refreshed on every wake. Evaluation owns "
            "the factual score fields and the orchestrator enriches the "
            "decision narrative. It contains the "
            "baseline entry for every base-model lineage once measured and one "
            "entry per completed trained candidate, with the fields documented in the "
            "Orchestrator platform contract. Held-out measurements are removed."
        ),
    )
    validation_rows: int = Field(
        0,
        description="Number of validation rows; one row defines score resolution.",
    )
    budget: BudgetSnapshot = Field(
        description=(
            "Initial/live cost and wall-clock snapshot; the orchestrator still "
            "refreshes it from the budget API before deciding on another loop."
        ),
    )
    dataset_profile: dict = Field(
        default_factory=dict,
        description="Cached profile of the original training dataset when one exists.",
    )
    runtime: RuntimeSpec

    @computed_field  # type: ignore[prop-decorator]
    @property
    def verifiable_reward_available(self) -> bool:
        """Scoring-contract signal, surfaced so the supervisor can see it.

        The loop optimizes against the Validation metric carried on
        `user_request` (Run setup copies the Validation contract into its
        `metric`/`metric_type`). When that metric is a verifiable correctness
        check, a +1/0 GRPO/RFT reward is derivable from the gold answer, which
        promotes the SFT -> RFT -> GRPO progression as a deliberate next Method
        lever after a reasonable SFT baseline — see
        playbook/agents/orchestrator/goal.md and the advisory
        `reinforcement_progression` policy in
        zevo.engine.method.loop_policy. Derived output only; it never appears in
        a stored request payload.
        """
        return verifiable_reward_available(
            self.user_request.metric, self.user_request.metric_type,
        )


class JournalNarrativeEntry(BaseModel):
    """Concise model-improvement narrative added to an evaluated model row."""

    model_config = ConfigDict(extra="forbid")
    iteration: int = Field(ge=0)
    source: Literal["baseline", "trained"]
    action: str = Field(
        min_length=1,
        description=(
            "Broad model-improvement direction tested. Do not list exact "
            "configuration values or narrate workflow activity."
        ),
    )
    result: str = Field(
        min_length=1,
        description=(
            "Measured Validation performance and its comparison with the "
            "selected parent model; for Baseline, the starting result."
        ),
    )
    analysis: str = Field(
        min_length=1,
        description=(
            "Concise evidence-based interpretation of model behavior and "
            "performance, not process, governance, or artifact bookkeeping."
        ),
    )
    next: str = Field(
        min_length=1,
        description=(
            "One concise, concrete next experiment: use one short direction "
            "sentence followed by only the exact planned changes, including "
            "before/after values when applicable (for example learning rate "
            "1e-5 -> 5e-6; epochs 3 -> 2; warmup ratio 3% -> 5%). Do not repeat "
            "unchanged configuration. A phrase such as 'further fine-tune' "
            "alone is too vague. When ending, give the model-level reason to "
            "retain the result."
        ),
    )

    @model_validator(mode="after")
    def require_validation_evidence(self) -> "JournalNarrativeEntry":
        if "validation" not in self.result.lower():
            raise ValueError(
                "Journal result must label its optimization measurement as Validation"
            )
        if re.search(r"(?i)\b(test|held[- ]out)\b", self.result):
            raise ValueError("the optimization Journal must not claim held-out Test evidence")
        self.analysis = _require_evidence_language(self.analysis, field="Journal analysis")
        return self


class PatchRunRequest(BaseModel):
    """The one accepted PATCH /api/runs/{run_id} request body."""

    model_config = ConfigDict(extra="forbid")

    status: RunStatus | None = None
    summary: str | None = None
    halted_reason: str | None = None
    history_entry: JournalNarrativeEntry | None = None

    @field_validator("summary")
    @classmethod
    def require_supported_terminal_summary(cls, value: str | None) -> str | None:
        if value is None:
            return value
        summary = _require_evidence_language(value, field="Run summary")
        if re.search(r"(?i)\b(improvement|improved|regression|score)\b", summary) and not re.search(
            r"(?i)\b(validation|test|held[- ]out)\b", summary
        ):
            raise ValueError(
                "Run summary score/improvement statements must say Validation or held-out Test"
            )
        return summary

    @field_validator("history_entry")
    @classmethod
    def require_complete_journal_narrative(
        cls, value: JournalNarrativeEntry | None,
    ) -> JournalNarrativeEntry | None:
        if value is None:
            return value
        missing = [
            name for name in ("action", "result", "analysis", "next")
            if not str(getattr(value, name, "") or "").strip()
        ]
        if missing:
            raise ValueError(
                "history_entry requires non-empty Journal fields: "
                + ", ".join(missing)
            )
        return value


class OrchestratorTaskInput(OrchestratePayload):
    """Resolved supervisor invocation: stored payload plus envelope facts."""

    ticket_id: str = Field(min_length=1)
    iteration: int = Field(ge=0)
    run_id: str = Field(min_length=1)
    supervisor_ticket_id: str = Field(min_length=1)
    api_routes: OrchestratorApiRoutes = Field(default_factory=OrchestratorApiRoutes)
    ticket_creation_schema: dict[str, Any] = Field(
        ...,
        description=(
            "Exact JSON Schema for POST /api/tickets. It is the sole outer "
            "request-key authority."
        ),
    )
    ticket_record_schema: dict[str, Any] = Field(
        ...,
        description=(
            "Exact Ticket record schema returned by POST /api/tickets and "
            "used for every item from the Ticket list. Ticket identity is id, "
            "never ticket_id."
        ),
    )
    ticket_creation_response_id_command: str = Field(
        ...,
        min_length=1,
        description=(
            "Side-effect-free command that validates the complete JSON response "
            "on stdin and prints its canonical id."
        ),
    )
    specialist_request_payload_schemas: dict[str, Any] = Field(
        ...,
        description=(
            "Exact JSON Schemas for child payloads authored by Orchestrator, "
            "before API-owned Run facts are stamped."
        ),
    )
    specialist_stored_payload_schemas: dict[str, Any] = Field(
        ...,
        description=(
            "Exact post-stamping payload schemas the worker ultimately receives. "
            "These are audit context, not fields to duplicate in the request."
        ),
    )
    specialist_input_binding_contracts: dict[str, Any] = Field(
        ...,
        description=(
            "Exact conditional input names, artifact_role values, source Agent, "
            "source status, iteration relationship, and same-source constraints "
            "enforced for every Specialist payload branch. Select the variant "
            "using its discriminator; do not infer lineage from examples."
        ),
    )
    artifact_binding_schema: dict[str, Any] = Field(
        ...,
        description="Exact JSON Schema for every value in the child inputs mapping.",
    )
    run_patch_schema: dict[str, Any] = Field(
        ...,
        description=(
            "Exact JSON Schema for PATCH /api/runs/{run_id}, including the "
            "complete four-part Journal narrative."
        ),
    )
    memory: AgentMemoryContext = Field(
        default_factory=AgentMemoryContext,
        description=(
            "Active shared_candidate entries reported by worker Agents in this "
            "Run. They are advisory evidence for the next suggestion and never "
            "override user pins or a Specialist's recorded YAML configuration."
        ),
    )


# =====================================================================
# Output: SupervisorAction (the orchestrator's per-heartbeat result)
# =====================================================================

class SupervisorAction(StrictResult):
    """What the reactive orchestrator emits each heartbeat.

    The orchestrator either:
      - 'emit_ticket' -- it created (via curl) a child ticket and
                         optionally tells us its id for breadcrumbs.
      - 'mark_done'   -- usable work is complete; the orchestrator has already
                         atomically PATCHed a non-empty Run summary and
                         Run.status=success.
      - 'mark_failed' -- the run cannot continue; the orchestrator has already
                         PATCHed Run.status=failed with a specific reason.
      - 'wait'        -- there's nothing to do this iteration; the runner
                         will re-wake on the next child completion.
    """
    ticket_id: str = Field(
        min_length=1,
        description="Exact Orchestrator Ticket id from OrchestratorTaskInput.",
    )
    action: Literal[
        "emit_ticket", "mark_done", "mark_failed", "wait",
    ]
    child_ticket_id: str = Field(default="", description="ID of the child ticket created this turn, if action=emit_ticket.")
    summary: str = Field(
        default="",
        description=(
            "Breadcrumb describing this heartbeat. It does not persist the Run "
            "summary; completion must first PATCH /runs/{run_id}. When this "
            "heartbeat crosses a Data, Method, or Base-model search boundary, "
            "this breadcrumb names the exhausted branch, its Validation evidence, "
            "and the next branch. Initial selection, switching, or explicit "
            "retention of an unpinned method also states the evidence-based "
            "method rationale and that it is Zevo-selected rather than user-pinned."
        ),
    )

    @model_validator(mode="after")
    def validate_action_payload(self) -> "SupervisorAction":
        if self.action == "emit_ticket" and not self.child_ticket_id:
            raise ValueError("emit_ticket requires child_ticket_id")
        if self.action != "emit_ticket" and self.child_ticket_id:
            raise ValueError("child_ticket_id is valid only for emit_ticket")
        return self
