"""Canonical stored Ticket payload and ArtifactBinding contracts."""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zevo.contracts.customizations import AgentCustomization
from zevo.contracts.data import (
    DATA_METHOD_IDS,
    SCOPING_ONLY_FIELDS,
    DataOperation,
    DataRecipeIntent,
    data_intent_signature,
    is_sha256,
)
from zevo.contracts.prompting import (
    canonical_prompt_contract,
    validate_decoding_config,
    validate_inference_config,
    validate_loss_objective_config,
)
from zevo.contracts.training_methods import METHOD_CONFIG_KEYS, method_config_errors

# auto = agent-derived scoring; the Data agent scopes the scoring contract first
# and the engine settles it, then the run proceeds exactly like full_pipeline.
RunMode = Literal["full_pipeline", "customized_pipeline", "single_stage", "auto"]
RunStatus = Literal[
    "planning", "running", "success", "degraded", "failed", "halted", "cancelled",
]
TERMINAL_RUN_STATUSES = frozenset(
    {"success", "degraded", "failed", "halted", "cancelled"}
)
TicketInputFormat = Literal["typed", "freeform"]
TicketLane = Literal["optimization", "held_out_test"]
TicketStatus = Literal[
    "queued", "running", "repairing", "awaiting_input", "waiting_external", "succeeded", "degraded",
    "failed", "skipped", "cancelled",
]
TERMINAL_TICKET_STATUSES = frozenset(
    {"succeeded", "failed", "degraded", "skipped", "cancelled"}
)
_DATA_CONFIGURATION_KEYS = frozenset({"method_ids", "target_size"})
_TRAIN_CONFIGURATION_KEYS = frozenset({
    "batch_size", "learning_rate", "num_epochs", "lora_r", "lora_alpha",
    "max_seq_len",
})
_TRAIN_SUGGESTION_KEYS = _TRAIN_CONFIGURATION_KEYS | frozenset({
    "direction", "training_method",
})
_INFERENCE_CONFIGURATION_KEYS = frozenset({
    "prompt_framing", "system_prompt", "inference_config",
    "decoding_config",
})
_INFERENCE_SUGGESTION_KEYS = frozenset({"direction"})


class StoredPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchBranchTransition(BaseModel):
    """Explicit audit record for crossing one exhausted autonomy-search level."""

    model_config = ConfigDict(extra="forbid")

    level: Literal["none", "data", "method", "base_model"] = "none"
    exhausted_branch: str = ""
    validation_evidence: str = ""
    next_branch: str = ""

    @model_validator(mode="after")
    def require_complete_transition(self) -> "SearchBranchTransition":
        values = (
            self.exhausted_branch.strip(),
            self.validation_evidence.strip(),
            self.next_branch.strip(),
        )
        if self.level == "none":
            if any(values):
                raise ValueError(
                    "search transition details require a data, method, or base_model level"
                )
        elif not all(values):
            raise ValueError(
                "a search branch transition requires exhausted_branch, "
                "validation_evidence, and next_branch"
            )
        return self


class ArtifactBinding(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "description": (
                "For a new child Ticket, set artifact_role and exactly one of "
                "source_ticket_id or path. Usually omit work_product_id; set it "
                "together with source_ticket_id only to select one exact artifact "
                "when a source Ticket exposes multiple products with that role. The resolver "
                "may later store source_ticket_id, work_product_id, and path "
                "together as resolved audit provenance."
            ),
        },
    )
    artifact_role: str = Field(
        min_length=1,
        description=(
            "Exact role required by specialist_input_binding_contracts for this "
            "input name."
        ),
    )
    source_ticket_id: str = Field(
        "",
        description=(
            "Whole upstream Ticket id. For ordinary pipeline authoring, set "
            "this and leave path/work_product_id empty."
        ),
    )
    work_product_id: str = Field(
        "",
        description=(
            "Exact WorkProduct selection/provenance. Omit for the source Ticket's "
            "canonical product; supply with source_ticket_id to select a retained "
            "intermediate checkpoint."
        ),
    )
    path: str = Field(
        "",
        description=(
            "Explicit direct artifact path. Use only when no source Ticket is "
            "available; do not combine it with source_ticket_id when authoring."
        ),
    )

    @model_validator(mode="after")
    def require_source_or_path(self) -> "ArtifactBinding":
        if not self.source_ticket_id and not self.path:
            raise ValueError("an artifact binding requires source_ticket_id or path")
        if self.work_product_id and not self.source_ticket_id:
            raise ValueError("work_product_id requires source_ticket_id")
        if self.source_ticket_id and self.path and not self.work_product_id:
            raise ValueError(
                "an unresolved binding cannot contain both source_ticket_id and path"
            )
        return self


class CreateTicketBody(BaseModel):
    """The one accepted POST /api/tickets request envelope."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1)
    input_format: TicketInputFormat = "typed"
    iteration: int = Field(0, ge=0)
    payload: dict[str, Any]
    customization: AgentCustomization | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    run_id: str = ""
    task_name: str = Field("", max_length=64)
    run_name: str = Field("", max_length=128)
    ticket_id: str = ""
    gpu_provider: Literal["cluster", "cloud", "instance"] | None = None
    num_gpus: int | None = Field(
        None, ge=0,
        description="Maximum GPUs for a standalone Run; None or zero means unlimited.",
    )
    generation_backend: Literal["hf", "vllm"] | None = None
    metric: str = Field("score", min_length=1, max_length=64)
    metric_direction: Literal["max", "min"] = "max"


class CreateTicketResponse(BaseModel):
    """Exact successful response from ``POST /api/tickets``.

    Ticket identity is named ``id`` everywhere in the persisted/API Ticket
    shape. ``ticket_id`` is reserved for references to a Ticket from another
    record; it is deliberately not an alias here.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        description=(
            "Canonical id of the newly created Ticket. Use this exact value as "
            "SupervisorAction.child_ticket_id."
        ),
    )
    run_id: str
    agent_id: str = Field(min_length=1)
    status: TicketStatus
    input_format: TicketInputFormat
    lane: TicketLane
    iteration: int = Field(ge=0)
    payload: dict[str, Any]
    customization: dict[str, Any]
    inputs: dict[str, Any]
    summary: str
    error_message: str
    repair_attempts: int = Field(0, ge=0, le=3)
    repair_route: Literal["", "self", "orchestrator", "terminal"] = ""
    created_at: str
    updated_at: str


class FreeformPayload(StoredPayload):
    request: str = Field(min_length=1)
    attachments: list[str] = Field(default_factory=list)


class InfrastructureProvisionPayload(StoredPayload):
    operation: Literal["provision"]
    purpose: Literal["train", "inference"]


class InfrastructureReleasePayload(StoredPayload):
    operation: Literal["release"]
    instance_id: str = Field(min_length=1)
    cloud_backend: Literal["", "vastai", "lambda"] = ""


class PipelineDataRequestPayload(StoredPayload):
    """Orchestrator-authored Data request before the API stamps Run facts."""

    operation: Literal["prepare_run_data"]
    dataset_source: str = ""
    dataset: str = ""
    dataset_split: str = ""
    dataset_config: str = ""
    data_query: str = ""
    training_method: str = Field(min_length=1)
    recipe_intent: DataRecipeIntent = Field(default_factory=DataRecipeIntent)
    branch_transition: SearchBranchTransition = Field(
        default_factory=SearchBranchTransition,
    )
    configuration_suggestions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_source_and_method(self) -> "PipelineDataRequestPayload":
        if not (self.dataset or self.data_query):
            raise ValueError("prepare_run_data requires dataset or data_query")
        self.training_method = self.training_method.strip().lower()
        if self.training_method not in METHOD_CONFIG_KEYS:
            raise ValueError(
                f"unsupported training_method={self.training_method!r}; installed "
                f"methods: {', '.join(sorted(METHOD_CONFIG_KEYS))}"
            )
        unknown = sorted(set(self.configuration_suggestions) - _DATA_CONFIGURATION_KEYS)
        if unknown:
            raise ValueError(
                "pipeline Data suggestions may contain only method_ids/target_size; "
                f"unsupported keys: {unknown}"
            )
        if "method_ids" in self.configuration_suggestions:
            methods = self.configuration_suggestions["method_ids"]
            if (
                not isinstance(methods, list)
                or any(
                    not isinstance(value, str) or value not in DATA_METHOD_IDS
                    for value in methods
                )
                or len(methods) != len(set(methods))
            ):
                raise ValueError(
                    "suggested Data method_ids must be a unique list of "
                    "installed canonical ids"
                )
        return self


class PipelineTrainRequestPayload(StoredPayload):
    """Orchestrator-authored Train request; user pins are API-stamped."""

    operation: Literal["train"]
    base_model: str = Field(min_length=1)
    model_source: Literal["base_model", "checkpoint"]
    parent_selection_rationale: str = Field(min_length=1)
    branch_transition: SearchBranchTransition = Field(
        default_factory=SearchBranchTransition,
    )
    configuration_suggestions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_high_level_advice(self) -> "PipelineTrainRequestPayload":
        unknown = sorted(
            set(self.configuration_suggestions) - {"direction", "training_method"}
        )
        if unknown:
            raise ValueError(
                "pipeline Train suggestions support only direction/training_method; "
                f"unsupported keys: {unknown}"
            )
        return self


class PipelineInferenceRequestPayload(StoredPayload):
    """Orchestrator-authored Inference request before immutable Run stamping."""

    operation: Literal["run_inference"]
    model_source: Literal["base_model", "checkpoint"]
    base_model: str = Field(min_length=1)
    branch_transition: SearchBranchTransition = Field(
        default_factory=SearchBranchTransition,
    )
    configuration_suggestions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_baseline_only_advice(self) -> "PipelineInferenceRequestPayload":
        unknown = sorted(set(self.configuration_suggestions) - {"direction"})
        if unknown:
            raise ValueError(
                "pipeline Inference suggestions support only direction; "
                f"unsupported keys: {unknown}"
            )
        return self


class PipelineEvaluationRequestPayload(StoredPayload):
    """Evaluation request; Task-owned scorer/metric fields are API-stamped."""


class DataPayload(StoredPayload):
    operation: DataOperation
    # Auto mode only: the engine-created scope_problem work order carries the
    # objective and optional hints; every other operation leaves these empty.
    task_objective: str = ""
    test_query: str = ""
    test_set_name: str = ""
    constraints: list[str] = Field(default_factory=list)
    dataset_source: str = ""
    dataset: str = ""
    dataset_split: str = ""
    dataset_config: str = ""
    data_query: str = ""
    training_method: str = ""
    recipe_intent: DataRecipeIntent = Field(default_factory=DataRecipeIntent)
    branch_transition: SearchBranchTransition = Field(
        default_factory=SearchBranchTransition,
    )
    data_intent_signature: str = Field(
        "",
        max_length=64,
        pattern=r"^(?:|[0-9a-f]{64})$",
        description=(
            "API-computed bare lowercase SHA-256 of the stored Data intent; "
            "callers leave it empty and the API stamps it."
        ),
    )
    # These fields belong only to the private held-out stripping operation.
    # Optimization Data is capability-isolated from Validation.
    validation_policy: Literal["supplied"] = "supplied"
    validation_fraction: Literal[0.0] = 0.0
    scoring_set: str = ""
    answer_fields: list[str] = Field(default_factory=list)
    metric_type: Literal["builtin", "custom"] = "builtin"
    # Private held-out preparation requires this value. Optimization Data sees
    # no scoring metric; scope_problem derives one rather than receiving it.
    metric: str = ""
    evaluation_script: str = ""
    evaluator_sha256: str = Field("", pattern=r"^(?:|[0-9a-f]{64})$")
    sample_submission: str = ""
    configuration_suggestions: dict[str, Any] = Field(default_factory=dict)
    configuration_pins: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_run_source(self) -> "DataPayload":
        if self.operation == "scope_problem":
            if not self.task_objective.strip():
                raise ValueError("scope_problem requires task_objective")
            if any(not str(c).strip() for c in self.constraints):
                raise ValueError("constraints must contain non-empty strings")
            if any((
                self.dataset_source, self.dataset, self.dataset_split,
                self.dataset_config, self.data_query, self.training_method, self.scoring_set,
                self.answer_fields, self.metric, self.evaluation_script,
                self.evaluator_sha256, self.sample_submission,
                self.test_set_name,
                self.configuration_suggestions, self.configuration_pins,
                self.data_intent_signature,
            )) or self.recipe_intent != DataRecipeIntent() or (
                self.branch_transition.level != "none"
            ):
                raise ValueError(
                    "scope_problem derives the scoring contract itself and cannot "
                    "carry training-source, method, metric, scoring, or recipe values"
                )
            return self
        if any(getattr(self, name) for name in SCOPING_ONLY_FIELDS):
            raise ValueError(f"{self.operation} must not carry Auto-mode scoping fields")
        if self.operation == "prepare_holdout_data" and not self.metric.strip():
            raise ValueError("prepare_holdout_data requires metric")
        if self.operation == "prepare_run_data" and not (
            self.dataset or self.data_query
        ):
            raise ValueError("prepare_run_data requires dataset or data_query")
        if self.operation == "prepare_run_data":
            method = self.training_method.strip().lower()
            if not method:
                raise ValueError("prepare_run_data requires training_method")
            if method not in METHOD_CONFIG_KEYS:
                raise ValueError(
                    f"unsupported training_method={method!r}; installed methods: "
                    + ", ".join(sorted(METHOD_CONFIG_KEYS))
                )
            expected_signature = data_intent_signature(self.model_dump())
            if self.data_intent_signature and self.data_intent_signature != expected_signature:
                raise ValueError("data_intent_signature does not match the stored recipe intent")
            self.data_intent_signature = expected_signature
            leaked = {
                "test_set_name": self.test_set_name,
                "scoring_set": self.scoring_set,
                "answer_fields": self.answer_fields,
                "metric": self.metric,
                "evaluation_script": self.evaluation_script,
                "evaluator_sha256": self.evaluator_sha256,
                "sample_submission": self.sample_submission,
            }
            exposed = sorted(name for name, value in leaked.items() if value)
            if exposed:
                raise ValueError(
                    "prepare_run_data is Validation-blind; forbidden fields: "
                    + ", ".join(exposed)
                )
        unknown_config = sorted(
            (set(self.configuration_suggestions) | set(self.configuration_pins))
            - _DATA_CONFIGURATION_KEYS
        )
        if unknown_config:
            raise ValueError(
                "Data configuration may contain only method_ids/target_size; "
                f"training or inference hyperparameters are forbidden: {unknown_config}"
            )
        if "target_size" in self.configuration_pins:
            target_size = self.configuration_pins["target_size"]
            if isinstance(target_size, bool) or not isinstance(target_size, int) or target_size < 1:
                raise ValueError("pinned Data target_size must be an integer >= 1")
        for owner, values in (
            ("suggested", self.configuration_suggestions),
            ("pinned", self.configuration_pins),
        ):
            if "method_ids" not in values:
                continue
            methods = values["method_ids"]
            if (
                not isinstance(methods, list)
                or any(
                    not isinstance(value, str) or value not in DATA_METHOD_IDS
                    for value in methods
                )
                or len(methods) != len(set(methods))
            ):
                raise ValueError(
                    f"{owner} Data method_ids must be a unique list of "
                    "installed canonical ids"
                )
        if self.operation == "prepare_holdout_data" and any((
            self.dataset_source, self.dataset, self.dataset_split,
            self.dataset_config, self.data_query, self.training_method,
            self.configuration_suggestions, self.configuration_pins,
        )):
            raise ValueError(
                "prepare_holdout_data cannot carry training-source, method, or configuration values"
            )
        if self.operation == "prepare_holdout_data" and (
            self.recipe_intent != DataRecipeIntent() or self.data_intent_signature
        ):
            raise ValueError(
                "prepare_holdout_data cannot carry a training data recipe"
            )
        if (
            self.operation == "prepare_holdout_data"
            and self.branch_transition.level != "none"
        ):
            raise ValueError(
                "prepare_holdout_data cannot carry an optimization search transition"
            )
        if self.operation == "prepare_holdout_data" and (
            self.validation_policy != "supplied"
            or not self.scoring_set
            or not self.answer_fields
            or not self.test_set_name.strip()
        ):
            raise ValueError(
                "prepare_holdout_data requires test_set_name, a supplied scoring "
                "set, and answer fields"
            )
        self.training_method = self.training_method.strip().lower()
        return self


class TrainPayload(StoredPayload):
    operation: Literal["train"]
    base_model: str = Field(min_length=1)
    training_method_pin: str = ""
    method_config_pins: dict[str, Any] = Field(default_factory=dict)
    loss_objective_pins: dict[str, Any] = Field(default_factory=dict)
    configuration_suggestions: dict[str, Any] = Field(default_factory=dict)
    configuration_pins: dict[str, Any] = Field(default_factory=dict)
    model_source: Literal["base_model", "checkpoint"]
    parent_selection_rationale: str = Field(min_length=1)
    branch_transition: SearchBranchTransition = Field(
        default_factory=SearchBranchTransition,
    )
    data_signature: str = Field(
        "",
        max_length=64,
        pattern=r"^(?:|[0-9a-f]{64})$",
        description=(
            "Engine-stamped identity of the exact Data artifact bound to Train: "
            "bare lowercase SHA-256 with no prefix."
        ),
    )

    @model_validator(mode="after")
    def validate_closed_train_configuration(self) -> "TrainPayload":
        self.training_method_pin = self.training_method_pin.strip().lower()
        if self.data_signature and not is_sha256(self.data_signature):
            raise ValueError("Train data_signature must contain 64 hexadecimal characters")
        unknown_pins = sorted(set(self.configuration_pins) - _TRAIN_CONFIGURATION_KEYS)
        if unknown_pins:
            raise ValueError(
                f"unsupported Train configuration_pins keys: {unknown_pins}; "
                f"allowed: {sorted(_TRAIN_CONFIGURATION_KEYS)}"
            )
        unknown_suggestions = sorted(
            set(self.configuration_suggestions) - _TRAIN_SUGGESTION_KEYS
        )
        if unknown_suggestions:
            raise ValueError(
                "Train configuration_suggestions may contain only a high-level "
                "direction/training_method or documented advisory Train controls; "
                f"unsupported keys: {unknown_suggestions}"
            )
        positive_ints = {"batch_size", "num_epochs", "max_seq_len"}
        nonnegative_ints = {"lora_r", "lora_alpha"}
        for key in positive_ints | nonnegative_ints:
            if key not in self.configuration_pins:
                continue
            value = self.configuration_pins[key]
            minimum = 1 if key in positive_ints else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"pinned Train {key} must be an integer >= {minimum}")
        if "learning_rate" in self.configuration_pins:
            value = self.configuration_pins["learning_rate"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) <= 0
            ):
                raise ValueError("pinned Train learning_rate must be a number > 0")
        suggested_method = self.configuration_suggestions.get("training_method")
        if suggested_method:
            method = str(suggested_method).strip().lower()
            if method not in METHOD_CONFIG_KEYS:
                raise ValueError(
                    f"unsupported suggested training_method={method!r}; installed "
                    f"methods: {', '.join(sorted(METHOD_CONFIG_KEYS))}"
                )
            self.configuration_suggestions["training_method"] = method
        if (self.method_config_pins or self.loss_objective_pins) and not self.training_method_pin:
            raise ValueError(
                "method_config_pins/loss_objective_pins require training_method_pin"
            )
        if self.training_method_pin:
            errors = method_config_errors(
                self.training_method_pin,
                self.method_config_pins,
                require_dependencies=True,
            )
            if errors:
                raise ValueError("; ".join(errors))
            validate_loss_objective_config(
                self.training_method_pin, self.loss_objective_pins,
            )
        return self


class InferencePayload(StoredPayload):
    operation: Literal["run_inference"]
    model_source: Literal["base_model", "checkpoint"]
    base_model: str = Field(min_length=1)
    branch_transition: SearchBranchTransition = Field(
        default_factory=SearchBranchTransition,
    )
    configuration_suggestions: dict[str, Any] = Field(default_factory=dict)
    configuration_pins: dict[str, Any] = Field(default_factory=dict)
    scoring_set: str = Field(min_length=1)
    sample_submission: str = Field(min_length=1)
    test_set_name: str = ""

    @model_validator(mode="after")
    def validate_closed_inference_configuration(self) -> "InferencePayload":
        unknown_pins = sorted(set(self.configuration_pins) - _INFERENCE_CONFIGURATION_KEYS)
        if unknown_pins:
            raise ValueError(
                f"unsupported Inference configuration_pins keys: {unknown_pins}; "
                f"allowed: {sorted(_INFERENCE_CONFIGURATION_KEYS)}"
            )
        unknown_suggestions = sorted(
            set(self.configuration_suggestions) - _INFERENCE_SUGGESTION_KEYS
        )
        if unknown_suggestions:
            raise ValueError(
                "Inference configuration_suggestions are high-level only; "
                f"unsupported keys: {unknown_suggestions}"
            )
        pins = self.configuration_pins
        prompt_keys = {"prompt_framing", "system_prompt"}
        if set(pins) & prompt_keys:
            framing, _reasoning_type, system = canonical_prompt_contract(
                prompt_framing=pins.get("prompt_framing", ""),
                model_reasoning_type="non_thinking",
                system_prompt=pins.get("system_prompt", ""),
                allow_empty=True,
            )
            pins.update({"prompt_framing": framing, "system_prompt": system})
        if "inference_config" in pins:
            pins["inference_config"] = validate_inference_config(
                pins["inference_config"]
            )
        if "decoding_config" in pins:
            pins["decoding_config"] = validate_decoding_config(
                pins["decoding_config"]
            )
        return self


class EvaluationPayload(StoredPayload):
    metric: str = Field(min_length=1)
    evaluation_config: dict[str, Any] = Field(default_factory=dict)
    scoring_set: str = Field(min_length=1)
    evaluation_script: str = ""
    evaluator_sha256: str = Field("", pattern=r"^(?:|[0-9a-f]{64})$")
    answer_fields: list[str] = Field(min_length=1)
    sample_submission: str = Field(min_length=1)
    test_set_name: str = ""


class RegistryPayload(StoredPayload):
    base_model: str = Field(min_length=1)
    training_method: str = Field(min_length=1)
    dataset_source: str = ""
    task_objective: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    metric_direction: Literal["max", "min"]
    checkpoint_is_remote: bool = False


class PipelineRegistryRequestPayload(StoredPayload):
    """A pipeline caller requests registration; the API derives all lineage."""


PIPELINE_REQUEST_PAYLOAD_BY_AGENT: dict[str, type[StoredPayload]] = {
    "infrastructure": InfrastructureProvisionPayload,
    "data": PipelineDataRequestPayload,
    "train": PipelineTrainRequestPayload,
    "inference": PipelineInferenceRequestPayload,
    "evaluation": PipelineEvaluationRequestPayload,
    "registry": PipelineRegistryRequestPayload,
}


PIPELINE_PAYLOAD_BY_AGENT: dict[str, type[StoredPayload]] = {
    "data": DataPayload,
    "train": TrainPayload,
    "inference": InferencePayload,
    "evaluation": EvaluationPayload,
    "registry": RegistryPayload,
}


def specialist_request_payload_schemas() -> dict[str, Any]:
    """Machine-readable child payloads accepted from Orchestrator.

    Pydantic cannot express closed keys inside ``dict[str, Any]``. Enrich those
    few advisory mappings here so the schema sent to Orchestrator exposes the
    same key boundaries enforced by model validators instead of hiding them
    until POST time.
    """
    schemas = {
        agent_id: model.model_json_schema()
        for agent_id, model in PIPELINE_REQUEST_PAYLOAD_BY_AGENT.items()
    }
    data_properties = schemas["data"]["properties"]
    data_properties["training_method"]["enum"] = sorted(METHOD_CONFIG_KEYS)
    data_properties["configuration_suggestions"] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "method_ids": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(DATA_METHOD_IDS)},
            },
            "target_size": {"type": "integer", "minimum": 1},
        },
        "description": (
            "Optional Data-only advice. Omit a key to delegate it; no other "
            "keys or value shapes are accepted."
        ),
    }
    schemas["train"]["properties"]["configuration_suggestions"] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "direction": {"type": "string"},
            "training_method": {
                "type": "string",
                "enum": sorted(METHOD_CONFIG_KEYS),
            },
        },
        "description": (
            "Optional high-level Train advice. Omit a key to delegate it; "
            "detailed hyperparameters are not accepted here."
        ),
    }
    schemas["inference"]["properties"]["configuration_suggestions"] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"direction": {"type": "string"}},
        "description": (
            "Optional baseline-only direction advice. No configuration keys "
            "are accepted from Orchestrator."
        ),
    }
    return schemas


def specialist_stored_payload_schemas() -> dict[str, Any]:
    """Machine-readable payloads after API-owned facts have been stamped."""
    return {
        "infrastructure": {
            "provision": InfrastructureProvisionPayload.model_json_schema(),
            "release": InfrastructureReleasePayload.model_json_schema(),
        }, **{
            agent_id: model.model_json_schema()
            for agent_id, model in PIPELINE_PAYLOAD_BY_AGENT.items()
        },
    }


def specialist_input_binding_contracts() -> dict[str, Any]:
    """Exact conditional input roles and lineage enforced by the Ticket API.

    The outer ArtifactBinding schema describes one value, but it cannot express
    which keys a particular Specialist branch requires. Orchestrator receives
    this table so required roles, source Agents, iteration relationships, and
    same-source constraints are never discoverable only by a rejected mutation.
    """
    common = {
        "same_run": True,
        "source_status": ["succeeded", "degraded"],
    }

    def source(agent: str, iteration: str = "any_iteration") -> dict[str, Any]:
        return {**common, "source_agent": agent, "source_iteration": iteration}

    return {
        "infrastructure": {
            "discriminator": "none",
            "variants": {"default": {"required": {}}},
        },
        "data": {
            "discriminator": "none",
            "variants": {"default": {"required": {}}},
        },
        "train": {
            "discriminator": "payload.model_source",
            "variants": {
                "base_model": {"required": _required_input_bindings(
                    "train", {"model_source": "base_model"}
                ), "lineage": {
                    "training_dataset": source("data", "not_later_than_child"),
                    "validation_dataset": {
                        **source("data", "not_later_than_child"),
                        "same_source_ticket_as": "training_dataset",
                    },
                    "inference_config": {
                        **source("inference", "not_later_than_child"),
                        "source_payload": {"model_source": "base_model"},
                    },
                    "device_info": source("infrastructure"),
                }},
                "checkpoint": {
                    "required": _required_input_bindings(
                        "train", {"model_source": "checkpoint"}
                    ),
                    "lineage": {
                        "training_dataset": source("data", "not_later_than_child"),
                        "validation_dataset": {
                            **source("data", "not_later_than_child"),
                            "same_source_ticket_as": "training_dataset",
                        },
                        "inference_config": {
                            **source("inference", "not_later_than_child"),
                            "source_payload": {"model_source": "base_model"},
                        },
                        "device_info": source("infrastructure"),
                        "parent_checkpoint": source(
                            "train", "earlier_optimization_iteration"
                        ),
                        "parent_train_config": {
                            **source("train", "earlier_optimization_iteration"),
                            "same_source_ticket_as": "parent_checkpoint",
                        },
                    },
                },
            },
        },
        "inference": {
            "discriminator": "payload.model_source",
            "variants": {
                "base_model": {
                    "required": _required_input_bindings(
                        "inference", {"model_source": "base_model"}
                    ),
                    "lineage": {
                        "device_info": source("infrastructure"),
                        "inference_data_profile": source("data", "exactly_0"),
                    },
                },
                "checkpoint": {
                    "required": _required_input_bindings(
                        "inference", {"model_source": "checkpoint"}
                    ),
                    "optional": {"predict_script": "script"},
                    "lineage": {
                        "device_info": source("infrastructure"),
                        "checkpoint": source("train", "same_iteration"),
                        "inference_config": {
                            **source("inference", "not_later_than_child"),
                            "source_payload": {"model_source": "base_model"},
                        },
                        "predict_script": {
                            **source("inference", "not_later_than_child"),
                            "source_payload": {"model_source": "base_model"},
                        },
                    },
                },
            },
        },
        "evaluation": {
            "discriminator": "none",
            "variants": {"default": {
                "required": _required_input_bindings("evaluation", {}),
                "lineage": {
                    "predictions": {
                        **source("inference", "same_iteration"),
                        "source_lane": "optimization",
                    },
                },
            }},
        },
        "registry": {
            "discriminator": "payload.checkpoint_is_remote",
            "variants": {
                "false": {
                    "required": _required_input_bindings(
                        "registry", {"checkpoint_is_remote": False}
                    ),
                    "lineage": {
                        "checkpoint": source("train", "same_iteration"),
                        "train_config": {
                            **source("train", "same_iteration"),
                            "same_source_ticket_as": "checkpoint",
                        },
                        "metrics": {
                            **source("evaluation", "same_iteration"),
                            "must_evaluate_input": "checkpoint",
                        },
                    },
                },
                "true": {
                    "required": _required_input_bindings(
                        "registry", {"checkpoint_is_remote": True}
                    ),
                    "lineage": {
                        "checkpoint": source("train", "same_iteration"),
                        "train_config": {
                            **source("train", "same_iteration"),
                            "same_source_ticket_as": "checkpoint",
                        },
                        "metrics": {
                            **source("evaluation", "same_iteration"),
                            "must_evaluate_input": "checkpoint",
                        },
                        "device_info": source("infrastructure"),
                    },
                },
            },
        },
    }


def validate_stored_payload(
    *, agent_id: str, input_format: str, payload: dict[str, Any]
) -> dict[str, Any]:
    if input_format == "freeform":
        return FreeformPayload.model_validate(payload).model_dump()
    if agent_id == "infrastructure" and payload.get("operation") == "provision":
        model: type[BaseModel] | None = InfrastructureProvisionPayload
    elif agent_id == "infrastructure" and payload.get("operation") == "release":
        model = InfrastructureReleasePayload
    else:
        model = PIPELINE_PAYLOAD_BY_AGENT.get(agent_id)
    if model is None:
        if agent_id == "orchestrator":
            from zevo.contracts.orchestrator import OrchestratePayload
            return OrchestratePayload.model_validate(payload).model_dump()
        raise ValueError(f"no typed payload contract for agent {agent_id!r}")
    return model.model_validate(payload).model_dump()


def _required_input_bindings(agent_id: str, payload: dict[str, Any]) -> dict[str, str]:
    """Return the one required binding-name → artifact-role mapping."""
    required: dict[str, str] = {}
    if agent_id == "train":
        required = {
            "training_dataset": "training_dataset",
            "validation_dataset": "validation_dataset",
            "inference_config": "inference_config",
            "device_info": "device_info",
        }
        if payload.get("model_source") == "checkpoint":
            required.update({
                "parent_checkpoint": "checkpoint",
                "parent_train_config": "train_config",
            })
    elif agent_id == "inference":
        required = {"device_info": "device_info"}
        if payload.get("model_source") == "base_model":
            required["inference_data_profile"] = "inference_data_profile"
        else:
            required.update({
                "checkpoint": "checkpoint",
                "inference_config": "inference_config",
            })
    elif agent_id == "evaluation":
        required = {"predictions": "predictions"}
    elif agent_id == "registry":
        required = {
            "checkpoint": "checkpoint",
            "metrics": "metrics",
            "train_config": "train_config",
        }
        if payload.get("checkpoint_is_remote"):
            required["device_info"] = "device_info"
    return required


def validate_bindings(
    *, agent_id: str, payload: dict[str, Any], inputs: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    parsed = {
        name: ArtifactBinding.model_validate(value).model_dump()
        for name, value in inputs.items()
    }
    required = _required_input_bindings(agent_id, payload)
    for name, role in required.items():
        binding = parsed.get(name)
        if binding is None:
            raise ValueError(f"{agent_id} requires input binding {name!r}")
        if binding["artifact_role"] != role:
            raise ValueError(
                f"input {name!r} requires artifact_role={role!r}; "
                f"got {binding['artifact_role']!r}"
            )
    return parsed


def _main(argv: list[str] | None = None) -> int:
    """Expose a non-guessing parser for the create-Ticket API receipt."""
    parser = ArgumentParser(prog="python -m zevo.contracts.tickets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "created-ticket-id",
        help="validate a POST /api/tickets JSON response from stdin and print its id",
    )
    subparsers.add_parser(
        "create-response-schema",
        help="print the exact successful POST /api/tickets response schema",
    )
    args = parser.parse_args(argv)
    if args.command == "create-response-schema":
        print(json.dumps(CreateTicketResponse.model_json_schema(), indent=2, sort_keys=True))
        return 0
    try:
        receipt = CreateTicketResponse.model_validate_json(sys.stdin.read())
    except Exception as exc:
        print(f"INVALID CreateTicketResponse: {exc}", file=sys.stderr)
        return 1
    print(receipt.id)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
