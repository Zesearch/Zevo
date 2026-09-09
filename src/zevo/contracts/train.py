"""Typed contract for one Train iteration.

Train receives one explicitly bound versioned Data corpus, the engine's frozen
raw Validation view, and baseline Inference's YAML. It
retains the active method branch unless the Orchestrator explicitly advances an
exhausted branch, selects concrete loss/hyperparameters for this iteration,
copies the matching model-lineage Baseline prompt and template identity, writes
``train_config.yaml``, and trains from the explicitly selected earlier Run
checkpoint or baseline model.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zevo.contracts._base import AgentResult, AgentTaskInput
from zevo.contracts.infrastructure import SlurmStageJobContract


class TrainExecutionContract(BaseModel):
    """Engine-owned requirements for the one real training command."""

    model_config = ConfigDict(extra="forbid")
    foreground: Literal[True] = True
    stream_to_local_log: Literal[True] = True
    timeout_seconds: Literal[14400] = 14400
    timeout_milliseconds: Literal[14400000] = 14400000
    required_environment: dict[str, str]
    secret_environment_names: list[Literal["WANDB_API_KEY"]] = Field(default_factory=list)
    tracking_provider: Literal["", "weights_and_biases"] = ""
    tracking_url: str = ""
    slurm_step_name: str = Field(
        min_length=1,
        pattern=r"^zevo-[A-Za-z0-9_-]+$",
        description=(
            "Engine-owned name for the remote training process. Use it exactly "
            "so Ticket cancellation targets only this process."
        ),
    )


class TrainTaskInput(AgentTaskInput):
    operation: Literal["train"] = "train"
    run_id: str = Field(min_length=1)
    iteration: int = Field(ge=1)

    dataset_path: str = Field(min_length=1)
    validation_dataset_path: str = Field(
        min_length=1,
        description=(
            "Engine-bound raw Validation scoring records for trainer evaluation "
            "only. Train renders a temporary method-specific eval view without "
            "using it to choose the training source or configuration."
        ),
    )
    validation_answer_fields: list[str] = Field(
        min_length=1,
        description="Ground-truth fields used only to render trainer evaluation labels.",
    )
    data_signature: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "Verified identity of the exact versioned training-data artifact: "
            "64 lowercase hexadecimal SHA-256 characters with no prefix. Copy "
            "this exact value into train_config.yaml."
        ),
    )
    inference_config_path: str = Field(
        min_length=1,
        description=(
            "Baseline Inference's complete YAML. Train must copy its prompt, "
            "tokenizer, chat-template identity, and special-token ids exactly "
            "into train_config.yaml and render training input accordingly."
        ),
    )
    expected_inference_config_sha256: str = Field(
        ...,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "Engine-computed bare SHA-256 of the exact baseline Inference YAML "
            "file bytes. Copy it exactly into "
            "train_config.yaml.inference_config_sha256; do not recompute or "
            "decorate it."
        ),
    )
    train_config_schema: dict[str, Any] = Field(
        ...,
        description=(
            "Exact JSON Schema for train_config.yaml. This supplied schema is "
            "the sole key/type authority; method Skill prose does not add or "
            "rename outer configuration keys."
        ),
    )
    training_method_contracts: dict[str, Any] = Field(
        ...,
        description=(
            "Exact per-method allowed/required keys for method_config and "
            "the complete loss_contract for every prompt framing, plus allowed "
            "loss_contract.objective_config keys, recommended starting values, "
            "recommended_training_starts (non-binding best-practice starting "
            "values for the common TrainingConfig knobs, e.g. LoRA "
            "rank/alpha/target-modules, learning_rate, effective batch, warmup, "
            "epochs, packing), required software-version keys, PEFT/LoRA state, "
            "and hash encoding. Select one method/framing entry and copy "
            "canonical literals; treat recommended_training_starts as guidance to "
            "start from and deviate from with rationale, never as caps; never "
            "invent objective/owner/framework/target_scope or transfer keys."
        ),
    )
    config_validation_command: str = Field(
        ...,
        min_length=1,
        description=(
            "Side-effect-free local command template that must pass before "
            "dependency/runtime preflight or GPU work. Replace "
            "<absolute-yaml-path> with train_config.yaml."
        ),
    )
    telemetry_helper_path: str = Field(
        ...,
        min_length=1,
        description=(
            "System-owned Python helper to copy beside train.py and import. It "
            "emits all numeric Trainer logs in Zevo's progress protocol; do not "
            "replace it with a loss-only callback."
        ),
    )
    telemetry_interval_steps: Literal[20] = Field(
        20,
        description="Fixed training-log cadence in optimizer steps.",
    )
    execution_contract: TrainExecutionContract = Field(
        ...,
        description=(
            "Mandatory foreground/streaming/deadline contract. Supply the seconds "
            "value to run_bash timeout_sec or the milliseconds value to a CLI "
            "Bash timeout; export every required_environment entry remotely."
        ),
    )

    base_model: str = Field(min_length=1)
    model_source: Literal["base_model", "checkpoint"]
    parent_selection_rationale: str = Field(min_length=1)
    parent_checkpoint_path: str = Field(
        "",
        description=(
            "Empty when this experiment branches from the baseline model; otherwise "
            "the explicitly selected earlier checkpoint from this Run."
        ),
    )
    parent_train_config_path: str = Field(
        "",
        description=(
            "Empty for a baseline branch; otherwise the train_config.yaml produced "
            "by the same Train Ticket as parent_checkpoint_path."
        ),
    )
    branch_transition: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Engine-validated exhausted-branch audit record. A Method change "
            "must carry the Orchestrator's method/base_model transition evidence."
        ),
    )

    training_method_pin: str = ""
    method_config_pins: dict[str, Any] = Field(default_factory=dict)
    loss_objective_pins: dict[str, Any] = Field(default_factory=dict)
    configuration_suggestions: dict[str, Any] = Field(default_factory=dict)
    configuration_pins: dict[str, Any] = Field(default_factory=dict)

    device_info_path: str = Field(min_length=1)
    slurm_job: SlurmStageJobContract = Field(
        default_factory=SlurmStageJobContract,
        description=(
            "Finite stage-owned job contract. Enabled only for cluster; the "
            "submitted file executes train.py itself and exits with it."
        ),
    )
    work_dir: str = Field(min_length=1)
    generation_backend: Literal["hf", "vllm"] = "vllm"

    @model_validator(mode="after")
    def require_selected_parent(self) -> "TrainTaskInput":
        if self.iteration == 1 and self.model_source != "base_model":
            raise ValueError("iteration 1 must start from the baseline model")
        if self.model_source == "base_model":
            if self.parent_checkpoint_path or self.parent_train_config_path:
                raise ValueError(
                    "a baseline branch cannot bind a parent checkpoint or Train config"
                )
        else:
            if not self.parent_checkpoint_path:
                raise ValueError(
                    "checkpoint model_source requires parent_checkpoint_path"
                )
            if not self.parent_train_config_path:
                raise ValueError(
                    "checkpoint model_source requires parent_train_config_path"
                )
        return self


class IntermediateCheckpoint(BaseModel):
    """One verified, weights-only branch point retained by Train."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    step: int = Field(0, ge=0)
    epoch: float = Field(0.0, ge=0, allow_inf_nan=False)
    retention_reason: str = Field(min_length=1)
    save_only_model: Literal[True] = True

    @model_validator(mode="after")
    def require_training_position(self) -> "IntermediateCheckpoint":
        if self.step == 0 and self.epoch == 0:
            raise ValueError("an intermediate checkpoint requires step or epoch")
        return self


class LossSeriesSummary(BaseModel):
    """Engine-derived summary of one loss series from the successful attempt."""

    model_config = ConfigDict(extra="forbid")

    points: int = Field(0, ge=0)
    first: float | None = Field(None, allow_inf_nan=False)
    final: float | None = Field(None, allow_inf_nan=False)
    minimum: float | None = Field(None, allow_inf_nan=False)
    minimum_step: int | None = Field(None, ge=0)


class TrainingDiagnostics(BaseModel):
    """Deterministic loss evidence materialized by the runner, not the Agent."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    source: Literal["engine_execution_events"] = "engine_execution_events"
    attempt_id: str = ""
    final_step: int = Field(0, ge=0)
    total_steps: int = Field(0, ge=0)
    training_loss: LossSeriesSummary = Field(default_factory=LossSeriesSummary)
    validation_loss: LossSeriesSummary = Field(default_factory=LossSeriesSummary)
    observations: list[Literal[
        "training_loss_decreased",
        "training_loss_increased",
        "validation_loss_decreased",
        "validation_loss_increased",
        "validation_loss_rose_after_minimum",
    ]] = Field(default_factory=list)


class TrainResult(AgentResult):
    status: Literal["succeeded", "deferred", "failed"]
    operation: Literal["train"] = "train"

    train_config_path: str = Field(
        "",
        description="Absolute path to the complete train_config.yaml that actually ran.",
    )
    train_script_path: str = ""
    slurm_script_path: str = ""
    log_path: str = ""
    checkpoint_path: str = Field(
        "",
        description=(
            "Verified checkpoint directory. A remote checkpoint must live in "
            "a Ticket-unique path containing ticket_id so later iterations cannot "
            "overwrite a selectable historical parent."
        ),
    )
    checkpoint_is_remote: bool = False
    intermediate_checkpoints: list[IntermediateCheckpoint] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Optional verified weights-only branch points retained by this Train. "
            "The final checkpoint_path is not repeated here."
        ),
    )
    n_examples: int = Field(0, ge=0)
    final_loss: float = -1.0
    training_seconds: int = Field(0, ge=0)
    tracking_url: str = Field(
        "",
        description="Public Weights & Biases Run URL when tracking is configured.",
    )

    @model_validator(mode="after")
    def require_success_artifacts(self) -> "TrainResult":
        if self.status == "succeeded" and not (
            self.train_config_path
            and self.train_script_path
            and self.log_path
            and self.checkpoint_path
        ):
            raise ValueError(
                "successful train requires config, script, log, and checkpoint"
            )
        if (
            self.status == "succeeded"
            and self.checkpoint_is_remote
            and self.ticket_id not in self.checkpoint_path
        ):
            raise ValueError(
                "a remote checkpoint path must contain ticket_id and remain "
                "unique to this Train Ticket"
            )
        return self
