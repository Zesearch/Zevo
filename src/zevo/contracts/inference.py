"""Typed contract for one Inference execution.

Baseline Inference selects and writes one ``inference_config.yaml`` for each
base-model lineage explored by the Run. Every trained-model Inference receives
the matching lineage artifact and must reuse it exactly. There is no separate
planning activation and no per-iteration re-selection of prompt rendering,
task mapping, parsing, or decoding parameters inside one model lineage.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zevo.contracts._base import AgentResult, AgentTaskInput
from zevo.contracts.infrastructure import SlurmStageJobContract


class InferenceMemoryPlanningContract(BaseModel):
    """Engine-owned policy for sizing vLLM memory from the actual workload."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["model_and_workload_sized"] = "model_and_workload_sized"
    weight_headroom_multiplier: Literal[1.2] = 1.2
    runtime_reserve_gib: Literal[4.0] = 4.0
    utilization_step: Literal[0.05] = 0.05
    min_utilization: Literal[0.1] = 0.1
    max_utilization: Literal[0.9] = 0.9
    free_memory_margin_gib: Literal[2.0] = 2.0
    target_formula: Literal[
        "ceil_gib((runtime_weight_gib * 1.2 + kv_cache_gib) / tensor_parallel_size + 4.0)"
    ] = (
        "ceil_gib((runtime_weight_gib * 1.2 + kv_cache_gib) / "
        "tensor_parallel_size + 4.0)"
    )
    utilization_formula: Literal[
        "clamp(ceil_step(target_gpu_memory_gib / total_gpu_memory_gib, 0.05), 0.1, 0.9)"
    ] = (
        "clamp(ceil_step(target_gpu_memory_gib / total_gpu_memory_gib, "
        "0.05), 0.1, 0.9)"
    )


class InferenceTaskInput(AgentTaskInput):
    operation: Literal["run_inference"] = "run_inference"
    run_id: str = Field(min_length=1)
    iteration: int = Field(ge=0)
    model_source: Literal["base_model", "checkpoint"]
    configuration_mode: Literal["select", "reuse"]
    base_model: str = Field(min_length=1)
    checkpoint_path: str = ""
    branch_transition: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Engine-validated exhausted-branch audit record. A later Base-model "
            "baseline carries level=base_model; ordinary executions use none."
        ),
    )

    scoring_set: str = Field(
        min_length=1,
        description="Questions-only scoring data; answers and scorer are absent.",
    )
    sample_submission: str
    inference_data_profile_path: str = ""

    inference_config_path: str = Field(
        "",
        description=(
            "Empty only for baseline: Inference selects and writes the complete "
            "configuration. Trained-model inference receives the baseline YAML "
            "and must execute it without modification."
        ),
    )
    inference_config_schema: dict[str, Any] = Field(
        ...,
        description=(
            "Exact JSON Schema for the InferenceRunConfig YAML. This supplied "
            "schema is the sole key/type authority; do not infer an envelope, "
            "copy an older example, or discover keys by trial and error."
        ),
    )
    inference_mapping_contract: dict[str, Any] = Field(
        ...,
        description=(
            "Exact allowed/required keys and types inside the otherwise generic "
            "measurement.inference_config mapping."
        ),
    )
    config_validation_command: str = Field(
        ...,
        min_length=1,
        description=(
            "Side-effect-free local command template that must pass after the "
            "YAML is written and before any remote command or GPU work. Replace "
            "<absolute-yaml-path> with the produced/supplied file path."
        ),
    )
    memory_planning_contract: InferenceMemoryPlanningContract = Field(
        default_factory=InferenceMemoryPlanningContract,
        description=(
            "Engine-owned vLLM memory-sizing formula. Baseline select mode "
            "records its measured model/workload evidence and absolute target "
            "in inference_config.yaml; predict.py derives the device-specific "
            "utilization from that target. Reuse mode preserves the plan."
        ),
    )
    memory_helper_path: str = Field(
        "",
        description=(
            "System-owned helper copied beside predict.py. Required for vLLM; "
            "use it for both Slurm free-memory preflight and LLM construction."
        ),
    )
    predictions_validation_command: str = Field(
        ...,
        min_length=1,
        description=(
            "Exact side-effect-free predictions validator. The questions-only "
            "and sample-submission paths are already bound by the engine; "
            "replace only <absolute-predictions-csv-path>."
        ),
    )
    reusable_predict_script_path: str = Field(
        "",
        description=(
            "Previously verified predict.py to reuse when it is model-artifact "
            "agnostic and executes the supplied YAML. Empty only when no "
            "compatible prior script is available."
        ),
    )
    option_scoring_helper_path: str = Field(
        "",
        description=(
            "System-owned per-option log-likelihood helper copied beside "
            "predict.py. Import it (both HF teacher-forcing and vLLM "
            "prompt_logprobs paths) only when the inference_config sets "
            "scoring_mode='option_loglikelihood'; it emits the per-option "
            "prediction JSON the mc_loglikelihood metric consumes. Unused in "
            "default generation mode."
        ),
    )
    configuration_suggestions: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Non-binding Orchestrator suggestions. 0, empty string, null, or a "
            "missing key means self-select. Baseline Inference records every "
            "realized choice and accept/adjust/self-select decision in YAML."
        ),
    )
    configuration_pins: dict[str, Any] = Field(
        default_factory=dict,
        description="Binding user/Customized-Pipeline values; suggestions cannot override these.",
    )

    device_info_path: str = Field(min_length=1)
    slurm_job: SlurmStageJobContract = Field(
        default_factory=SlurmStageJobContract,
        description=(
            "Finite stage-owned job contract. Enabled only for cluster; the "
            "submitted file executes predict.py itself and exits with it."
        ),
    )
    work_dir: str = Field(min_length=1)
    generation_backend: Literal["hf", "vllm"] = "vllm"

    @model_validator(mode="after")
    def validate_model_and_config_source(self) -> "InferenceTaskInput":
        if self.generation_backend == "vllm" and not self.memory_helper_path:
            raise ValueError("vLLM inference requires memory_helper_path")
        if self.model_source == "base_model":
            if self.checkpoint_path:
                raise ValueError("baseline inference must not receive checkpoint_path")
            if self.configuration_mode == "select" and self.inference_config_path:
                raise ValueError("baseline selection must create inference config")
            if self.configuration_mode == "select" and not self.inference_data_profile_path:
                raise ValueError("baseline inference requires Data's safe inference profile")
        else:
            if self.iteration < 1:
                raise ValueError("checkpoint inference requires iteration >= 1")
            if not self.checkpoint_path:
                raise ValueError("checkpoint inference requires checkpoint_path")
            if self.configuration_mode != "reuse":
                raise ValueError("checkpoint inference must reuse baseline configuration")
            if not self.inference_config_path:
                raise ValueError("checkpoint inference requires baseline inference_config.yaml")
        if self.configuration_mode == "reuse" and not self.inference_config_path:
            raise ValueError("configuration_mode=reuse requires inference_config_path")
        if self.configuration_mode == "reuse" and any(
            value not in (None, "", 0, {}, [])
            for value in self.configuration_suggestions.values()
        ):
            # Reused inference must execute the existing baseline YAML exactly;
            # configuration_suggestions are baseline-only advice with no effect
            # here. A stray non-empty suggestion (e.g. an Orchestrator that
            # emitted {"direction": ...} on a checkpoint/reuse ticket) is
            # dropped and the run proceeds, rather than hard-failing and
            # wedging the ticket on every wakeup. All other reuse guarantees
            # above still hold.
            self.configuration_suggestions = {}
        return self


class GenerationRecord(BaseModel):
    """Termination evidence for one real generation request, without its text."""

    model_config = ConfigDict(extra="forbid")

    request_index: int = Field(ge=0)
    row_index: int = Field(ge=0)
    turn: int = Field(ge=1)
    finish_reason: str = Field(min_length=1)
    stop_reason: str | int | None = None
    generated_tokens: int = Field(ge=0)


class GenerationDiagnostics(BaseModel):
    """Auditable per-request finish reasons emitted by the inference runtime."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    records: list[GenerationRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def require_dense_request_indices(self) -> "GenerationDiagnostics":
        indices = [record.request_index for record in self.records]
        if indices != list(range(len(indices))):
            raise ValueError(
                "generation diagnostics request_index values must be dense and ordered from 0"
            )
        return self


class GenerationTerminationSummary(BaseModel):
    """Compact, text-free termination evidence safe for iteration history."""

    model_config = ConfigDict(extra="forbid")

    total_requests: int = Field(ge=1)
    stopped_requests: int = Field(ge=0)
    length_limited_requests: int = Field(ge=0)
    finish_reason_counts: dict[str, int]
    stop_reason_counts: dict[str, int]
    max_generated_tokens: int = Field(ge=0)


def summarize_generation_diagnostics(
    diagnostics: GenerationDiagnostics,
) -> GenerationTerminationSummary:
    """Reduce the per-request ledger without copying private row content."""
    finish_counts: dict[str, int] = {}
    stop_counts: dict[str, int] = {}
    for record in diagnostics.records:
        finish_key = record.finish_reason.strip().lower()
        finish_counts[finish_key] = finish_counts.get(finish_key, 0) + 1
        stop_key = "null" if record.stop_reason is None else str(record.stop_reason)
        stop_counts[stop_key] = stop_counts.get(stop_key, 0) + 1
    return GenerationTerminationSummary(
        total_requests=len(diagnostics.records),
        stopped_requests=finish_counts.get("stop", 0),
        length_limited_requests=finish_counts.get("length", 0),
        finish_reason_counts=finish_counts,
        stop_reason_counts=stop_counts,
        max_generated_tokens=max(record.generated_tokens for record in diagnostics.records),
    )


class InferenceResult(AgentResult):
    status: Literal["succeeded", "deferred", "failed"]
    operation: Literal["run_inference"] = "run_inference"
    inference_config_path: str = Field(
        "",
        description=(
            "Absolute path to complete inference_config.yaml. Baseline writes it; "
            "candidate inference reports the exact supplied file it reused."
        ),
    )
    predict_script_path: str = ""
    slurm_script_path: str = ""
    log_path: str = ""
    predictions_path: str = ""
    generation_diagnostics_path: str = Field(
        "",
        description=(
            "Absolute path to GenerationDiagnostics JSON containing one finish "
            "record per real generation request and no prediction text."
        ),
    )
    n_rows: int = Field(0, ge=0)
    n_requests: int = Field(0, ge=0)
    n_unparseable: int = Field(0, ge=0)

    @model_validator(mode="after")
    def require_success_artifacts(self) -> "InferenceResult":
        if self.status == "succeeded" and not (
            self.inference_config_path
            and self.predictions_path
            and self.generation_diagnostics_path
        ):
            raise ValueError(
                "successful inference requires config, predictions, and generation "
                "diagnostics paths"
            )
        if self.status == "succeeded" and self.n_requests < self.n_rows:
            raise ValueError("successful inference requires n_requests >= n_rows")
        return self


def load_generation_diagnostics(path: str) -> GenerationDiagnostics:
    """Load the runtime-owned JSON finish ledger."""
    import json
    from pathlib import Path

    source = Path(path)
    try:
        body = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read generation diagnostics {source}: {exc}") from exc
    return GenerationDiagnostics.model_validate(body)
