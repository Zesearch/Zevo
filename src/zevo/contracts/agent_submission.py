"""Agent-facing submissions, separate from the engine's persisted Results.

Project only fields owned by the current operation. Convert once at the driver
boundary; the existing Result validators and engine finalization remain authoritative.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import ClassVar, Literal

from pydantic import Field, create_model, model_validator

from zevo.contracts._base import AgentResult, StrictResult
from zevo.contracts.failure import RecoveryText, TrainingConfigurationChange


class RecoveryEvidence(StrictResult):
    reason: RecoveryText
    attempted_actions: list[RecoveryText] = Field(default_factory=list)
    evidence: list[RecoveryText] = Field(min_length=1)
    next_action: RecoveryText


class CurrentTaskRecovery(RecoveryEvidence):
    scope: Literal["current_task"]
    execution_retry_reason: str = Field("", max_length=2000,
        description="Empty for artifact-only repair; otherwise explain why another execution is needed.")
    execution_retry_action: Literal["resubmit", "replace_resource"] = "resubmit"


class ReviewRecovery(RecoveryEvidence):
    scope: Literal["upstream_input", "resource_plan", "run_decision"]


class ConfigurationRecovery(RecoveryEvidence):
    scope: Literal["fixed_configuration"]


class TrainConfigurationRecovery(ConfigurationRecovery):
    training_configuration_change: TrainingConfigurationChange | None = None


class ResourceIdentity(StrictResult):
    provider: Literal["", "cluster", "cloud", "instance"] = ""
    instance_id: str = ""
    cloud_backend: Literal["", "vastai", "lambda", "aws"] = ""


class FailedResource(ResourceIdentity):
    auto_release: bool = False


DATA_ENGINE_FIELDS = frozenset({
    "remote_data_receipt_path", "remote_training_package_path",
    "remote_materialization_mode", "remote_control_retries",
    "system_scoring_duplicates_removed", "decontamination_checked",
    "decontamination_removed_rows", "validation_source_path",
    "validation_answer_fields", "validation_dataset_path",
})
DATA_OPERATION_FIELDS = {
    "scope_problem": {"scoping_result_path"},
    "prepare_run_data": {
        "training_dataset_path", "remote_dataset_path", "remote_dataset_spec_path",
        "remote_data_profile_path", "prepare_script_path", "data_recipe_path",
        "slurm_script_path", "n_rows_in", "n_rows_out", "synthesis_teacher_model",
        "synthesis_output_format", "synthesis_generated_rows",
    },
    "prepare_holdout_data": {
        "scoring_public_path", "inference_data_profile_path", "sample_submission_path",
        "suite_members", "prepare_script_path", "n_rows_in", "n_rows_out",
    },
}
_RECOVERY_FIELDS = {"execution_retry_reason", "execution_retry_action", "failure_recovery"}
_RESOURCE_FIELDS = {"provider", "instance_id", "cloud_backend", "auto_release"}


class AgentSubmission(StrictResult):
    """Wire format only; never persisted in place of the canonical Result."""

    canonical_result: ClassVar[type[AgentResult]]

    def to_result(self) -> AgentResult:
        body = self.model_dump()
        recovery = body.get("failure_recovery")
        if recovery and recovery["scope"] == "current_task":
            body["execution_retry_reason"] = recovery.pop("execution_retry_reason", "")
            body["execution_retry_action"] = recovery.pop("execution_retry_action", "resubmit")
        for key in ("failure_resource", "released_resource"):
            resource = body.pop(key, None)
            if resource is not None:
                if key == "failure_resource" and body["status"] != "failed":
                    raise ValueError("failure_resource is only valid on failed provision")
                body.update(resource)
        return self.canonical_result.model_validate(body)

    @model_validator(mode="after")
    def validate_result_contract(self):
        self.to_result()
        return self


@lru_cache(maxsize=None)
def submission_schema(result: type[AgentResult], operation: str = "") -> type[AgentSubmission]:
    from zevo.contracts.data import DataResult
    from zevo.contracts.infrastructure import InfraResult
    from zevo.contracts.train import TrainResult

    allowed = set(result.model_fields) - _RECOVERY_FIELDS
    if result is DataResult:
        if operation not in DATA_OPERATION_FIELDS:
            raise ValueError(f"unsupported Data submission operation: {operation!r}")
        allowed &= set(AgentResult.model_fields) | {"operation"} | DATA_OPERATION_FIELDS[operation]
    elif result is InfraResult:
        if operation not in {"provision", "release"}:
            raise ValueError(f"unsupported Infra submission operation: {operation!r}")
        allowed -= _RESOURCE_FIELDS
        if operation == "release":
            allowed.discard("device_info_path")

    fields = {
        name: (field.annotation, deepcopy(field))
        for name, field in result.model_fields.items() if name in allowed
    }
    if "operation" in fields and operation:
        fields["operation"] = (Literal[operation], Field(default=operation))
    configuration = TrainConfigurationRecovery if result is TrainResult else ConfigurationRecovery
    fields["failure_recovery"] = (
        CurrentTaskRecovery | ReviewRecovery | configuration | None,
        Field(default=None, description=(
            "Failed results only. current_task may request another execution using nested "
            "execution_retry_reason/action. Other scopes request review and have no retry fields."
        )),
    )
    if result is InfraResult:
        key, resource = (("failure_resource", FailedResource) if operation == "provision"
                         else ("released_resource", ResourceIdentity))
        fields[key] = (resource | None, Field(default=None, description=(
            "On failed provision only, preserve resource identity and cleanup evidence. "
            "Leave null on success; device_info.json owns usable provision facts."
            if operation == "provision" else "Resource identity for release bookkeeping."
        )))
    name = result.__name__ + "Submission" + "".join(word.title() for word in operation.split("_"))
    model = create_model(name, __base__=AgentSubmission, **fields)
    model.canonical_result = result
    return model
