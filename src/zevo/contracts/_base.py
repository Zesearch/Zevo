"""Shared validation policy for agent result contracts."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zevo.contracts.customizations import AgentCustomization
from zevo.contracts.failure import FailureRecovery
from zevo.contracts.memory import AgentMemoryContext, MemoryUpdate


class AgentTaskInput(BaseModel):
    """Fields shared by every resolved typed Agent invocation."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(
        min_length=1,
        description=(
            "Exact full Ticket id assigned by the engine. Copy it byte-for-byte "
            "into the result ticket_id; never shorten or derive it."
        ),
    )
    activation_contract: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine-owned activation mode, completion boundary, and repair allowances.",
    )
    gpu_runtime_helper_path: str = Field(
        "", description=(
            "System-owned, stage-independent GPU runtime/cache wrapper. Available to every "
            "specialist; use for GPU workloads only, after environment activation on each "
            "compute node. CPU-only work does not require it."
        ),
    )
    stage_resume_helper_path: str = Field(
        "", description="System phase reporter and durable batch journal. Use for bounded Data/Inference batches; upload beside other runtime helpers."
    )
    customization: AgentCustomization = Field(default_factory=AgentCustomization)
    run_context: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Engine-owned, held-out-safe Run evidence such as prior Validation "
            "history, used methods, runtime, and budget. Advisory only."
        ),
    )
    memory: AgentMemoryContext = Field(
        default_factory=AgentMemoryContext,
        description=(
            "Relevant lessons from this Agent's earlier Tickets in the same "
            "Run and lane. Advisory only; current contracts and Ticket fields win."
        ),
    )


class StrictBody(BaseModel):
    """Mutation request bodies reject misspelled or obsolete keys."""

    model_config = ConfigDict(extra="forbid")


class StrictResult(BaseModel):
    """Agent stdout must match its declared result schema exactly."""

    model_config = ConfigDict(extra="forbid", strict=True)


class AgentResult(StrictResult):
    """Fields present in every Agent's structured heartbeat result."""

    status: Literal["succeeded", "degraded", "deferred", "failed"]
    ticket_id: str = Field(
        min_length=1,
        description=(
            "Exact full ticket_id from the work order. The engine hard-checks "
            "this value; copy it byte-for-byte."
        ),
    )
    error_message: str = Field(
        ..., description="Empty on succeeded/degraded/deferred; specific failure detail otherwise."
    )
    notes: str = Field(..., description="Concise execution notes; empty is allowed.")
    execution_retry_reason: str = Field("", max_length=2000, description=(
        "For a failed Result: explain why recovery is necessary on the same Ticket. "
        "resubmit is for collection failures needing another execution; replace_resource "
        "is valid in preparation/submission or collection, including checkpoint continuation. "
        "Pair with "
        "failure_recovery.scope=current_task and concrete evidence. The engine owns submission "
        "and the repair limit. Leave empty for artifact/collection fixes. This grants no "
        "permission to change frozen settings, inputs, resource requirements or the training plan. "
        "Replacing unusable compute within those requirements is an engine-owned action."
    ))
    failure_recovery: FailureRecovery | None = Field(
        None, description=(
            "For failed results, explain the evidence, attempted actions and authority "
            "needed for recovery. Use this field to request Orchestrator review instead "
            "of relying on phrases in notes/error_message. Null preserves legacy results."
        ),
    )
    execution_retry_action: Literal["resubmit", "replace_resource"] = Field(
        "resubmit", description="With execution_retry_reason: reuse healthy compute after collection, or request replacement of assigned unusable compute in any execution phase within the unchanged resource plan. Never release resources yourself.",
    )
    memory_updates: list[MemoryUpdate] = Field(
        default_factory=list,
        description=(
            "Small durable lessons for later Tickets of this Agent in this Run. "
            "When execution hits a problem, reflection identifies the cause, "
            "record verified findings that may help later work, even if the task fails. "
            "Failed work requires verification, evidence, environment, and agent-local visibility. "
            "Do not include routine status, "
            "unverified guesses, full transcripts, secrets, or held-out information."
        ),
    )

    @model_validator(mode="after")
    def require_status_error_consistency(self) -> "AgentResult":
        if self.execution_retry_action == "replace_resource" and not self.execution_retry_reason.strip():
            raise ValueError("replace_resource requires execution_retry_reason and evidence")
        if self.execution_retry_action == "replace_resource" and (
            self.failure_recovery is None or not self.failure_recovery.evidence
        ):
            raise ValueError("replace_resource requires failure_recovery evidence")
        if self.execution_retry_reason.strip():
            if self.status != "failed":
                raise ValueError("execution_retry_reason is only valid for failed results")
            if self.failure_recovery is not None and self.failure_recovery.scope != "current_task":
                raise ValueError("execution retry must stay within current_task; other scopes require review")
        if self.failure_recovery is not None and self.status != "failed":
            raise ValueError("failure_recovery is only valid for failed results")
        if self.status == "failed" and not self.error_message.strip():
            raise ValueError("failed results require a non-empty error_message")
        if self.status in {"succeeded", "degraded", "deferred"} and self.error_message.strip():
            raise ValueError(
                "succeeded/degraded/deferred results require an empty error_message"
            )
        return self
