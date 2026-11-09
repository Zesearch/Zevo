"""Deterministic ownership policy for failed Ticket activations.

The model may explain a failure, but it does not decide whether the system
spends again.  This module assigns one of three routes:

``self``
    The Specialist owns the generated output, script, command, or artifact and
    gets up to twenty repair activations on the same Ticket. All agents stop
    local repair earlier after five consecutive failures with unchanged
    evidence, including execution identity, input/output artifacts and measured
    workload progress. Specialists hand off for Orchestrator review; the
    Orchestrator itself cannot escalate to itself.
``orchestrator``
    The work order/upstream lineage must change.  Re-running the Specialist
    unchanged would be wasteful, so its terminal failure is handed upstream.
``terminal``
    An engine-confirmed cancellation, isolation or budget boundary, or
    exhausted recovery with no upstream owner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zevo.contracts.failure import FailureRecovery


RepairRoute = Literal["self", "orchestrator", "terminal"]
MAX_REPAIR_ATTEMPTS = 20
MAX_UNCHANGED_FAILURES = 5

# Legacy reconciliation threshold for old queued rows without versioned
# validation notices. New work orders stop on their FIRST validation failure;
# they require coordinator correction, not repeated execution of the same input.
MAX_PAYLOAD_VALIDATION_ATTEMPTS = 3

# Stable, greppable marker stamped into an AgentWakeupRequest's `reason` when the
# wakeup failed because the ticket's stored typed payload could not be validated.
# The reconciler counts historical rows as a safety net. Keep it stable.
PAYLOAD_VALIDATION_FAILURE_SIGNATURE = "stored-payload-validation-failed"


@dataclass(frozen=True)
class FailureDisposition:
    route: RepairRoute
    code: str
    reason: str


# Hard boundaries are supplied by engine checks, never inferred from error prose.
FailureBoundary = Literal["cancelled", "held_out_isolation", "budget"]
FailureOrigin = Literal["agent_execution", "result_validation", "engine_postprocessing", "external_workload", "runner"]

# Kept for producers of hardware diagnostics. The message itself grants no
# authority to release or replace an allocation.
HARDWARE_DEFECT_PREFIX = "GPU hardware defect:"


@dataclass(frozen=True)
class FailureContext:
    origin: FailureOrigin
    message: str
    boundary: FailureBoundary | None = None


def classify_failure(
    error_message: str,
    *,
    agent_id: str,
    cancelled: bool = False,
    recovery: FailureRecovery | None = None,
    boundary: FailureBoundary | None = None,
    can_diagnose: bool = True,
) -> FailureDisposition:
    """Route by the authority needed for the next action, not error keywords.

    Missing recovery proposals get one bounded diagnostic/repair activation on
    the same Ticket. They do not authorize rerunning expensive work or changing
    the work order. Engine-enforced cancellation and isolation still win.
    """
    if cancelled or boundary == "cancelled":
        return FailureDisposition("terminal", "cancelled", "the user cancelled this execution")
    if boundary is not None:
        return FailureDisposition("terminal", boundary, "an engine-enforced boundary prevents repair")
    if recovery is not None:
        if recovery.scope != "current_task":
            return FailureDisposition(
                "terminal" if agent_id == "orchestrator" else "orchestrator",
                "recovery_" + recovery.scope,
                "Review required for the proposed next action: " + recovery.reason,
            )
        return FailureDisposition("self", "recovery_current_task", recovery.next_action)
    if not can_diagnose:
        return FailureDisposition(
            "orchestrator", "diagnostic_review",
            "the deterministic executor has no diagnostic Agent; coordinator review is required",
        )
    return FailureDisposition(
        "self", "diagnosis_required", "diagnose the failure within the existing work order",
    )


def repair_instruction(
    *, agent_id: str, attempt: int, error_message: str, origin: FailureOrigin = "agent_execution",
) -> str:
    """Bounded system message injected into the same Ticket conversation."""
    return (
        "__REPAIR__\n\n"
        f"Repair attempt {attempt}/{MAX_REPAIR_ATTEMPTS} for {agent_id}. "
        "The previous activation did not produce an acceptable committed result. "
        f"Failure origin: {origin}. First diagnose the cause and the authority needed for the next action. "
        "Inspect its existing files, transcript evidence, and the exact failure below. "
        "Reuse any already verified expensive work; fix only the invalid output, "
        "generated implementation, command, dependency environment, or artifact. "
        "For engine post-processing failures, preserve the already prepared artifacts, "
        "repair only within this work order, and re-emit the original result for engine revalidation. "
        "Do not change engine code, scoring contracts, hidden evaluation assets, credentials, "
        "user pins or the resource allocation. If those must change, request review. "
        "Do not restart training or "
        "inference when the completed artifact can be verified and reported correctly. "
        "Run the required artifact validators and checks relevant to the repair, then emit the typed "
        "Result schema. If a new expensive execution is genuinely required, explain "
        "why in execution_retry_reason on a failed collect Result, with "
        "failure_recovery.scope=current_task and evidence. The engine can issue a new submit "
        "contract even when the previous process exited zero/COMPLETED. Keep the same Ticket, "
        "resources, frozen configuration and resumable state. Needing permission to submit "
        "again is not a resource-plan change. If recovery needs "
        "an upstream input, resource-plan, fixed-configuration or Run-decision change, "
        "return failed with failure_recovery describing that scope, evidence and "
        "the requested next action; the Orchestrator will review it. For current-task "
        "repair or investigation, name the concrete change or discriminating check. "
        "Diagnostic activations count toward the same twenty-repair limit. "
        "Updating a report alone is not progress.\n\n"
        f"Exact failure: {error_message[:6000]}"
    )
