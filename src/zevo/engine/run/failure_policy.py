"""Deterministic ownership policy for failed Ticket activations.

The model may explain a failure, but it does not decide whether the system
spends again.  This module assigns one of three routes:

``self``
    The Specialist owns the generated output, script, command, or artifact and
    gets up to three repair activations on the same Ticket.
``orchestrator``
    The work order/upstream lineage must change.  Re-running the Specialist
    unchanged would be wasteful, so its terminal failure is handed upstream.
``terminal``
    Cancellation, isolation/security violations, authentication, or another
    boundary that must never be retried automatically.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


RepairRoute = Literal["self", "orchestrator", "terminal"]
MAX_REPAIR_ATTEMPTS = 3

# A ticket's STORED typed payload is validated BEFORE the agent runs. A `queued`
# ticket whose payload fails that check cannot be repaired by re-running it
# unchanged -- the alarm-clock cron would re-enqueue the identical payload every
# tick forever (no escalation, no re-plan), idle-burning a rented GPU. Bound it:
# after this many consecutive validation-failing wakeups the ticket is failed
# terminally instead of looping.
MAX_PAYLOAD_VALIDATION_ATTEMPTS = 3

# Stable, greppable marker stamped into an AgentWakeupRequest's `reason` when the
# wakeup failed because the ticket's stored typed payload could not be validated.
# The runner counts these rows to bound retries; the reconciler counts them as
# its safety net. Both match on this exact string, so keep it stable.
PAYLOAD_VALIDATION_FAILURE_SIGNATURE = "stored-payload-validation-failed"


@dataclass(frozen=True)
class FailureDisposition:
    route: RepairRoute
    code: str
    reason: str


_TERMINAL: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("cancelled", re.compile(r"cancel(?:led|ed) by user|received sigterm", re.I),
     "the user cancelled this execution"),
    ("held_out_access", re.compile(r"held[-_ ]out test isolation violated", re.I),
     "held-out Test isolation was violated"),
    ("authentication", re.compile(
        r"authentication failed|unauthorized|invalid api key|login required|"
        r"not logged in|expired token|permission denied \(publickey\)", re.I),
     "credentials or operator authorization must change"),
    ("budget", re.compile(r"budget (?:is )?(?:exhausted|exceeded)|over budget", re.I),
     "the Run budget boundary was reached"),
)

_ORCHESTRATOR: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("upstream_binding", re.compile(
        r"inputs? unresolvable|unresolved input binding|upstream ticket|"
        r"source_ticket_id|work product.*(?:missing|not found)", re.I),
     "an upstream artifact binding or Ticket must change"),
    ("pinned_contract", re.compile(
        r"(?:pin|pinned|user[- ]fixed).*(?:unsupported|incompatible|violat)|"
        r"unsupported.*(?:pin|pinned)", re.I),
     "a user-owned pin conflicts with executable constraints"),
    ("data_method_incompatible", re.compile(
        r"(?:training method|record family|dataset|reward model).*(?:incompatible|unsupported)|"
        r"(?:incompatible|unsupported).*(?:training method|record family|dataset|reward model)",
        re.I,
    ), "the selected method/data direction must change"),
    ("capacity", re.compile(
        r"no (?:idle|usable|available) gpu|insufficient (?:gpu|vram|capacity)|"
        r"no allocation can satisfy", re.I),
     "the requested resource plan cannot currently be satisfied"),
)


def classify_failure(
    error_message: str,
    *,
    agent_id: str,
    cancelled: bool = False,
) -> FailureDisposition:
    """Return the one retry owner for a failed activation.

    Unknown implementation failures default to the producing Agent.  This is
    intentionally different from the older manual-rerun classifier: malformed
    Result JSON, generated YAML drift, missing generated artifacts, OOM, and
    version-specific command mistakes are exactly what reflection can repair.
    """
    message = (error_message or "").strip()
    if cancelled:
        return FailureDisposition("terminal", "cancelled", "the user cancelled this execution")
    for code, pattern, reason in _TERMINAL:
        if pattern.search(message):
            return FailureDisposition("terminal", code, reason)
    for code, pattern, reason in _ORCHESTRATOR:
        if pattern.search(message):
            return FailureDisposition("orchestrator", code, reason)
    if agent_id == "evaluation" and re.search(
        r"evaluation_script|scoring_set|sample_submission|answer_fields", message, re.I
    ):
        return FailureDisposition(
            "orchestrator", "evaluation_input", "the deterministic evaluator input must change",
        )
    return FailureDisposition(
        "self", "agent_execution", "the producing Agent owns this output or execution defect",
    )


def repair_instruction(
    *, agent_id: str, attempt: int, error_message: str,
) -> str:
    """Bounded system message injected into the same Ticket conversation."""
    return (
        "__REPAIR__\n\n"
        f"Repair attempt {attempt}/{MAX_REPAIR_ATTEMPTS} for {agent_id}. "
        "The previous activation did not produce an acceptable committed result. "
        "Inspect its existing files, transcript evidence, and the exact failure below. "
        "Reuse any already verified expensive work; fix only the invalid output, "
        "generated implementation, command, or artifact. Do not restart training or "
        "inference when the completed artifact can be verified and reported correctly. "
        "Run the supplied deterministic validators again, then emit exactly the typed "
        "Result schema. If a new expensive execution is genuinely required, explain "
        "why in notes and keep it within the original work order.\n\n"
        f"Exact failure: {error_message[:6000]}"
    )
