"""Create a Run's one reactive supervisor (Orchestrator) Ticket.

Two callers build the same initial work order:

* ``POST /runs`` for full/customized pipelines, immediately after Run creation
  settles the Validation/Test split;
* ``zevo.engine.run.scoping`` for an ``auto`` Run, after the Data agent's
  ``scope_problem`` Ticket has derived the scoring contract and the engine has
  settled it onto the Run.

From that point both kinds of Run proceed identically, which is why the payload
is assembled in exactly one place.
"""
from __future__ import annotations

from typing import Any
import hashlib
import json

from zevo.contracts.orchestrator import UserRequest
from zevo.contracts.tickets import validate_stored_payload
from zevo.db import Run, Ticket


def supervisor_ticket_id(run: Run) -> str:
    """`orchestrate-<run8>-001`: the stable id re-woken for the whole Run."""
    return f"orchestrate-{run.id[:8]}-001"


def supervisor_runtime(run: Run) -> dict[str, Any]:
    """Current engine-owned configuration, not a ticket's cached copy."""
    return {
        "gpu_provider": run.gpu_provider,
        "num_gpus": run.num_gpus,
        "generation_backend": run.generation_backend,
        "gpu_allocation_mode": run.gpu_allocation_mode or "per_stage",
        "adaptive_mid_training_evaluation": bool((run.lifecycle or {}).get("adaptive_mid_training_evaluation")),
        "max_queue_wait_hours": float(run.max_queue_wait_hours or 0.0),
    }


def supervisor_activation_payload(run, payload, activation_payload=None, *, source="", reason=""):
    """Rebuild only engine-owned fields; preserve decisions and frozen inputs."""
    from zevo.engine.run.adaptive_training import activation_validation_context
    activation = activation_payload or {}
    context = activation_validation_context(run.lifecycle or {}, reason, activation)
    if context:
        trigger = {"type": "checkpoint_ready", "ticket_id": context["train_ticket_id"],
                   "agent_id": "train", "status": ""}
    elif activation.get("orchestration_trigger"):
        trigger = activation["orchestration_trigger"]
    elif source == "assignment":
        trigger = {"type": "run_created", "status": "running"}
    else:
        trigger = {"type": "on_demand", "status": ""}
    return {**payload, "runtime": supervisor_runtime(run), "trigger": trigger}


def supervisor_input_signature(run, ticket, activation_payload):
    """Identify corrected inputs without hashing changing execution telemetry."""
    payload = supervisor_activation_payload(run, ticket.payload or {}, activation_payload)
    envelope = {"payload": payload, "inputs": ticket.inputs, "customization": ticket.customization,
                "input_format": ticket.input_format}
    return hashlib.sha256(json.dumps(envelope, sort_keys=True, default=str).encode()).hexdigest()


def initial_supervisor_payload(
    run: Run, *, agent_request: UserRequest, holdout: dict[str, Any],
    dataset_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    """The validated `run_created` OrchestratePayload for a freshly settled Run."""
    payload: dict[str, object] = {
        "task_objective": run.task_objective,
        "agent_objective": run.agent_objective,
        "user_request": agent_request.model_dump(),
        "trigger": {"type": "run_created", "ticket_id": "", "status": "running"},
        "history": [],  # empty until the baseline lands — nothing measured yet
        "mode": run.mode,
        "customizations": run.customizations or {},
        "runtime": supervisor_runtime(run),
        "validation_rows": int(holdout.get("validation_rows", 0) or 0),
        "dataset_profile": dataset_profile or {},
        "budget": {
            "max_cost_usd": float(run.max_cost_usd or 0.0),
            "spent_usd": 0.0,
            "remaining_usd": float(run.max_cost_usd or 0.0),
            "max_runtime_hours": float(run.max_runtime_hours or 0.0),
            "max_queue_wait_hours": float(run.max_queue_wait_hours or 0.0),
            "queue_wait_hours": 0.0,
            "elapsed_runtime_hours": 0.0,
            "remaining_runtime_hours": float(run.max_runtime_hours or 0.0),
            "over_time_limit": False,
        },
    }
    # G.1 — give the orchestrator visibility into the budget cap from
    # turn 0. Each subsequent supervisor wakeup recomputes via the
    # /runs/{id}/budget endpoint (or by reading run.max_cost_usd +
    # totaling its own spend).
    return validate_stored_payload(
        agent_id="orchestrator", input_format="typed", payload=payload,
    )


def new_supervisor_ticket(run: Run, payload: dict[str, Any]) -> Ticket:
    """The queued supervisor Ticket row; the caller adds and commits it."""
    return Ticket(
        id=supervisor_ticket_id(run), run_id=run.id, agent_id="orchestrator",
        status="queued", input_format="typed", lane="optimization", iteration=0,
        payload=payload, customization={}, inputs={}, summary="",
    )
