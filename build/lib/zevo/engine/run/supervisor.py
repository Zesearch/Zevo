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

from zevo.contracts.orchestrator import UserRequest
from zevo.contracts.tickets import validate_stored_payload
from zevo.db import Run, Ticket


def supervisor_ticket_id(run: Run) -> str:
    """`orchestrate-<run8>-001`: the stable id re-woken for the whole Run."""
    return f"orchestrate-{run.id[:8]}-001"


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
        "runtime": {
            "gpu_provider": run.gpu_provider,
            "num_gpus": run.num_gpus,
            "generation_backend": run.generation_backend,
            "max_queue_wait_hours": float(run.max_queue_wait_hours or 0.0),
        },
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
