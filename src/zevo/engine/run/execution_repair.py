"""Coordinator-reviewed repair of an existing work order, without budget resets."""
from __future__ import annotations

import json
from typing import Literal

from fastapi import HTTPException
from pydantic import Field
from sqlalchemy import select

from zevo.contracts._base import StrictBody
from zevo.contracts.failure import RecoveryText
from zevo.db import HeartbeatRun, Run, Ticket, TicketMessage, TicketNotice
from zevo.engine.run.failure_policy import MAX_REPAIR_ATTEMPTS
from zevo.engine.run.repair_progress import record_execution_repair


class RepairTicketRequest(StrictBody):
    action: Literal["collect", "resubmit"]
    reason: RecoveryText = Field(max_length=2000)
    evidence: list[RecoveryText] = Field(min_length=1, max_length=20)
    execution_attempt: int = Field(ge=1, description="Exact registered attempt being repaired")


async def repair_ticket(session, ticket: Ticket, request: RepairTicketRequest):
    from zevo.engine.cost.budget import snapshot_for_run
    from zevo.engine.run.runner import _latest_slurm_resource_request, _SLURM_TERMINAL_STATES
    from zevo.engine.run.steering import instruction_gate_active
    from zevo.engine.run.wakeup import queue_wakeup

    # Same admission lock as workload publication; recheck all mutable state.
    run = (await session.scalars(select(Run).where(Run.id == ticket.run_id)
                                .with_for_update().execution_options(populate_existing=True))).one()
    await session.refresh(ticket)
    if run.status != "running" or run.cancel_requested_at is not None:
        raise HTTPException(409, "Repair requires an active Run")
    if await instruction_gate_active(session, run.id):
        raise HTTPException(409, "Run instruction must be resolved before repair")
    if (ticket.lane != "optimization" or ticket.agent_id not in {"data", "train", "inference", "evaluation"}
            or ticket.status != "failed" or ticket.repair_route != "orchestrator"):
        raise HTTPException(409, "Only a failed optimization workload referred to Orchestrator can be repaired")
    supervisor = await session.get(Ticket, run.supervisor_ticket_id)
    supervisor_heartbeat = await session.scalar(select(HeartbeatRun.id).where(
        HeartbeatRun.ticket_id == run.supervisor_ticket_id,
        HeartbeatRun.agent_id == "orchestrator", HeartbeatRun.finished_at.is_(None),
    ).order_by(HeartbeatRun.started_at.desc()).limit(1))
    if not supervisor or supervisor.status != "running" or not supervisor_heartbeat:
        raise HTTPException(409, "Repair requires a live Run supervisor decision")
    if int(ticket.repair_attempts or 0) >= MAX_REPAIR_ATTEMPTS:
        raise HTTPException(409, "Ticket repair limit reached")
    if await session.scalar(select(HeartbeatRun.id).where(
        HeartbeatRun.ticket_id == ticket.id, HeartbeatRun.finished_at.is_(None),
    ).limit(1)):
        raise HTTPException(409, "Previous activation is still finishing")
    budget = await snapshot_for_run(session, run.id)
    if budget.over_budget or budget.over_time_limit:
        raise HTTPException(409, "Run budget prevents repair")
    row = await _latest_slurm_resource_request(session, ticket)
    meta = dict(row.meta or {}) if row else {}
    state = str(meta.get("scheduler_state") or "").split(" ")[0].upper().rstrip("+")
    if (row is None or state not in _SLURM_TERMINAL_STATES
            or int(meta.get("execution_attempt") or 1) != request.execution_attempt):
        raise HTTPException(409, "Repair must name the latest terminal execution attempt")

    # No payload, bindings, work products, checkpoint, iteration or resource
    # changes are accepted here. Preparation and publication validate them again.
    record_execution_repair(row, action=request.action, reason=request.reason,
                            heartbeat_id=supervisor_heartbeat)
    ticket.repair_attempts = int(ticket.repair_attempts or 0) + 1
    ticket.repair_route = "self"
    ticket.status = "repairing"
    ticket.summary = f"Repair {ticket.repair_attempts}/{MAX_REPAIR_ATTEMPTS}: {request.reason}"
    session.add(TicketNotice(ticket_id=ticket.id, code="repair.authorized", severity="info",
        body=json.dumps({**request.model_dump(), "heartbeat_id": supervisor_heartbeat,
                         "resource_id": row.id, "repair_attempt": ticket.repair_attempts})))
    session.add(TicketMessage(ticket_id=ticket.id, author="system", body=(
        f"__REPAIR__\n\nOrchestrator approved {request.action} on this same Ticket "
        f"(repair {ticket.repair_attempts}/{MAX_REPAIR_ATTEMPTS}). {request.reason}\n"
        f"Evidence: {json.dumps(request.evidence)}\n"
        "Follow the engine-supplied execution contract. Preserve frozen configuration, "
        "inputs, plan, allocation and verified resumable checkpoints. Resume committed "
        "training state; do not restart completed steps or create a new iteration."
    )))
    await queue_wakeup(session, agent_id=ticket.agent_id, ticket_id=ticket.id, source="retry",
        reason=f"Orchestrator approved repair: {request.reason}",
        payload={"repair_attempt": ticket.repair_attempts, "review_heartbeat_id": supervisor_heartbeat})
    return {"ticket_id": ticket.id, "status": ticket.status,
            "repair_attempts": ticket.repair_attempts, "action": request.action}
