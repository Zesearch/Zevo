"""Execution gate for unresolved Run-level user instructions.

Posting an instruction is a control-plane operation.  Specialists must not
start another activation while the Orchestrator is still deciding what the
instruction means, otherwise the old plan can submit a job in the gap between
the user's request and the decision.

``scheduled`` is deliberately not blocking: it is the Orchestrator's explicit
decision that current work may continue and the request belongs at a later
safe point.  ``needs_input`` remains blocking until the user supplies the
missing information and the Orchestrator updates the decision.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.db import RunInstruction, AgentWakeupRequest, Ticket


BLOCKING_INSTRUCTION_STATUSES = ("queued", "delivered", "needs_input")


async def requested_slurm_gpus(session: AsyncSession, ticket_id: str) -> int:
    """Keep the latest routed resource request across activations and repairs."""
    ticket = await session.get(Ticket, ticket_id)
    if ticket is None or ticket.lane != "optimization":
        return 0
    requests = (await session.execute(
        select(AgentWakeupRequest).join(Ticket, Ticket.id == AgentWakeupRequest.ticket_id).where(
            Ticket.run_id == ticket.run_id,
            AgentWakeupRequest.payload["requested_gpus"].as_integer().is_not(None),
        ).order_by(AgentWakeupRequest.created_at.desc())
    )).scalars().all()
    for request in requests:
        payload = request.payload or {}
        applies = request.ticket_id == ticket_id or ticket.agent_id in payload.get("gpu_stages", [])
        if applies and payload.get("run_instruction_id") and payload.get("instruction_action") == "restart_activation":
            return int(payload["requested_gpus"])
    return 0


def instruction_gate_exists(run_id):
    """Return a correlated EXISTS expression for a Run id/value column."""
    return select(RunInstruction.id).where(
        RunInstruction.run_id == run_id,
        RunInstruction.status.in_(BLOCKING_INSTRUCTION_STATUSES),
    ).exists()


async def instruction_gate_active(session: AsyncSession, run_id: str) -> bool:
    """Whether optimization work must wait for an instruction decision."""
    if not run_id:
        return False
    return bool((await session.execute(
        select(RunInstruction.id).where(
            RunInstruction.run_id == run_id,
            RunInstruction.status.in_(BLOCKING_INSTRUCTION_STATUSES),
        ).limit(1)
    )).scalar_one_or_none())
