"""Activation replacement leaves the owning task alive."""
from sqlalchemy.ext.asyncio import AsyncSession
from zevo.db import HeartbeatRun, Ticket
from zevo.contracts.tickets import TERMINAL_TICKET_STATUSES


async def finish_superseded_activation(
    session: AsyncSession, ticket: Ticket, activation: HeartbeatRun,
) -> bool:
    # Read only the control field: preserve the runner's pending usage and exit data.
    await session.refresh(activation, attribute_names=["superseded_by_instruction_id"])
    if not activation.superseded_by_instruction_id:
        return False
    await session.refresh(ticket)
    if ticket.status in TERMINAL_TICKET_STATUSES:
        return False  # An explicit cancellation takes precedence.
    ticket.status = "queued"
    ticket.summary = "Waiting to apply the updated instruction"
    ticket.error_message = ""
    return True
