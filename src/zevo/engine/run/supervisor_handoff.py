"""Admission barriers for work authored by an unfinished supervisor decision."""
from sqlalchemy import select

from zevo.db import TicketNotice

WAIT_CODE = "ticket.awaiting_supervisor_completion"


async def waiting(session, ticket) -> bool:
    if ticket.agent_id == "orchestrator":
        return False
    return await session.scalar(select(TicketNotice.id).where(
        TicketNotice.ticket_id == ticket.id, TicketNotice.code == WAIT_CODE,
    ).limit(1)) is not None
