"""GET /wakeups, GET /wakeups/{id}."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.api.database import get_db
from zevo.db import AgentWakeupRequest


router = APIRouter()


class WakeupDTO(BaseModel):
    id: str
    agent_id: str
    ticket_id: str | None
    source: str
    status: str
    heartbeat_run_id: str | None
    trigger_detail: str
    reason: str
    scheduled_for: str
    created_at: str
    finished_at: str | None


def _dto(w: AgentWakeupRequest) -> WakeupDTO:
    return WakeupDTO(
        id=w.id, agent_id=w.agent_id, ticket_id=w.ticket_id,
        source=w.source, status=w.status,
        heartbeat_run_id=w.heartbeat_run_id,
        trigger_detail=w.trigger_detail or "", reason=w.reason or "",
        scheduled_for=w.scheduled_for.isoformat() if w.scheduled_for else "",
        created_at=w.created_at.isoformat() if w.created_at else "",
        finished_at=w.finished_at.isoformat() if w.finished_at else None,
    )


@router.get("/wakeups", response_model=list[WakeupDTO])
async def list_wakeups(
    agent_id: str = "",
    ticket_id: str = "",
    status: str = "",
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
) -> list[WakeupDTO]:
    q = select(AgentWakeupRequest).order_by(desc(AgentWakeupRequest.created_at)).limit(limit)
    if agent_id:
        q = q.where(AgentWakeupRequest.agent_id == agent_id)
    if ticket_id:
        q = q.where(AgentWakeupRequest.ticket_id == ticket_id)
    if status:
        q = q.where(AgentWakeupRequest.status == status)
    rows = (await db.execute(q)).scalars().all()
    return [_dto(r) for r in rows]


@router.get("/wakeups/{wakeup_id}", response_model=WakeupDTO)
async def get_wakeup(wakeup_id: str, db: AsyncSession = Depends(get_db)) -> WakeupDTO:
    r = (await db.execute(
        select(AgentWakeupRequest).where(AgentWakeupRequest.id == wakeup_id)
    )).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"wakeup {wakeup_id} not found")
    return _dto(r)
