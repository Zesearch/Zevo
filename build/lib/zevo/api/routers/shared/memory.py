"""Run-scoped Agent memory inspection and supervisor disposition."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.api.database import get_db
from zevo.api.ui_access import is_trusted_ui_request
from zevo.contracts._base import StrictBody
from zevo.db import AgentMemoryEntry, Run


router = APIRouter()

MemoryStatus = Literal["active", "superseded", "accepted", "dismissed"]
MemoryLane = Literal["optimization", "held_out_test"]


class MemoryEntryDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    run_id: str
    agent_id: str
    lane: MemoryLane
    iteration: int
    kind: str
    key: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    visibility: Literal["agent_local", "shared_candidate"]
    applies_to: dict[str, str] = Field(default_factory=dict)
    status: MemoryStatus
    source_ticket_id: str
    supersedes_id: str = ""
    created_at: str


class MemoryDisposition(StrictBody):
    status: Literal["accepted", "dismissed"]


def _dto(row: AgentMemoryEntry) -> MemoryEntryDTO:
    return MemoryEntryDTO(
        id=row.id,
        run_id=row.run_id,
        agent_id=row.agent_id,
        lane=row.lane,
        iteration=int(row.iteration or 0),
        kind=row.kind,
        key=row.key,
        summary=row.summary,
        details=dict(row.details or {}),
        visibility=row.visibility,
        applies_to={str(k): str(v) for k, v in dict(row.applies_to or {}).items()},
        status=row.status,
        source_ticket_id=row.source_ticket_id,
        supersedes_id=row.supersedes_id or "",
        created_at=row.created_at.isoformat() if row.created_at else "",
    )


@router.get("/runs/{run_id}/memory", response_model=list[MemoryEntryDTO])
async def list_run_memory(
    run_id: str,
    request: Request,
    agent_id: str = "",
    lane: MemoryLane | None = None,
    status: MemoryStatus | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[MemoryEntryDTO]:
    """Inspect memory without ever searching or merging another Run.

    The trusted dashboard may audit the whole Run.  Agent-side callers see
    only active cross-Agent candidates from the optimization lane; private
    worker memory is delivered by the runner, not exposed as a shared feed.
    """
    if not await db.scalar(select(Run.id).where(Run.id == run_id)):
        raise HTTPException(status_code=404, detail="run not found")

    query = select(AgentMemoryEntry).where(AgentMemoryEntry.run_id == run_id)
    if agent_id:
        query = query.where(AgentMemoryEntry.agent_id == agent_id)
    trusted = is_trusted_ui_request(request)
    if trusted:
        if lane is not None:
            query = query.where(AgentMemoryEntry.lane == lane)
        if status is not None:
            query = query.where(AgentMemoryEntry.status == status)
    else:
        query = query.where(
            AgentMemoryEntry.lane == "optimization",
            AgentMemoryEntry.visibility == "shared_candidate",
            AgentMemoryEntry.status == "active",
        )
    rows = (await db.execute(
        query.order_by(desc(AgentMemoryEntry.created_at), desc(AgentMemoryEntry.id))
    )).scalars().all()
    return [_dto(row) for row in rows]


@router.patch(
    "/runs/{run_id}/memory/{entry_id}", response_model=MemoryEntryDTO,
)
async def dispose_shared_candidate(
    run_id: str,
    entry_id: str,
    body: MemoryDisposition,
    db: AsyncSession = Depends(get_db),
) -> MemoryEntryDTO:
    """Mark advisory shared memory accepted or dismissed by the supervisor."""
    row = (await db.execute(select(AgentMemoryEntry).where(
        AgentMemoryEntry.id == entry_id,
        AgentMemoryEntry.run_id == run_id,
    ))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="memory entry not found")
    if (
        row.lane != "optimization"
        or row.visibility != "shared_candidate"
        or row.status != "active"
    ):
        raise HTTPException(
            status_code=409,
            detail="only an active optimization shared_candidate can be disposed",
        )
    row.status = body.status
    await db.commit()
    await db.refresh(row)
    return _dto(row)
