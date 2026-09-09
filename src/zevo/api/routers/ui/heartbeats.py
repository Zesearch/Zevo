"""GET /heartbeats, GET /heartbeats/{id}, GET /heartbeats/{id}/stdout.

Lets the UI + CLI stream the per-heartbeat transcript (agent stdout).
Polling-based for v1; SSE/WebSocket can come later if needed.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.api.database import get_db
from zevo.api.ui_access import is_trusted_ui_request
from zevo.db import HeartbeatResult, HeartbeatRun, Run
from zevo.contracts.tickets import TERMINAL_RUN_STATUSES
from zevo.db.models import Ticket


router = APIRouter()


async def _by_id_prefix(db: AsyncSession, heartbeat_id: str) -> HeartbeatRun | None:
    """Look a heartbeat up by full id OR short id-prefix.

    The prefix is escaped so `%`/`_` in the path match literally instead of
    acting as LIKE wildcards, and an ambiguous prefix resolves to the NEWEST
    matching heartbeat (`.first()` after ordering) rather than raising
    MultipleResultsFound and 500ing.
    """
    escaped = (
        heartbeat_id.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    )
    return (await db.execute(
        select(HeartbeatRun)
        .where(HeartbeatRun.id.like(escaped + "%", escape="\\"))
        .order_by(desc(HeartbeatRun.started_at))
        .limit(1)
    )).scalars().first()


class HeartbeatDTO(BaseModel):
    id: str
    ticket_id: str
    agent_id: str
    driver: str
    model: str
    started_at: str
    finished_at: str = ""
    exit_code: int
    error_message: str = ""
    stdout_path: str = ""
    stdout_size_bytes: int = 0
    is_live: bool = False  # finished_at is null
    # Per-heartbeat usage + cost.
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    # The immutable operation executed by this heartbeat. It is stored on the
    # heartbeat so the audit record never depends on mutable display state.
    operation: str = ""
    activation_phase: str = ""
    # A compact, UI-safe projection of an Orchestrator result. The Timeline
    # needs the real action/child to represent sequential supervision.
    action: str = ""
    action_summary: str = ""
    child_ticket_id: str = ""


def _dto(h: HeartbeatRun, result_output: dict | None = None) -> HeartbeatDTO:
    size = 0
    is_live = h.finished_at is None
    if h.stdout_path:
        p = Path(h.stdout_path)
        if p.exists():
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
    output = result_output if isinstance(result_output, dict) else {}
    child_id = str(output.get("child_ticket_id") or "")
    return HeartbeatDTO(
        id=h.id, ticket_id=h.ticket_id, agent_id=h.agent_id,
        driver=h.driver, model=h.model,
        started_at=h.started_at.isoformat() if h.started_at else "",
        finished_at=h.finished_at.isoformat() if h.finished_at else "",
        exit_code=h.exit_code, error_message=h.error_message or "",
        stdout_path=h.stdout_path or "",
        stdout_size_bytes=size,
        is_live=is_live,
        input_tokens=int(h.input_tokens or 0),
        output_tokens=int(h.output_tokens or 0),
        cached_input_tokens=int(h.cached_input_tokens or 0),
        reasoning_output_tokens=int(h.reasoning_output_tokens or 0),
        estimated_cost_usd=float(h.estimated_cost_usd or 0.0),
        operation=str(h.operation or output.get("operation") or ""),
        activation_phase=str(h.activation_phase or ""),
        action=str(output.get("action") or "") if h.agent_id == "orchestrator" else "",
        action_summary=str(output.get("summary") or "") if h.agent_id == "orchestrator" else "",
        child_ticket_id=child_id if h.agent_id == "orchestrator" else "",
    )


async def _result_outputs(
    db: AsyncSession, heartbeat_ids: list[str],
) -> dict[str, dict]:
    if not heartbeat_ids:
        return {}
    rows = (await db.execute(
        select(HeartbeatResult.heartbeat_id, HeartbeatResult.output)
        .where(HeartbeatResult.heartbeat_id.in_(heartbeat_ids))
        .order_by(HeartbeatResult.created_at)
    )).all()
    return {
        heartbeat_id: output
        for heartbeat_id, output in rows
        if isinstance(output, dict)
    }


async def _heartbeat_is_visible(
    db: AsyncSession, heartbeat: HeartbeatRun, request: Request,
) -> bool:
    if is_trusted_ui_request(request):
        return True
    row = (await db.execute(
        select(Ticket.lane, Run.status)
        .join(Run, Run.id == Ticket.run_id)
        .where(Ticket.id == heartbeat.ticket_id)
    )).first()
    return bool(
        row is None
        or row.lane != "held_out_test"
        or row.status in TERMINAL_RUN_STATUSES
    )


@router.get("/heartbeats", response_model=list[HeartbeatDTO])
async def list_heartbeats(
    request: Request,
    agent_id: str = "",
    ticket_id: str = "",
    run_id: str = "",
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> list[HeartbeatDTO]:
    """Heartbeats, newest first.

    `run_id` filters to one run's heartbeats via its tickets. Without it a
    caller that wants a run's totals has to pull a global window and filter in
    the browser — which is right only while that window still reaches back far
    enough, and silently under-reports once it doesn't.
    """
    q = select(HeartbeatRun)
    if not is_trusted_ui_request(request):
        visible_ticket_ids = (
            select(Ticket.id)
            .join(Run, Run.id == Ticket.run_id)
            .where(
                (Ticket.lane != "held_out_test")
                | Run.status.in_(TERMINAL_RUN_STATUSES)
            )
        )
        q = q.where(HeartbeatRun.ticket_id.in_(visible_ticket_ids))
    q = q.order_by(desc(HeartbeatRun.started_at)).limit(limit)
    if agent_id:
        q = q.where(HeartbeatRun.agent_id == agent_id)
    if ticket_id:
        q = q.where(HeartbeatRun.ticket_id == ticket_id)
    if run_id:
        q = q.where(HeartbeatRun.ticket_id.in_(
            select(Ticket.id).where(Ticket.run_id == run_id)
        ))
    rows = (await db.execute(q)).scalars().all()
    outputs = await _result_outputs(db, [row.id for row in rows])
    return [_dto(row, outputs.get(row.id)) for row in rows]


@router.get("/heartbeats/{heartbeat_id}", response_model=HeartbeatDTO)
async def get_heartbeat(
    heartbeat_id: str, request: Request, db: AsyncSession = Depends(get_db),
) -> HeartbeatDTO:
    # Accept short id-prefix too
    r = await _by_id_prefix(db, heartbeat_id)
    if r is None:
        raise HTTPException(404, f"heartbeat {heartbeat_id} not found")
    if not await _heartbeat_is_visible(db, r, request):
        raise HTTPException(404, f"heartbeat {heartbeat_id} not found")
    outputs = await _result_outputs(db, [r.id])
    return _dto(r, outputs.get(r.id))


@router.post("/heartbeats/{heartbeat_id}/cancel")
async def cancel_heartbeat(
    heartbeat_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Kill the agent subprocess for a running heartbeat.

    Also marks the ticket as 'cancelled' so the UI/reconciler see a
    clean terminal state. If the heartbeat has already finished, this
    is a no-op.
    """
    r = await _by_id_prefix(db, heartbeat_id)
    if r is None:
        raise HTTPException(404, f"heartbeat {heartbeat_id} not found")
    if r.finished_at is not None:
        return {"status": "already_finished", "heartbeat_id": r.id}

    from zevo.engine.run import process_registry
    # Two keys: a CLI driver's own subprocess is under the heartbeat id, an
    # in-process driver's shell commands are under the ticket id.
    killed = process_registry.cancel(r.id) | process_registry.cancel(r.ticket_id)

    # Even if we can't kill (different process, already gone), mark
    # the ticket cancelled so the dashboard is honest.
    from zevo.db import Ticket
    tk = (await db.execute(
        select(Ticket).where(Ticket.id == r.ticket_id)
    )).scalar_one_or_none()
    if tk is not None and tk.status in ("queued", "running", "repairing", "awaiting_input", "waiting_external"):
        tk.status = "cancelled"
        tk.error_message = "cancelled by user"
        await db.commit()
    remote_job = None
    if tk is not None:
        from zevo.engine.run.remote_jobs import cancel_ticket_remote_job
        remote_job = await cancel_ticket_remote_job(db, tk)

    return {
        "status": "cancelled" if killed else "marked_cancelled",
        "heartbeat_id": r.id,
        "process_killed": killed,
        "remote_job": remote_job,
    }


@router.get("/heartbeats/{heartbeat_id}/stdout", response_class=PlainTextResponse)
async def heartbeat_stdout(
    heartbeat_id: str,
    request: Request,
    tail: int = Query(default=0, description="If >0, return only the last N lines."),
    db: AsyncSession = Depends(get_db),
) -> str:
    r = await _by_id_prefix(db, heartbeat_id)
    if r is None:
        raise HTTPException(404, f"heartbeat {heartbeat_id} not found")
    if not await _heartbeat_is_visible(db, r, request):
        raise HTTPException(404, f"heartbeat {heartbeat_id} not found")
    if not r.stdout_path:
        return ""
    p = Path(r.stdout_path)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    if tail > 0:
        lines = text.splitlines()
        text = "\n".join(lines[-tail:])
    return text
