"""Run-level unified timeline -- REST + WebSocket.

Stitches every "thing that happened in this run" from the existing
tables into a single feed, ordered by timestamp:

  type                source                          payload
  ------------------  ------------------------------  --------------------------------
  ticket_created      tickets.created_at              {input_format, lane, agent_id}
  ticket_status       tickets.updated_at (snapshot)   {status, summary}
  message             ticket_messages                 {author, body}
  notice              ticket_notices                  {code, severity, body}
  heartbeat_started   heartbeat_runs.started_at       {heartbeat_id, driver, model}
  heartbeat_finished  heartbeat_runs.finished_at      {heartbeat_id, exit_code, ...}
  transcript_event    transcript_events               {inner_type, payload}

REST:
  GET /api/runs/{id}/timeline?since_ts=ISO
      Returns all events with ts > since_ts (default: epoch).

WebSocket:
  WS /api/ws/runs/{id}/timeline
      On connect, sends the full backlog. Then polls the DB ~every 500ms
      and pushes any new events. Closes when the run reaches a terminal
      status (or 'finished' arrives via transcript_events).

The timeline doesn't introduce a new table -- it's a projection. That
keeps writes cheap and means existing flows (orchestrator emit, agent
runs, reconciler) keep working untouched.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.api.database import get_db
from zevo.api.ui_access import is_trusted_ui_request, is_trusted_ui_websocket
from zevo.contracts.tickets import TERMINAL_RUN_STATUSES
from zevo.db import (
    TicketMessage,
    TicketNotice,
    HeartbeatRun,
    Run,
    Ticket,
    TranscriptEvent,
    get_session_factory,
)


router = APIRouter()
log = logging.getLogger(__name__)


class TimelineEvent(BaseModel):
    ts: str
    type: str
    ticket_id: str = ""
    agent_id: str = ""
    payload: dict[str, Any]


def _iso(dt: datetime | None) -> str:
    return dt.isoformat() if dt else ""


async def _ticket_index(
    s: AsyncSession, run_id: str, *, reveal_holdout: bool,
) -> dict[str, Ticket]:
    stmt = select(Ticket).where(Ticket.run_id == run_id)
    if not reveal_holdout:
        stmt = stmt.where(Ticket.lane != "held_out_test")
    rows = (await s.execute(stmt)).scalars().all()
    return {t.id: t for t in rows}


async def _build_timeline(
    s: AsyncSession, run_id: str, since_ts: datetime | None,
    *, trusted_ui: bool = False,
) -> list[TimelineEvent]:
    """Compose the full timeline for a run, ordered by ts ASC."""
    # Verify the run exists; 404 if not.
    r = (await s.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"run {run_id} not found")

    tickets = await _ticket_index(
        s, run_id,
        reveal_holdout=trusted_ui or r.status in TERMINAL_RUN_STATUSES,
    )
    ticket_ids = list(tickets.keys())

    events: list[TimelineEvent] = []

    # --- ticket_created (always) + ticket_status (current snapshot) ---
    for t in tickets.values():
        events.append(TimelineEvent(
            ts=_iso(t.created_at),
            type="ticket_created",
            ticket_id=t.id,
            agent_id=t.agent_id or "",
            payload={
                "input_format": t.input_format,
                "lane": t.lane,
                "iteration": t.iteration,
                "summary": (t.summary or "")[:300],
            },
        ))
        # Use updated_at as the "current status" snapshot. This is
        # lossy -- we don't get the full history of transitions -- but
        # it's good enough until we add a ticket_transitions table.
        if t.updated_at and t.updated_at != t.created_at:
            events.append(TimelineEvent(
                ts=_iso(t.updated_at),
                type="ticket_status",
                ticket_id=t.id,
                agent_id=t.agent_id or "",
                payload={
                    "status": t.status,
                    "summary": (t.summary or "")[:300],
                },
            ))

    # --- conversation messages + system notices ---
    if ticket_ids:
        messages = (await s.execute(
            select(TicketMessage).where(TicketMessage.ticket_id.in_(ticket_ids))
        )).scalars().all()
        from zevo.api.artifacts import host_paths_in_text
        for c in messages:
            tk = tickets.get(c.ticket_id)
            events.append(TimelineEvent(
                ts=_iso(c.created_at),
                type="message",
                ticket_id=c.ticket_id,
                agent_id=(tk.agent_id if tk else "") or c.author or "",
                # host-path the body: agents write container paths (/app/data/runs/…)
                payload={"author": c.author, "body": host_paths_in_text((c.body or "")[:2000])},
            ))
        notices = (await s.execute(
            select(TicketNotice).where(TicketNotice.ticket_id.in_(ticket_ids))
        )).scalars().all()
        for notice in notices:
            tk = tickets.get(notice.ticket_id)
            events.append(TimelineEvent(
                ts=_iso(notice.created_at), type="notice",
                ticket_id=notice.ticket_id,
                agent_id=(tk.agent_id if tk else ""),
                payload={"code": notice.code, "severity": notice.severity,
                         "body": host_paths_in_text((notice.body or "")[:2000])},
            ))

    # --- heartbeats: start + (optional) finish ---
    if ticket_ids:
        hbs = (await s.execute(
            select(HeartbeatRun).where(HeartbeatRun.ticket_id.in_(ticket_ids))
        )).scalars().all()
        hb_index: dict[str, HeartbeatRun] = {h.id: h for h in hbs}
        for h in hbs:
            tk = tickets.get(h.ticket_id)
            events.append(TimelineEvent(
                ts=_iso(h.started_at),
                type="heartbeat_started",
                ticket_id=h.ticket_id,
                agent_id=h.agent_id or (tk.agent_id if tk else ""),
                payload={
                    "heartbeat_id": h.id,
                    "driver": h.driver,
                    "model": h.model,
                },
            ))
            if h.finished_at is not None:
                events.append(TimelineEvent(
                    ts=_iso(h.finished_at),
                    type="heartbeat_finished",
                    ticket_id=h.ticket_id,
                    agent_id=h.agent_id or (tk.agent_id if tk else ""),
                    payload={
                        "heartbeat_id": h.id,
                        "exit_code": h.exit_code,
                        "error_message": (h.error_message or "")[:500],
                    },
                ))

        # --- transcript_events (one per row; the UI reuses LiveTranscript
        # per-event renderers, so we ship the inner type as `type` and the
        # payload verbatim) ---
        hb_ids = [h.id for h in hbs]
        if hb_ids:
            tev = (await s.execute(
                select(TranscriptEvent).where(TranscriptEvent.heartbeat_id.in_(hb_ids))
            )).scalars().all()
            for e in tev:
                h = hb_index.get(e.heartbeat_id)
                tk = tickets.get(h.ticket_id) if h else None
                events.append(TimelineEvent(
                    ts=_iso(e.ts),
                    type=e.type,  # e.g. 'agent_message' | 'tool_call' | ...
                    ticket_id=(h.ticket_id if h else ""),
                    agent_id=(tk.agent_id if tk else (h.agent_id if h else "")),
                    payload=e.payload or {},
                ))

    # Sort by ts ASC; events with empty ts (rare) sort first.
    def _key(e: TimelineEvent) -> str:
        return e.ts or ""
    events.sort(key=_key)

    if since_ts is not None:
        events = [e for e in events if e.ts > _iso(since_ts)]
    return events


@router.get("/runs/{run_id}/timeline", response_model=list[TimelineEvent])
async def get_run_timeline(
    run_id: str,
    request: Request,
    since_ts: str = "",
    db: AsyncSession = Depends(get_db),
) -> list[TimelineEvent]:
    since: datetime | None = None
    if since_ts:
        try:
            since = datetime.fromisoformat(since_ts)
        except ValueError:
            raise HTTPException(400, f"bad since_ts: {since_ts!r}")
    return await _build_timeline(
        db, run_id, since, trusted_ui=is_trusted_ui_request(request),
    )


@router.websocket("/ws/runs/{run_id}/timeline")
async def run_timeline_ws(websocket: WebSocket, run_id: str) -> None:
    """Backlog + live-tail run timeline.

    Closes when the parent Run reaches a terminal status (success /
    failed / halted) AND the latest snapshot has been delivered.
    """
    await websocket.accept()
    Session = get_session_factory()
    trusted_ui = is_trusted_ui_websocket(websocket)
    last_sent_ts: str = ""
    try:
        # Initial backlog
        async with Session() as s:
            try:
                events = await _build_timeline(
                    s, run_id, None, trusted_ui=trusted_ui,
                )
            except HTTPException as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "payload": {"message": e.detail},
                }))
                await websocket.close(code=1011)
                return
        for ev in events:
            await websocket.send_text(ev.model_dump_json())
            if ev.ts > last_sent_ts:
                last_sent_ts = ev.ts

        # Tail
        while True:
            await asyncio.sleep(0.5)
            async with Session() as s:
                since = None
                if last_sent_ts:
                    try:
                        since = datetime.fromisoformat(last_sent_ts)
                    except ValueError:
                        since = None
                try:
                    new = await _build_timeline(
                        s, run_id, since, trusted_ui=trusted_ui,
                    )
                except HTTPException:
                    break
                r = (await s.execute(
                    select(Run).where(Run.id == run_id)
                )).scalar_one_or_none()
            for ev in new:
                await websocket.send_text(ev.model_dump_json())
                if ev.ts > last_sent_ts:
                    last_sent_ts = ev.ts
            # Close once the run is terminal AND no more events came in.
            if r is not None and r.status in TERMINAL_RUN_STATUSES and not new:
                # send one synthesized run_finished frame for symmetry
                await websocket.send_text(json.dumps({
                    "ts": _iso(r.finished_at) or last_sent_ts,
                    "type": "run_finished",
                    "ticket_id": "",
                    "agent_id": "",
                    "payload": {
                        "status": r.status,
                        "best_validation_score": r.best_validation_score,
                        "halted_reason": r.halted_reason or "",
                    },
                }))
                break
    except WebSocketDisconnect:
        return
    except Exception as e:
        log.exception("run_timeline_ws crashed: %s", e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
