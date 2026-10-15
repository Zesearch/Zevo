"""Live + historical transcript-event endpoints.

REST:
  GET /api/heartbeats/{id}/events?since_seq=N
      Returns every TranscriptEvent for this heartbeat with seq > N,
      ordered by seq ASC. Used by clients to fetch the initial backlog
      and to resync after a WebSocket drop.

WebSocket:
  WS /api/ws/heartbeats/{id}
      On connect, sends the full backlog (every event so far) as a
      sequence of JSON frames. Then subscribes to the in-process
      transcript_bus and forwards every new event live until the
      heartbeat finishes (a 'finished' event arrives or the WS drops).
      Each frame is a single JSON object: {"seq", "ts", "type", "payload"}.

Why a process-local bus: today the wakeup-daemon runs the runner
in-process inside the scheduler container, while the backend runs in
its own container. They DO NOT share memory. WebSocket subscribers are
in the backend; events are emitted in the scheduler. So for live push
we tail the `transcript_events` Postgres table by seq + short sleep --
~250ms latency, no extra infra. (When/if we collapse the two into one
process, this falls back to the in-memory bus for ~0 latency.)
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.api.database import get_db
from zevo.api.ui_access import is_trusted_ui_request, is_trusted_ui_websocket
from zevo.engine.observe import transcript_bus
from zevo.db import HeartbeatRun, Run, Ticket, TranscriptEvent, get_session_factory
from zevo.contracts.tickets import TERMINAL_RUN_STATUSES


router = APIRouter()
log = logging.getLogger(__name__)


class EventDTO(BaseModel):
    seq: int
    ts: str
    type: str
    payload: dict[str, Any]


def _to_dto(e: TranscriptEvent) -> EventDTO:
    return EventDTO(
        seq=e.seq,
        ts=e.ts.isoformat() if e.ts else "",
        type=e.type,
        payload=e.payload or {},
    )


async def _resolve_heartbeat_id(db: AsyncSession, prefix_or_id: str) -> str:
    """Accept either a full uuid or a short prefix (first 8 chars are unique enough).

    The prefix is escaped so `%`/`_` in the path match literally instead of
    acting as LIKE wildcards (which previously turned e.g. `/heartbeats/_/events`
    into a broad match), and an ambiguous prefix resolves to the NEWEST match
    (`.first()` after ordering) rather than raising MultipleResultsFound → 500.
    Mirrors heartbeats._by_id_prefix.
    """
    escaped = (
        prefix_or_id.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    )
    r = (await db.execute(
        select(HeartbeatRun)
        .where(HeartbeatRun.id.like(escaped + "%", escape="\\"))
        .order_by(desc(HeartbeatRun.started_at))
        .limit(1)
    )).scalars().first()
    if r is None:
        raise HTTPException(404, f"heartbeat {prefix_or_id} not found")
    return r.id


async def _heartbeat_visible(
    db: AsyncSession, heartbeat_id: str, *, trusted_ui: bool,
) -> bool:
    if trusted_ui:
        return True
    row = (await db.execute(
        select(Ticket.lane, Run.status)
        .join(HeartbeatRun, HeartbeatRun.ticket_id == Ticket.id)
        .join(Run, Run.id == Ticket.run_id)
        .where(HeartbeatRun.id == heartbeat_id)
    )).first()
    return bool(
        row is None
        or row.lane != "held_out_test"
        or row.status in TERMINAL_RUN_STATUSES
    )


@router.get("/heartbeats/{heartbeat_id}/events", response_model=list[EventDTO])
async def list_events(
    heartbeat_id: str,
    request: Request,
    since_seq: int = 0,
    db: AsyncSession = Depends(get_db),
) -> list[EventDTO]:
    hb_id = await _resolve_heartbeat_id(db, heartbeat_id)
    if not await _heartbeat_visible(
        db, hb_id, trusted_ui=is_trusted_ui_request(request),
    ):
        raise HTTPException(404, f"heartbeat {heartbeat_id} not found")
    rows = (await db.execute(
        select(TranscriptEvent)
        .where(TranscriptEvent.heartbeat_id == hb_id, TranscriptEvent.seq > since_seq)
        .order_by(TranscriptEvent.seq)
    )).scalars().all()
    return [_to_dto(r) for r in rows]


@router.websocket("/ws/heartbeats/{heartbeat_id}")
async def heartbeat_ws(websocket: WebSocket, heartbeat_id: str) -> None:
    """Push every transcript event for one heartbeat as it lands.

    Protocol:
      1. accept connection
      2. resolve heartbeat
      3. send the entire backlog (in order)
      4. tail: every ~250ms poll for new rows with seq > last_sent;
         also drain the in-process bus if we share a process with the
         runner. Close cleanly on 'finished' event or peer disconnect.
    """
    await websocket.accept()
    Session = get_session_factory()
    last_sent_seq = 0
    try:
        # Resolve heartbeat id (404 closes the WS politely).
        async with Session() as s:
            try:
                hb_id = await _resolve_heartbeat_id(s, heartbeat_id)
                visible = await _heartbeat_visible(
                    s, hb_id, trusted_ui=is_trusted_ui_websocket(websocket),
                )
                if not visible:
                    raise HTTPException(404, f"heartbeat {heartbeat_id} not found")
            except HTTPException as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "payload": {"message": e.detail},
                }))
                await websocket.close(code=1011)
                return

        # Subscribe to in-process bus first so we don't miss events that
        # land while we're sending the backlog.
        live_queue = await transcript_bus.subscribe(hb_id)

        try:
            # Send historical backlog so the client renders the full
            # timeline immediately, then transition to live tail.
            # Track whether the backlog ALREADY contains the real `finished`
            # event -- if so the heartbeat is complete and we must SKIP the tail
            # loop, else the safety net below (synthesize-on-finished_at) would
            # emit a SECOND finished with a bumped seq and the client would
            # render two "exit N" rows.
            finished = False
            async with Session() as s:
                rows = (await s.execute(
                    select(TranscriptEvent)
                    .where(TranscriptEvent.heartbeat_id == hb_id)
                    .order_by(TranscriptEvent.seq)
                )).scalars().all()
            for r in rows:
                await websocket.send_text(json.dumps({
                    "seq": r.seq,
                    "ts": r.ts.isoformat() if r.ts else "",
                    "type": r.type,
                    "payload": r.payload or {},
                }))
                last_sent_seq = max(last_sent_seq, r.seq)
                if r.type == "finished":
                    finished = True

            # Live tail: prefer the in-process bus (zero latency,
            # cross-process is impossible anyway). Also poll the DB
            # every 500ms as a safety net in case the runner lives in
            # a sibling process — captures persisted events the bus
            # never saw. (Skipped entirely when the backlog already delivered a
            # `finished` event -- see the flag set during backlog above.)
            while not finished:
                # 1) drain bus quickly
                try:
                    ev = await asyncio.wait_for(live_queue.get(), timeout=0.5)
                    if ev.get("seq", 0) > last_sent_seq:
                        await websocket.send_text(json.dumps(ev))
                        last_sent_seq = ev["seq"]
                        if ev.get("type") == "finished":
                            finished = True
                            break
                    continue
                except asyncio.TimeoutError:
                    pass

                # 2) DB tail (catches sibling-process emits)
                async with Session() as s:
                    new_rows = (await s.execute(
                        select(TranscriptEvent)
                        .where(
                            TranscriptEvent.heartbeat_id == hb_id,
                            TranscriptEvent.seq > last_sent_seq,
                        )
                        .order_by(TranscriptEvent.seq)
                    )).scalars().all()
                    # Also peek at heartbeat finished_at so we can close
                    # WS even when no `finished` event was emitted.
                    hb = (await s.execute(
                        select(HeartbeatRun).where(HeartbeatRun.id == hb_id)
                    )).scalar_one_or_none()
                for r in new_rows:
                    await websocket.send_text(json.dumps({
                        "seq": r.seq,
                        "ts": r.ts.isoformat() if r.ts else "",
                        "type": r.type,
                        "payload": r.payload or {},
                    }))
                    last_sent_seq = r.seq
                    if r.type == "finished":
                        finished = True
                if not finished and hb is not None and hb.finished_at is not None:
                    # Safety: heartbeat closed without a 'finished' event.
                    # Synthesize one and close.
                    await websocket.send_text(json.dumps({
                        "seq": last_sent_seq + 1,
                        "ts": hb.finished_at.isoformat(),
                        "type": "finished",
                        "payload": {"exit_code": hb.exit_code, "synthesized": True},
                    }))
                    finished = True
        finally:
            transcript_bus.unsubscribe(hb_id, live_queue)
    except WebSocketDisconnect:
        return
    except Exception as e:
        log.exception("heartbeat_ws crashed: %s", e)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
        return
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
