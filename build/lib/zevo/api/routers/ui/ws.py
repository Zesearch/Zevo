"""WebSocket: /ws/runs/{run_id}.

Simple polling-based push for now: the WebSocket re-queries the DB every
second and sends a diff message when ticket statuses or phase counts change.
PostgreSQL NOTIFY/LISTEN remains a possible replacement if polling becomes a
measured bottleneck.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import func, select

from zevo.db import ExecutionEvent, Ticket, get_session_factory
from zevo.api.ui_access import is_trusted_ui_websocket


router = APIRouter()


@router.websocket("/ws/runs/{run_id}")
async def run_ws(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    Session = get_session_factory()
    trusted_ui = is_trusted_ui_websocket(websocket)
    last_state: dict[str, dict] = {}

    async def _watch_disconnect() -> None:
        # The send loop never receives, so without this a closed tab watching
        # a finished (never-changing) run keeps the polling task alive forever.
        # Drain incoming frames until the client goes away.
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code") or 1000)

    watcher = asyncio.create_task(_watch_disconnect())
    try:
        while True:
            async with Session() as s:
                tickets = (
                    await s.execute(select(Ticket).where(Ticket.run_id == run_id))
                ).scalars().all()
                # One grouped COUNT per snapshot, not a full materialisation of
                # every ExecutionEvent row per ticket just to len() them.
                phase_counts: dict[str, int] = {}
                if tickets:
                    counted = await s.execute(
                        select(ExecutionEvent.ticket_id, func.count())
                        .where(ExecutionEvent.ticket_id.in_([t.id for t in tickets]))
                        .group_by(ExecutionEvent.ticket_id)
                    )
                    phase_counts = {tid: int(n) for tid, n in counted.all()}
            state: dict[str, dict] = {
                t.id: {
                    # The trusted dashboard receives live held-out metadata;
                    # untrusted websocket callers receive only a structural
                    # outline. Payloads, paths, scores and transcripts never
                    # travel on this channel.
                    "agent_id": t.agent_id,
                    "lane": t.lane,
                    "iteration": t.iteration,
                    "status": t.status,
                    "summary": (
                        (t.summary or "")[:200]
                        if trusted_ui or t.lane != "held_out_test" else ""
                    ),
                    "created_at": t.created_at.isoformat() if t.created_at else "",
                    "execution_event_rows": (
                        0 if t.lane == "held_out_test" and not trusted_ui
                        else phase_counts.get(t.id, 0)
                    ),
                }
                for t in tickets
            }
            if state != last_state:
                await websocket.send_text(json.dumps({"type": "snapshot", "tickets": state}))
                last_state = state
            # Sleep AND watch: waiting on the watcher task doubles as the
            # 1-second poll interval, so a disconnect ends the loop at once.
            done, _ = await asyncio.wait({watcher}, timeout=1.0)
            if done:
                break
    except WebSocketDisconnect:
        return
    finally:
        watcher.cancel()
        try:
            await watcher
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception:
            pass
