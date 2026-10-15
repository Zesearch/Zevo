"""In-process pub/sub for live transcript events.

The runner emits events as a heartbeat executes; subscribers (WebSocket
connections in the backend) read them off a per-heartbeat asyncio.Queue.
Each new event also gets a monotonically-increasing `seq` so subscribers
that reconnect mid-run can ask for everything since their last seen seq.

This is process-local. The backend and the runner share a process when
the daemon runs in-process (current default). If we later split them,
we'll replace this with a Postgres LISTEN/NOTIFY or Redis pub/sub --
the API stays the same.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Any


log = logging.getLogger(__name__)


_lock = threading.Lock()
# heartbeat_id -> { "seq": int, "subscribers": set[asyncio.Queue] }
_state: dict[str, dict[str, Any]] = {}


def _bucket(heartbeat_id: str) -> dict[str, Any]:
    with _lock:
        b = _state.get(heartbeat_id)
        if b is None:
            b = {"seq": 0, "subscribers": set()}
            _state[heartbeat_id] = b
        return b


def next_seq(heartbeat_id: str) -> int:
    """Atomically bump and return the next seq for this heartbeat."""
    with _lock:
        b = _state.get(heartbeat_id)
        if b is None:
            b = {"seq": 0, "subscribers": set()}
            _state[heartbeat_id] = b
        b["seq"] += 1
        return b["seq"]


def publish(heartbeat_id: str, event: dict) -> None:
    """Push one event to every subscriber of this heartbeat.

    `event` should already carry `seq`, `ts`, `type`, `payload`. This
    fans out to subscriber queues; queues are bounded but drop oldest
    on overflow (a slow subscriber must not stall the runner).
    """
    b = _bucket(heartbeat_id)
    dead: list[asyncio.Queue] = []
    for q in list(b["subscribers"]):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            # Drop the oldest, push the new one. Better to skip than
            # block the runner on a single slow client.
            try:
                q.get_nowait()
                q.put_nowait(event)
            except Exception:
                dead.append(q)
        except Exception as e:  # pragma: no cover - defensive
            log.warning("transcript_bus.publish: dropping bad subscriber: %s", e)
            dead.append(q)
    if dead:
        with _lock:
            b["subscribers"].difference_update(dead)


async def subscribe(heartbeat_id: str) -> asyncio.Queue:
    """Return a queue that receives every future event for this heartbeat."""
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    b = _bucket(heartbeat_id)
    with _lock:
        b["subscribers"].add(q)
    return q


def unsubscribe(heartbeat_id: str, q: asyncio.Queue) -> None:
    with _lock:
        b = _state.get(heartbeat_id)
        if b is not None:
            b["subscribers"].discard(q)


def cleanup(heartbeat_id: str) -> None:
    """Drop all state for a heartbeat once it has terminated."""
    with _lock:
        _state.pop(heartbeat_id, None)


def make_event(*, seq: int, ev_type: str, payload: dict) -> dict:
    """Build a canonical event envelope. Callers pass this to publish()."""
    return {
        "seq": seq,
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": ev_type,
        "payload": payload or {},
    }
