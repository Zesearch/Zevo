"""Cross-worker progress for the synchronous part of ``POST /runs``.

A Run is committed only after its complete scoring contract has been
materialized. The browser therefore supplies an unguessable UUID and polls a
small endpoint while the POST remains open. Web workers share ``data/`` in
production, so the progress record lives there as a tiny atomic JSON file as
well as in memory. A heartbeat lets a surviving worker distinguish a slow
setup from one whose worker was killed (for example by the container OOM
killer).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4


_CURRENT_SETUP: ContextVar[str] = ContextVar("zevo_run_setup_id", default="")
_PROGRESS: dict[str, dict[str, Any]] = {}
_STALE_AFTER_SECONDS = 10 * 60
_FAILED_AFTER_SECONDS = 30
_IDENT = re.compile(r"^[0-9a-fA-F-]{36}$")
_LAST_DISK_PRUNE = 0.0


def _root() -> Path:
    configured = (os.environ.get("ZEVO_RUN_SETUP_DIR") or "").strip()
    if configured:
        return Path(configured)
    from zevo.paths import work_dir_root

    return Path(work_dir_root()).parent / "run-setups"


def _path(setup_id: str) -> Path | None:
    ident = (setup_id or "").strip()
    if not _IDENT.fullmatch(ident):
        return None
    return _root() / f"{ident}.json"


def _prune(now: float) -> None:
    global _LAST_DISK_PRUNE

    stale = [
        setup_id
        for setup_id, state in _PROGRESS.items()
        if now - float(state.get("updated_at") or 0.0) > _STALE_AFTER_SECONDS
    ]
    for setup_id in stale:
        _PROGRESS.pop(setup_id, None)
    if now - _LAST_DISK_PRUNE < 60:
        return
    _LAST_DISK_PRUNE = now
    try:
        for path in _root().glob("*.json"):
            if now - path.stat().st_mtime > _STALE_AFTER_SECONDS:
                path.unlink(missing_ok=True)
    except OSError:
        pass


def _write(ident: str, state: dict[str, Any]) -> None:
    """Best-effort atomic persistence; UI feedback must never fail a Run."""
    path = _path(ident)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        try:
            temporary.write_text(
                json.dumps(state, separators=(",", ":")), encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    except OSError:
        return


def _read_file(ident: str) -> tuple[dict[str, Any], float] | None:
    path = _path(ident)
    if path is None:
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            return None
        return state, path.stat().st_mtime
    except (OSError, ValueError, TypeError):
        return None


def bind(setup_id: str) -> Token[str]:
    """Bind one launch request so engine code can emit without API coupling."""
    return _CURRENT_SETUP.set((setup_id or "").strip())


def reset(token: Token[str]) -> None:
    _CURRENT_SETUP.reset(token)


def update(
    setup_id: str = "",
    *,
    phase: str,
    completed: int = 0,
    total: int = 0,
    label: str = "",
    status: str = "active",
) -> None:
    ident = (setup_id or _CURRENT_SETUP.get()).strip()
    if not ident:
        return
    now = time.time()
    _prune(now)
    state = {
        "status": status,
        "phase": phase,
        "completed": max(0, int(completed)),
        "total": max(0, int(total)),
        "label": str(label or ""),
        "updated_at": now,
    }
    _PROGRESS[ident] = state
    _write(ident, state)


def touch(setup_id: str = "") -> None:
    """Refresh only liveness, without racing a newer phase back to an old one."""
    ident = (setup_id or _CURRENT_SETUP.get()).strip()
    path = _path(ident)
    if path is None:
        return
    try:
        if path.is_file():
            path.touch()
    except OSError:
        return


@asynccontextmanager
async def heartbeat(setup_id: str, interval: float = 5.0) -> AsyncIterator[None]:
    """Keep an active setup alive while its worker is still executing.

    Dataset conversion includes deliberately synchronous CPU and filesystem
    work.  Running the heartbeat as an asyncio task made that work look like a
    dead worker whenever it held the request event loop for more than the
    failure threshold.  A small daemon thread follows process liveness instead:
    it survives an event-loop stall, but still stops immediately if the worker
    process is killed.
    """
    ident = (setup_id or "").strip()
    stop = threading.Event()
    thread: threading.Thread | None = None
    if ident:
        def beat() -> None:
            while not stop.wait(interval):
                touch(ident)

        thread = threading.Thread(
            target=beat,
            name=f"zevo-run-setup-{ident[:8]}",
            daemon=True,
        )
        thread.start()
    try:
        yield
    finally:
        stop.set()
        if thread is not None:
            # Event.set wakes Event.wait immediately. Keep the join away from
            # the request loop in case the final EFS touch is still returning.
            await asyncio.to_thread(thread.join, max(1.0, interval * 2))


def read(setup_id: str) -> dict[str, Any] | None:
    ident = (setup_id or "").strip()
    now = time.time()
    _prune(now)
    state = _PROGRESS.get(ident)
    updated_at = float(state.get("updated_at") or 0.0) if state else 0.0
    stored = _read_file(ident)
    if stored is not None:
        disk_state, disk_mtime = stored
        # The file's mtime is the liveness clock: heartbeats intentionally
        # touch it without rewriting a potentially newer phase payload.
        state = disk_state
        updated_at = disk_mtime
    if state is None:
        return None
    public = {key: value for key, value in state.items() if key != "updated_at"}
    if public.get("status") == "active" and now - updated_at > _FAILED_AFTER_SECONDS:
        return {
            **public,
            "status": "failed",
            "phase": "failed",
            "label": "Run setup stopped unexpectedly. Try starting it again.",
        }
    return public
