"""Global registry of live agent subprocess handles.

Two kinds of process end up here, under two kinds of key:

  heartbeat id   the CLI driver's own subprocess (Claude Code / Codex)
  ticket id      a shell command an in-process driver is running as a tool

The second one is why this is a registry of SETS rather than one handle
per key. The bedrock driver has no CLI subprocess of its own -- it calls
the model API in-process and shells out per tool call -- so cancelling by
heartbeat id found nothing to kill, and a remote process waiting on a busy
allocation kept running long after its run was cancelled and its GPU
freed. Whatever a driver spawns, it registers here, and cancel reaches
all of it.

The registry is process-local (the scheduler container); that is fine
because the runner and the cancel signal share the same process while the
daemon runs in-process.

Thread-safe via a threading.Lock (drain coroutines run on the event loop;
the cancel endpoint may arrive on a different thread in uvicorn).
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading

log = logging.getLogger(__name__)

_lock = threading.Lock()
_procs: dict[str, set[asyncio.subprocess.Process]] = {}


def register(key: str, proc: asyncio.subprocess.Process) -> None:
    """Track `proc` under `key` (a heartbeat id or a ticket id)."""
    if not key:
        return
    with _lock:
        _procs.setdefault(key, set()).add(proc)


def unregister(key: str, proc: asyncio.subprocess.Process | None = None) -> None:
    """Forget one process under `key`, or every one when `proc` is None."""
    if not key:
        return
    with _lock:
        if proc is None:
            _procs.pop(key, None)
            return
        live = _procs.get(key)
        if live is None:
            return
        live.discard(proc)
        if not live:
            _procs.pop(key, None)


def cancel(key: str) -> bool:
    """Terminate everything registered under `key`.

    Returns True if at least one process was found and signalled, False
    if there was nothing to kill (already finished, or spawned by another
    process).
    """
    with _lock:
        procs = list(_procs.get(key) or ())
    signalled = False
    for proc in procs:
        if proc.returncode is not None:
            continue
        try:
            # SIGTERM the whole process group, so a shell's children -- the
            # ssh it opened, and the remote wrapper behind that -- die with it. Killing
            # only the shell leaves those orphaned and still holding
            # whatever they were holding.
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            log.info("cancelled %s (pid=%s, process group)", key, proc.pid)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
                log.info("cancelled %s (pid=%s, process only)", key, proc.pid)
            except (ProcessLookupError, OSError):
                continue
        signalled = True
    return signalled
