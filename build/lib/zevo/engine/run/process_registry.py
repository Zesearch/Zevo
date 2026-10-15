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
from typing import Awaitable, TypeVar

log = logging.getLogger(__name__)

_lock = threading.Lock()
_procs: dict[str, set[asyncio.subprocess.Process]] = {}
_paused: set[str] = set()
_T = TypeVar("_T")


def _signal_process_group(
    proc: asyncio.subprocess.Process, sig: signal.Signals,
) -> bool:
    if proc.returncode is not None:
        return False
    try:
        os.killpg(os.getpgid(proc.pid), sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(proc.pid, sig)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False


def register(key: str, proc: asyncio.subprocess.Process) -> None:
    """Track `proc` under `key` (a heartbeat id or a ticket id)."""
    if not key:
        return
    with _lock:
        _procs.setdefault(key, set()).add(proc)
        paused = key in _paused
    # A tool process can be created in the small interval between the runner's
    # pause poll and the driver's next tool call.  Honour the already-active
    # gate as soon as that process registers.
    if paused:
        _signal_process_group(proc, signal.SIGSTOP)


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
            # SIGTERM remains pending for a stopped process on some platforms.
            # Continue it first so an operator cancel cannot leave a paused
            # activation stranded forever.
            _signal_process_group(proc, signal.SIGCONT)
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


def pause(key: str) -> bool:
    """Cooperatively freeze every live process owned by ``key``.

    The key remains paused even when no process exists yet, so an in-process
    driver cannot race the gate by spawning its next shell just afterwards.
    """
    if not key:
        return False
    with _lock:
        _paused.add(key)
        procs = list(_procs.get(key) or ())
    signalled = False
    for proc in procs:
        signalled = _signal_process_group(proc, signal.SIGSTOP) or signalled
    return signalled


def resume(key: str) -> bool:
    """Release a prior :func:`pause` and continue its live processes."""
    if not key:
        return False
    with _lock:
        _paused.discard(key)
        procs = list(_procs.get(key) or ())
    signalled = False
    for proc in procs:
        signalled = _signal_process_group(proc, signal.SIGCONT) or signalled
    return signalled


def is_paused(key: str) -> bool:
    with _lock:
        return key in _paused


async def wait_until_resumed(key: str, poll_seconds: float = 0.1) -> None:
    """Hold an in-process driver between tool calls while a Run is gated."""
    while key and is_paused(key):
        await asyncio.sleep(poll_seconds)


async def wait_for_active_time(
    awaitable: Awaitable[_T], *, key: str, timeout: float, poll_seconds: float = 0.25,
) -> _T:
    """Wait with a timeout that excludes instruction-paused wall time."""
    task = asyncio.ensure_future(awaitable)
    if timeout <= 0:
        return await task
    loop = asyncio.get_running_loop()
    active_elapsed = 0.0
    last_tick = loop.time()
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=poll_seconds)
            now = loop.time()
            if not is_paused(key):
                active_elapsed += max(0.0, now - last_tick)
            last_tick = now
            if done:
                return await task
            if active_elapsed >= timeout:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise asyncio.TimeoutError()
    except BaseException:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        raise
