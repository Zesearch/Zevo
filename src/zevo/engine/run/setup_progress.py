"""Ephemeral progress for the synchronous part of ``POST /runs``.

A Run does not exist until its complete scoring contract has been materialized
and committed, so ordinary Run/Ticket progress cannot describe this interval.
The browser supplies an unguessable UUID and polls a small read endpoint while
the request is open. State is deliberately process-local and short-lived: it is
UI feedback, never durable Run state or part of the transaction.
"""
from __future__ import annotations

import time
from contextvars import ContextVar, Token
from typing import Any


_CURRENT_SETUP: ContextVar[str] = ContextVar("zevo_run_setup_id", default="")
_PROGRESS: dict[str, dict[str, Any]] = {}
_STALE_AFTER_SECONDS = 10 * 60


def _prune(now: float) -> None:
    stale = [
        setup_id
        for setup_id, state in _PROGRESS.items()
        if now - float(state.get("updated_monotonic") or 0.0) > _STALE_AFTER_SECONDS
    ]
    for setup_id in stale:
        _PROGRESS.pop(setup_id, None)


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
    now = time.monotonic()
    _prune(now)
    _PROGRESS[ident] = {
        "status": status,
        "phase": phase,
        "completed": max(0, int(completed)),
        "total": max(0, int(total)),
        "label": str(label or ""),
        "updated_monotonic": now,
    }


def read(setup_id: str) -> dict[str, Any] | None:
    now = time.monotonic()
    _prune(now)
    state = _PROGRESS.get((setup_id or "").strip())
    if state is None:
        return None
    return {key: value for key, value in state.items() if key != "updated_monotonic"}
