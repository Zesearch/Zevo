"""Wake-up queue helpers.

Every code path that creates or reassigns a ticket calls
`queue_wakeup(...)`. The daemon (`zevo.engine.run.scheduler.wakeup_daemon`)
drains the queue: groups by agent_id, respects each agent's
`max_concurrent_runs`, takes a per-WAKEUP advisory lock, and runs the
ticket in-process.

Two helpers exported:

  queue_wakeup(session, *, agent_id, ticket_id=None, source, ...)
      Inserts an AgentWakeupRequest row. Coalesces against any
      existing 'queued' wakeup for the same (agent_id, ticket_id) —
      first by a SELECT, and as the backstop by the partial unique
      index `uq_wakeup_queued_per_ticket`, so two concurrent enqueues
      cannot both land as 'queued'. Returns the AgentWakeupRequest
      (which may already be marked 'coalesced' if a duplicate was
      suppressed).

  advisory_lock(session, key) -> async context manager
      `async with advisory_lock(session, f"wakeup:{id}") as got:`
      Tries a named Postgres advisory lock keyed on the WAKEUP (one key
      per wakeup, not per agent) via `pg_try_advisory_lock(hashtext(key))`
      on a dedicated connection, and releases it on the way out. The
      lock is session-level, so it does NOT come back by itself when
      the connection returns to the pool — the unlock-failure path
      closes the connection outright, and the daemon's reap pass
      reclaims anything that still leaks.

Both work against asyncpg + the shared Zevo session factory.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError, PendingRollbackError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from zevo.db import AgentWakeupRequest
from zevo.db.session import get_engine

log = logging.getLogger(__name__)


async def queue_wakeup(
    session: AsyncSession,
    *,
    agent_id: str,
    ticket_id: str | None = None,
    source: str,
    trigger_detail: str = "",
    reason: str = "",
    payload: dict[str, Any] | None = None,
) -> AgentWakeupRequest:
    """Enqueue a wakeup request. Coalesces against existing queued.

    Coalescing rule (per the agreed defaults): if a `queued` wakeup
    already exists for this (agent_id, ticket_id) pair, the new request
    is recorded as 'coalesced' instead of duplicating work. Running
    wakeups DO NOT block new ones -- new state since the run started
    can deserve a fresh wake.
    """
    if ticket_id:
        existing = (
            await session.execute(
                select(AgentWakeupRequest).where(
                    AgentWakeupRequest.agent_id == agent_id,
                    AgentWakeupRequest.ticket_id == ticket_id,
                    AgentWakeupRequest.status == "queued",
                )
            )
        ).scalars().first()
        if existing is not None:
            row = AgentWakeupRequest(
                agent_id=agent_id, ticket_id=ticket_id, source=source,
                status="coalesced", trigger_detail=trigger_detail,
                reason=f"coalesced into {existing.id}: {reason}",
                payload=payload or {},
            )
            session.add(row)
            await session.commit()
            return row

    row = AgentWakeupRequest(
        agent_id=agent_id, ticket_id=ticket_id, source=source,
        status="queued", trigger_detail=trigger_detail, reason=reason,
        payload=payload or {},
    )
    session.add(row)
    try:
        await session.commit()
    except IntegrityError:
        # Lost the race against a concurrent enqueue: the partial unique
        # index (one 'queued' row per agent+ticket) rejected the insert.
        # Record the coalescing the same way the SELECT path does.
        await session.rollback()
        existing = (
            await session.execute(
                select(AgentWakeupRequest).where(
                    AgentWakeupRequest.agent_id == agent_id,
                    AgentWakeupRequest.ticket_id == ticket_id,
                    AgentWakeupRequest.status == "queued",
                )
            )
        ).scalars().first()
        row = AgentWakeupRequest(
            agent_id=agent_id, ticket_id=ticket_id, source=source,
            status="coalesced", trigger_detail=trigger_detail,
            reason=f"coalesced into {existing.id if existing else '(drained)'}: {reason}",
            payload=payload or {},
        )
        session.add(row)
        await session.commit()
        return row
    await session.refresh(row)
    return row


@asynccontextmanager
async def advisory_lock(
    session: AsyncSession,
    key: str,
) -> AsyncIterator[bool]:
    """Try to acquire a named Postgres advisory lock. Yields True iff the
    lock was obtained.

    `key` is any string; it is hashed. It used to be the agent id, which
    made the lock a per-agent mutex and capped every agent at one wakeup
    at a time no matter what `max_concurrent_runs` said. The thing that
    must never happen twice is processing the same WAKEUP, so that is
    what the key names now. Concurrency per agent is accounted for by the
    daemon instead, where the limit actually lives.

    Usage:
        async with advisory_lock(session, f"wakeup:{w.id}") as got:
            if not got:
                return  # someone else owns it; try again next tick
            # ... do work ...

    The lock is taken on a connection of its own rather than on
    `session`, and this is the whole point of the design. An advisory
    lock belongs to the connection that took it, and the caller commits
    while holding it -- that is how a wakeup gets marked `running`. A
    commit hands the session's connection back to the pool, so under any
    concurrency the next statement runs on a different connection: the
    unlock would be issued by someone who never held the lock (postgres
    answers "not held" and shrugs), while the real holder sits in the
    pool owning it forever. That agent then never gets woken again, its
    wakeups pile up as `queued`, and from the UI a launched run simply
    never starts. Closing our own connection at the end releases the
    lock unconditionally, whatever happened inside.

    `session` is still taken as an argument -- it identifies the engine,
    and it keeps the call sites honest about which unit of work the lock
    is guarding.
    """
    engine = session.get_bind() if session.bind is not None else get_engine()
    if not isinstance(engine, AsyncEngine):
        engine = get_engine()

    conn = await engine.connect()
    got = False
    try:
        got = bool((await conn.execute(
            text("SELECT pg_try_advisory_lock(hashtext(:k))").bindparams(k=key)
        )).scalar())
        yield got
    finally:
        # close() returns the connection to the pool AND drops every
        # advisory lock it holds, so this is correct even when the body
        # raised, and even when it was cancelled mid-await.
        try:
            if got:
                await conn.execute(
                    text("SELECT pg_advisory_unlock(hashtext(:k))").bindparams(k=key)
                )
        except (SQLAlchemyError, PendingRollbackError) as exc:
            log.warning("advisory unlock failed for %s (%s); closing the "
                        "connection instead", key, exc)
        finally:
            try:
                await conn.close()
            except Exception:
                log.exception("could not close the connection holding %s's lock",
                              key)
