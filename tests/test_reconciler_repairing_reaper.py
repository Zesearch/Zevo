"""A ticket stuck in 'repairing' with no retry wakeup in flight (its
reactivation was lost) must be re-queued by the reconciler, not left to wedge
its run forever. Recently-flipped tickets, ones with an active wakeup, and
tickets on terminal runs are left alone.
"""
from __future__ import annotations

import datetime as _dt
from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import Agent, AgentWakeupRequest, Base, Run, Ticket
from zevo.engine.run.scheduler.reconciler import _reactivate_stuck_repairing_tickets


async def _session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


_RUN = dict(
    task_name="t", task_objective="o", agent_objective="ao",
    metric="accuracy", metric_direction="max",
    validation_metric="accuracy", validation_metric_direction="max",
)


async def _seed(db, *, ticket_updated, run_status="running"):
    db.add(Agent(id="train", name="train", title="Trainer", identity_path=""))
    db.add(Run(id="r1", status=run_status, **_RUN))
    db.add(Ticket(
        id="train-1", run_id="r1", agent_id="train", status="repairing",
        payload={}, summary="", error_message="", repair_attempts=1,
        updated_at=ticket_updated,
    ))
    await db.commit()


async def _wakeup_count(db) -> int:
    return (await db.execute(
        select(func.count()).select_from(AgentWakeupRequest)
        .where(AgentWakeupRequest.ticket_id == "train-1")
    )).scalar_one()


@pytest.mark.asyncio
async def test_stuck_repairing_is_requeued() -> None:
    Session = await _session()
    stale = datetime.now(timezone.utc) - _dt.timedelta(hours=1)
    async with Session() as db:
        await _seed(db, ticket_updated=stale)
        n = await _reactivate_stuck_repairing_tickets(db, stale_ticket_seconds=60)
        assert n == 1
        # A fresh retry wakeup now exists for the ticket.
        assert await _wakeup_count(db) == 1
        wk = (await db.execute(select(AgentWakeupRequest))).scalars().first()
        assert wk.source == "retry" and wk.status == "queued"


@pytest.mark.asyncio
async def test_recently_flipped_is_left_alone() -> None:
    Session = await _session()
    fresh = datetime.now(timezone.utc)  # just flipped to repairing
    async with Session() as db:
        await _seed(db, ticket_updated=fresh)
        n = await _reactivate_stuck_repairing_tickets(db, stale_ticket_seconds=60)
        assert n == 0
        assert await _wakeup_count(db) == 0


@pytest.mark.asyncio
async def test_active_wakeup_not_duplicated() -> None:
    Session = await _session()
    stale = datetime.now(timezone.utc) - _dt.timedelta(hours=1)
    async with Session() as db:
        await _seed(db, ticket_updated=stale)
        db.add(AgentWakeupRequest(
            id="wk-existing", agent_id="train", ticket_id="train-1",
            status="queued", source="retry",
        ))
        await db.commit()
        n = await _reactivate_stuck_repairing_tickets(db, stale_ticket_seconds=60)
        assert n == 0
        assert await _wakeup_count(db) == 1  # still just the existing one


@pytest.mark.asyncio
async def test_terminal_run_not_reactivated() -> None:
    Session = await _session()
    stale = datetime.now(timezone.utc) - _dt.timedelta(hours=1)
    async with Session() as db:
        await _seed(db, ticket_updated=stale, run_status="halted")
        n = await _reactivate_stuck_repairing_tickets(db, stale_ticket_seconds=60)
        assert n == 0
        assert await _wakeup_count(db) == 0
