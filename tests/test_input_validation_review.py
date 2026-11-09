"""Rejected inputs cannot trigger cron loops or stale coordinator activations."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db import Agent, AgentWakeupRequest, Base, Run, Ticket, TicketNotice
from zevo.engine.run.input_validation import EVENTS_KEY, NOTICE_CODE, report_input_failure
from zevo.engine.run.scheduler import wakeup_daemon as wd
from zevo.engine.run.scheduler.reconciler import _fail_validation_looping_tickets
from zevo.engine.run.wakeup import queue_wakeup


@pytest_asyncio.fixture
async def database(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/review.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(wd, "get_session_factory", lambda: factory)

    @asynccontextmanager
    async def lock(*args):
        yield True

    monkeypatch.setattr(wd, "advisory_lock", lock)
    async with factory() as db:
        db.add_all([
            Agent(id="infrastructure", name="infra", title="infra", identity_path="unused"),
            Run(id="run", metric="score", status="running", supervisor_ticket_id="orch"),
            Ticket(id="orch", run_id="run", agent_id="orchestrator", status="succeeded"),
            Ticket(id="child", run_id="run", agent_id="infrastructure", status="queued",
                   payload={"version": 1}, repair_attempts=7),
        ])
        await db.commit()
    yield factory
    await engine.dispose()


async def report(db, ticket_id="child"):
    await report_input_failure(db, await db.get(Ticket, ticket_id), await db.get(Run, "run"),
                               "work order rejected before execution")
    return (await db.scalars(select(AgentWakeupRequest).where(
        AgentWakeupRequest.ticket_id == "orch", AgentWakeupRequest.status == "queued",
    ))).one()


@pytest.mark.asyncio
@pytest.mark.parametrize("review_status", ["queued", "running", "completed"])
async def test_first_failure_stops_cron_and_duplicate_notifications(database, review_status):
    async with database() as db:
        wake = await report(db)
        wake.status = review_status
        await db.commit()
        # Reproduce a second validation callback while review is already under way.
        await report_input_failure(db, await db.get(Ticket, "child"), await db.get(Run, "run"),
                                   "the same rejection with different wording")
    await wd._cron_tick(300, "optimization")
    async with database() as db:
        child = await db.get(Ticket, "child")
        assert (child.status, child.repair_route, child.repair_attempts) == ("failed", "orchestrator", 7)
        assert len((await db.scalars(select(AgentWakeupRequest))).all()) == 1
        assert len((await db.scalars(select(TicketNotice).where(TicketNotice.code == NOTICE_CODE))).all()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["cancelled", "succeeded", "corrected", "handled", "unresolved", "user"])
async def test_review_rechecks_current_state_before_starting_an_agent(database, monkeypatch, change):
    async with database() as db:
        wake = await report(db)
        child = await db.get(Ticket, "child")
        if change in {"cancelled", "succeeded"}:
            child.status = change
        elif change == "corrected":
            child.payload = {"version": 2}
        elif change == "handled":
            db.add(AgentWakeupRequest(agent_id="orchestrator", ticket_id="orch", source="handoff",
                                     status="completed", payload=dict(wake.payload)))
        elif change == "user":
            wake.source = "on_demand"
        await db.commit()
        wake_id = wake.id
    calls = []

    async def run(*args, **kwargs):
        calls.append(kwargs["ticket_id"])

    monkeypatch.setattr(wd, "run_ticket", run)
    await wd._run_one("orchestrator", wake_id, datetime.now(timezone.utc))
    should_run = change in {"unresolved", "user"}
    assert calls == (["orch"] if should_run else [])
    async with database() as db:
        wake = await db.get(AgentWakeupRequest, wake_id)
        assert wake.status == "completed" and wake.finished_at


@pytest.mark.asyncio
async def test_corrected_work_order_can_be_dispatched_and_new_rejection_can_notify(database, monkeypatch):
    async with database() as db:
        first = await report(db)
        first.status = "completed"
        child = await db.get(Ticket, "child")
        child.payload = {"version": 2}
        child.status = "queued"
        await db.commit()
        # Even old legacy failure counters cannot reject this corrected version.
        for _ in range(3):
            db.add(AgentWakeupRequest(agent_id="infrastructure", ticket_id="child", source="cron",
                                     status="failed", reason="stored-payload-validation-failed"))
        await db.commit()
        assert await _fail_validation_looping_tickets(db) == 0
    await wd._cron_tick(300, "optimization")
    async with database() as db:
        scheduled = (await db.scalars(select(AgentWakeupRequest).where(
            AgentWakeupRequest.ticket_id == "child", AgentWakeupRequest.status == "queued",
        ))).one()
        wake_id = scheduled.id
    calls = []

    async def run(*args, **kwargs):
        calls.append(kwargs["ticket_id"])

    monkeypatch.setattr(wd, "run_ticket", run)
    await wd._run_one("infrastructure", wake_id, datetime.now(timezone.utc))
    assert calls == ["child"]
    async with database() as db:
        completed = await db.get(AgentWakeupRequest, wake_id)
        assert completed.status == "completed" and completed.finished_at
        new = await report(db)
        assert new.payload[EVENTS_KEY] != first.payload[EVENTS_KEY]


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["review_first", "completion_first"])
async def test_coalesced_real_completion_is_not_discarded_with_stale_review(database, monkeypatch, order):
    async with database() as db:
        if order == "completion_first":
            await queue_wakeup(db, agent_id="orchestrator", ticket_id="orch", source="handoff",
                               payload={"optimization_completion_key": "other:succeeded"})
        wake = await report(db)
        if order == "review_first":
            await queue_wakeup(db, agent_id="orchestrator", ticket_id="orch", source="handoff",
                               payload={"optimization_completion_key": "other:succeeded"})
        (await db.get(Ticket, "child")).status = "cancelled"
        await db.commit()
        wake_id = wake.id
    calls = []

    async def run(*args, **kwargs):
        calls.append(kwargs["ticket_id"])

    monkeypatch.setattr(wd, "run_ticket", run)
    await wd._run_one("orchestrator", wake_id, datetime.now(timezone.utc))
    assert calls == ["orch"]


@pytest.mark.asyncio
async def test_coalesced_reviews_keep_other_unresolved_children(database, monkeypatch):
    async with database() as db:
        wake = await report(db)
        db.add(Ticket(id="other", run_id="run", agent_id="data", status="queued", payload={}))
        await db.commit()
        await report(db, "other")
        assert len(wake.payload[EVENTS_KEY]) == 2
        (await db.get(Ticket, "child")).status = "cancelled"
        await db.commit()
        wake_id = wake.id
    calls = []

    async def run(*args, **kwargs):
        calls.append(kwargs["ticket_id"])

    monkeypatch.setattr(wd, "run_ticket", run)
    await wd._run_one("orchestrator", wake_id, datetime.now(timezone.utc))
    assert calls == ["orch"]


@pytest.mark.asyncio
async def test_enqueue_race_preserves_failed_ticket_and_notice(database, monkeypatch):
    async with database() as db:
        await queue_wakeup(db, agent_id="orchestrator", ticket_id="orch", source="handoff")
        child = await db.get(Ticket, "child")
        child.status = "failed"
        notice = TicketNotice(ticket_id="child", code=NOTICE_CODE, body="{}", severity="error")
        db.add(notice)
        execute = db.execute
        first = True

        async def miss_first_lookup(*args, **kwargs):
            nonlocal first
            if first:
                first = False
                return SimpleNamespace(scalars=lambda: SimpleNamespace(first=lambda: None))
            return await execute(*args, **kwargs)

        # Simulate a stale SELECT, then exercise the real unique-index conflict.
        monkeypatch.setattr(db, "execute", miss_first_lookup)
        wake = await queue_wakeup(db, agent_id="orchestrator", ticket_id="orch", source="handoff")
        assert wake.status == "coalesced"
    async with database() as db:
        assert (await db.get(Ticket, "child")).status == "failed"
        assert await db.get(TicketNotice, notice.id) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["cancelled_ticket", "cancelled_run", "held_out"])
async def test_failure_notification_preserves_cancellation_and_isolation(database, case):
    async with database() as db:
        child, run = await db.get(Ticket, "child"), await db.get(Run, "run")
        if case == "cancelled_ticket":
            child.status = "cancelled"
        elif case == "cancelled_run":
            run.status = "cancelled"
        else:
            child.lane = "held_out_test"
        await db.commit()
        await report_input_failure(db, child, run, "invalid work order")
        assert not (await db.scalars(select(AgentWakeupRequest))).all()
        if case == "cancelled_ticket":
            assert child.status == "cancelled"
        elif case == "held_out":
            assert child.status == "failed" and child.repair_route == "terminal"
