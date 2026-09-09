"""`max_concurrent_runs` has to actually reach the dispatcher.

The setting existed and was read into the slot arithmetic, but two other
things pinned every agent to one wakeup at a time regardless: `_inflight`
held a single task per agent, and the advisory lock was keyed on the agent
id. Raising the number in the database changed nothing, and nothing failed
— two runs simply took turns, which looks like one of them being stuck.

That is the shape of bug these tests exist for: a limit that is configured,
believed, and silently not applied. So they assert on how many wakeups the
dispatcher actually STARTS, not on what the column says.

`_run_one` is stubbed out. It takes a Postgres advisory lock, which SQLite
has no equivalent for, and what is under test here is the slot accounting
that decides whether to call it at all.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import Agent, AgentWakeupRequest, Base, Ticket, TicketNotice
from zevo.engine.run.scheduler import wakeup_daemon as wd


@pytest_asyncio.fixture
async def db(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/wake.db", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(wd, "get_session_factory", lambda: Session)
    wd._inflight.clear()
    yield Session
    wd._inflight.clear()
    await engine.dispose()


@pytest.fixture
def started(monkeypatch):
    """Record which wakeups the dispatcher hands to `_run_one`."""
    seen: list[str] = []

    async def _fake(agent_id: str, wakeup_id: str, queued_at: datetime) -> None:
        seen.append(wakeup_id)
        # Long enough that the dispatcher cannot see it finish within the test:
        # a task that completes immediately frees its slot and hides an
        # over-start behind timing.
        await asyncio.sleep(60)

    monkeypatch.setattr(wd, "_run_one", _fake)
    return seen


async def _seed(Session, agent_id: str, cap: int, n_queued: int) -> None:
    async with Session() as s:
        s.add(Agent(
            id=agent_id, name=agent_id, title=agent_id,
            identity_path=f"playbook/agents/{agent_id}/identity.md",
            max_concurrent_runs=cap,
        ))
        for i in range(n_queued):
            s.add(AgentWakeupRequest(
                id=f"{agent_id}-w{i}", agent_id=agent_id, status="queued",
                source="assignment", scheduled_for=datetime.now(timezone.utc),
            ))
        await s.commit()


async def _drain(started: list[str]) -> list[str]:
    n = await wd._drain_once()
    # `create_task` only schedules; the coroutine body has not run yet when
    # the dispatcher returns. Yield so "started" means started.
    await asyncio.sleep(0)
    assert n == len(started), f"reported {n} processed but started {started}"
    return started


@pytest.mark.asyncio
async def test_two_slots_start_two_wakeups(db, started):
    """The case that was broken: cap 2, two waiting, both go."""
    await _seed(db, "inference", cap=2, n_queued=2)
    assert sorted(await _drain(started)) == ["inference-w0", "inference-w1"]


@pytest.mark.asyncio
async def test_one_slot_still_starts_one(db, started):
    """An Agent's configured concurrency cap is enforced independently."""
    await _seed(db, "registry", cap=1, n_queued=3)
    assert len(await _drain(started)) == 1


@pytest.mark.asyncio
async def test_never_exceeds_the_cap(db, started):
    await _seed(db, "train", cap=2, n_queued=5)
    assert len(await _drain(started)) == 2


@pytest.mark.asyncio
async def test_a_second_tick_does_not_double_start(db, started):
    """The first tick's tasks are still in flight; the cap already spent."""
    await _seed(db, "data", cap=2, n_queued=4)
    await wd._drain_once()
    await asyncio.sleep(0)
    first = list(started)
    await wd._drain_once()
    await asyncio.sleep(0)
    assert started == first, f"a later tick restarted work: {started}"


@pytest.mark.asyncio
async def test_running_rows_count_against_the_cap(db, started):
    """A wakeup another daemon marked `running` occupies a slot too.

    `_inflight` only sees this process. Without also counting the DB rows a
    restarted daemon would start a second worker beside one already going.
    """
    await _seed(db, "inference", cap=2, n_queued=2)
    async with db() as s:
        s.add(AgentWakeupRequest(
            id="inference-elsewhere", agent_id="inference", status="running",
            source="assignment", scheduled_for=datetime.now(timezone.utc),
        ))
        await s.commit()
    assert len(await _drain(started)) == 1


@pytest.mark.asyncio
async def test_agents_do_not_block_each_other(db, started):
    await _seed(db, "registry", cap=1, n_queued=2)
    await _seed(db, "train", cap=2, n_queued=2)
    got = await _drain(started)
    assert sum(w.startswith("registry") for w in got) == 1
    assert sum(w.startswith("train") for w in got) == 2


@pytest.mark.asyncio
async def test_finished_work_frees_its_slot(db, started, monkeypatch):
    """The done-callback has to remove the right entry, not the whole agent."""
    async def _instant(agent_id: str, wakeup_id: str, queued_at: datetime) -> None:
        started.append(wakeup_id)

    monkeypatch.setattr(wd, "_run_one", _instant)
    await _seed(db, "train", cap=2, n_queued=4)
    await wd._drain_once()
    for _ in range(5):  # task body, then its done-callback
        await asyncio.sleep(0)
    assert wd._inflight.get("train") in (None, {}), wd._inflight


@pytest.mark.asyncio
async def test_each_daemon_starts_only_its_ticket_lane(db, started):
    """Physical scheduler separation starts with queue ownership."""
    async with db() as s:
        s.add(Agent(
            id="data", name="data", title="data",
            identity_path="playbook/agents/data/identity.md",
            max_concurrent_runs=4,
        ))
        for lane in ("optimization", "held_out_test"):
            ticket_id = f"{lane}-ticket"
            s.add(Ticket(
                id=ticket_id, run_id="run-1", agent_id="data",
                status="queued", lane=lane, payload={}, customization={}, inputs={},
            ))
            s.add(AgentWakeupRequest(
                id=f"{lane}-wakeup", agent_id="data", ticket_id=ticket_id,
                status="queued", source="assignment",
                scheduled_for=datetime.now(timezone.utc),
            ))
        await s.commit()

    assert await wd._drain_once("optimization") == 1
    await asyncio.sleep(0)
    assert started == ["optimization-wakeup"]

    wd._inflight.clear()
    started.clear()
    assert await wd._drain_once("held_out_test") == 1
    await asyncio.sleep(0)
    assert started == ["held_out_test-wakeup"]


@pytest.mark.asyncio
async def test_other_lane_running_work_does_not_consume_slots(db, started):
    """The two daemons have independent per-agent slot accounting."""
    async with db() as s:
        s.add(Agent(
            id="evaluation", name="evaluation", title="evaluation",
            identity_path="playbook/agents/evaluation/identity.md",
            max_concurrent_runs=1,
        ))
        for lane, status in (("optimization", "queued"), ("held_out_test", "running")):
            ticket_id = f"evaluation-{lane}"
            s.add(Ticket(
                id=ticket_id, run_id="run-1", agent_id="evaluation",
                status="queued", lane=lane, payload={}, customization={}, inputs={},
            ))
            s.add(AgentWakeupRequest(
                id=f"evaluation-{lane}-wakeup", agent_id="evaluation",
                ticket_id=ticket_id, status=status, source="assignment",
                scheduled_for=datetime.now(timezone.utc),
            ))
        await s.commit()

    assert await wd._drain_once("optimization") == 1
    await asyncio.sleep(0)
    assert started == ["evaluation-optimization-wakeup"]


@pytest.mark.asyncio
async def test_cron_does_not_bypass_a_pending_supervisor_handoff(db):
    """Orphan recovery must not start a child before Orchestrator returns."""
    async with db() as s:
        s.add(Agent(
            id="inference", name="inference", title="inference",
            identity_path="playbook/agents/inference/identity.md",
            max_concurrent_runs=1,
        ))
        child = Ticket(
            id="waiting-child", run_id="run-1", agent_id="inference",
            status="queued", lane="optimization", payload={},
            customization={}, inputs={},
        )
        ordinary = Ticket(
            id="ordinary-child", run_id="run-2", agent_id="inference",
            status="queued", lane="optimization", payload={},
            customization={}, inputs={},
        )
        s.add_all([child, ordinary, TicketNotice(
            ticket_id=child.id,
            code="ticket.awaiting_supervisor_completion",
            severity="info",
            body="waiting for the active supervisor heartbeat",
        )])
        await s.commit()

    await wd._cron_tick(300, "optimization")

    async with db() as s:
        wakes = (await s.execute(
            select(AgentWakeupRequest)
        )).scalars().all()
    assert [w.ticket_id for w in wakes] == [ordinary.id]
