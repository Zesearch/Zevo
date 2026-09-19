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
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import (
    Agent, AgentWakeupRequest, Base, InfraInstance, RunInstruction, Ticket, TicketNotice,
)
from zevo.engine.run.scheduler import wakeup_daemon as wd
from zevo.engine.run.wakeup import queue_wakeup


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
async def test_pending_run_instruction_holds_specialist_wakeup(db, started):
    await _seed(db, "train", cap=1, n_queued=0)
    async with db() as s:
        s.add(Ticket(
            id="train-paused", run_id="run-steered", agent_id="train",
            status="queued", lane="optimization", payload={},
            customization={}, inputs={},
        ))
        s.add(RunInstruction(
            id="instruction-1", run_id="run-steered", body="Change the plan",
            status="queued", agent_response="",
        ))
        s.add(AgentWakeupRequest(
            id="train-paused-wakeup", agent_id="train", ticket_id="train-paused",
            status="queued", source="assignment",
            scheduled_for=datetime.now(timezone.utc),
        ))
        await s.commit()

    assert await wd._drain_once() == 0
    await asyncio.sleep(0)
    assert started == []

    async with db() as s:
        instruction = await s.get(RunInstruction, "instruction-1")
        instruction.status = "scheduled"
        await s.commit()

    assert await wd._drain_once() == 1
    await asyncio.sleep(0)
    assert started == ["train-paused-wakeup"]


@pytest.mark.asyncio
async def test_instruction_identity_promotes_the_queued_wakeup(db) -> None:
    await _seed(db, "train", cap=1, n_queued=0)
    async with db() as s:
        s.add(Ticket(
            id="train-directed", run_id="run-directed", agent_id="train",
            status="cancelled", lane="optimization", payload={},
            customization={}, inputs={},
            error_message="current activation superseded by Run instruction instruction-1",
        ))
        s.add(AgentWakeupRequest(
            id="old-assignment", agent_id="train", ticket_id="train-directed",
            status="queued", source="assignment",
            scheduled_for=datetime.now(timezone.utc),
        ))
        await s.commit()
        promoted = await queue_wakeup(
            s,
            agent_id="train",
            ticket_id="train-directed",
            source="on_demand",
            reason="message by orchestrator",
            payload={"run_instruction_id": "instruction-1"},
        )
        assert promoted.id == "old-assignment"
        assert promoted.source == "on_demand"
        assert promoted.payload == {"run_instruction_id": "instruction-1"}


@pytest.mark.asyncio
async def test_new_slurm_job_replaces_stale_queued_collect_payload(db) -> None:
    await _seed(db, "inference", cap=1, n_queued=0)
    async with db() as s:
        s.add(Ticket(
            id="infer-repaired", run_id="run-repaired", agent_id="inference",
            status="queued", lane="optimization", payload={},
            customization={}, inputs={},
        ))
        s.add(AgentWakeupRequest(
            id="old-collect", agent_id="inference", ticket_id="infer-repaired",
            status="queued", source="slurm_watcher", reason="old job ended",
            payload={"job_id": "100"}, scheduled_for=datetime.now(timezone.utc),
        ))
        await s.commit()
        current = await queue_wakeup(
            s,
            agent_id="inference",
            ticket_id="infer-repaired",
            source="slurm_watcher",
            reason="new job ended",
            payload={"job_id": "200", "scheduler_state": "COMPLETED"},
        )
        assert current.id == "old-collect"
        assert current.payload["job_id"] == "200"
        assert "superseded stale Slurm wake for 100" in current.reason


@pytest.mark.asyncio
async def test_instruction_waits_for_old_slurm_release_before_specialist_runs(
    db, monkeypatch,
) -> None:
    await _seed(db, "train", cap=1, n_queued=0)
    async with db() as s:
        s.add(Ticket(
            id="train-replace", run_id="run-replace", agent_id="train",
            status="cancelled", lane="optimization", payload={},
            customization={}, inputs={},
            error_message="current activation superseded by Run instruction instruction-1",
        ))
        s.add(RunInstruction(
            id="instruction-1", run_id="run-replace", body="Use a better plan",
            status="scheduled", agent_response="Applying now",
        ))
        s.add(InfraInstance(
            id="stage-row", provider="cluster", instance_id="12345",
            run_id="run-replace", ticket_id="train-replace",
            status="provisioning", meta={"resource_request": True},
        ))
        s.add(AgentWakeupRequest(
            id="instruction-wake", agent_id="train", ticket_id="train-replace",
            status="queued", source="on_demand", reason="message by orchestrator",
            payload={"run_instruction_id": "instruction-1"},
            scheduled_for=datetime.now(timezone.utc),
        ))
        await s.commit()

    @asynccontextmanager
    async def _lock(*_args, **_kwargs):
        yield True

    calls: list[dict] = []

    async def _run(*_args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(wd, "advisory_lock", _lock)
    monkeypatch.setattr(wd, "run_ticket", _run)
    await wd._run_one("train", "instruction-wake", datetime.now(timezone.utc))
    assert calls == []

    async with db() as s:
        stage = await s.get(InfraInstance, "stage-row")
        stage.released_at = datetime.now(timezone.utc)
        stage.status = "released"
        wake = await s.get(AgentWakeupRequest, "instruction-wake")
        wake.scheduled_for = datetime.now(timezone.utc)
        await s.commit()

    await wd._run_one("train", "instruction-wake", datetime.now(timezone.utc))
    assert calls[0]["activation_payload"] == {"run_instruction_id": "instruction-1"}
    async with db() as s:
        ticket = await s.get(Ticket, "train-replace")
        assert ticket.status == "queued"


@pytest.mark.asyncio
async def test_pending_instruction_does_not_pause_held_out_lane(db, started):
    await _seed(db, "evaluation", cap=1, n_queued=0)
    async with db() as s:
        s.add(Ticket(
            id="private-eval", run_id="run-steered", agent_id="evaluation",
            status="queued", lane="held_out_test", payload={},
            customization={}, inputs={},
        ))
        s.add(RunInstruction(
            id="instruction-private", run_id="run-steered",
            body="Change optimization", status="queued", agent_response="",
        ))
        s.add(AgentWakeupRequest(
            id="private-eval-wakeup", agent_id="evaluation",
            ticket_id="private-eval", status="queued", source="assignment",
            scheduled_for=datetime.now(timezone.utc),
        ))
        await s.commit()

    assert await wd._drain_once("held_out_test") == 1
    await asyncio.sleep(0)
    assert started == ["private-eval-wakeup"]


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
async def test_cancelled_background_work_is_logged_and_frees_its_slot(caplog):
    task = asyncio.create_task(asyncio.sleep(60))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    wd._inflight["data"] = {"wake-cancelled": task}

    with caplog.at_level("ERROR"):
        wd._inflight_done("data", "wake-cancelled", task)

    assert "task cancelled unexpectedly" in caplog.text
    assert wd._inflight.get("data") is None


@pytest.mark.asyncio
async def test_background_exception_is_retrieved_logged_and_frees_slot(caplog):
    async def _boom() -> None:
        raise RuntimeError("worker exploded")

    task = asyncio.create_task(_boom())
    await asyncio.sleep(0)
    wd._inflight["train"] = {"wake-failed": task}

    with caplog.at_level("ERROR"):
        wd._inflight_done("train", "wake-failed", task)

    assert "background task escaped with an exception" in caplog.text
    assert "worker exploded" in caplog.text
    assert wd._inflight.get("train") is None


@pytest.mark.asyncio
async def test_stale_automatic_wakeup_cannot_revive_terminal_specialist(
    db, monkeypatch,
) -> None:
    async with db() as s:
        s.add(Ticket(
            id="data-finished", run_id="run-1", agent_id="data",
            status="succeeded", payload={}, customization={}, inputs={},
        ))
        s.add(AgentWakeupRequest(
            id="stale-collect", agent_id="data", ticket_id="data-finished",
            status="queued", source="slurm_watcher",
            scheduled_for=datetime.now(timezone.utc),
        ))
        await s.commit()

    @asynccontextmanager
    async def _lock(*_args, **_kwargs):
        yield True

    async def _must_not_run(*_args, **_kwargs):
        raise AssertionError("obsolete wakeup revived a terminal specialist")

    monkeypatch.setattr(wd, "advisory_lock", _lock)
    monkeypatch.setattr(wd, "run_ticket", _must_not_run)

    await wd._run_one("data", "stale-collect", datetime.now(timezone.utc))

    async with db() as s:
        ticket = await s.get(Ticket, "data-finished")
        wakeup = await s.get(AgentWakeupRequest, "stale-collect")
        assert ticket.status == "succeeded"
        assert wakeup.status == "completed"
        assert wakeup.finished_at is not None
        assert "skipped obsolete wakeup" in wakeup.reason


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
