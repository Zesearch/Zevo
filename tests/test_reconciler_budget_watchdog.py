"""The reconciler must enforce a run's cost / runtime / iteration caps as a
backstop when the orchestrator doesn't stop itself: over-cap runs are halted and
their in-flight tickets failed + remote work reaped; under-cap / uncapped runs
are left alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import zevo.engine.cost.budget as budget_mod
import zevo.engine.run.remote_jobs as remote_jobs
from zevo.db.models import Base, Run, Ticket
from zevo.engine.run.scheduler.reconciler import _watchdog_halt_over_budget_runs


_ENGINES = []


@pytest_asyncio.fixture(autouse=True)
async def _close_test_engines():
    yield
    while _ENGINES:
        await _ENGINES.pop().dispose()


async def _session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    _ENGINES.append(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


@dataclass
class _Snap:
    over_budget: bool = False
    over_time_limit: bool = False
    spent_usd: float = 0.0
    elapsed_runtime_hours: float = 0.0


def _patch_snapshot(monkeypatch, snap: _Snap) -> None:
    async def fake(session, run_id):
        return snap
    monkeypatch.setattr(budget_mod, "snapshot_for_run", fake)


def _reap_spy(monkeypatch) -> list[str]:
    reaped: list[str] = []
    async def fake(session, tickets):
        reaped.extend(t.id for t in tickets)
        return len(tickets)
    monkeypatch.setattr(remote_jobs, "cancel_run_remote_jobs", fake)
    return reaped


_RUN = dict(
    task_name="t", task_objective="o", agent_objective="ao",
    metric="accuracy", metric_direction="max",
    validation_metric="accuracy", validation_metric_direction="max",
    status="running",
)


async def _seed(db, **run_over):
    run = Run(id="r1", **{**_RUN, **run_over})
    db.add(run)
    db.add(Ticket(id="train-1", run_id="r1", agent_id="train", status="running",
                  payload={}, summary="", error_message=""))
    db.add(Ticket(id="orch-1", run_id="r1", agent_id="orchestrator", status="running",
                  payload={}, summary="", error_message=""))
    await db.commit()


@pytest.mark.asyncio
async def test_over_cost_stops_training_but_permits_finalization(monkeypatch) -> None:
    _patch_snapshot(monkeypatch, _Snap(over_budget=True, spent_usd=12.0))
    reaped = _reap_spy(monkeypatch)
    Session = await _session()
    async with Session() as db:
        await _seed(db, max_cost_usd=10.0)
        n = await _watchdog_halt_over_budget_runs(db)
        assert n == 1
        run = await db.get(Run, "r1")
        assert run.status == "running" and run.halted_reason == ""
        assert run.lifecycle["finalization"]["stop_trigger"]["code"] == "cost_limit"
        assert run.lifecycle["finalization"]["deadline_at"]
        assert (await db.get(Ticket, "train-1")).status == "failed"
        assert (await db.get(Ticket, "orch-1")).status == "running"
        assert "train-1" in reaped  # orphaned remote work reaped


@pytest.mark.asyncio
async def test_iteration_cap_starts_bounded_finalization(monkeypatch) -> None:
    _patch_snapshot(monkeypatch, _Snap())  # under cost/time
    _reap_spy(monkeypatch)
    Session = await _session()
    async with Session() as db:
        await _seed(db, iteration_budget=3, iterations_completed=3)
        n = await _watchdog_halt_over_budget_runs(db)
        assert n == 1
        run = await db.get(Run, "r1")
        assert run.status == "running" and run.halted_reason == ""
        assert run.lifecycle["finalization"]["stop_trigger"] == {
            "code": "iteration_limit",
            "current": 3,
            "limit": 3,
            "message": "Stopped after completing 3 of 3 iterations.",
        }
        assert run.lifecycle["finalization"]["deadline_at"]


@pytest.mark.asyncio
async def test_under_all_caps_left_running(monkeypatch) -> None:
    _patch_snapshot(monkeypatch, _Snap())
    _reap_spy(monkeypatch)
    Session = await _session()
    async with Session() as db:
        await _seed(db, max_cost_usd=10.0, max_runtime_hours=5.0,
                    iteration_budget=10, iterations_completed=2)
        n = await _watchdog_halt_over_budget_runs(db)
        assert n == 0
        assert (await db.get(Run, "r1")).status == "running"


@pytest.mark.asyncio
async def test_no_caps_never_halts(monkeypatch) -> None:
    # Even if a snapshot somehow reports over-budget, an uncapped run stands.
    _patch_snapshot(monkeypatch, _Snap(over_budget=True, over_time_limit=True))
    _reap_spy(monkeypatch)
    Session = await _session()
    async with Session() as db:
        await _seed(db)  # no caps set (all default 0)
        n = await _watchdog_halt_over_budget_runs(db)
        assert n == 0
        assert (await db.get(Run, "r1")).status == "running"


@pytest.mark.asyncio
async def test_cap_does_not_kill_final_test_or_registry(monkeypatch):
    _patch_snapshot(monkeypatch, _Snap())
    reaped = _reap_spy(monkeypatch)
    Session = await _session()
    async with Session() as db:
        await _seed(db, iteration_budget=1, iterations_completed=1)
        db.add(Ticket(id="test",run_id="r1",agent_id="inference",lane="held_out_test",status="running",payload={}))
        db.add(Ticket(id="registry",run_id="r1",agent_id="registry",status="queued",payload={}))
        await db.commit()
        await _watchdog_halt_over_budget_runs(db)
        assert (await db.get(Ticket,"test")).status == "running"
        assert (await db.get(Ticket,"registry")).status == "queued"
        assert "test" not in reaped and "registry" not in reaped
        assert (await db.get(Run,"r1")).cancel_requested_at is None
    await Session.kw["bind"].dispose()


@pytest.mark.asyncio
async def test_finalization_deadline_enters_preservation_before_cleanup(monkeypatch):
    from datetime import datetime,timedelta,timezone
    _patch_snapshot(monkeypatch, _Snap())
    _reap_spy(monkeypatch)
    Session = await _session()
    async with Session() as db:
        await _seed(db, iteration_budget=1, iterations_completed=1,
                    lifecycle={"finalization":{"reason":"iteration limit", "deadline_at":(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()}})
        await _watchdog_halt_over_budget_runs(db)
        run = await db.get(Run,"r1")
        assert run.status == "running"  # resource cleanup must wait for retention
        assert run.cancel_requested_at is not None
        assert run.cancel_policy["weights"] == "download"
        assert run.lifecycle["rescue_terminal_status"] == "failed"
        assert run.lifecycle["finalization"]["expired_at"]
        assert (await db.get(Ticket,"orch-1")).status == "failed"
    await Session.kw["bind"].dispose()


@pytest.mark.asyncio
async def test_failed_tickets_cannot_skip_final_checkpoint_retention(monkeypatch):
    from zevo.engine.run.scheduler.reconciler import _close_finished_runs
    from datetime import datetime, timedelta, timezone
    Session = await _session()
    async with Session() as db:
        run=Run(id="limited",**_RUN,lifecycle={"finalization":{
            "reason":"limit", "deadline_at":(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat()}})
        db.add(run)
        db.add(Ticket(id="only-train",run_id="limited",agent_id="train",status="failed",payload={}))
        await db.commit()
        counts=await _close_finished_runs(db)
        assert not any(counts.values())
        assert run.status == "running"


def _final_state(deadline_delta, started_delta=timedelta(minutes=30)):
    now = datetime.now(timezone.utc)
    return {"finalization": {"reason": "iteration limit",
                             "started_at": (now - started_delta).isoformat(),
                             "deadline_at": (now + deadline_delta).isoformat()}}


@pytest.mark.asyncio
async def test_deadline_holds_while_final_test_is_in_flight(monkeypatch):
    # A champion held-out pass can outlast the window on its own; the deadline
    # must not reap it (or the queued registry behind it) mid-flight.
    _patch_snapshot(monkeypatch, _Snap())
    reaped = _reap_spy(monkeypatch)
    Session = await _session()
    async with Session() as db:
        await _seed(db, iteration_budget=1, iterations_completed=1,
                    lifecycle=_final_state(timedelta(seconds=-1)))
        db.add(Ticket(id="test", run_id="r1", agent_id="inference", lane="held_out_test",
                      status="running", payload={}))
        db.add(Ticket(id="registry", run_id="r1", agent_id="registry", status="queued", payload={}))
        await db.commit()
        await _watchdog_halt_over_budget_runs(db)
        run = await db.get(Run, "r1")
        assert run.cancel_requested_at is None
        assert (await db.get(Ticket, "test")).status == "running"
        assert (await db.get(Ticket, "registry")).status == "queued"
        assert (await db.get(Ticket, "orch-1")).status == "running"
        assert "test" not in reaped and "registry" not in reaped


@pytest.mark.asyncio
async def test_completed_final_step_restarts_the_window(monkeypatch):
    # Held-out eval finished a minute ago, past the original deadline: the
    # orchestrator still gets a full window to register the champion.
    _patch_snapshot(monkeypatch, _Snap())
    _reap_spy(monkeypatch)
    Session = await _session()
    async with Session() as db:
        await _seed(db, iteration_budget=1, iterations_completed=1,
                    lifecycle=_final_state(timedelta(seconds=-1)))
        db.add(Ticket(id="holdout-eval", run_id="r1", agent_id="evaluation", lane="held_out_test",
                      status="succeeded", payload={},
                      updated_at=datetime.now(timezone.utc) - timedelta(minutes=1)))
        await db.commit()
        await _watchdog_halt_over_budget_runs(db)
        run = await db.get(Run, "r1")
        assert run.cancel_requested_at is None
        new_deadline = datetime.fromisoformat(run.lifecycle["finalization"]["deadline_at"])
        assert new_deadline > datetime.now(timezone.utc) + timedelta(minutes=13)
        assert (await db.get(Ticket, "orch-1")).status == "running"


@pytest.mark.asyncio
async def test_stale_final_step_does_not_extend_the_window(monkeypatch):
    # The last finalization step finished well over a window ago and nothing
    # is in flight: bounded rescue proceeds as before.
    _patch_snapshot(monkeypatch, _Snap())
    _reap_spy(monkeypatch)
    Session = await _session()
    async with Session() as db:
        await _seed(db, iteration_budget=1, iterations_completed=1,
                    lifecycle=_final_state(timedelta(seconds=-1)))
        db.add(Ticket(id="holdout-eval", run_id="r1", agent_id="evaluation", lane="held_out_test",
                      status="succeeded", payload={},
                      updated_at=datetime.now(timezone.utc) - timedelta(minutes=20)))
        await db.commit()
        await _watchdog_halt_over_budget_runs(db)
        run = await db.get(Run, "r1")
        assert run.cancel_requested_at is not None
        assert run.lifecycle["rescue_terminal_status"] == "failed"
        assert (await db.get(Ticket, "orch-1")).status == "failed"
