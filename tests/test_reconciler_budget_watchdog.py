"""The reconciler must enforce a run's cost / runtime / iteration caps as a
backstop when the orchestrator doesn't stop itself: over-cap runs are halted and
their in-flight tickets failed + remote work reaped; under-cap / uncapped runs
are left alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import zevo.engine.cost.budget as budget_mod
import zevo.engine.run.remote_jobs as remote_jobs
from zevo.db.models import Base, Run, Ticket
from zevo.engine.run.scheduler.reconciler import _watchdog_halt_over_budget_runs


async def _session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
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
async def test_over_cost_halts_and_reaps(monkeypatch) -> None:
    _patch_snapshot(monkeypatch, _Snap(over_budget=True, spent_usd=12.0))
    reaped = _reap_spy(monkeypatch)
    Session = await _session()
    async with Session() as db:
        await _seed(db, max_cost_usd=10.0)
        n = await _watchdog_halt_over_budget_runs(db)
        assert n == 1
        run = await db.get(Run, "r1")
        assert run.status == "halted" and "cost" in run.halted_reason
        assert (await db.get(Ticket, "train-1")).status == "failed"
        assert (await db.get(Ticket, "orch-1")).status == "failed"
        assert "train-1" in reaped  # orphaned remote work reaped


@pytest.mark.asyncio
async def test_over_iterations_halts(monkeypatch) -> None:
    _patch_snapshot(monkeypatch, _Snap())  # under cost/time
    _reap_spy(monkeypatch)
    Session = await _session()
    async with Session() as db:
        await _seed(db, iteration_budget=3, iterations_completed=3)
        n = await _watchdog_halt_over_budget_runs(db)
        assert n == 1
        run = await db.get(Run, "r1")
        assert run.status == "halted" and "iterations" in run.halted_reason


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
