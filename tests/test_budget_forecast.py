from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import Agent, Base, HeartbeatRun, InfraInstance, Run, Ticket
from zevo.engine.cost.budget import snapshot_for_run


@pytest.mark.asyncio
async def test_next_iteration_budget_forecast_is_advisory_and_preemptive() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async with Session() as db:
        db.add(Agent(
            id="train", name="Train", title="Train", reports_to="orchestrator",
            identity_path="playbook/agents/train/identity.md",
        ))
        run = Run(
            id="budget-run", metric="accuracy", metric_direction="max",
            max_cost_usd=50.0, iterations_completed=1,
        )
        baseline = Ticket(
            id="infer-budget-000", run_id=run.id, agent_id="train", iteration=0,
        )
        iteration = Ticket(
            id="train-budget-001", run_id=run.id, agent_id="train", iteration=1,
        )
        db.add_all([run, baseline, iteration])
        db.add_all([
            HeartbeatRun(
                id="hb-budget-000", ticket_id=baseline.id, agent_id="train",
                driver="stub", model="stub", estimated_cost_usd=10.0,
            ),
            HeartbeatRun(
                id="hb-budget-001", ticket_id=iteration.id, agent_id="train",
                driver="stub", model="stub", estimated_cost_usd=15.0,
            ),
        ])
        await db.commit()

        snapshot = await snapshot_for_run(db, run.id)
        assert snapshot.spent_usd == 25.0
        assert snapshot.remaining_usd == 25.0
        assert snapshot.projected_next_iteration_usd == 15.0
        assert snapshot.can_afford_next_iteration is True
        assert snapshot.over_budget is False

        # The advisory turns false before the hard cap is reached, giving the
        # Orchestrator a chance to stop cleanly with the retained model.
        run.max_cost_usd = 35.0
        await db.commit()
        snapshot = await snapshot_for_run(db, run.id)
        assert snapshot.remaining_usd == 10.0
        assert snapshot.can_afford_next_iteration is False
        assert snapshot.over_budget is False

    await engine.dispose()


@pytest.mark.asyncio
async def test_next_iteration_time_forecast_uses_completed_rounds() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    now = datetime.now(timezone.utc)
    async with Session() as db:
        db.add(Agent(
            id="train", name="Train", title="Train", reports_to="orchestrator",
            identity_path="playbook/agents/train/identity.md",
        ))
        run = Run(
            id="timed-run", metric="accuracy", metric_direction="max",
            max_runtime_hours=10.0, iterations_completed=2,
            started_at=now - timedelta(hours=5),
        )
        first = Ticket(
            id="train-timed-001", run_id=run.id, agent_id="train", iteration=1,
        )
        second = Ticket(
            id="train-timed-002", run_id=run.id, agent_id="train", iteration=2,
        )
        db.add_all([run, first, second])
        db.add_all([
            HeartbeatRun(
                id="hb-timed-001", ticket_id=first.id, agent_id="train",
                driver="stub", model="stub",
                started_at=now - timedelta(hours=4),
                finished_at=now - timedelta(hours=3),
            ),
            HeartbeatRun(
                id="hb-timed-002", ticket_id=second.id, agent_id="train",
                driver="stub", model="stub",
                started_at=now - timedelta(hours=2),
                finished_at=now,
            ),
        ])
        await db.commit()

        snapshot = await snapshot_for_run(db, run.id)
        assert snapshot.elapsed_runtime_hours == pytest.approx(5.0, abs=0.01)
        assert snapshot.remaining_runtime_hours == pytest.approx(5.0, abs=0.01)
        assert snapshot.projected_next_iteration_hours == pytest.approx(2.0)
        assert snapshot.can_finish_next_iteration_in_time is True
        assert snapshot.over_time_limit is False

        run.max_runtime_hours = 6.0
        await db.commit()
        snapshot = await snapshot_for_run(db, run.id)
        assert snapshot.remaining_runtime_hours == pytest.approx(1.0, abs=0.01)
        assert snapshot.can_finish_next_iteration_in_time is False
        assert snapshot.over_time_limit is False

    await engine.dispose()


@pytest.mark.asyncio
async def test_cluster_queue_wait_is_excluded_from_run_runtime() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    now = datetime.now(timezone.utc)
    async with Session() as db:
        run = Run(
            id="queued-run", metric="accuracy", metric_direction="max",
            max_runtime_hours=4.0, max_queue_wait_hours=8.0,
            started_at=now - timedelta(hours=5),
        )
        db.add(run)
        db.add(InfraInstance(
            id="queued-instance", instance_id="12345", provider="cluster",
            status="ready", run_id=run.id,
            created_at=now - timedelta(hours=4),
            ready_at=now - timedelta(hours=2),
        ))
        await db.commit()

        snapshot = await snapshot_for_run(db, run.id)
        assert snapshot.queue_wait_hours == pytest.approx(2.0, abs=0.01)
        assert snapshot.elapsed_runtime_hours == pytest.approx(3.0, abs=0.01)
        assert snapshot.remaining_runtime_hours == pytest.approx(1.0, abs=0.01)
        assert snapshot.max_queue_wait_hours == 8.0
        assert snapshot.over_time_limit is False

    await engine.dispose()
