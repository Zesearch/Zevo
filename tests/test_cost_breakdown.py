from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import Agent, Base, HeartbeatRun, InfraInstance, Run, Ticket
from zevo.engine.cost.budget import cost_breakdown_for_run


async def _session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


@pytest.mark.asyncio
async def test_cost_breakdown_groups_llm_by_agent_and_sums_to_total() -> None:
    engine, Session = await _session()
    async with Session() as db:
        for role in ("orchestrator", "train", "data"):
            db.add(Agent(
                id=role, name=role, title=role, reports_to="orchestrator",
                identity_path=f"playbook/agents/{role}/identity.md",
            ))
        run = Run(id="cb-run", metric="accuracy", metric_direction="max")
        orch = Ticket(id="t-orch", run_id=run.id, agent_id="orchestrator", iteration=0)
        train = Ticket(id="t-train", run_id=run.id, agent_id="train", iteration=1)
        data = Ticket(id="t-data", run_id=run.id, agent_id="data", iteration=1)
        db.add_all([run, orch, train, data])
        db.add_all([
            # Orchestrator spends across two heartbeats — must be summed.
            HeartbeatRun(id="hb-o1", ticket_id=orch.id, agent_id="orchestrator",
                         driver="stub", model="stub", estimated_cost_usd=60.0),
            HeartbeatRun(id="hb-o2", ticket_id=orch.id, agent_id="orchestrator",
                         driver="stub", model="stub", estimated_cost_usd=34.92),
            HeartbeatRun(id="hb-t1", ticket_id=train.id, agent_id="train",
                         driver="stub", model="stub", estimated_cost_usd=5.0),
            HeartbeatRun(id="hb-d1", ticket_id=data.id, agent_id="data",
                         driver="stub", model="stub", estimated_cost_usd=3.0),
        ])
        await db.commit()

        cb = await cost_breakdown_for_run(db, run.id)

        by_agent = {a.agent_id: a for a in cb.agents}
        assert by_agent["orchestrator"].cost_usd == pytest.approx(94.92)
        assert by_agent["orchestrator"].heartbeats == 2
        assert by_agent["train"].cost_usd == pytest.approx(5.0)
        assert by_agent["data"].cost_usd == pytest.approx(3.0)

        # Pipeline order, not alphabetical: orchestrator, data, then train.
        assert [a.agent_id for a in cb.agents] == ["orchestrator", "data", "train"]

        # Per-agent rows add up to the LLM total, and with no rented GPU the
        # total is exactly the agent cost.
        assert cb.agent_cost_usd == pytest.approx(94.92 + 5.0 + 3.0)
        assert cb.gpu_cost_usd == 0.0
        assert cb.total_cost_usd == pytest.approx(cb.agent_cost_usd)
        assert sum(a.cost_usd for a in cb.agents) == pytest.approx(cb.agent_cost_usd)

    await engine.dispose()


@pytest.mark.asyncio
async def test_cost_breakdown_includes_rented_gpu_in_total() -> None:
    engine, Session = await _session()
    now = datetime.now(timezone.utc)
    async with Session() as db:
        db.add(Agent(
            id="train", name="train", title="train", reports_to="orchestrator",
            identity_path="playbook/agents/train/identity.md",
        ))
        run = Run(id="cb-gpu", metric="accuracy", metric_direction="max")
        train = Ticket(id="t-g", run_id=run.id, agent_id="train", iteration=1)
        db.add_all([run, train])
        db.add(HeartbeatRun(
            id="hb-g", ticket_id=train.id, agent_id="train",
            driver="stub", model="stub", estimated_cost_usd=10.0,
        ))
        # Rented box: 2h at $2/h -> $4 GPU. `cluster`/`instance` would be free.
        db.add(InfraInstance(
            id="inst-g", run_id=run.id, provider="cloud", dph=2.0,
            gpu_name="A100", gpu_count=1,
            created_at=now - timedelta(hours=2), released_at=now,
        ))
        await db.commit()

        cb = await cost_breakdown_for_run(db, run.id)
        assert cb.agent_cost_usd == pytest.approx(10.0)
        assert cb.gpu_cost_usd == pytest.approx(4.0, abs=1e-3)
        assert cb.total_cost_usd == pytest.approx(14.0, abs=1e-3)

    await engine.dispose()


@pytest.mark.asyncio
async def test_cost_breakdown_empty_run_is_zeroed() -> None:
    engine, Session = await _session()
    async with Session() as db:
        run = Run(id="cb-empty", metric="accuracy", metric_direction="max")
        db.add(run)
        await db.commit()

        cb = await cost_breakdown_for_run(db, run.id)
        assert cb.agents == []
        assert cb.agent_cost_usd == 0.0
        assert cb.gpu_cost_usd == 0.0
        assert cb.total_cost_usd == 0.0

    await engine.dispose()
