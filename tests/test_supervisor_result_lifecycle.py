"""Exercise supervisor output settlement and bounded failure handoff together."""
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.contracts.orchestrator import SupervisorAction
from zevo.db import Base, Run, Ticket, HeartbeatRun, HeartbeatResult, AgentWakeupRequest
from zevo.engine.agent.drivers.base import DriverRunResult
from zevo.engine.run import runner
from zevo.engine.run.activation_failure import persist_activation_failure


@pytest_asyncio.fixture
async def database(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/supervisor.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr("zevo.db.get_session_factory", lambda: factory)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["wait", "emit_ticket", "mark_done", "mark_failed"])
async def test_supervisor_action_settles_without_specialist_status(database, tmp_path, monkeypatch, action):
    supervisor_id = "orchestrate-run-001"
    child_id = "data-run-001"

    class Driver:
        name = "test"

        async def run_agent(self, **kwargs):
            # Model the side effects the coordinator performs through the API
            # before returning its action. The actual runner settles the result.
            async with database() as db:
                run = await db.get(Run, "run")
                if action == "emit_ticket":
                    db.add(Ticket(id=child_id, run_id=run.id, agent_id="data", status="queued"))
                elif action in {"mark_done", "mark_failed"}:
                    run.status = "success" if action == "mark_done" else "failed"
                    run.summary = "Completed decision"
                await db.commit()
            return DriverRunResult(output=SupervisorAction(
                ticket_id=supervisor_id, action=action,
                child_ticket_id=child_id if action == "emit_ticket" else "",
                summary="Coordinator decision",
            ))

    monkeypatch.setattr(runner, "get_driver", lambda _: Driver())
    async with database() as db:
        db.add(Run(id="run", metric="accuracy", status="running", supervisor_ticket_id=supervisor_id))
        db.add(Ticket(id=supervisor_id, run_id="run", agent_id="orchestrator", status="queued", payload={
            "task_objective": "Improve accuracy", "agent_objective": "Improve accuracy",
            "user_request": {
                "task_objective": "Improve accuracy", "metric": "accuracy", "metric_direction": "max",
                "training_method": "", "dataset": "", "base_model": "",
                "test_set": "", "test_sample_submission": "", "constraints": [],
            },
            "trigger": {"type": "run_created"}, "mode": "full_pipeline",
            "budget": {"max_cost_usd": 0, "spent_usd": 0, "remaining_usd": 0},
            "runtime": {"gpu_provider": "instance", "num_gpus": 1, "generation_backend": "vllm"},
        }))
        await db.commit()
        ticket = await runner.run_ticket(db, ticket_id=supervisor_id, work_dir_root=str(tmp_path), driver_name="test")
        assert ticket.status == "succeeded", ticket.error_message
        assert ticket.repair_attempts == 0
        result = (await db.scalars(select(HeartbeatResult))).one()
        assert result.status == "succeeded" and result.output["action"] == action
        hb = (await db.scalars(select(HeartbeatRun))).one()
        assert hb.finished_at is not None and hb.exit_code == 0
        wakes = (await db.scalars(select(AgentWakeupRequest))).all()
        assert [w.ticket_id for w in wakes] == ([child_id] if action == "emit_ticket" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["unchanged", "total_limit"])
async def test_supervisor_failure_cannot_bypass_repair_stop_by_waking_itself(database, stop):
    supervisor_id = "orchestrate-run-001"
    async with database() as db:
        db.add(Run(id="run", metric="accuracy", status="running", supervisor_ticket_id=supervisor_id))
        db.add(Ticket(id=supervisor_id, run_id="run", agent_id="orchestrator", status="running",
                      repair_attempts=20 if stop == "total_limit" else 0))
        await db.commit()
    count = 5 if stop == "unchanged" else 1
    for index in range(count):
        async with database() as db:
            ticket = await db.get(Ticket, supervisor_id)
            ticket.status = "running"
            # Complete the preceding retry exactly as a worker would, so a
            # queued retry cannot accidentally hide a fresh self-handoff.
            for wake in (await db.scalars(select(AgentWakeupRequest))).all():
                wake.status = "completed"
            db.add(HeartbeatRun(id=f"hb-{index}", ticket_id=supervisor_id, agent_id="orchestrator",
                driver="test", model="test", started_at=datetime.now(timezone.utc) + timedelta(seconds=index)))
            await db.commit()
        await persist_activation_failure(database, (supervisor_id, f"hb-{index}"), RuntimeError("same runner failure"))
    async with database() as db:
        ticket = await db.get(Ticket, supervisor_id)
        assert ticket.status == "failed"
        assert ticket.repair_route == "terminal"
        assert ticket.repair_attempts == (4 if stop == "unchanged" else 20)
        wakes = (await db.scalars(select(AgentWakeupRequest))).all()
        assert len(wakes) == count - 1
        assert all(w.source == "retry" and w.status == "completed" for w in wakes)
        # Repeated terminal failure handling and direct completion notifications
        # must leave both the failure and the empty pending queue intact.
        await runner._maybe_wake_supervisor(db, ticket)
        await db.refresh(ticket)
        assert ticket.status == "failed"
        assert not (await db.scalars(select(AgentWakeupRequest).where(AgentWakeupRequest.status == "queued"))).all()
