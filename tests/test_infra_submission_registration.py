"""A registered Slurm job is observable before its submit Result returns."""
from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.routers.agent.infra import create_instance
from zevo.contracts.infrastructure import CreateInfraInstanceBody
from zevo.db.models import Base, HeartbeatRun, Run, Ticket


@pytest.mark.asyncio
async def test_resource_request_registration_stamps_observation_not_handoff() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        db.add(Run(
            id="registration-run", task_name="suite", status="running",
            metric="accuracy", metric_direction="max",
        ))
        db.add(Ticket(
            id="registration-infer", run_id="registration-run",
            agent_id="inference", status="running", payload={}, inputs={},
        ))
        db.add(HeartbeatRun(
            id="registration-heartbeat", ticket_id="registration-infer",
            agent_id="inference", driver="codex_cli", model="test",
            activation_phase="submit",
        ))
        await db.commit()

        result = await create_instance(CreateInfraInstanceBody(
            instance_id="42390534", provider="cluster",
            run_id="registration-run", ticket_id="registration-infer",
            meta={
                "resource_request": True,
                "status_path": "/remote/registration-infer/.zevo-slurm-status",
                "remote_workdir": "/remote/registration-infer",
                "scheduler_state": "PENDING",
            },
        ), db)
        assert result.meta["submission_heartbeat_id"] == "registration-heartbeat"
        assert result.meta["log_path"] == "/remote/registration-infer/slurm-42390534.out"
        assert result.meta["stderr_path"] == "/remote/registration-infer/slurm-42390534.err"
        assert "submission_committed" not in result.meta
    await engine.dispose()
