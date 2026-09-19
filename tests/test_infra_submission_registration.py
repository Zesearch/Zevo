"""Agents cannot register a finite job before their submit Result validates."""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.routers.agent.infra import create_instance, list_instances
from zevo.contracts.infrastructure import CreateInfraInstanceBody
from zevo.db.models import Base, InfraInstance, Run, Ticket


@pytest.mark.asyncio
async def test_resource_request_registration_is_engine_owned() -> None:
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
        await db.commit()

        with pytest.raises(HTTPException, match="submitted and registered by the engine"):
            await create_instance(CreateInfraInstanceBody(
                instance_id="42390534", provider="cluster",
                run_id="registration-run", ticket_id="registration-infer",
                meta={
                    "resource_request": True,
                    "status_path": "/remote/registration-infer/.zevo-slurm-status",
                    "remote_workdir": "/remote/registration-infer",
                    "scheduler_state": "PENDING",
                },
            ), db)
        assert (await db.execute(select(InfraInstance))).scalars().all() == []
        assert await list_instances(active_only=False, run_id="registration-run", db=db) == []
    await engine.dispose()
