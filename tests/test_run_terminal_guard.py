"""run_ticket must no-op when the run is already terminal.

A wakeup can outlive its run (reconciler-queued supervisor wakeups don't flip
the ticket status; cancel/delete mark the run terminal without clearing queued
wakeups). Draining such a wakeup must NOT run an orchestrator heartbeat against
a cancelled/finished run.
"""
from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.contracts.tickets import TERMINAL_RUN_STATUSES
from zevo.db.models import Base, HeartbeatRun, Run, Ticket
from zevo.engine.run.runner import run_ticket


async def _session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", sorted(TERMINAL_RUN_STATUSES))
async def test_run_ticket_noops_on_terminal_run(terminal_status: str) -> None:
    Session = await _session()
    async with Session() as db:
        db.add(Run(
            id="r1", task_name="t", status=terminal_status, metric="accuracy",
            metric_direction="max", agent_objective="", summary="",
            halted_reason="", registry_version_tag="",
        ))
        db.add(Ticket(
            id="orch-1", run_id="r1", agent_id="orchestrator", status="queued",
            payload={}, summary="", error_message="",
        ))
        await db.commit()

        # Must return immediately without executing anything.
        result = await run_ticket(db, ticket_id="orch-1")

        assert result.id == "orch-1"
        assert result.status == "queued"  # untouched — no execution occurred
        hbs = (await db.execute(select(HeartbeatRun))).scalars().all()
        assert hbs == []  # no heartbeat was created
