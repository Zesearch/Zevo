"""PATCH /runs/{id}/policy must not 500 (regression for the max_queue_wait_hours
KeyError) and must persist the change + write a coherent audit diff.
"""
from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.routers.shared.runs import PolicyPatchBody, patch_run_policy
from zevo.db.models import AuditEvent, Base, Run


async def _session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.mark.asyncio
async def test_policy_patch_persists_and_audits_without_500() -> None:
    Session = await _session()
    async with Session() as db:
        db.add(Run(
            id="r1", task_name="t", status="running", metric="accuracy",
            iteration_budget=3, max_cost_usd=5.0, max_queue_wait_hours=0.0,
        ))
        await db.commit()

        # Previously raised KeyError('max_queue_wait_hours') → 500.
        resp = await patch_run_policy(
            "r1",
            PolicyPatchBody(iteration_budget=7, max_queue_wait_hours=2.0),
            db,
        )
        assert resp is not None

        run = await db.get(Run, "r1")
        assert run.iteration_budget == 7
        assert float(run.max_queue_wait_hours) == 2.0

        # The audit diff must record both changed fields.
        ev = (await db.execute(
            select(AuditEvent).where(AuditEvent.event_type == "run.policy_patch")
        )).scalars().first()
        assert ev is not None
        assert "iteration_budget" in ev.after
        assert "max_queue_wait_hours" in ev.after
