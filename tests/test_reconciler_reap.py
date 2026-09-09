"""The reconciler's stuck-ticket sweep must reap orphaned remote work.

When a runner crashes mid-training (SSH drop / process death), the sweep marks
the ticket failed — but the remote trainer can keep holding GPUs on a
cluster/instance allocation. The sweep must call cancel_run_remote_jobs for the
swept tickets so the orphaned step is killed.
"""
from __future__ import annotations

import datetime as _dt
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import zevo.engine.run.remote_jobs as remote_jobs
from zevo.db.models import Base, Ticket
from zevo.engine.run.scheduler.reconciler import _sweep_stuck_tickets


async def _session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.mark.asyncio
async def test_sweep_reaps_remote_jobs_for_crashed_ticket(monkeypatch) -> None:
    reaped_ids: list[str] = []

    async def fake_reap(session, tickets):
        reaped_ids.extend(t.id for t in tickets)
        return len(tickets)

    monkeypatch.setattr(remote_jobs, "cancel_run_remote_jobs", fake_reap)

    Session = await _session()
    async with Session() as db:
        stale = datetime.now(timezone.utc) - _dt.timedelta(hours=1)
        # running train ticket, no heartbeat, stale updated_at → "runner gone"
        db.add(Ticket(
            id="train-1", run_id="r1", agent_id="train", status="running",
            payload={}, summary="", error_message="", updated_at=stale,
        ))
        await db.commit()

        swept = await _sweep_stuck_tickets(db, stale_ticket_seconds=60)

        assert swept == 1
        tk = await db.get(Ticket, "train-1")
        assert tk.status == "failed"
        # the orphaned remote step must have been reaped
        assert "train-1" in reaped_ids
