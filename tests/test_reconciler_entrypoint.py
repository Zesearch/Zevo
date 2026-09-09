"""The reconciler's PUBLIC entrypoint must actually run.

Every other reconciler test calls the private sweep helpers directly, so an
import-time or dispatch-level defect in `reconcile_runs_and_tickets` — a
missing import used only inside `_release_run_instances`, a renamed helper —
ships silently: the daemon catches the exception, logs "[reconciler] failed"
every 30 seconds, and no run ever closes. Exactly that happened once (a
`resolve_ssh_key` import landed in the runner and the API but not here), so
the whole pass now gets exercised end to end through the real entrypoint.
"""
from __future__ import annotations

import datetime as dt

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import zevo.db.session as db_session
from zevo.db.models import Base, Run, Ticket, WorkProduct
from zevo.engine.run.scheduler.reconciler import reconcile_runs_and_tickets


def _ago(seconds: float) -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=seconds)


@pytest_asyncio.fixture
async def global_factory(monkeypatch):
    """Point the process-wide session factory at an in-memory DB, because
    `reconcile_runs_and_tickets` opens its own sessions internally."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(db_session, "_SessionLocal", Session)
    yield Session
    await engine.dispose()


@pytest.mark.asyncio
async def test_full_pass_closes_a_finished_run(global_factory) -> None:
    Session = global_factory
    async with Session() as s:
        s.add(Run(metric="accuracy", id="r-done", task_name="t", status="running",
                  started_at=_ago(600)))
        # One failed ticket, long past the stale cutoff — the run should
        # close as failed on this pass.
        s.add(Ticket(id="t-1", run_id="r-done", agent_id="data",
                     status="failed", payload={}, inputs={}))
        await s.commit()

    report = await reconcile_runs_and_tickets(stale_ticket_seconds=1)

    assert report["runs_closed"]["failed"] == 1
    async with Session() as s:
        run = (await s.execute(select(Run).where(Run.id == "r-done"))).scalar_one()
        assert run.status == "failed"
        assert run.finished_at is not None


@pytest.mark.asyncio
async def test_full_pass_is_idempotent_and_quiet_on_empty_db(global_factory) -> None:
    report = await reconcile_runs_and_tickets(stale_ticket_seconds=1)
    assert report["tickets_failed"] == 0
    assert all(v == 0 for v in report["runs_closed"].values())
    # Second pass: still a no-op, still no exception.
    report = await reconcile_runs_and_tickets(stale_ticket_seconds=1)
    assert report["tickets_failed"] == 0
