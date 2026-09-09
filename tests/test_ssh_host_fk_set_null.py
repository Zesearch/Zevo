"""Deleting an SshHost must NULL out runs.ssh_host_id (ON DELETE SET NULL),
not leave a dangling id. Exercised with sqlite FK enforcement turned on.
"""
from __future__ import annotations

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import Base, Run, SshHost


async def _session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    # sqlite ignores FK actions unless PRAGMA foreign_keys=ON per connection.
    @event.listens_for(engine.sync_engine, "connect")
    def _fk_on(dbapi_conn, _rec):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


_RUN_KW = dict(
    task_name="t", task_objective="o", agent_objective="ao",
    metric="accuracy", metric_direction="max",
    validation_metric="accuracy", validation_metric_direction="max",
)


@pytest.mark.asyncio
async def test_delete_ssh_host_nulls_run_ssh_host_id() -> None:
    Session = await _session()
    async with Session() as db:
        # The host exists first (created in an earlier request in production),
        # then a run references it.
        db.add(SshHost(
            id="h1", label="box", host="10.0.0.9", port=22, username="root",
            category="cluster", status="verified", key_path="/k", password_path="",
        ))
        await db.commit()
        db.add(Run(id="r1", gpu_provider="cluster", ssh_host_id="h1", **_RUN_KW))
        await db.commit()

        host = await db.get(SshHost, "h1")
        await db.delete(host)
        await db.commit()

        run = await db.get(Run, "r1")
        await db.refresh(run)
        # The FK action must have cleared the reference, not orphaned it.
        assert run.ssh_host_id is None


@pytest.mark.asyncio
async def test_run_may_have_null_ssh_host_id() -> None:
    Session = await _session()
    async with Session() as db:
        db.add(Run(id="r2", gpu_provider="cloud", **_RUN_KW))  # no ssh_host_id
        await db.commit()
        run = await db.get(Run, "r2")
        assert run.ssh_host_id is None
