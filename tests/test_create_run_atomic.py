"""create_run must be atomic: a failure after the Run is inserted (e.g. a
recoverable 400 while deriving the validation split) must leave NO orphan
`planning` Run behind.

The API path calls persistence.create_run(commit=False) so the Run is flushed
(id assigned) but not committed; the rest of creation runs in the same
transaction and commits once. These tests pin that contract at the primitive
level: with commit=False a rollback discards the Run; with the default
commit=True it survives.
"""
from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import Base, Run
from zevo.engine.persistence import create_run as persist_create


async def _session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


_KW = dict(
    task_name="t", run_name="r", task_objective="o", agent_objective="ao",
    metric="accuracy", metric_direction="max",
    validation_metric="accuracy", validation_metric_direction="max",
)


async def _count(db) -> int:
    return (await db.execute(select(func.count()).select_from(Run))).scalar_one()


@pytest.mark.asyncio
async def test_commit_false_rolls_back_on_downstream_failure() -> None:
    Session = await _session()
    async with Session() as db:
        run = await persist_create(db, commit=False, **_KW)
        assert run.id  # id assigned by the flush
        # Simulate a later step raising (split settlement 400, etc.).
        await db.rollback()
    # A fresh session must see zero rows — no orphan planning run.
    async with Session() as db2:
        assert await _count(db2) == 0


@pytest.mark.asyncio
async def test_commit_true_persists() -> None:
    Session = await _session()
    async with Session() as db:
        run = await persist_create(db, commit=True, **_KW)
        rid = run.id
    async with Session() as db2:
        got = await db2.get(Run, rid)
        assert got is not None and got.status == "planning"
