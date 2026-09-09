"""Rows written later in one transaction receive later application timestamps."""
from __future__ import annotations

import asyncio
import datetime as dt

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import Base, Run, Ticket, TicketMessage, WorkProduct, _utcnow


@pytest_asyncio.fixture
async def Session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


def test_timestamp_default_is_callable() -> None:
    a, b = _utcnow(), _utcnow()
    assert a.tzinfo is not None and b >= a


async def _ticket(session, ticket_id: str) -> None:
    session.add(Run(metric="accuracy", id=f"run-{ticket_id}", agent_objective="stamps", status="running"))
    session.add(Ticket(
        id=ticket_id, run_id=f"run-{ticket_id}", agent_id="train", status="running",
        input_format="typed", lane="optimization", iteration=1,
        payload={}, customization={}, inputs={},
    ))
    await session.commit()


@pytest.mark.asyncio
async def test_messages_keep_insert_order(Session) -> None:
    async with Session() as session:
        await _ticket(session, "train-001")
    async with Session() as session:
        session.add(TicketMessage(ticket_id="train-001", author="train", body="Starting"))
        await session.flush()
        await asyncio.sleep(0.05)
        session.add(TicketMessage(ticket_id="train-001", author="train", body="Done"))
        await session.commit()
    async with Session() as session:
        rows = (await session.execute(
            select(TicketMessage).where(TicketMessage.ticket_id == "train-001")
            .order_by(TicketMessage.created_at)
        )).scalars().all()
    assert [row.body for row in rows] == ["Starting", "Done"]
    assert rows[1].created_at > rows[0].created_at


@pytest.mark.asyncio
async def test_work_product_timestamp_is_end_time(Session) -> None:
    async with Session() as session:
        await _ticket(session, "train-002")
    opened = dt.datetime.now(dt.timezone.utc)
    async with Session() as session:
        session.add(TicketMessage(ticket_id="train-002", author="train", body="Starting"))
        await session.flush()
        await asyncio.sleep(0.05)
        product = WorkProduct(ticket_id="train-002", role="checkpoint", path="/remote/model", meta={})
        session.add(product)
        await session.commit()
    created = product.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=dt.timezone.utc)
    assert created > opened
