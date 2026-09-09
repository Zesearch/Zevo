"""A missing upstream WorkProduct fails the downstream Ticket visibly."""
from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from zevo.db.models import Base, Run, Ticket
from zevo.engine.run.scheduler.bindings import resolve_input_bindings


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_unresolvable_input_is_persisted_as_failed(session: AsyncSession) -> None:
    session.add(Run(metric="accuracy", id="r1", task_name="t", agent_objective="t", status="running"))
    session.add(Ticket(
        id="data-001", run_id="r1", agent_id="data", status="succeeded",
        input_format="typed", lane="optimization", iteration=0,
        payload={}, customization={}, inputs={},
    ))
    train = Ticket(
        id="train-001", run_id="r1", agent_id="train", status="queued",
        input_format="typed", lane="optimization", iteration=1,
        payload={}, customization={}, inputs={
            "training_dataset": {
                "artifact_role": "training_dataset", "source_ticket_id": "data-001",
                "work_product_id": "", "path": "",
            },
        },
    )
    session.add(train)
    await session.commit()

    try:
        await resolve_input_bindings(session, run_id="r1", inputs=train.inputs)
    except ValueError as exc:
        train.status = "failed"
        train.summary = f"inputs unresolvable: {exc}"
        await session.commit()

    refreshed = (await session.execute(select(Ticket).where(Ticket.id == "train-001"))).scalar_one()
    assert refreshed.status == "failed"
    assert "training_dataset" in refreshed.summary
    assert "Work Product" in refreshed.summary
