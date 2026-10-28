"""Data recovery must not consume an iteration before any Train is created."""
import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.routers.shared.tickets import (
    _guard_data_revision_before_train, _optimization_data_tickets,
)
from zevo.db.models import Base, Run, Ticket


def ticket(tid, agent, iteration, status, run_id="r"):
    return Ticket(id=tid, run_id=run_id, agent_id=agent, iteration=iteration,
                  status=status, lane="optimization", input_format="typed",
                  payload={"operation": "prepare_run_data" if agent == "data" else "train"})


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        session.add_all([Run(id="r", metric="accuracy"), Run(id="other", metric="accuracy")])
        session.add_all([ticket("base", "data", 0, "succeeded"),
                         ticket("previous-train", "train", 3, "succeeded"),
                         ticket("partial-data", "data", 4, "succeeded")])
        await session.commit()
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_partial_data_can_be_replaced_in_same_untrained_iteration(db):
    # Reproduces 7d9dfe58: succeeded base-only Data at iteration 4, Train only at 3.
    db.add(ticket("unrelated-train", "train", 4, "succeeded", run_id="other"))
    await db.commit()
    await _guard_data_revision_before_train(db, "r", 4, await _optimization_data_tickets(db, "r"))
    db.add(ticket("replacement-data", "data", 4, "succeeded"))
    await db.commit()
    rows = await _optimization_data_tickets(db, "r")
    assert {row.id for row in rows if row.iteration == 4} == {"partial-data", "replacement-data"}
    await _guard_data_revision_before_train(db, "r", 4, rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["queued", "running", "repairing", "waiting_external", "awaiting_input"])
async def test_active_data_cannot_be_replaced_concurrently(db, status):
    db.add(ticket("active-data", "data", 4, status))
    await db.commit()
    with pytest.raises(HTTPException) as exc:
        await _guard_data_revision_before_train(db, "r", 4, await _optimization_data_tickets(db, "r"))
    assert exc.value.status_code == 409
    assert exc.value.detail["ticket_id"] == "active-data"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["queued", "running", "succeeded", "failed", "cancelled"])
async def test_train_freezes_iteration_data_even_if_not_running(db, status):
    db.add(ticket("bound-train", "train", 4, status))
    await db.commit()
    with pytest.raises(HTTPException) as exc:
        await _guard_data_revision_before_train(db, "r", 4, await _optimization_data_tickets(db, "r"))
    assert exc.value.status_code == 409
    assert exc.value.detail["ticket_id"] == "bound-train"
