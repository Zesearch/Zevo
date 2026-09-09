from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from zevo.db.models import Base, Run, Ticket, TicketMessage
from zevo.engine.run.failure_policy import MAX_REPAIR_ATTEMPTS, classify_failure
from zevo.engine.run.runner import _apply_ticket_failure_policy


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as value:
        yield value
    await engine.dispose()


def test_failure_ownership_is_deterministic() -> None:
    assert classify_failure(
        "ValidationError: malformed TrainResult", agent_id="train"
    ).route == "self"
    assert classify_failure(
        "inputs unresolvable: upstream ticket has no Work Product", agent_id="train"
    ).route == "orchestrator"
    assert classify_failure(
        "held-out test isolation violated", agent_id="inference"
    ).route == "terminal"


@pytest.mark.asyncio
async def test_three_repairs_stay_on_one_ticket_then_escalate(session: AsyncSession) -> None:
    session.add(Run(
        id="r", task_name="t", agent_objective="t", metric="score", status="running",
    ))
    ticket = Ticket(
        id="train-r-001", run_id="r", agent_id="train", status="running",
        input_format="typed", lane="optimization", iteration=1,
        payload={}, customization={}, inputs={},
    )
    session.add(ticket)
    await session.commit()

    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        scheduled = await _apply_ticket_failure_policy(
            session, ticket, error_message="generated train_config schema mismatch",
        )
        assert scheduled is True
        assert ticket.status == "repairing"
        assert ticket.repair_attempts == attempt

    scheduled = await _apply_ticket_failure_policy(
        session, ticket, error_message="generated train_config schema mismatch",
    )
    await session.commit()
    assert scheduled is False
    assert ticket.status == "failed"
    assert ticket.repair_route == "orchestrator"
    messages = (await session.execute(select(TicketMessage).where(
        TicketMessage.ticket_id == ticket.id,
    ).order_by(TicketMessage.created_at))).scalars().all()
    assert len(messages) == 3
    assert "Repair attempt 1/3" in messages[0].body
    assert "Repair attempt 3/3" in messages[-1].body


@pytest.mark.asyncio
async def test_terminal_failure_never_enters_repairing(session: AsyncSession) -> None:
    ticket = Ticket(id="infer-r-001", run_id="r", agent_id="inference", status="cancelled")
    scheduled = await _apply_ticket_failure_policy(
        session, ticket, error_message="cancelled by user", cancelled=True,
    )
    assert scheduled is False
    assert ticket.status == "cancelled"
    assert ticket.repair_attempts == 0
    assert ticket.repair_route == "terminal"
