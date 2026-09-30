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
    assert classify_failure(
        "engine could not prepare frozen Validation artifacts: "
        "Validation sample submission must be an absolute path",
        agent_id="data",
    ).route == "terminal"


@pytest.mark.asyncio
async def test_ten_repairs_stay_on_one_ticket_then_escalate(session: AsyncSession) -> None:
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
    assert len(messages) == 10
    assert "Repair attempt 1/10" in messages[0].body
    assert "Repair attempt 10/10" in messages[-1].body


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


@pytest.mark.asyncio
async def test_engine_scoring_failure_does_not_retry_data_agent(
    session: AsyncSession,
) -> None:
    ticket = Ticket(
        id="data-r-001", run_id="r", agent_id="data", status="running",
        repair_attempts=0,
    )
    scheduled = await _apply_ticket_failure_policy(
        session, ticket,
        error_message=(
            "engine could not prepare frozen Validation artifacts: "
            "Validation sample submission does not exist"
        ),
    )
    assert scheduled is False
    assert ticket.status == "failed"
    assert ticket.repair_attempts == 0
    assert ticket.repair_route == "terminal"


@pytest.mark.asyncio
@pytest.mark.parametrize('agent', ['data', 'train', 'inference'])
async def test_failed_workload_enters_one_repair_without_collect(session, agent):
    from zevo.db import InfraInstance
    from zevo.engine.run.runner import _activate_failed_workload_repair
    ticket = Ticket(id='stage', run_id='r', agent_id=agent, status='queued')
    session.add_all([Run(id='r', metric='accuracy', status='running'), ticket])
    row = InfraInstance(id='job-row', run_id='r', ticket_id='stage', provider='cluster',
                        instance_id='123', status='failed', meta={
        'resource_request': True, 'scheduler_state': 'FAILED',
        'log_path': '/work/slurm-123.out', 'stderr_path': '/work/slurm-123.err',
    })
    session.add(row)
    await session.commit()
    assert await _activate_failed_workload_repair(session, ticket)
    assert ticket.status == 'repairing'
    assert ticket.repair_attempts == 1
    assert await _activate_failed_workload_repair(session, ticket)
    assert ticket.repair_attempts == 1  # A repeated watcher observation is not another repair.
    messages = (await session.execute(select(TicketMessage))).scalars().all()
    assert len(messages) == 1
    assert '/work/slurm-123.err' in messages[0].body


@pytest.mark.asyncio
@pytest.mark.parametrize('state,used,allowed,expected', [
    ('COMPLETED', 0, True, 'queued'),
    ('FAILED', 10, False, 'failed'),
])
async def test_external_repair_preserves_collection_and_enforces_budget(session, state, used, allowed, expected):
    from zevo.db import InfraInstance
    from zevo.engine.run.runner import _activate_failed_workload_repair
    ticket = Ticket(id='stage', run_id='r', agent_id='inference', status='queued', repair_attempts=used)
    session.add_all([Run(id='r', metric='accuracy', status='running'), ticket,
        InfraInstance(id='job-row', run_id='r', ticket_id='stage', provider='cluster',
                      instance_id='123', status='released', meta={
                          'resource_request': True, 'scheduler_state': state,
                      })])
    await session.commit()
    assert await _activate_failed_workload_repair(session, ticket) is allowed
    assert ticket.status == expected
    assert ticket.repair_attempts == used
