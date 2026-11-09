from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
import hashlib

import pytest
import pytest_asyncio
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.contracts._base import AgentResult
from zevo.db import Base, Run, Ticket, HeartbeatRun, InfraInstance, RunInstruction, AgentWakeupRequest, WorkProduct
from zevo.engine.run.execution_repair import RepairTicketRequest, repair_ticket
from zevo.engine.run.repair_progress import execution_retry_authorized, verify_repair_training_config
from zevo.api.routers.ui.heartbeats import _repair_sources


@pytest_asyncio.fixture
async def repair_db(monkeypatch):
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session = async_sessionmaker(engine, expire_on_commit=False)()
    run = Run(id='r', metric='accuracy', status='running', supervisor_ticket_id='orch')
    supervisor = Ticket(id='orch', run_id='r', agent_id='orchestrator', status='running')
    ticket = Ticket(id='train', run_id='r', agent_id='train', iteration=2, status='failed',
                    repair_attempts=2, repair_route='orchestrator',
                    payload={'fixed': 'plan'}, inputs={'parent': 'original'})
    hb = HeartbeatRun(id='decision', ticket_id='orch', agent_id='orchestrator', driver='mock', model='test')
    execution = InfraInstance(id='job', run_id='r', ticket_id='train', provider='cluster',
        instance_id='123', status='released', meta={'resource_request': True,
            'execution_attempt': 2, 'scheduler_state': 'COMPLETED', 'scheduler_exit_code': '0:0'})
    product = WorkProduct(id='product', ticket_id='train', role='checkpoint', path='/durable/checkpoint-735')
    session.add_all([run, supervisor, ticket, hb, execution, product])
    await session.commit()
    from zevo.engine.cost import budget
    monkeypatch.setattr(budget, 'snapshot_for_run', AsyncMock(return_value=SimpleNamespace(over_budget=False, over_time_limit=False)))
    yield session, ticket, execution
    await session.close()
    await engine.dispose()


def request(action='resubmit'):
    return RepairTicketRequest(action=action, execution_attempt=2,
        reason='Launcher corrected; the prior process completed without training',
        evidence=['attempt-2/workload.out: declined before first step; checkpoint-735 intact'])


@pytest.mark.asyncio
@pytest.mark.parametrize('action', ['collect', 'resubmit'])
async def test_review_repairs_same_ticket_preserving_state_and_budget(repair_db, action):
    db, ticket, execution = repair_db
    result = await repair_ticket(db, ticket, request(action))
    assert result['status'] == 'repairing' and ticket.repair_attempts == 3
    assert ticket.iteration == 2 and ticket.payload == {'fixed': 'plan'}
    assert ticket.inputs == {'parent': 'original'}
    assert (await db.get(WorkProduct, 'product')).path == '/durable/checkpoint-735'
    assert execution_retry_authorized(execution.meta) is (action == 'resubmit')
    assert not execution_retry_authorized({**execution.meta, 'execution_attempt': 3})
    wakes = (await db.scalars(select(AgentWakeupRequest))).all()
    assert len(wakes) == 1 and wakes[0].ticket_id == ticket.id
    assert wakes[0].payload['repair_attempt'] == 3
    with pytest.raises(HTTPException):
        await repair_ticket(db, ticket, request(action))
    assert ticket.repair_attempts == 3  # repeated HTTP request cannot spend twice


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['limit', 'pause', 'cancel', 'terminal', 'held_out', 'active', 'stale', 'no_supervisor', 'budget'])
async def test_review_cannot_override_execution_boundaries(repair_db, monkeypatch, boundary):
    db, ticket, execution = repair_db
    run = await db.get(Run, 'r')
    if boundary == 'limit': ticket.repair_attempts = 20
    if boundary == 'pause': db.add(RunInstruction(run_id='r', body='Pause until explicitly resumed', status='needs_input'))
    if boundary == 'cancel': run.cancel_requested_at = datetime.now(timezone.utc)
    if boundary == 'terminal': ticket.repair_route = 'terminal'
    if boundary == 'held_out': ticket.lane = 'held_out_test'
    if boundary == 'active': execution.meta = {**execution.meta, 'scheduler_state': 'RUNNING'}
    if boundary == 'stale': execution.meta = {**execution.meta, 'execution_attempt': 3}
    if boundary == 'no_supervisor': (await db.get(HeartbeatRun, 'decision')).finished_at = datetime.now(timezone.utc)
    if boundary == 'budget':
        from zevo.engine.cost import budget
        monkeypatch.setattr(budget, 'snapshot_for_run', AsyncMock(return_value=SimpleNamespace(over_budget=True, over_time_limit=False)))
    await db.commit()
    count = ticket.repair_attempts
    with pytest.raises(HTTPException):
        await repair_ticket(db, ticket, request())
    assert ticket.repair_attempts == count and ticket.status == 'failed'
    assert not (await db.scalars(select(AgentWakeupRequest))).all()


def test_execution_retry_requires_failure_and_same_work_order():
    args = dict(ticket_id='train', status='failed', error_message='no work', notes='', execution_retry_reason='run corrected launcher')
    assert AgentResult(**args).execution_retry_reason
    with pytest.raises(ValidationError):
        AgentResult(**{**args, 'status': 'succeeded', 'error_message': ''})
    with pytest.raises(ValidationError):
        AgentResult(**args, failure_recovery=dict(scope='resource_plan', reason='need new host',
                    evidence=['host lost'], next_action='replace allocation'))


def test_training_repair_cannot_change_or_lose_the_registered_config(tmp_path):
    config = tmp_path / 'train_config.yaml'
    config.write_text('frozen plan')
    previous = SimpleNamespace(meta={'execution_attempt': 2,
        'execution_repair': {'execution_attempt': 2, 'action': 'resubmit', 'reason': 'launcher fix'},
        'training_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest()})
    verify_repair_training_config(previous, config)
    config.write_text('changed plan')
    with pytest.raises(ValueError, match='exact registered training configuration'):
        verify_repair_training_config(previous, config)
    previous.meta.pop('training_config_sha256')
    with pytest.raises(ValueError):
        verify_repair_training_config(previous, config)


@pytest.mark.asyncio
async def test_timeline_uses_historical_trigger_not_mutable_ticket_status(repair_db):
    db, ticket, _ = repair_db
    hb = await db.get(HeartbeatRun, 'decision')
    hb.resolved_config = {'orchestration_trigger': {'ticket_id': ticket.id, 'status': 'failed'}}
    ticket.status = 'succeeded'
    await db.commit()
    assert (await _repair_sources(db, [hb]))[hb.id] == {'ticket_id': ticket.id, 'agent_id': 'train'}
    hb.resolved_config = {'orchestration_trigger': {'ticket_id': ticket.id, 'status': 'succeeded'}}
    ticket.status = 'failed'
    await db.commit()
    assert await _repair_sources(db, [hb]) == {}
    hb.resolved_config = {}
    db.add(AgentWakeupRequest(agent_id='orchestrator', ticket_id='orch', heartbeat_run_id=hb.id,
                             source='handoff', status='completed',
                             payload={'optimization_completion_key': 'train:failed-hb:failed'}))
    await db.commit()
    assert (await _repair_sources(db, [hb]))[hb.id]['ticket_id'] == 'train'
    ticket.lane = 'held_out_test'
    await db.commit()
    assert await _repair_sources(db, [hb]) == {}
