from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db import Base, Run, Ticket, HeartbeatRun, AgentWakeupRequest, HeartbeatResult
from sqlalchemy import select
from zevo.engine.run.activation_failure import ACTIVATION_KEY, track_activation_failures, persist_activation_failure


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['running', 'cancelled', 'newer', 'finished', 'before_activation'])
async def test_unhandled_failure_only_closes_its_own_active_heartbeat(tmp_path, mode):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path}/failure.db')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(timezone.utc)
    async with factory() as db:
        db.add_all([Run(id='run', metric='accuracy'),
            Ticket(id='ticket', run_id='run', agent_id='inference', status='cancelled' if mode=='cancelled' else 'running'),
            HeartbeatRun(id='old', ticket_id='ticket', agent_id='inference', driver='test', model='test',
                         started_at=now, finished_at=now if mode=='finished' else None)])
        if mode=='newer':
            db.add(HeartbeatRun(id='new', ticket_id='ticket', agent_id='inference', driver='test', model='test', started_at=now+timedelta(seconds=1)))
        await db.commit()
    @track_activation_failures
    async def broken(session):
        if mode!='before_activation':session.info[ACTIVATION_KEY]=('ticket','old')
        raise RuntimeError('result registration failed')
    async with factory() as db:
        # A stale identity from another invocation must never be reused.
        db.info[ACTIVATION_KEY]=('ticket','old')
        with pytest.raises(RuntimeError, match='result registration failed'):
            await broken(db)
        assert ACTIVATION_KEY not in db.info
    async with factory() as db:
        hb=await db.get(HeartbeatRun,'old');t=await db.get(Ticket,'ticket')
        if mode in ['running','cancelled','newer']:
            assert hb.finished_at and 'result registration failed' in hb.error_message
        else:assert not hb.error_message
        assert t.status == ('repairing' if mode=='running' else 'cancelled' if mode=='cancelled' else 'running')
        if mode == 'running':
            assert t.repair_attempts == 1 and t.repair_route == 'self'
            wake = (await db.scalars(select(AgentWakeupRequest))).one()
            assert wake.ticket_id == t.id and wake.source == 'retry'
            result = (await db.scalars(select(HeartbeatResult))).one()
            assert result.heartbeat_id == 'old' and result.status == 'failed'
    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize('agent,driver,lane,attempts,run_status,expected_route,retry', [
    ('data', 'test', 'optimization', 19, 'running', 'self', True),
    ('data', 'test', 'optimization', 20, 'running', 'orchestrator', False),
    ('orchestrator', 'test', 'optimization', 20, 'running', 'terminal', False),
    ('evaluation', 'evaluation_runner', 'optimization', 3, 'running', 'orchestrator', False),
    ('evaluation', 'evaluation_runner', 'held_out_test', 3, 'running', 'terminal', False),
    ('data', 'test', 'optimization', 3, 'cancelled', 'terminal', False),
])
async def test_runner_failure_preserves_limits_and_does_not_queue_duplicate_repairs(
    tmp_path, agent, driver, lane, attempts, run_status, expected_route, retry,
):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path}/bounded.db')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as db:
        db.add_all([
            Run(id='run', metric='accuracy', status=run_status),
            Ticket(id='ticket', run_id='run', agent_id=agent, lane=lane,
                   status='running', repair_attempts=attempts),
            HeartbeatRun(id='hb', ticket_id='ticket', agent_id=agent,
                         driver=driver, model='test'),
        ])
        await db.commit()
    for _ in range(2):
        await persist_activation_failure(factory, ('ticket', 'hb'), RuntimeError('result registration failed'))
    async with factory() as db:
        ticket = await db.get(Ticket, 'ticket')
        assert ticket.status == ('repairing' if retry else 'failed')
        assert ticket.repair_attempts == attempts + int(retry)
        assert ticket.repair_route == expected_route
        wakes = (await db.scalars(select(AgentWakeupRequest))).all()
        assert len(wakes) == int(retry)
        if retry:
            assert wakes[0].payload['repair_attempt'] == 20
        results = (await db.scalars(select(HeartbeatResult))).all()
        assert len(results) == (0 if run_status == 'cancelled' else 1)
    await engine.dispose()
