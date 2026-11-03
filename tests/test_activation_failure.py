from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db import Base, Run, Ticket, HeartbeatRun
from zevo.engine.run.activation_failure import ACTIVATION_KEY, track_activation_failures


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
        assert t.status == ('failed' if mode=='running' else 'cancelled' if mode=='cancelled' else 'running')
    await engine.dispose()
