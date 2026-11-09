"""Exercise real VARCHAR bounds and rollback semantics; SQLite cannot catch these."""
import os
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select, text, func
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db import Agent, Base, Run, Ticket, HeartbeatRun, WorkloadExecution, ExecutionEvent
from zevo.engine.run.activation_failure import ACTIVATION_KEY, track_activation_failures
from zevo.engine.run.inference_completion import record_completion


@pytest_asyncio.fixture
async def pg():
    url = os.getenv('ZEVO_TEST_POSTGRES_URL') or (
        os.getenv('ZEVO_DATABASE_URL', '') if os.getenv('CI') == 'true' else '')
    if not url.startswith('postgresql+asyncpg://'):
        pytest.skip('requires ZEVO_TEST_POSTGRES_URL (CI uses its test PostgreSQL service)')
    schema = 'completion_test_' + uuid4().hex
    admin = create_async_engine(url)
    engine = None
    try:
        async with admin.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        engine = create_async_engine(url, connect_args={'server_settings': {'search_path': schema}})
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        if engine:
            await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin.dispose()


async def seed(factory):
    hid = str(uuid4())
    async with factory() as db:
        db.add(Agent(id='inference', name='Inference', title='Inference', identity_path='playbook/agents/inference/identity.md'))
        db.add(Run(id='run', metric='accuracy'))
        await db.flush()
        db.add(Ticket(id='infer', run_id='run', agent_id='inference', status='running', lane='optimization'))
        await db.flush()
        db.add(HeartbeatRun(id=hid, ticket_id='infer', agent_id='inference', driver='test', model='test'))
        await db.commit()
    return hid


@pytest.mark.asyncio
async def test_cloud_result_uuid_fits_postgres_and_redelivery_is_idempotent(pg, tmp_path):
    hid = await seed(pg)
    artifact = tmp_path / 'validated'; artifact.write_text('verified')
    member = dict(n_rows=271, predictions_path=str(artifact), generation_diagnostics_path=str(artifact),
                  inference_config_path=str(artifact))
    result = NS(slurm_script_path='', log_path='infer.log', suite_members=[], **member)
    inp = NS(benchmark_id='validation:0', suite_members=[], slurm_job=NS(enabled=False))
    for _ in range(2):
        async with pg() as db:
            await record_completion(db, ticket=await db.get(Ticket, 'infer'),
                heartbeat=await db.get(HeartbeatRun, hid), inp=inp, result=result, primary_name='Validation')
            await db.commit()
    async with pg() as db:
        executions = (await db.scalars(select(WorkloadExecution))).all()
        assert len(executions) == 1 and len(executions[0].id) == 36
        assert await db.scalar(select(func.count()).select_from(ExecutionEvent)) == 1


@pytest.mark.asyncio
async def test_database_failure_rolls_back_and_immediately_closes_activation(pg):
    hid = await seed(pg)
    @track_activation_failures
    async def broken(session):
        session.info[ACTIVATION_KEY] = ('infer', hid)
        ticket = await session.get(Ticket, 'infer')
        ticket.status = 'succeeded'  # must roll back this uncommitted outcome
        session.add(WorkloadExecution(id='verified-' + hid, ticket_id='infer', heartbeat_id=hid,
                                     runtime_key='verified_result', log_path='infer.log'))
        await session.flush()
    async with pg() as db:
        with pytest.raises(DBAPIError):
            await broken(db)
        assert ACTIVATION_KEY not in db.info
    async with pg() as db:
        hb = await db.get(HeartbeatRun, hid)
        ticket = await db.get(Ticket, 'infer')
        assert hb.finished_at is not None
        assert 'character varying(36)' in hb.error_message
        assert ticket.status == 'repairing' and ticket.error_message == hb.error_message
        assert ticket.repair_attempts == 1
        assert await db.scalar(select(func.count()).select_from(WorkloadExecution)) == 0
