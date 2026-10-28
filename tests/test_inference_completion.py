"""Artifact success is independent of resource cleanup and scoped to one execution."""
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from zevo.db import Base, Run, Ticket, InfraInstance, ExecutionEvent
from zevo.engine.run.inference_completion import record_completion, file_digest
from zevo.engine.run.workload_execution import register_execution, classify_event


@pytest.mark.asyncio
@pytest.mark.parametrize('matching', [True, False])
async def test_verified_completion_survives_cleanup_failure_without_reusing_gpu(tmp_path, monkeypatch, matching):
    from zevo.engine.run import runner
    monkeypatch.setattr(runner, '_slurm_stage_implementation_sha256', lambda _: 'implementation')
    script = tmp_path / 'job.sh'
    script.write_text('true')
    artifact = tmp_path / 'result'
    artifact.write_text('validated')
    member = dict(n_rows=5, predictions_path=str(artifact), generation_diagnostics_path=str(artifact),
                  inference_config_path=str(artifact))
    result = NS(slurm_script_path=str(script), log_path='log', suite_members=[NS(name='B', **member)], **member)
    inp = NS(benchmark_id='validation:0', suite_members=[NS(name='B', benchmark_id='validation:1')], slurm_job=NS(enabled=True, bookkeeping_row_id='job'))
    heartbeat = NS(id='collect', resolved_config={})
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as db:
        ticket = Ticket(id='infer', run_id='run', agent_id='inference', lane='optimization')
        infra = InfraInstance(id='job', run_id='run', ticket_id='infer', provider='cluster', instance_id='123',
            meta={'scheduler_state': 'FAILED', 'cleanup_confirmed': False,
                  'script_sha256': file_digest(str(script)) if matching else 'other', 'implementation_sha256': 'implementation'})
        db.add_all([Run(id='run', metric='accuracy'), ticket, infra])
        await db.flush()
        original = await register_execution(db, execution_id='job', ticket_id='infer', heartbeat_id='submit', runtime_key='slurm', log_path='log')
        for _ in range(2):
            await record_completion(db, ticket=ticket, heartbeat=heartbeat, inp=inp, result=result, primary_name='A')
            await db.flush()
        events = (await db.execute(select(ExecutionEvent))).scalars().all()
        assert len(events) == 2
        assert {e.extras['benchmark_name'] for e in events} == {'A', 'B'}
        assert all(e.extras['artifact_verified'] and e.current_step == e.total_steps == 5 for e in events)
        assert infra.meta['scheduler_state'] == 'FAILED'
        assert infra.meta['cleanup_confirmed'] is False
        if matching:
            assert all(e.attempt_id == 'job' for e in events)
            assert classify_event(events[0], {'job': original})['execution_purpose'] == 'workload'
            assert heartbeat.resolved_config['execution_phase'] == 'collect'
            assert len(infra.meta['verified_workload_result']['members']) == 2
        else:
            assert all(e.attempt_id == 'verified-collect' for e in events)
            assert 'verified_workload_result' not in infra.meta
    await engine.dispose()
