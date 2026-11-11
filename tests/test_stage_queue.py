"""Shared-allocation admission survives retries and scheduler restarts."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.contracts.infrastructure import InfrastructureDeviceInfo, SlurmStageJobContract, slurm_lifecycle_prologue, slurm_runtime_prologue
from zevo.db.models import Base, Run, Ticket, InfraInstance
from zevo.engine.run.stage_queue import blocking_stage, dispatch_waiting_stages, preparation_already_queued
from zevo.engine.run.runner import _submit_validated_slurm_stage, _commit_slurm_submission


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_dispatch_uses_released_checkpoint_execution_on_live_owner(db, tmp_path, monkeypatch):
    from zevo.engine.run import remote_jobs
    commands = []
    async def ssh(info, command, **kwargs):
        commands.append(command)
        return {'ok': True, 'stdout': 'RUNNING\n' if 'squeue -h' in command else '123\n'}
    monkeypatch.setattr(remote_jobs, '_ssh', ssh)
    run = Run(id='run', metric='accuracy', gpu_allocation_mode='per_run', status='running')
    train = Ticket(id='train', run_id=run.id, agent_id='train', status='waiting_external')
    child = Ticket(id='next', run_id=run.id, agent_id='inference', status='running')
    now = datetime.now(timezone.utc)
    owner = InfraInstance(id='owner', run_id=run.id, ticket_id=train.id, provider='cluster',
        instance_id='123', gpu_count=4, created_at=now - timedelta(hours=1),
        meta={'resource_request': True, 'controller_directory': '/controller', 'nodes': 1,
              'scheduler_state': 'COMPLETED', 'checkpoint_yield': 'old', 'checkpoint_resume': 'old'})
    db.add_all([run, train, child, owner]); await db.commit()
    inp = inputs(tmp_path)
    row = await _submit_validated_slurm_stage(db, run=run, ticket=child, inp=inp, heartbeat_id='hb')
    _commit_slurm_submission(row, heartbeat_id='hb', stdout_path=inp.slurm_job.stdout_path,
                            stderr_path=inp.slurm_job.stderr_path)
    assert row.instance_id == ''
    child.status = 'waiting_external'
    request = {'request_id': 'latest', 'ticket_id': train.id, 'step': 2447}
    child.payload = {'mid_training_checkpoint': request}
    run.lifecycle = {'adaptive_checkpoints': {'latest': {
        'request': request, 'job_row_id': 'checkpoint', 'inference_ticket_id': child.id}}}
    db.add(InfraInstance(id='checkpoint', run_id=run.id, ticket_id=train.id, provider='cluster',
        instance_id='123', gpu_count=4, created_at=now, released_at=now,
        meta={'allocation_owner_row_id': owner.id, 'controller_directory': '/controller',
              'scheduler_state': 'COMPLETED', 'checkpoint_yield': 'latest'}))
    await db.commit()
    async with async_sessionmaker(db.bind, expire_on_commit=False)() as restarted:
        assert await dispatch_waiting_stages(restarted) == 1
        assert await dispatch_waiting_stages(restarted) == 0
    await db.refresh(row)
    assert row.instance_id == '123' and row.meta['allocation_owner_row_id'] == owner.id
    assert len([cmd for cmd in commands if 'requests/hb.json' in cmd]) == 1
    assert not any('sbatch --parsable' in cmd for cmd in commands)


def inputs(tmp_path):
    script = tmp_path / 'predict.sbatch'
    script.write_text('#!/bin/bash\necho prepared\n')
    (tmp_path / 'predict.py').write_text('print("prepared")\n')
    device = InfrastructureDeviceInfo(
        run_id='run', ticket_id='infra', purpose='train', provider='cluster',
        auto_release=False, host='login.example',
        ssh=dict(host='login.example', port=22, user='user', key_path='/key'),
        cluster=dict(requested_gpus=4, nodes=1, workdir='/remote', hf_cache='/cache',
                     gpu_constraints=dict(min_gpus_per_job=1, allocation_step=1,
                                          gpus_per_node=4, source='test')),
        cost=dict(dph_total=0), probe_source='none', probed_at='2026-11-01T00:00:00Z',
        resource_plan=dict(purpose='train', required_working_set_gib=40,
            host_memory_components_gib={'runtime': 40}, host_memory_formula_gib=64,
            site_min_ram_gib=0, num_gpus=4, nodes=1, min_vram_gb=40,
            min_ram_gb=64, min_cpus=8, time_limit_hours=8, disk_gb=0, rationale='test'),
    )
    path = tmp_path / 'device.json'
    path.write_text(device.model_dump_json())
    contract = SlurmStageJobContract(enabled=True, stage='inference',
        estimated_gpus=4, resource_plan_source='test', resource_plan_rationale='test',
        script_path=str(script), remote_script_path='/remote/predict.sbatch',
        remote_work_dir='/remote', job_name='zevo-test', status_path='/remote/.zevo-slurm-status',
        lifecycle_prologue=slurm_lifecycle_prologue('/remote/.zevo-slurm-status'), runtime_prologue=slurm_runtime_prologue(),
        stdout_path='/remote/slurm-%j.out', stderr_path='/remote/slurm-%j.err', num_gpus=4)
    return SimpleNamespace(slurm_job=contract, device_info_path=str(path))


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['per_run', 'per_stage', 'per_submission'])
async def test_fifo_uses_ready_requests_not_ticket_creation_and_keeps_repair_turn(db, mode):
    run = Run(id='run', metric='accuracy', gpu_allocation_mode=mode)
    a = Ticket(id='inference', run_id='run', agent_id='inference', lane='held_out_test', status='repairing')
    b = Ticket(id='train', run_id='run', agent_id='train', status='waiting_external',
               created_at=datetime.now(timezone.utc)-timedelta(days=1))
    old = InfraInstance(id='old', run_id='run', ticket_id=a.id, provider='cluster', instance_id='123',
        released_at=datetime.now(timezone.utc), created_at=datetime.now(timezone.utc)-timedelta(hours=1),
        meta={'resource_request': True, 'scheduler_state': 'TIMEOUT'})
    queued = InfraInstance(id='queued', run_id='run', ticket_id=b.id, provider='cluster', instance_id='',
                          meta={'resource_request': True, 'stage_queue': {}})
    db.add_all([run, a, b, old, queued]); await db.commit()
    assert (await blocking_stage(db, run, b)) is (a if mode == 'per_run' else None)
    assert await blocking_stage(db, run, a) is None
    a.status = 'succeeded'; await db.commit()
    assert await blocking_stage(db, run, b) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('end', ['succeeded', 'cancelled', 'failed'])
async def test_deferred_submission_dispatches_once_after_owner_finishes(db, tmp_path, monkeypatch, end):
    from zevo.engine.run import remote_jobs
    commands = []
    async def ssh(info, command, **kwargs):
        commands.append(command)
        return {'ok': True, 'stdout': 'RUNNING\n' if 'squeue -h' in command else '123\n'}
    monkeypatch.setattr(remote_jobs, '_ssh', ssh)
    run = Run(id='run', metric='accuracy', gpu_allocation_mode='per_run', status='running')
    a = Ticket(id='heldout', run_id='run', agent_id='inference', lane='held_out_test', status='running')
    b = Ticket(id='next', run_id='run', agent_id='inference', status='running')
    owner = InfraInstance(id='owner', run_id='run', ticket_id=a.id, provider='cluster', instance_id='123',
        gpu_count=4, meta={'resource_request': True, 'controller_directory': '/controller', 'nodes': 1})
    db.add_all([run, a, b, owner]); await db.commit()
    inp = inputs(tmp_path)
    row = await _submit_validated_slurm_stage(db, run=run, ticket=b, inp=inp, heartbeat_id='hb')
    assert row.instance_id == '' and commands == []
    _commit_slurm_submission(row, heartbeat_id='hb', stdout_path=inp.slurm_job.stdout_path,
                            stderr_path=inp.slurm_job.stderr_path)
    b.status = 'waiting_external'; await db.commit()
    assert await preparation_already_queued(db, run, b, {})
    assert await dispatch_waiting_stages(db) == 0
    a.status = 'repairing'; owner.released_at = datetime.now(timezone.utc); await db.commit()
    assert await dispatch_waiting_stages(db) == 0
    a.status = end; owner.released_at = None; await db.commit()
    # A fresh session represents a scheduler restart. No Agent needs to rerun.
    factory = async_sessionmaker(db.bind, expire_on_commit=False)
    async with factory() as restarted:
        assert await dispatch_waiting_stages(restarted) == 1
        assert await dispatch_waiting_stages(restarted) == 0
    await db.refresh(row)
    assert row.instance_id == '123'
    assert 'stage_queue' not in row.meta
    assert row.meta['log_path'] == '/remote/slurm-123.out'
    assert len([command for command in commands if 'requests/hb.json' in command]) == 1


@pytest.mark.asyncio
async def test_cancelled_waiter_is_never_published(db):
    run = Run(id='run', metric='accuracy', gpu_allocation_mode='per_run', status='running')
    ticket = Ticket(id='waiter', run_id='run', agent_id='train', status='cancelled')
    row = InfraInstance(id='queue', run_id='run', ticket_id='waiter', provider='cluster', instance_id='',
                       meta={'resource_request': True, 'stage_queue': {'heartbeat_id': 'hb'}})
    db.add_all([run, ticket, row]); await db.commit()
    assert await dispatch_waiting_stages(db) == 0
    assert row.released_at is not None


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['per_stage', 'per_submission'])
async def test_independent_allocations_submit_while_other_lane_runs(db, tmp_path, monkeypatch, mode):
    from zevo.engine.run import remote_jobs
    commands = []
    async def ssh(info, command, **kwargs):
        commands.append(command)
        return {'ok': True, 'stdout': '456\n'}
    monkeypatch.setattr(remote_jobs, '_ssh', ssh)
    run = Run(id='run', metric='accuracy', gpu_allocation_mode=mode, status='running')
    a = Ticket(id='heldout', run_id='run', agent_id='inference', lane='held_out_test', status='running')
    b = Ticket(id='next', run_id='run', agent_id='inference', status='running')
    owner = InfraInstance(id='owner', run_id='run', ticket_id=a.id, provider='cluster', instance_id='123',
        gpu_count=4, meta={'resource_request': True, 'controller_directory': '/controller', 'nodes': 1})
    db.add_all([run, a, b, owner]); await db.commit()
    row = await _submit_validated_slurm_stage(db, run=run, ticket=b, inp=inputs(tmp_path), heartbeat_id='hb')
    assert row.instance_id == '456' and 'stage_queue' not in row.meta
    assert any('sbatch --parsable' in command for command in commands)


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', ['script', 'implementation'])
async def test_queued_artifacts_cannot_change_before_dispatch(db, tmp_path, monkeypatch, changed):
    from zevo.engine.run import remote_jobs
    async def forbidden(*args, **kwargs):
        raise AssertionError('changed artifact must never reach SSH')
    monkeypatch.setattr(remote_jobs, '_ssh', forbidden)
    run = Run(id='run', metric='accuracy', gpu_allocation_mode='per_run', status='running')
    a = Ticket(id='heldout', run_id='run', agent_id='inference', lane='held_out_test', status='running')
    b = Ticket(id='next', run_id='run', agent_id='inference', status='running')
    owner = InfraInstance(id='owner', run_id='run', ticket_id=a.id, provider='cluster', instance_id='123',
        gpu_count=4, meta={'resource_request': True, 'controller_directory': '/controller', 'nodes': 1})
    db.add_all([run, a, b, owner]); await db.commit()
    inp = inputs(tmp_path)
    row = await _submit_validated_slurm_stage(db, run=run, ticket=b, inp=inp, heartbeat_id='hb')
    _commit_slurm_submission(row, heartbeat_id='hb', stdout_path=inp.slurm_job.stdout_path,
                            stderr_path=inp.slurm_job.stderr_path)
    b.status = 'waiting_external'; a.status = 'succeeded'; await db.commit()
    (tmp_path / ('predict.sbatch' if changed == 'script' else 'predict.py')).write_text('changed')
    assert await dispatch_waiting_stages(db) == 0
    await db.refresh(b); await db.refresh(row)
    assert b.status == 'awaiting_input'
    assert 'changed after validation' in row.meta['stage_queue_error']
    assert row.instance_id == ''


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['data', 'train', 'inference', 'evaluation'])
async def test_successful_workload_keeps_turn_until_collect_accepts_result(db, stage):
    run = Run(id='run', metric='accuracy', gpu_allocation_mode='per_run')
    a = Ticket(id='owner', run_id='run', agent_id=stage, status='waiting_external')
    b = Ticket(id='next', run_id='run', agent_id='train', status='running')
    row = InfraInstance(id='claim', run_id='run', ticket_id=a.id, provider='cluster', instance_id='123',
                       meta={'resource_request': True, 'scheduler_state': 'COMPLETED'})
    db.add_all([run, a, b, row]); await db.commit()
    assert await blocking_stage(db, run, b) is a
    a.status = 'running'; await db.commit()  # collect activation
    assert await blocking_stage(db, run, b) is a
    a.status = 'repairing'; await db.commit()  # collect rejected missing outputs
    assert await blocking_stage(db, run, b) is a
    a.status = 'succeeded'; await db.commit()
    assert await blocking_stage(db, run, b) is None


@pytest.mark.asyncio
async def test_interrupted_publication_is_not_replayed_after_restart(db):
    run = Run(id='run', metric='accuracy', gpu_allocation_mode='per_run', status='running')
    ticket = Ticket(id='waiter', run_id='run', agent_id='train', status='waiting_external')
    row = InfraInstance(id='queue', run_id='run', ticket_id='waiter', provider='cluster', instance_id='',
        meta={'resource_request': True, 'submission_committed': True,
              'stage_queue': {'heartbeat_id': 'hb'},
              'stage_queue_dispatch_started_at': (datetime.now(timezone.utc)-timedelta(minutes=5)).isoformat()})
    db.add_all([run, ticket, row]); await db.commit()
    assert await dispatch_waiting_stages(db) == 0
    assert ticket.status == 'awaiting_input'
    assert 'interrupted' in ticket.error_message
    assert await preparation_already_queued(db, run, ticket,
        {'run_instruction_id': 'instruction', 'instruction_action': 'restart_activation'})
    assert row.released_at is None
