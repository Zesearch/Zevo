"""A completed finite job can bank a partial plan without failing its ticket."""
import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.contracts.train import TrainingContinuation, TrainResult
from zevo.engine.run.training_continuation import authorized, verify_files, queue_continuation
from zevo.db import Run, Ticket, InfraInstance, HeartbeatRun, AgentWakeupRequest
from zevo.db.models import Base


@pytest.fixture
def saved(tmp_path):
    root = tmp_path / 'train-ticket'
    root.mkdir()
    config = root / 'train_config.yaml'
    config.write_text('unchanged 450 step configuration')
    checkpoint = root / 'state' / 'checkpoint-15'
    checkpoint.mkdir(parents=True)
    plan = root / 'state' / 'training_plan.json'
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    plan.write_text(json.dumps(dict(ticket_id='train-ticket', config_sha256=digest, planned_steps=450)))
    for name in ['model.safetensors', 'optimizer.pt', 'scheduler.pt', 'training_args.bin',
                 *[f'rng_state_{i}.pth' for i in range(4)]]:
        (checkpoint / name).write_bytes(b'full state test fixture')
    (checkpoint / 'trainer_state.json').write_text(json.dumps(dict(global_step=15, max_steps=450)))
    request = TrainingContinuation(checkpoint_path=str(checkpoint), plan_path=str(plan),
        completed_steps=15, planned_steps=450, config_sha256=digest).model_dump()
    return dict(root=str(root), request=request, ticket_id='train-ticket', executed_config=str(config), world_size=4)


def test_partial_complete_job_has_valid_continuation(saved):
    proof = verify_files(**saved)
    assert proof['completed_steps'] == 15 and proof['planned_steps'] == 450
    assert proof['verified']
    result = TrainResult(status='deferred', ticket_id='train-ticket', error_message='', notes='', continuation=saved['request'])
    assert not result.checkpoint_path
    with pytest.raises(ValueError, match='continuation requires deferred'):
        TrainResult(status='succeeded', ticket_id='train-ticket', error_message='', notes='', continuation=saved['request'])


@pytest.mark.parametrize('damage', ['optimizer', 'rng', 'empty', 'step', 'plan', 'config', 'outside', 'symlink', 'weights'])
def test_incomplete_or_changed_state_cannot_continue(saved, damage, tmp_path):
    from pathlib import Path
    checkpoint = Path(saved['request']['checkpoint_path'])
    if damage == 'optimizer': (checkpoint / 'optimizer.pt').unlink()
    if damage == 'rng': (checkpoint / 'rng_state_3.pth').unlink()
    if damage == 'empty': (checkpoint / 'scheduler.pt').write_bytes(b'')
    if damage == 'weights': (checkpoint / 'model.safetensors').unlink()
    if damage == 'step': (checkpoint / 'trainer_state.json').write_text('{"global_step": 14, "max_steps":450}')
    if damage == 'plan': Path(saved['request']['plan_path']).write_text('{}')
    if damage == 'config': Path(saved['executed_config']).write_text('shortened plan')
    if damage == 'outside': saved['request']['checkpoint_path'] = str(tmp_path)
    if damage == 'symlink':
        outside = tmp_path / 'foreign-optimizer'; outside.write_text('other ticket')
        (checkpoint / 'optimizer.pt').unlink(); (checkpoint / 'optimizer.pt').symlink_to(outside)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_files(**saved)


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['cluster', 'cloud', 'instance'])
async def test_verified_exit_zero_queues_same_ticket_without_repair(saved, monkeypatch, provider):
    from zevo.engine.run import resource_recovery, remote_jobs
    from zevo.contracts.infrastructure import InfrastructureDeviceInfo
    from tests.test_ssh_provider_contract import _device
    monkeypatch.setattr(remote_jobs, '_device_info_for_ticket',
                        AsyncMock(return_value=InfrastructureDeviceInfo.model_validate(_device('cluster'))))
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        run = Run(id='run', task_name='task', metric='accuracy', status='running')
        ticket = Ticket(id='train-ticket', run_id='run', agent_id='train', status='running', repair_attempts=2)
        hb = HeartbeatRun(id='collect-hb', agent_id='train', ticket_id=ticket.id, driver='test', model='test')
        row = InfraInstance(id='old', instance_id='123', provider=provider, run_id='run', ticket_id=ticket.id,
            status='released', released_at=datetime.now(timezone.utc),
            meta={'scheduler_state': 'COMPLETED', 'resource_request': True})
        session.add_all([run, ticket, hb, row]); await session.commit()
        proof = {**verify_files(**saved), 'source_resource_id': row.id, 'source_job_id': row.instance_id}
        await queue_continuation(session, ticket, hb, row, proof)
        await session.commit()
        if provider == 'cluster':
            assert ticket.status == 'waiting_external'
            assert await resource_recovery.reconcile(session) == 1
        assert ticket.status == 'queued' and ticket.repair_attempts == 2
        assert hb.finished_at and hb.activation_phase == 'continue'
        assert authorized(ticket, row)
        assert not authorized(SimpleNamespace(id='different', agent_id='train'), row)
        # Redelivery coalesces; it must not produce two new submissions.
        await queue_continuation(session, ticket, hb, row, proof)
        await session.commit()
        assert await resource_recovery.reconcile(session) == 0
        queued = (await session.execute(select(AgentWakeupRequest).where(AgentWakeupRequest.status == 'queued'))).scalars().all()
        assert len(queued) == 1
        assert queued[0].ticket_id == ticket.id
        assert row.meta['training_continuation']['planned_steps'] == 450
        row.meta = {**row.meta, 'scheduler_state': 'RUNNING'}
        assert not authorized(ticket, row)
    await engine.dispose()


@pytest.fixture
def zero3(saved):
    from pathlib import Path
    ck = Path(saved['request']['checkpoint_path'])
    (ck / 'optimizer.pt').unlink()
    (ck / 'latest').write_text('global_step15')
    shards = ck / 'global_step15'
    shards.mkdir()
    for rank in range(4):
        (shards / f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt').write_bytes(b'optimizer')
        (shards / f'zero_pp_rank_{rank}_mp_rank_00_model_states.pt').write_bytes(b'model and scheduler')
    return saved


def test_runtime_config_and_zero3_continuation(zero3):
    from pathlib import Path
    old = Path(zero3['executed_config'])
    config = old.parent / 'runtime' / 'train_config.yaml'
    config.parent.mkdir()
    old.rename(config)
    zero3['executed_config'] = str(config)
    assert verify_files(**zero3)['completed_steps'] == 15


@pytest.mark.parametrize('damage', ['optimizer_rank', 'model_rank', 'empty', 'tag', 'escape', 'rng', 'world_size', 'shard_symlink'])
def test_zero3_partial_or_misdirected_checkpoint_rejected(zero3, damage, tmp_path):
    from pathlib import Path
    ck = Path(zero3['request']['checkpoint_path'])
    optimizer = ck / 'global_step15/bf16_zero_pp_rank_3_mp_rank_00_optim_states.pt'
    if damage == 'optimizer_rank': optimizer.unlink()
    if damage == 'model_rank': (ck / 'global_step15/zero_pp_rank_3_mp_rank_00_model_states.pt').unlink()
    if damage == 'empty': optimizer.write_bytes(b'')
    if damage == 'tag': (ck / 'latest').write_text('global_step14')
    if damage == 'escape': (ck / 'latest').write_text('../../foreign')
    if damage == 'rng': (ck / 'rng_state_3.pth').unlink()
    if damage == 'world_size': zero3['world_size'] = 2
    if damage == 'shard_symlink':
        foreign = tmp_path / 'foreign'; foreign.write_bytes(b'foreign state')
        optimizer.unlink(); optimizer.symlink_to(foreign)
    with pytest.raises(ValueError): verify_files(**zero3)


@pytest.mark.asyncio
async def test_collect_uses_recorded_runtime_path_and_portable_verifier(zero3, monkeypatch):
    from pathlib import Path
    import shlex
    import subprocess
    import sys
    from zevo.engine.run import training_continuation as c, remote_jobs
    from zevo.contracts import configuration
    root = Path(zero3['root'])
    config = root / 'runtime/train_config.yaml'; config.parent.mkdir()
    Path(zero3['executed_config']).rename(config)
    row = SimpleNamespace(id='resource', instance_id='123', meta={
        'scheduler_state': 'COMPLETED', 'remote_workdir': str(root / 'attempt-5'),
        'training_config_remote_path': str(config),
        'training_config_sha256': zero3['request']['config_sha256']})
    async def route(*args): return object()
    async def ssh(info, command, **kwargs):
        argv = shlex.split(command)
        result = subprocess.run([sys.executable, *argv[1:]], capture_output=True, text=True)
        return dict(ok=result.returncode == 0, stdout=result.stdout, error=result.stderr)
    monkeypatch.setattr(remote_jobs, '_device_info_for_ticket', route)
    monkeypatch.setattr(remote_jobs, '_ssh', ssh)
    monkeypatch.setattr(configuration, 'load_train_config', lambda p: SimpleNamespace(training=SimpleNamespace(world_size=4)))
    inp = SimpleNamespace(slurm_job=SimpleNamespace(phase='collect', bookkeeping_row_id='resource', job_id='123'))
    output = SimpleNamespace(continuation=TrainingContinuation(**zero3['request']), train_config_path=str(config))
    proof = await c.verify_request(None, SimpleNamespace(id='train-ticket'), inp, output, row)
    assert proof['verified'] and proof['source_job_id'] == '123'
    del row.meta['training_config_remote_path']
    with pytest.raises(ValueError, match='cannot verify full-state continuation'):
        await c.verify_request(None, SimpleNamespace(id='train-ticket'), inp, output, row)


@pytest.mark.asyncio
async def test_remote_config_binding_checks_bytes_and_ticket_boundary(saved, monkeypatch, tmp_path):
    import shlex
    import subprocess
    import sys
    from zevo.engine.run import training_continuation as c, remote_jobs
    async def ssh(info, command, **kwargs):
        result = subprocess.run([sys.executable, *shlex.split(command)[1:]], capture_output=True, text=True)
        return dict(ok=result.returncode == 0, error=result.stderr)
    monkeypatch.setattr(remote_jobs, '_ssh', ssh)
    workdir = saved['root'] + '/attempt-5'
    await c.verify_remote_config(None, workdir, saved['executed_config'], saved['request']['config_sha256'])
    with pytest.raises(ValueError, match='hash mismatch'):
        await c.verify_remote_config(None, workdir, saved['executed_config'], 'changed')
    foreign = tmp_path / 'foreign.yaml'; foreign.write_text('foreign')
    with pytest.raises(ValueError, match='outside Train ticket'):
        await c.verify_remote_config(None, workdir, str(foreign), saved['request']['config_sha256'])


@pytest.mark.asyncio
@pytest.mark.parametrize('valid', [True, False])
async def test_submit_binds_config_before_launch_and_records_path(tmp_path, monkeypatch, valid):
    from zevo.engine.run import runner, training_continuation as c, gpu_controller
    from zevo.contracts.train import TrainTaskInput
    from zevo.contracts.infrastructure import SlurmStageJobContract
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    (tmp_path / 'train_config.yaml').write_text('frozen configuration')
    script = tmp_path / 'train.sbatch'; script.write_text('launch runtime/train_config.yaml')
    (tmp_path / 'train.py').write_text('training worker')
    device = tmp_path / 'device.json'; device.write_text('{}')
    remote = '/remote/train-ticket/runtime/train_config.yaml'
    contract = SlurmStageJobContract.model_construct(enabled=True, phase='submit', stage='train',
        attempt=1, script_path=str(script), remote_work_dir='/remote/train-ticket/attempt-1',
        remote_script_path='/remote/train-ticket/attempt-1/train.sbatch', num_gpus=4)
    inp = TrainTaskInput.model_construct(slurm_job=contract, device_info_path=str(device))
    info = SimpleNamespace(provider='cluster', cluster=object(), gpu=None,
                           ssh=SimpleNamespace(host='host', port=22, user='user'))
    monkeypatch.setattr(runner.InfrastructureDeviceInfo, 'model_validate_json', lambda data: info)
    calls = []
    async def bind(info, workdir, path, digest):
        calls.append('bind')
        assert path == remote and digest == hashlib.sha256((tmp_path / 'train_config.yaml').read_bytes()).hexdigest()
        if not valid: raise ValueError('remote configuration hash mismatch')
    async def submit(*args, **kwargs):
        calls.append('launch')
        return '123', {}
    monkeypatch.setattr(c, 'verify_remote_config', bind)
    monkeypatch.setattr(gpu_controller, 'submit_workload', submit)
    async with Session() as s:
        r=Run(id='run',task_name='task',metric='accuracy',status='running')
        t=Ticket(id='train-ticket',run_id='run',agent_id='train',status='running')
        s.add_all([r,t]); await s.commit()
        args=dict(run=r,ticket=t,inp=inp,heartbeat_id='hb',publication={'attempted':False},remote_train_config_path=remote)
        if valid:
            row=await runner._submit_validated_slurm_stage_once(s,**args)
            assert row.meta['training_config_remote_path']==remote
            assert calls==['bind','launch']
        else:
            with pytest.raises(ValueError,match='hash mismatch'):
                await runner._submit_validated_slurm_stage_once(s,**args)
            assert calls==['bind']
    await engine.dispose()
