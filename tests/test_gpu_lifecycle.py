"""A failed child must not lose its GPU allocation or start a duplicate worker."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import Base, Run, Ticket, InfraInstance
from zevo.engine.run.gpu_controller import CONTROLLER_SOURCE, submit_workload
from zevo.engine.run.gpu_lifecycle import needs_stage_release


def wait_outcome(path: Path, process: subprocess.Popen):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text())
        assert process.poll() is None, "allocation controller exited unexpectedly"
        time.sleep(.05)
    raise AssertionError("workload outcome was not published")


def publish(root, name, script, *, digest=None):
    payload = dict(script=str(script), sha256=digest or hashlib.sha256(script.read_bytes()).hexdigest(),
                   ticket_id="train-1", run_id="run-1", cwd=str(root),
                   stdout=str(root / f"{name}.out"), stderr=str(root / f"{name}.err"))
    (root / 'requests' / f'{name}.json').write_text(json.dumps(payload))


def test_controller_survives_failure_then_runs_repaired_script(tmp_path):
    for name in ('requests', 'outcomes', 'cancel'):
        (tmp_path / name).mkdir()
    script = tmp_path / 'train.sh'
    script.write_text('echo traceback >&2\nexit 1\n')
    publish(tmp_path, '01', script)
    process = subprocess.Popen([sys.executable, '-c', CONTROLLER_SOURCE, str(tmp_path), '20'])
    try:
        assert wait_outcome(tmp_path / 'outcomes/01.json', process)['exit_code'] == 1
        assert process.poll() is None
        script.write_text('echo checkpoint > checkpoint.txt\n')
        publish(tmp_path, '02', script)
        assert wait_outcome(tmp_path / 'outcomes/02.json', process)['exit_code'] == 0
        assert (tmp_path / 'checkpoint.txt').read_text().strip() == 'checkpoint'
        assert process.poll() is None
        (tmp_path / 'release').touch()
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_controller_rejects_modified_script_and_has_idle_deadline(tmp_path):
    for name in ('requests', 'outcomes', 'cancel'):
        (tmp_path / name).mkdir()
    script = tmp_path / 'train.sh'
    script.write_text('touch should-not-exist\n')
    publish(tmp_path, '01', script, digest='wrong')
    process = subprocess.Popen([sys.executable, '-c', CONTROLLER_SOURCE, str(tmp_path), '1'])
    try:
        result = wait_outcome(tmp_path / 'outcomes/01.json', process)
        assert result['exit_code'] != 0 and 'changed' in result['error']
        assert not (tmp_path / 'should-not-exist').exists()
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


@pytest.mark.parametrize('mode,state,expected', [
    ('per_stage', 'repairing', False), ('per_run', 'repairing', False),
    ('per_stage', 'waiting_external', False), ('per_stage', 'succeeded', True),
    ('per_run', 'succeeded', False), ('per_stage', 'failed', True),
])
def test_release_boundary(mode, state, expected):
    assert needs_stage_release(SimpleNamespace(gpu_allocation_mode=mode),
                               SimpleNamespace(status=state, agent_id='train')) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,next_ticket,reuses,live', [
    ('per_stage', 'train-1', True, True), ('per_run', 'infer-1', True, True),
    ('per_stage', 'infer-1', False, True), ('per_run', 'infer-1', False, False),
])
async def test_repair_and_cross_stage_allocation_reuse(tmp_path, monkeypatch, mode, next_ticket, reuses, live):
    from zevo.engine.run import remote_jobs
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    commands = []
    async def ssh(info, command, **kwargs):
        commands.append(command)
        if "squeue -h" in command:
            return dict(ok=True, stdout='RUNNING\n' if live else '')
        return dict(ok=True, stdout='123\n')
    monkeypatch.setattr(remote_jobs, '_ssh', ssh)
    script = tmp_path / 'train.sbatch'
    script.write_text('#!/bin/bash\n#SBATCH --gpus=8\nexit 0\n')
    contract = SimpleNamespace(num_gpus=8, nodes=1, script_path=str(script),
        remote_script_path='/remote/train.sh', remote_work_dir='/remote',
        stdout_path='/remote/out', stderr_path='/remote/err')
    async with Session() as session:
        run = Run(id='run-1', metric='accuracy', gpu_allocation_mode=mode)
        ticket = Ticket(id=next_ticket, run_id=run.id, agent_id='train')
        owner = InfraInstance(id='owner', run_id=run.id, ticket_id='train-1',
            instance_id='123', provider='cluster', gpu_count=8,
            meta=dict(controller_directory='/controller', scheduler_state='FAILED', nodes=1))
        session.add_all([run, ticket, owner])
        await session.commit()
        job, meta = await submit_workload(session, run=run, ticket=ticket, contract=contract,
            info=SimpleNamespace(cluster=SimpleNamespace(workdir='/remote'), resource_plan=SimpleNamespace(min_ram_gb=64, min_cpus=8, time_limit_hours=4)),
            heartbeat_id='h2', sha256='digest')
        assert job == '123'
        assert bool(meta['allocation_owner_row_id']) is reuses
        assert any('sbatch --parsable' in c for c in commands) is not reuses
    await engine.dispose()


def test_cancel_stops_child_but_retains_controller(tmp_path):
    for name in ('requests', 'outcomes', 'cancel'):
        (tmp_path / name).mkdir()
    script = tmp_path / 'train.sh'
    script.write_text('echo $$ > child.pid\nexec sleep 60\n')
    publish(tmp_path, '01', script)
    process = subprocess.Popen([sys.executable, '-c', CONTROLLER_SOURCE, str(tmp_path), '20'])
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / 'child.pid').exists() and time.monotonic() < deadline:
            time.sleep(.05)
        child_pid = int((tmp_path / 'child.pid').read_text())
        (tmp_path / 'cancel/01').touch()
        assert wait_outcome(tmp_path / 'outcomes/01.json', process)['exit_code'] != 0
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert process.poll() is None
        (tmp_path / 'release').touch()
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


@pytest.mark.parametrize('estimate,buffer,platform,user,minutes', [
    (8, None, 336, 0, 600), (8, 3, 10, 0, 600),
    (8, 3, 336, 9, 540), (8, 0, None, 0, 480),
    (8, 3, 336, 1.009, 60), (None, None, None, 0, 240),
])
def test_whole_run_walltime_caps(estimate, buffer, platform, user, minutes):
    from zevo.engine.run.gpu_controller import run_walltime_plan
    plan = SimpleNamespace(estimated_run_hours=estimate, runtime_buffer_hours=buffer,
                           platform_max_runtime_hours=platform, time_limit_hours=4)
    assert run_walltime_plan(plan, user)['requested_minutes'] == minutes


def test_run_controller_waits_for_next_stage_without_idle_expiry(tmp_path):
    for name in ('requests', 'outcomes', 'cancel'):
        (tmp_path / name).mkdir()
    process = subprocess.Popen([sys.executable, '-c', CONTROLLER_SOURCE, str(tmp_path), '0'])
    try:
        time.sleep(1.2)
        assert process.poll() is None
        script = tmp_path / 'next.sh'
        script.write_text('echo next-stage\n')
        publish(tmp_path, 'next', script)
        assert wait_outcome(tmp_path / 'outcomes/next.json', process)['exit_code'] == 0
        (tmp_path / 'release').touch()
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize('state,released', [('COMPLETED', True), ('RUNNING', False), ('', False)])
async def test_owner_monitor_after_source_ticket_completed(monkeypatch, state, released):
    from zevo.engine.run.scheduler import reconciler
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async def connection(*args):
        return dict(host='test', user='test')
    async def query(**kwargs):
        assert not kwargs.get('outcome_path')
        return state, '0:0', '', None
    monkeypatch.setattr(reconciler, '_slurm_connection', connection)
    monkeypatch.setattr(reconciler, '_query_slurm_job', query)
    async with Session() as session:
        run = Run(id='run', metric='accuracy', status='running')
        ticket = Ticket(id='data', run_id='run', agent_id='data', status='succeeded')
        owner = InfraInstance(id='owner', run_id='run', ticket_id='data',
            provider='cluster', instance_id='123', status='ready',
            meta=dict(controller_directory='/controller', scheduler_state='COMPLETED'))
        alias = InfraInstance(id='alias', run_id='run', ticket_id='data',
            provider='cluster', instance_id='123', status='ready',
            meta=dict(controller_directory='/controller', allocation_owner_row_id='owner'))
        session.add_all([run, ticket, owner, alias])
        await session.commit()
        assert await reconciler._reconcile_slurm_allocation_owners(session) == 1
        assert (owner.released_at is not None) is released
        assert (alias.released_at is not None) is released
        assert ticket.status == 'succeeded'
        assert owner.meta['scheduler_state'] == 'COMPLETED'
    await engine.dispose()


@pytest.mark.parametrize('provider,lane,state,expected', [
    # A successful optimization Inference on a direct host feeds the engine's
    # held-out chain on the same device: keep the rental.
    ('cloud', 'optimization', 'succeeded', False),
    ('cloud', 'optimization', 'degraded', False),
    ('instance', 'optimization', 'succeeded', False),
    # The held-out Inference is the end of that chain: release.
    ('cloud', 'held_out_test', 'succeeded', True),
    # A failed Inference spawns nothing: release.
    ('cloud', 'optimization', 'failed', True),
    # Cluster allocations are controller-owned; the rule does not apply.
    ('cluster', 'optimization', 'succeeded', True),
])
def test_per_stage_release_waits_for_the_engine_held_out_chain(provider, lane, state, expected):
    run = SimpleNamespace(gpu_allocation_mode='per_stage', gpu_provider=provider)
    ticket = SimpleNamespace(status=state, agent_id='inference', lane=lane)
    assert needs_stage_release(run, ticket) is expected


def test_train_release_boundary_is_unchanged_on_cloud():
    run = SimpleNamespace(gpu_allocation_mode='per_stage', gpu_provider='cloud')
    assert needs_stage_release(run, SimpleNamespace(status='succeeded', agent_id='train', lane='optimization')) is True
    assert needs_stage_release(run, SimpleNamespace(status='repairing', agent_id='train', lane='optimization')) is False
