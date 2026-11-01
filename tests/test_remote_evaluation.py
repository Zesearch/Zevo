from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from zevo.contracts.evaluation import EvaluationTaskInput, EvaluationResult, EvaluationSuiteMemberInput
from zevo.contracts.infrastructure import InfrastructureDeviceInfo, SlurmStageJobContract, slurm_lifecycle_prologue, slurm_runtime_prologue
from zevo.engine import remote_evaluation as remote
from zevo.engine.agent.drivers.evaluation_runner import EvaluationRunnerDriver
from tests.test_ssh_provider_contract import _device


def work(tmp_path):
    prediction = tmp_path / 'predictions.csv'
    prediction.write_text('id,prediction\n1,A\n')
    gold = tmp_path / 'gold.csv'
    gold.write_text('id,answer\n1,A\n')
    return EvaluationTaskInput(ticket_id='eval-test-001', predictions_path=str(prediction),
        scoring_set=str(gold), sample_submission=str(prediction), metric='accuracy',
        answer_fields=['answer'], evaluation_script='')


def cluster_contract(tmp_path):
    status = '/scratch/eval/attempt-1/.zevo-slurm-status'
    return SlurmStageJobContract(enabled=True, stage='evaluation', estimated_gpus=1,
        resource_plan_source='site minimum', resource_plan_rationale='CPU scoring on allocated node',
        script_path=str(tmp_path / 'evaluate.sbatch'), remote_work_dir='/scratch/eval/attempt-1',
        remote_script_path='/scratch/eval/attempt-1/evaluate.sbatch', job_name='zevo-eval-test-001',
        status_path=status, stdout_path='/scratch/eval/attempt-1/slurm-%j.out',
        stderr_path='/scratch/eval/attempt-1/slurm-%j.err',
        lifecycle_prologue=slurm_lifecycle_prologue(status), runtime_prologue=slurm_runtime_prologue())


def test_bundle_runs_same_scoring_and_reuses_only_verified_results(tmp_path, monkeypatch):
    inp = work(tmp_path)
    staging = tmp_path / 'staging'; staging.mkdir()
    host = tmp_path / 'compute'; host.mkdir()
    archive = remote.make_bundle(inp, staging, str(host))
    with tarfile.open(archive) as tar:
        tar.extractall(host, filter='data')
    monkeypatch.setenv('ZEVO_EVALUATION_WORKER', '1')
    def run():
        env = host / '.judge-env.json'; env.write_text('{}')
        return remote.worker(host / 'bundle/input.json', host / 'results', env)
    assert run() == 0
    output = EvaluationResult.model_validate_json((host / 'results/result.json').read_text())
    assert json.loads(Path(output.metrics_path).read_text())['score'] == 1
    before = Path(output.metrics_path).stat().st_mtime_ns
    assert run() == 0
    assert Path(output.metrics_path).stat().st_mtime_ns == before
    # A changed prediction invalidates the receipt rather than retaining score=1.
    frozen = json.loads((host / 'bundle/input.json').read_text())
    Path(frozen['predictions_path']).write_text('id,prediction\n1,B\n')
    assert run() == 0
    assert json.loads(Path(output.metrics_path).read_text())['score'] == 0
    # Tampering with cached metrics also invalidates reuse.
    Path(output.metrics_path).write_text('{"score":0.99}')
    assert run() == 0
    assert json.loads(Path(output.metrics_path).read_text())['score'] == 0


def test_bundle_preserves_frozen_evaluator_bytes_and_rejects_changed_script(tmp_path):
    inp = work(tmp_path)
    script = tmp_path / 'score.py'; script.write_text('print("original")')
    inp.evaluation_script = str(script); inp.evaluator_sha256 = remote.digest(script)
    staging = tmp_path / 'one'; staging.mkdir()
    remote.make_bundle(inp, staging, '/remote')
    copied = staging / 'bundle/inputs/0/evaluation_script/score.py'
    assert copied.read_bytes() == script.read_bytes()
    script.write_text('print("changed")')
    staging = tmp_path / 'two'; staging.mkdir()
    with pytest.raises(ValueError, match='digest|sha256|changed|match'):
        remote.make_bundle(inp, staging, '/remote')


def test_cluster_script_cannot_execute_on_login_and_contains_no_credentials(tmp_path, monkeypatch):
    inp = work(tmp_path); inp.slurm_job = cluster_contract(tmp_path)
    info = InfrastructureDeviceInfo.model_validate(_device('cluster'))
    monkeypatch.setenv('OPENAI_API_KEY', 'never-in-script')
    script = remote.launch_script(inp, info, '/scratch/eval/evaluation')
    assert 'never-in-script' not in script
    path = tmp_path / 'evaluate.sbatch'; path.write_text(script)
    completed = subprocess.run(['bash', str(path)], capture_output=True, text=True, env={'PATH': '/usr/bin:/bin'})
    assert completed.returncode != 0
    assert 'requires a Slurm allocation' in completed.stderr
    assert 'zevo_slurm_event EXITED' in script


@pytest.mark.parametrize('provider', ['instance', 'cloud'])
def test_direct_host_script_uses_compute_route_without_sbatch(tmp_path, provider):
    inp = work(tmp_path)
    raw = _device('instance')
    if provider == 'cloud':
        raw.update(provider='cloud', cloud_backend='vastai', auto_release=True)
        raw['resource_plan']['cloud_backend'] = 'vastai'
    info = InfrastructureDeviceInfo.model_validate(raw)
    script = remote.launch_script(inp, info, '/compute/eval')
    assert '#SBATCH' not in script
    assert 'ZEVO_TICKET_ID=eval-test-001' in script
    assert 'flock -n' in script
    assert 'python3 -m zevo.engine.remote_evaluation' in script


def test_judge_environment_cannot_forward_compute_or_service_secrets(monkeypatch):
    scope = pytest.importorskip('zevo.engine.agent.provider_scope')
    ProviderScope, set_scope, reset_scope = scope.ProviderScope, scope.set_scope, scope.reset_scope
    monkeypatch.setenv('ZEVO_DATABASE_URL', 'private-db')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'compute-secret')
    monkeypatch.setenv('OPENAI_API_KEY', 'wrong-owner')
    token = set_scope(ProviderScope(keys={'OPENAI_API_KEY': 'scoped-key', 'VASTAI_API_KEY': 'rental'}))
    try:
        env = remote.judge_environment()
    finally:
        reset_scope(token)
    assert env['OPENAI_API_KEY'] == 'scoped-key'
    assert not {'AWS_SECRET_ACCESS_KEY', 'VASTAI_API_KEY', 'ZEVO_DATABASE_URL'} & env.keys()


@pytest.mark.asyncio
async def test_cluster_submit_only_prepares_then_collect_does_not_rerun(tmp_path, monkeypatch):
    inp = work(tmp_path)
    info = InfrastructureDeviceInfo.model_validate(_device('cluster'))
    device = tmp_path / 'device.json'; device.write_text(info.model_dump_json())
    inp.device_info_path = str(device)
    inp.remote_work_dir = '/scratch/eval/evaluation'
    inp.slurm_job = cluster_contract(tmp_path)
    calls = []
    async def command(argv, **kwargs):
        calls.append(argv)
        return ''
    monkeypatch.setattr(remote, 'command', command)
    result = await EvaluationRunnerDriver().run_agent(blueprint=None, input_payload=inp, workspace_dir=str(tmp_path))
    assert result.output.status == 'deferred'
    assert not any('sbatch --' in ' '.join(call) for call in calls)
    assert not any('bash evaluate' in ' '.join(call) for call in calls)
    assert (tmp_path / 'evaluation-bundle.sha256').is_file()
    calls.clear()
    inp.slurm_job = inp.slurm_job.model_copy(update={'phase': 'collect', 'scheduler_state': 'RUNNING'})
    result = await remote.run_remote(inp, str(tmp_path))
    assert result.output.status == 'deferred'
    assert calls == []
    inp.slurm_job = inp.slurm_job.model_copy(update={'scheduler_state': 'COMPLETED'})
    expected = EvaluationResult(ticket_id=inp.ticket_id, status='succeeded', metrics_path='/metrics', notes='', error_message='')
    async def collect(*args): return expected
    monkeypatch.setattr(remote, 'collect', collect)
    result = await remote.run_remote(inp, str(tmp_path))
    assert result.output == expected
    assert calls == []


def test_evaluation_deferred_does_not_require_metrics_artifact(tmp_path):
    from zevo.engine.run.runner import _extract_summary_artifact_meta
    inp = work(tmp_path)
    result = EvaluationResult(ticket_id=inp.ticket_id, status='deferred', metrics_path='', error_message='', notes='submitted')
    assert _extract_summary_artifact_meta(result, inp=inp, agent_id='evaluation', work_dir=str(tmp_path)) == ('deferred', 'submitted', '', {})


@pytest.mark.parametrize('lane,expected', [('optimization', False), ('held_out_test', True)])
def test_evaluation_is_the_direct_host_release_boundary(lane, expected):
    from zevo.engine.run.gpu_lifecycle import needs_stage_release
    assert needs_stage_release(SimpleNamespace(gpu_provider='cloud', gpu_allocation_mode='per_stage'),
        SimpleNamespace(agent_id='evaluation', status='succeeded', lane=lane)) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['cluster', 'instance', 'cloud'])
async def test_engine_inherits_prediction_compute_route_and_rejects_released_host(tmp_path, monkeypatch, provider):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from zevo.db import Run, Ticket, WorkProduct, InfraInstance, GpuLease
    from zevo.db.models import Base
    from zevo.engine.run import runner
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    raw = _device('cluster' if provider == 'cluster' else 'instance')
    if provider == 'cloud':
        raw.update(provider='cloud', cloud_backend='vastai', auto_release=True)
        raw['resource_plan']['cloud_backend'] = 'vastai'
    device = tmp_path / 'device.json'; device.write_text(json.dumps(raw))
    async def probe(*args): raise ValueError('no live capacity in unit test')
    monkeypatch.setattr(runner, 'probe_slurm_capacity', probe)
    async with Session() as session:
        run = Run(id=raw['run_id'], metric='accuracy', gpu_provider=provider)
        infra = Ticket(id=raw['ticket_id'], run_id=run.id, agent_id='infrastructure', status='succeeded')
        product = WorkProduct(id='device-product', ticket_id=infra.id, role='device_info', path=str(device))
        source = Ticket(id='infer-test-001', run_id=run.id, agent_id='inference', status='succeeded', inputs={
            'device_info': {'source_ticket_id': infra.id, 'artifact_role': 'device_info', 'work_product_id': product.id}})
        ticket = Ticket(id='eval-test-001', run_id=run.id, agent_id='evaluation', status='queued', inputs={
            'predictions': {'source_ticket_id': source.id, 'artifact_role': 'predictions'}})
        session.add_all([run, infra, product, source, ticket]); await session.flush()
        inp = await runner._bind_evaluation_compute(session, run, ticket, ticket.inputs, str(tmp_path), work(tmp_path))
        assert inp.device_info_path == str(device)
        assert inp.slurm_job.enabled == (provider == 'cluster')
        assert inp.remote_work_dir.endswith('/eval-test-001/evaluation')
        assert ticket.inputs['device_info']['work_product_id'] == product.id
        if provider == 'cloud':
            from datetime import datetime, timezone
            session.add(InfraInstance(instance_id=raw['instance_id'], provider='cloud', run_id=run.id,
                ticket_id=infra.id, released_at=datetime.now(timezone.utc)))
            await session.flush()
            with pytest.raises(ValueError, match='released'):
                await runner._bind_evaluation_compute(session, run, ticket, ticket.inputs, str(tmp_path), inp)
    await engine.dispose()


def test_worker_rejects_modified_submitted_bundle_before_scoring(tmp_path, monkeypatch):
    inp = work(tmp_path)
    stage = tmp_path / 'stage'; stage.mkdir()
    host = tmp_path / 'host'; host.mkdir()
    archive = remote.make_bundle(inp, stage, str(host))
    with tarfile.open(archive) as tar:
        tar.extractall(host, filter='data')
    monkeypatch.setenv('ZEVO_EVALUATION_MANIFEST_SHA256', remote.digest(host / 'bundle/manifest.json'))
    staged = json.loads((host / 'bundle/input.json').read_text())
    Path(staged['scoring_set']).write_text('id,answer\n1,changed\n')
    environment = host / '.judge-env.json'; environment.write_text('{}')
    with pytest.raises(ValueError, match='changed after submission'):
        remote.worker(host / 'bundle/input.json', host / 'results', environment)
    assert not (host / 'results/metrics.json').exists()
    assert not environment.exists()


def test_scoring_reuses_environment_without_installing_dependencies(tmp_path):
    inp = work(tmp_path); inp.slurm_job = cluster_contract(tmp_path)
    info = InfrastructureDeviceInfo.model_validate(_device('cluster'))
    info.cluster.env_setup = 'source /scratch/envs/env.sh'
    script = remote.launch_script(inp, info, '/scratch/eval/evaluation')
    assert script.index(info.cluster.env_setup) < script.index('python3 -m zevo.engine.remote_evaluation')
    assert 'pip install' not in script
    assert 'venv' not in script
    assert 'python install' not in script
    assert 'bootstrap-runtimes' not in script


@pytest.mark.asyncio
async def test_cancel_direct_evaluation_removes_queued_judge_credentials(monkeypatch):
    from zevo.engine.run import remote_jobs
    raw = _device('instance')
    raw.update(provider='cloud', cloud_backend='vastai', auto_release=True)
    raw['resource_plan']['cloud_backend'] = 'vastai'
    info = InfrastructureDeviceInfo.model_validate(raw)
    commands = []
    async def device(*args): return info
    async def ssh(*args, **kwargs):
        commands.append(args[1]); return {'ok': True, 'exit_code': 0}
    monkeypatch.setattr(remote_jobs, '_device_info_for_ticket', device)
    monkeypatch.setattr(remote_jobs, '_ssh', ssh)
    result = await remote_jobs.cancel_ticket_remote_job(None, SimpleNamespace(
        id='eval-test-001', run_id=info.run_id, agent_id='evaluation'))
    assert result['ok']
    assert commands[0] == 'rm -f -- /scratch/zevo/run/eval-test-001/evaluation/.judge-env.json'
    assert commands[1].startswith('python3 -c ')


def test_container_mounts_include_stage_script_and_sibling_environment(tmp_path):
    inp = work(tmp_path); inp.slurm_job = cluster_contract(tmp_path)
    info = InfrastructureDeviceInfo.model_validate(_device('cluster'))
    info.cluster.workdir = '/scratch/eval'
    info.cluster.env_setup = 'source /scratch/envs/env.sh'
    info.cluster.container_image = 'test.sqsh'
    script = remote.launch_script(inp, info, '/scratch/eval/evaluation')
    assert '#SBATCH --container-mounts=/scratch:/scratch' in script
    assert '/app/data' not in next(line for line in script.splitlines() if '--container-mounts=' in line)


def test_container_mounts_do_not_mount_host_root(tmp_path):
    inp = work(tmp_path); inp.slurm_job = cluster_contract(tmp_path)
    info = InfrastructureDeviceInfo.model_validate(_device('cluster'))
    info.cluster.workdir = '/scratch/eval'
    info.cluster.env_setup = 'source /projects/user/envs/env.sh'
    mounts = remote.container_mounts(inp, info, '/scratch/eval/evaluation')
    assert '/scratch/eval:/scratch/eval' in mounts
    assert '/projects/user/envs:/projects/user/envs' in mounts
    assert '/:/' not in mounts


@pytest.mark.parametrize('root_variable', ['ZEVO_EVALUATION_ROOT', 'ZEVO_HOLDOUT_ROOT'])
@pytest.mark.parametrize('inherited_root', [None, '/app/data/.holdout'])
def test_worker_relocates_judge_audits_to_durable_compute_results(tmp_path, monkeypatch, inherited_root, root_variable):
    import os
    for key in ['ZEVO_EVALUATION_ROOT', 'ZEVO_HOLDOUT_ROOT']:
        if inherited_root is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, inherited_root + '/' + key)
    inp = work(tmp_path)
    script = tmp_path / 'judge.py'
    script.write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        f"root = Path(os.environ.get({root_variable!r}, '/app/data/.holdout'))\n"
        "assert os.environ['ZEVO_EVALUATION_ROOT'] == os.environ['ZEVO_HOLDOUT_ROOT']\n"
        "audit = root / 'evaluation-audits' / 'alpaca'\n"
        "audit.mkdir(parents=True, exist_ok=True)\n"
        "cache = audit / 'calls.txt'\n"
        "cache.write_text(cache.read_text() + 'x' if cache.exists() else 'x')\n"
        "Path(sys.argv[3]).write_text(json.dumps({'score': 1, 'audit_root': str(root)}))\n"
    )
    inp.evaluation_script = str(script)
    inp.evaluator_sha256 = remote.digest(script)
    staging = tmp_path / 'staging'; staging.mkdir()
    host = tmp_path / 'compute'; host.mkdir()
    archive = remote.make_bundle(inp, staging, str(host))
    with tarfile.open(archive) as tar:
        tar.extractall(host, filter='data')
    output = host / 'results'
    # Force two executions to check that the audit location is persistent,
    # rather than having a cached receipt mask the second scorer invocation.
    monkeypatch.delenv('ZEVO_EVALUATION_WORKER', raising=False)
    for _ in range(2):
        environment = host / '.judge-env.json'; environment.write_text('{}')
        assert remote.worker(host / 'bundle/input.json', output, environment) == 0
        metrics = json.loads((output / 'metrics.json').read_text())
        assert Path(metrics['audit_root']) == output
        for key in ['ZEVO_EVALUATION_ROOT', 'ZEVO_HOLDOUT_ROOT']:
            assert os.environ.get(key) == (inherited_root + '/' + key if inherited_root else None)
        assert not environment.exists()
    assert (output / 'evaluation-audits/alpaca/calls.txt').read_text() == 'xx'
