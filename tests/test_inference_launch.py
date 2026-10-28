from types import SimpleNamespace
from pathlib import Path
import hashlib

import pytest

from zevo.engine.run.inference_launch import (
    digest, render, snapshot, frozen_inputs, verify_frozen_inputs, effective_snapshot,
)


def input_for(root, ticket='infer-1', model='org/model'):
    work = root / ticket
    work.mkdir(parents=True, exist_ok=True)
    (work / 'predict.py').write_text('print("predict")')
    config = root / 'frozen.yaml'
    if not config.exists():
        config.write_text('temperature: 0\n')
    job = SimpleNamespace(remote_work_dir=f'/remote/{ticket}/attempt-1',
        job_name=f'zevo-{ticket}', nodes=1, gpus_per_node=4,
        script_path=str(work / 'predict.sbatch'),
        stdout_path=f'/remote/{ticket}/attempt-1/slurm-%j.out',
        stderr_path=f'/remote/{ticket}/attempt-1/slurm-%j.err',
        lifecycle_prologue=f'export STATUS=/remote/{ticket}/attempt-1/status',
        phase='submit', enabled=True, attempt=1, previous_execution={})
    job.model_dump = lambda **_: {k:v for k,v in vars(job).items() if k != 'model_dump'}
    return SimpleNamespace(ticket_id=ticket, work_dir=str(work), slurm_job=job,
        checkpoint_path='' if model == 'org/model' else model, base_model='org/model',
        inference_config_path=str(config), scoring_set=f'/questions/{ticket}.csv',
        sample_submission=f'/samples/{ticket}.csv', reusable_predict_script_path=str(work/'predict.py'),
        suite_members=[], configuration_mode='reuse')


def successful_body(inp):
    return f'''#!/bin/bash
#SBATCH --job-name={inp.slurm_job.job_name}
#SBATCH --partition=beta
#SBATCH --gres=gpu:b200:4
#SBATCH --nodes=1
#SBATCH --output={inp.slurm_job.stdout_path}
#SBATCH --error={inp.slurm_job.stderr_path}
{inp.slurm_job.lifecycle_prologue}
source /verified/env.sh
python {inp.slurm_job.remote_work_dir}/predict.py --model org/model --tokenizer org/model --suite {inp.slurm_job.remote_work_dir}/suite.json --ticket-id {inp.ticket_id}
'''


def test_replay_changes_model_and_paths_but_preserves_tokenizer_environment(tmp_path):
    old = input_for(tmp_path)
    new = input_for(tmp_path, 'mid-infer-2', '/checkpoints/train-2/checkpoint-111')
    body = render(snapshot(old, successful_body(old)), new)
    assert '--model /checkpoints/train-2/checkpoint-111 --tokenizer org/model' in body
    assert 'source /verified/env.sh' in body
    assert '#SBATCH --partition=beta' in body
    assert '--gpus-per-node=b200:4' in body and '--gres=' not in body
    assert '/remote/infer-1/' not in body
    assert '/remote/mid-infer-2/attempt-1/suite.json' in body
    assert new.slurm_job.lifecycle_prologue in body
    assert '--ticket-id mid-infer-2' in body


def test_model_variable_rebinding_is_narrow_and_shell_safe(tmp_path):
    old = input_for(tmp_path)
    new = input_for(tmp_path, 'infer-2', '/checkpoint with spaces/model')
    body = successful_body(old).replace('python ', 'MODEL="org/model"\npython ').replace('--model org/model', '--model "$MODEL"')
    result = render(snapshot(old, body), new)
    assert "MODEL='/checkpoint with spaces/model'" in result
    assert '--tokenizer org/model' in result
    with pytest.raises(ValueError, match='safely rebindable'):
        render(snapshot(old, successful_body(old).replace('--model org/model', '--model "$(resolve_model)"')), new)


def test_only_config_is_frozen_while_predict_is_advisory(tmp_path):
    inp = input_for(tmp_path)
    pins = frozen_inputs(inp)
    verify_frozen_inputs(pins)
    Path(inp.inference_config_path).write_text('temperature: 1\n')
    with pytest.raises(ValueError, match='changed after input binding'):
        verify_frozen_inputs(pins)
    Path(inp.inference_config_path).write_text('temperature: 0\n')
    Path(inp.reusable_predict_script_path).write_text('changed code')
    assert str(Path(inp.reusable_predict_script_path).resolve()) not in pins
    verify_frozen_inputs(pins)


def test_snapshot_cannot_be_modified(tmp_path):
    inp = input_for(tmp_path)
    source = snapshot(inp, successful_body(inp))
    source['body'] += 'pip install different-runtime\n'
    with pytest.raises(ValueError, match='snapshot changed'):
        render(source, input_for(tmp_path, 'infer-2', '/checkpoint'))


def test_retained_allocation_does_not_activate_inert_container_directives(tmp_path):
    inp = input_for(tmp_path)
    source = snapshot(inp, successful_body(inp).replace('#SBATCH --partition=beta',
        '#SBATCH --partition=beta\n#SBATCH --container-image=never-used.sqsh\n#SBATCH --container-mounts=/a:/a'))
    effective = effective_snapshot(source, '#!/bin/bash\n#SBATCH --partition=beta\nexec python3 controller.py\n')
    assert '--container-' not in effective['body']
    assert 'source /verified/env.sh' in effective['body']
    assert effective['sha256'] == hashlib.sha256(effective['body'].encode()).hexdigest()
    used = effective_snapshot(source, '#SBATCH --container-image=actually-used.sqsh')
    assert '--container-image=actually-used.sqsh' in used['body']
    assert 'never-used.sqsh' not in used['body']


@pytest.mark.asyncio
@pytest.mark.parametrize('source_lane,target_lane,expected', [
    ('optimization','optimization','reuse'), ('held_out_test','held_out_test','reuse'),
    ('optimization','held_out_test','reuse'), ('held_out_test','optimization','unavailable'),
])
async def test_only_successful_compatible_launches_and_lane_isolation(tmp_path, source_lane,target_lane,expected):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from zevo.db import Base, Run, Ticket, InfraInstance
    from zevo.engine.run.inference_launch import prepare_launch
    engine=create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    old=input_for(tmp_path)
    new=input_for(tmp_path,'infer-2','/checkpoint')
    new.model_copy=lambda *,update: SimpleNamespace(**{**vars(new), **update})
    async with async_sessionmaker(engine,expire_on_commit=False)() as session:
        run=Run(id='run',metric='accuracy')
        owner=Ticket(id=old.ticket_id,run_id='run',agent_id='inference',status='succeeded',lane=source_lane,payload={'base_model':'org/model'})
        target=Ticket(id=new.ticket_id,run_id='run',agent_id='inference',status='queued',lane=target_lane)
        job=InfraInstance(id='source-job',run_id='run',ticket_id=owner.id,provider='cluster',instance_id='123',
            meta={'scheduler_state':'COMPLETED','submission_committed':True,'inference_launch':snapshot(old,successful_body(old))})
        session.add_all([run,owner,target,job]);await session.commit()
        result=await prepare_launch(session,run,target,new)
        assert result.slurm_launch_reuse['mode']==expected
        if expected=='reuse':
            assert result.slurm_launch_reuse['source_execution_id']=='source-job'
            assert Path(new.slurm_job.script_path).is_file()
        owner.status='failed';await session.commit()
        result=await prepare_launch(session,run,target,new)
        # Later ticket failures do not erase a completed execution's evidence.
        assert result.slurm_launch_reuse['mode']==expected
    await engine.dispose()

@pytest.mark.asyncio
async def test_real_failure_allows_repair_and_retries_keep_original_config_pins(tmp_path):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from zevo.db import Base, Run, Ticket, InfraInstance
    from zevo.engine.run.inference_launch import prepare_launch
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    old = input_for(tmp_path)
    new = input_for(tmp_path, 'infer-2', '/checkpoint')
    new.model_copy = lambda *,update: SimpleNamespace(**{**vars(new), **update})
    async with async_sessionmaker(engine,expire_on_commit=False)() as session:
        run = Run(id='run',metric='accuracy')
        owner = Ticket(id=old.ticket_id,run_id='run',agent_id='inference',status='succeeded',payload={'base_model':'org/model'})
        target = Ticket(id=new.ticket_id,run_id='run',agent_id='inference',status='queued')
        job = InfraInstance(id='source',run_id='run',ticket_id=owner.id,provider='cluster',instance_id='123',
            meta={'scheduler_state':'COMPLETED','submission_committed':True,'inference_launch':snapshot(old,successful_body(old))})
        session.add_all([run,owner,target,job]);await session.commit()
        assert (await prepare_launch(session,run,target,new)).slurm_launch_reuse['mode'] == 'reuse'
        await session.commit()
        # A normal retry has no authority to rewrite a previously successful launch.
        assert (await prepare_launch(session,run,target,new)).slurm_launch_reuse['mode'] == 'reuse'
        new.slurm_job.attempt = 2
        new.slurm_job.previous_execution = {'scheduler_state':'FAILED'}
        repaired = await prepare_launch(session,run,target,new)
        assert repaired.slurm_launch_reuse['mode'] == 'repair'
        assert 'expected_sha256' not in repaired.slurm_launch_reuse
        # But repair cannot make an in-place decoding change the new baseline.
        Path(new.inference_config_path).write_text('temperature: 1\n')
        with pytest.raises(ValueError,match='changed after input binding'):
            await prepare_launch(session,run,target,new)
    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize('directly_rebindable', [True, False])
async def test_first_test_baseline_inherits_launch_while_selecting_its_config(tmp_path, directly_rebindable):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from zevo.db import Base, Run, Ticket, InfraInstance
    from zevo.engine.run.inference_launch import prepare_launch
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    old = input_for(tmp_path)
    new = input_for(tmp_path, 'test-baseline')
    new.configuration_mode = 'select'
    new.inference_config_path = ''
    new.reusable_predict_script_path = old.reusable_predict_script_path
    new.model_copy = lambda *, update: SimpleNamespace(**{**vars(new), **update})
    body = successful_body(old)
    if not directly_rebindable:
        body = body.replace('--model org/model', '--model "$(resolve_model)"')
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            run = Run(id='run', metric='accuracy')
            owner = Ticket(id=old.ticket_id, run_id=run.id, agent_id='inference',
                status='succeeded', lane='optimization', payload={'base_model':'org/model'})
            target = Ticket(id=new.ticket_id, run_id=run.id, agent_id='inference',
                status='queued', lane='held_out_test')
            job = InfraInstance(id='validation-job', run_id=run.id, ticket_id=owner.id,
                provider='cluster', instance_id='123', meta={'scheduler_state':'COMPLETED',
                    'submission_committed':True, 'inference_launch':snapshot(old, body)})
            session.add_all([run, owner, target, job])
            await session.commit()
            result = await prepare_launch(session, run, target, new)
            launch = result.slurm_launch_reuse
            assert result.configuration_mode == 'select'
            assert result.inference_config_path == ''
            assert launch['mode'] == 'adapt'
            assert launch['source_ticket_id'] == owner.id
            assert launch['source_execution_id'] == job.id
            assert Path(launch['source_script_path']).read_text() == body
            assert 'expected_sha256' not in launch
            # Script references are advisory; no Test config has been chosen yet.
            assert result.frozen_reuse_artifacts == {}
            if directly_rebindable:
                rendered = Path(new.slurm_job.script_path).read_text()
                assert 'source /verified/env.sh' in rendered
                assert new.slurm_job.remote_work_dir in rendered
                assert old.slurm_job.remote_work_dir not in rendered
            else:
                assert 'safely rebindable' in launch['reason']
            # New target artifacts can be written without mutating Validation.
            Path(new.work_dir, 'inference_config.yaml').write_text('max_new_tokens: 4096\n')
            verify_frozen_inputs(result.frozen_reuse_artifacts)
            Path(old.reusable_predict_script_path).unlink()
            await prepare_launch(session, run, target, new)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_initial_validation_without_source_still_selects_from_scratch(tmp_path):
    from zevo.engine.run.inference_launch import prepare_launch
    inp = input_for(tmp_path)
    inp.configuration_mode = 'select'
    inp.inference_config_path = inp.reusable_predict_script_path = ''
    assert await prepare_launch(None, None, None, inp) is inp


@pytest.mark.parametrize("reference_mode", ["reuse", "unavailable"])
def test_launch_adaptation_allowed_before_submission_but_not_during_collect(tmp_path, reference_mode):
    from zevo.contracts.inference import InferenceTaskInput
    from zevo.contracts.infrastructure import SlurmStageJobContract
    from zevo.engine.run.runner import _slurm_stage_result_problem
    inp = input_for(tmp_path)
    path = Path(inp.slurm_job.script_path)
    path.write_text(successful_body(inp))
    contract = SlurmStageJobContract.model_construct(**{
        **inp.slurm_job.model_dump(), 'num_gpus':4, 'status_path':'/remote/status',
        'runtime_prologue':'export TMPDIR=/tmp/job'})
    path.write_text(path.read_text().replace('source /verified/env.sh',
        'export TMPDIR=/tmp/job\n# /remote/status\nsource /verified/env.sh'))
    typed = InferenceTaskInput.model_construct(slurm_job=contract,
        frozen_reuse_artifacts=frozen_inputs(inp),
        slurm_launch_reuse={'mode':reference_mode,'expected_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    path.write_text(path.read_text()+'# adapt current suite and config guards\n')
    assert _slurm_stage_result_problem(typed,str(path),'predict.py') == ''
    contract.phase = 'collect'
    assert 'launch changed' in _slurm_stage_result_problem(typed,str(path),'predict.py')
    Path(inp.inference_config_path).write_text('temperature: 1\n')
    assert 'changed after input binding' in _slurm_stage_result_problem(typed,str(path),'predict.py')


def test_controller_initializes_environment_before_python(tmp_path):
    import subprocess
    from zevo.engine.run.gpu_controller import controller_script
    marker = tmp_path / 'environment-ready'
    # Deliberately fail command discovery after proving the configured bootstrap ran.
    script = controller_script('#!/bin/bash\n#SBATCH --nodes=1\n', str(tmp_path),
        env_setup=f'touch {marker}\nexport PATH=/nonexistent')
    result = subprocess.run(['/bin/bash','-c',script],capture_output=True,text=True)
    assert result.returncode == 127
    assert marker.exists()
    assert 'requires python3 after env_setup' in result.stderr


@pytest.mark.asyncio
async def test_existing_launch_import_checks_real_inference_implementation(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock
    from zevo.engine.run.inference_launch import verified_existing_launch, digest
    from zevo.engine.run.runner import _slurm_stage_implementation_sha256
    from zevo.engine.run import remote_jobs
    inp = input_for(tmp_path)
    script = Path(inp.slurm_job.script_path)
    script.write_text(successful_body(inp))
    contract = SimpleNamespace(stage="inference", script_path=str(script))
    row = SimpleNamespace(meta={
        "script_sha256": digest(script),
        "implementation_sha256": _slurm_stage_implementation_sha256(contract),
        "remote_workdir": inp.slurm_job.remote_work_dir,
        "status_path": "/remote/status", "controller_directory": "/remote/controller",
    })
    owner = SimpleNamespace(id=inp.ticket_id, inputs={}, payload={"base_model": "org/model"})
    result = SimpleNamespace(output={"slurm_script_path": str(script),
        "predict_script_path": inp.reusable_predict_script_path})
    session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(
        scalars=lambda: SimpleNamespace(all=lambda: [result]))))
    monkeypatch.setattr(remote_jobs, "_device_info_for_ticket", AsyncMock(return_value=object()))
    monkeypatch.setattr(remote_jobs, "_ssh", AsyncMock(return_value={
        "ok": True, "stdout": "#!/bin/bash\n#SBATCH --partition=beta\n"}))
    source = await verified_existing_launch(session, row, owner)
    assert source is not None
    assert source["predict_sha256"] == digest(inp.reusable_predict_script_path)
    different = tmp_path / "different_predict.py"
    different.write_text("different implementation")
    result.output["predict_script_path"] = str(different)
    assert await verified_existing_launch(session, row, owner) is None
    result.output["predict_script_path"] = inp.reusable_predict_script_path
    Path(inp.reusable_predict_script_path).write_text("changed implementation")
    assert await verified_existing_launch(session, row, owner) is None


def test_shared_model_variable_updates_model_and_tokenizer_together(tmp_path):
    old = input_for(tmp_path)
    new = input_for(tmp_path, 'infer-2', '/checkpoint/new')
    body = successful_body(old).replace('python ', 'MODEL="org/model"\npython ').replace(
        '--model org/model --tokenizer org/model', '--model "$MODEL" --tokenizer "$MODEL"')
    rendered = render(snapshot(old, body), new)
    assert 'MODEL=/checkpoint/new' in rendered
    assert '--model "$MODEL" --tokenizer "$MODEL"' in rendered


@pytest.mark.asyncio
async def test_controller_headers_are_complete_or_unavailable(monkeypatch):
    from unittest.mock import AsyncMock
    from zevo.engine.run import remote_jobs
    from zevo.engine.run.gpu_controller import read_controller_headers
    headers = '#SBATCH --container-mounts=' + '/mount,' * 300 + '\n'
    ssh = AsyncMock(return_value={'ok': True, 'stdout': headers, 'stdout_truncated': False})
    monkeypatch.setattr(remote_jobs, '_ssh', ssh)
    assert await read_controller_headers(object(), '/controller') == headers
    assert ssh.call_args.kwargs['stdout_limit'] == 65536
    ssh.return_value = {'ok': True, 'stdout': headers[:1000], 'stdout_truncated': True}
    assert await read_controller_headers(object(), '/controller') is None


@pytest.mark.asyncio
async def test_ssh_marks_truncation_and_supports_full_header_reads(monkeypatch):
    from unittest.mock import AsyncMock
    from zevo.engine.run import remote_jobs
    stdout = b'x' * 2000
    process = SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(stdout, b'')))
    monkeypatch.setattr(remote_jobs.asyncio, 'create_subprocess_exec', AsyncMock(return_value=process))
    monkeypatch.setattr(remote_jobs, 'ssh_base_args', lambda _: [])
    info = SimpleNamespace(ssh=SimpleNamespace(user='user', host='host'))
    limited = await remote_jobs._ssh(info, 'read headers')
    assert len(limited['stdout']) == 1000 and limited['stdout_truncated']
    complete = await remote_jobs._ssh(info, 'read headers', stdout_limit=65536)
    assert len(complete['stdout']) == 2000 and not complete['stdout_truncated']


@pytest.mark.asyncio
async def test_remeasurement_prefers_own_completed_execution_and_preserves_repairs(tmp_path, monkeypatch):
    from datetime import datetime, timezone, timedelta
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from zevo.db import Base, Run, Ticket, InfraInstance
    from zevo.engine.run.inference_launch import prepare_launch
    from zevo.engine.run import remeasurement
    revision = {'revision':'length-1024'}
    monkeypatch.setattr(remeasurement, 'active_attempt', lambda *_: revision)
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    old = input_for(tmp_path, 'current', '/checkpoint/current')
    source = snapshot(old, successful_body(old).replace('--model org/model',
        '--model /checkpoint/current') + '# current successful execution\n')
    new = input_for(tmp_path, 'current', '/checkpoint/current')
    new.slurm_job.attempt = 3
    new.slurm_job.remote_work_dir = '/remote/current/attempt-3'
    new_config = tmp_path / 'revised-1024.yaml'
    new_config.write_text('temperature: 0\nmax_new_tokens: 1024\n')
    new.inference_config_path = str(new_config)
    new.model_copy = lambda *, update: SimpleNamespace(**{**vars(new), **update})
    other = input_for(tmp_path, 'other')
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            run = Run(id='run', metric='accuracy')
            ticket = Ticket(id='current', run_id=run.id, agent_id='inference',
                status='running', payload={'base_model':'org/model'})
            older_ticket = Ticket(id='other', run_id=run.id, agent_id='inference',
                status='succeeded', payload={'base_model':'org/model'})
            now = datetime.now(timezone.utc)
            jobs = [InfraInstance(id='own-completed', run_id=run.id, ticket_id=ticket.id,
                provider='cluster', instance_id='123', created_at=now-timedelta(hours=1),
                meta={'scheduler_state':'COMPLETED', 'submission_committed':True, 'inference_launch':source}),
                InfraInstance(id='other-completed', run_id=run.id, ticket_id=older_ticket.id,
                provider='cluster', instance_id='456', created_at=now,
                meta={'scheduler_state':'COMPLETED', 'submission_committed':True,
                      'inference_launch':snapshot(other, successful_body(other))}),
                InfraInstance(id='own-failed', run_id=run.id, ticket_id=ticket.id,
                provider='cluster', instance_id='789', created_at=now,
                meta={'scheduler_state':'FAILED', 'submission_committed':True, 'inference_launch':source})]
            session.add_all([run, ticket, older_ticket, *jobs])
            await session.commit()
            result = await prepare_launch(session, run, ticket, new)
            assert result.slurm_launch_reuse['source_execution_id'] == 'own-completed'
            assert 'expected_sha256' not in result.slurm_launch_reuse
            path = Path(new.slurm_job.script_path)
            repaired = path.read_text() + '# single MedExQA member; guard 1024 config hash\n'
            path.write_text(repaired)
            await session.commit()
            # A fresh activation/session must preserve the staged repair too.
            session.expire_all()
            run = await session.get(Run, 'run')
            ticket = await session.get(Ticket, 'current')
            result = await prepare_launch(session, run, ticket, new)
            assert path.read_text() == repaired
            assert result.slurm_launch_reuse['source_execution_id'] == 'own-completed'
            # An approved new measurement gets a fresh draft; the prior draft's
            # target settings cannot leak into a new budget/work order.
            revision['revision'] = 'length-2048'
            another = tmp_path / 'revised-2048.yaml'
            another.write_text('temperature: 0\nmax_new_tokens: 2048\n')
            new.inference_config_path = str(another)
            await prepare_launch(session, run, ticket, new)
            assert path.read_text() != repaired
            assert 'single MedExQA member' not in path.read_text()
            assert str(another.resolve()) in run.lifecycle['inference_reuse_artifacts']['current']['length-2048']
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_missing_reference_allows_fresh_draft_and_preserves_it(tmp_path):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from zevo.db import Base, Run, Ticket
    from zevo.engine.run.inference_launch import prepare_launch
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    inp = input_for(tmp_path)
    inp.reusable_predict_script_path = str(tmp_path / 'missing-reference.py')
    inp.model_copy = lambda *, update: SimpleNamespace(**{**vars(inp), **update})
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            run = Run(id='run', metric='accuracy')
            ticket = Ticket(id=inp.ticket_id, run_id=run.id, agent_id='inference', status='running')
            session.add_all([run, ticket])
            await session.commit()
            result = await prepare_launch(session, run, ticket, inp)
            assert result.slurm_launch_reuse['mode'] == 'unavailable'
            draft = Path(inp.slurm_job.script_path)
            draft.write_text(successful_body(inp) + '# newly prepared for the work order\n')
            before = draft.read_bytes()
            await prepare_launch(session, run, ticket, inp)
            assert draft.read_bytes() == before
            # Optional script references do not relax frozen configuration checks.
            Path(inp.inference_config_path).write_text('temperature: 1\n')
            with pytest.raises(ValueError, match='changed after input binding'):
                await prepare_launch(session, run, ticket, inp)
    finally:
        await engine.dispose()

@pytest.mark.asyncio
@pytest.mark.parametrize('verified', [False, True])
async def test_failed_controller_requires_verified_artifacts_for_script_reuse(tmp_path, monkeypatch, verified):
    from unittest.mock import AsyncMock
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from zevo.db import Base, Run, Ticket, InfraInstance
    from zevo.engine.run import inference_launch
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    old = input_for(tmp_path)
    new = input_for(tmp_path, 'infer-2', '/checkpoint')
    new.model_copy = lambda *, update: SimpleNamespace(**{**vars(new), **update})
    source = snapshot(old, successful_body(old))
    verifier = AsyncMock(return_value=source if verified else None)
    monkeypatch.setattr(inference_launch, 'verified_existing_launch', verifier)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        run = Run(id='run', metric='accuracy')
        owner = Ticket(id=old.ticket_id, run_id='run', agent_id='inference', status='succeeded', lane='optimization', payload={'base_model':'org/model'})
        target = Ticket(id=new.ticket_id, run_id='run', agent_id='inference', status='queued', lane='optimization')
        job = InfraInstance(id='source-job', run_id='run', ticket_id=owner.id, provider='cluster', instance_id='123',
            meta={'scheduler_state':'FAILED', 'submission_committed':True, 'inference_launch': source})
        session.add_all([run, owner, target, job])
        await session.commit()
        result = await inference_launch.prepare_launch(session, run, target, new)
        assert result.slurm_launch_reuse['mode'] == ('reuse' if verified else 'unavailable')
        verifier.assert_awaited()
        assert job.meta['scheduler_state'] == 'FAILED'
    await engine.dispose()
