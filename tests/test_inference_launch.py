from types import SimpleNamespace
from pathlib import Path
import hashlib

import pytest

from zevo.engine.run.inference_launch import (
    render, snapshot, frozen_inputs, verify_frozen_inputs, effective_snapshot,
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


def test_freeze_detects_in_place_config_and_predict_changes(tmp_path):
    inp = input_for(tmp_path)
    pins = frozen_inputs(inp)
    verify_frozen_inputs(pins)
    Path(inp.inference_config_path).write_text('temperature: 1\n')
    with pytest.raises(ValueError, match='changed after input binding'):
        verify_frozen_inputs(pins)
    Path(inp.inference_config_path).write_text('temperature: 0\n')
    Path(inp.reusable_predict_script_path).write_text('changed code')
    with pytest.raises(ValueError, match='changed after input binding'):
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
        assert result.slurm_launch_reuse['mode']=='unavailable'
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


def test_replay_validation_rejects_modified_launcher_and_in_place_config(tmp_path):
    from zevo.contracts.inference import InferenceTaskInput
    from zevo.contracts.infrastructure import SlurmStageJobContract
    from zevo.engine.run.runner import _slurm_stage_result_problem
    inp = input_for(tmp_path)
    path = Path(inp.slurm_job.script_path)
    path.write_text(successful_body(inp))
    contract = SlurmStageJobContract.model_construct(**inp.slurm_job.model_dump())
    typed = InferenceTaskInput.model_construct(slurm_job=contract,
        frozen_reuse_artifacts=frozen_inputs(inp),
        slurm_launch_reuse={'mode':'reuse','expected_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    path.write_text(path.read_text()+'pip install another-runtime\n')
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
