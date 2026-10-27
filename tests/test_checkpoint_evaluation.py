"""Checkpoint handoff advances only after each independently completed stage."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from zevo.db import Base, Run, Ticket, HeartbeatRun, ExecutionEvent, WorkProduct, AgentWakeupRequest, InfraInstance
from zevo.engine.run.adaptive_training import (
    reconcile_adaptive_training, validate_checkpoint_request, build_checkpoint_evaluation_input,
    handle_checkpoint_yield,
)


def bundle(root):
    root.mkdir()
    files = {}
    for role, content in {'questions': 'id,question\n1,Q\n', 'labels': 'id,answer\n1,A\n',
                          'sample_submission': 'id,prediction\n1,A\n',
                          'inference_config': 'frozen', 'predict_script': 'frozen'}.items():
        path = root / f'{role}.csv'
        path.write_text(content)
        files[role] = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    contract = {'split': 'validation', 'direction': 'max', 'members': [{'name': 'A', 'files': files,
        'metric': 'accuracy', 'answer_fields': ['answer'], 'evaluation_config': {}}]}
    path = root / 'contract.json'
    path.write_text(json.dumps(contract))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize('final', [False, True])
@pytest.mark.parametrize('streamed', [False, True])
async def test_checkpoint_pipeline_serial_and_retryable(tmp_path, monkeypatch, final, streamed):
    from zevo.engine.run import remote_jobs
    from zevo.engine.agent.drivers.evaluation_runner import EvaluationRunnerDriver
    from zevo.engine.run.runner import _advance_measurements, _maybe_wake_supervisor, _apply_run_score_monitor
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    path, digest = bundle(tmp_path / 'bundle')
    key = 'a'*32
    request = {'request_id': key, 'ticket_id': 'train-1', 'step': 20, 'planned_steps': 20 if final else 100,
        'contract_sha256': digest, 'metrics_path': '/remote/train-1/adaptive/validation-20.json',
        'decision_path': f'/remote/train-1/adaptive/decision-{key}.json',
        'checkpoint': '/remote/train-1/adaptive/checkpoint-20', 'world_size': 1,
        'measurements': [], 'decisions': [], 'direction': 'max'}
    deliveries = []
    async def route(*args): return SimpleNamespace(ssh=None)
    async def ssh(info, command, **kwargs):
        if command.startswith('tail '):
            return {'ok': True, 'stdout': '__PROGRESS__:train-1:' + json.dumps({
                'phase': 'adaptive_checkpoint', 'adaptive_checkpoint_request': request})}
        if 'base64' in command:
            return {'ok': True, 'stdout': 'checkpoint-verified'}
        deliveries.append(command)
        return {'ok': len(deliveries) > 1}
    monkeypatch.setattr(remote_jobs, '_device_info_for_ticket', route)
    monkeypatch.setattr(remote_jobs, '_ssh', ssh)
    async with Session() as db:
        run = Run(id='r', task_name='t', metric='accuracy', status='running', supervisor_ticket_id='orch',
            lifecycle={'adaptive_mid_training_evaluation': True, 'adaptive_validation_contracts': {'train-1': digest},
                       'adaptive_validation_contract_paths': {'train-1': str(path)}})
        train = Ticket(id='train-1', run_id='r', agent_id='train', status='waiting_external', iteration=1,
                       payload={'base_model': 'model'}, inputs={'device_info': {'path': '/device.json'}})
        job = InfraInstance(id='job', provider='cluster', instance_id='123', run_id='r', ticket_id=train.id,
                            meta={'scheduler_state': 'COMPLETED', 'submission_committed': True,
                                  'remote_workdir': '/remote/train-1', 'log_path': '/remote/train-1/slurm-123.out'})
        db.add_all([run, train, job, Ticket(id='orch', run_id='r', agent_id='orchestrator', status='succeeded'),
                    HeartbeatRun(id='hb', ticket_id=train.id, agent_id='train', driver='stub', model='stub')])
        await db.commit()
        if streamed:
            db.add(ExecutionEvent(ticket_id=train.id, heartbeat_id='hb', attempt_id='attempt', event_type='progress',
            phase='adaptive_checkpoint', current_step=20, extras={'execution_purpose': 'workload',
                'execution_id': job.id, 'adaptive_checkpoint_request': request}))
        await db.commit()
        from zevo.engine.run.scheduler.reconciler import _reconcile_slurm_stage_jobs
        assert await _reconcile_slurm_stage_jobs(db) == (0, 0)  # No final-model collect wake.
        assert await handle_checkpoint_yield(db, run, train, job, dict(job.meta))
        infer = await db.get(Ticket, f'mid-infer-{key}')
        assert infer is not None and train.status == 'waiting_external'
        assert (await db.execute(select(Ticket).where(Ticket.agent_id == 'evaluation'))).scalars().all() == []
        assert [w.agent_id for w in (await db.execute(select(AgentWakeupRequest))).scalars()] == ['inference']
        infer.status = 'repairing'
        await db.commit()
        await reconcile_adaptive_training(db)
        assert not (await db.execute(select(Ticket).where(Ticket.agent_id == 'evaluation'))).scalars().all()
        artifact = tmp_path / 'predictions.csv'
        artifact.write_text('id,prediction\n1,A\n')
        infer.status = 'succeeded'
        db.add(WorkProduct(ticket_id=infer.id, role='predictions', path=str(artifact)))
        await db.commit()
        await reconcile_adaptive_training(db)
        await reconcile_adaptive_training(db)
        evaluations = (await db.execute(select(Ticket).where(Ticket.agent_id == 'evaluation'))).scalars().all()
        assert len(evaluations) == 1
        evaluation = evaluations[0]
        wakes = (await db.execute(select(AgentWakeupRequest))).scalars().all()
        assert [w.agent_id for w in wakes] == ['inference', 'evaluation']
        work = tmp_path / 'evaluation'
        work.mkdir()
        inp = await build_checkpoint_evaluation_input(db, run, evaluation, str(work))
        result = await EvaluationRunnerDriver().run_agent(blueprint=None, input_payload=inp, workspace_dir=str(work))
        assert result.output.status == 'succeeded'
        for internal in (infer, evaluation):
            _apply_run_score_monitor(run, internal, result.output, {'score': 1})
            await _advance_measurements(db, internal, 1, result.output)
            await _maybe_wake_supervisor(db, internal)
        assert run.best_validation_score is None and not run.history
        evaluation.status = 'succeeded'
        db.add(WorkProduct(ticket_id=evaluation.id, role='metrics', path=result.output.metrics_path))
        await db.commit()
        await reconcile_adaptive_training(db)  # Failed remote delivery does not advance.
        assert not run.lifecycle['adaptive_checkpoints'][key]['metrics_delivered']
        await reconcile_adaptive_training(db)
        assert run.lifecycle['adaptive_checkpoints'][key]['metrics_delivered']
        await reconcile_adaptive_training(db)
        assert len(deliveries) == 2
        wakes = (await db.execute(select(AgentWakeupRequest))).scalars().all()
        if final:
            assert [w.agent_id for w in wakes] == ['inference', 'evaluation', 'train']
        else:
            assert [w.agent_id for w in wakes] == ['inference', 'evaluation', 'orchestrator']
            assert train.status == 'waiting_external'
            decisions = dict(run.lifecycle['adaptive_training_requests'])
            decisions[key] = {**decisions[key], 'status': 'decided', 'decision': {'action': 'continue'}}
            run.lifecycle = {**run.lifecycle, 'adaptive_training_requests': decisions}
            active = HeartbeatRun(id='decision', ticket_id='orch', agent_id='orchestrator', driver='stub', model='stub')
            db.add(active)
            await db.commit()
            await reconcile_adaptive_training(db)
            assert train.status == 'waiting_external'  # Decision activation must finish first.
            from datetime import datetime, timezone
            active.finished_at = datetime.now(timezone.utc)
            await db.commit()
            await reconcile_adaptive_training(db)
        assert train.status == 'queued'
        assert job.meta['checkpoint_resume'] == key
        assert run.lifecycle['adaptive_checkpoints'][key]['resume_queued']
        await reconcile_adaptive_training(db)
        wakes = (await db.execute(select(AgentWakeupRequest).where(AgentWakeupRequest.agent_id == 'train'))).scalars().all()
        assert len(wakes) == 1 and wakes[0].payload['adaptive_resume_request_id'] == key
        with pytest.raises(ValueError, match='path'):
            validate_checkpoint_request({**request, 'metrics_path': '/other/validation-20.json'}, train, digest)
    await engine.dispose()


@pytest.mark.asyncio
async def test_checkpoint_inference_reuses_all_frozen_benchmarks(tmp_path, monkeypatch):
    from zevo.engine.run import runner
    from zevo.engine.run.adaptive_training import build_checkpoint_inference_input
    from zevo.contracts.infrastructure import SlurmStageJobContract
    path, _ = bundle(tmp_path / 'bundle')
    contract = json.loads(path.read_text())
    contract['members'] += [{**contract['members'][0], 'name': name} for name in ['B', 'C']]
    path.write_text(json.dumps(contract))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    train = Ticket(id='train-1', run_id='r', agent_id='train', lane='optimization')
    run = Run(id='r', gpu_provider='cluster', lifecycle={
        'adaptive_validation_contracts': {train.id: digest},
        'adaptive_validation_contract_paths': {train.id: str(path)}})
    request = {'ticket_id': train.id, 'request_id': 'a'*32, 'contract_sha256': digest,
        'step': 20, 'planned_steps': 100, 'world_size': 1, 'measurements': [], 'decisions': [],
        'checkpoint': '/remote/train-1/checkpoint-20', 'metrics_path': '/remote/train-1/validation-20.json',
        'decision_path': '/remote/train-1/decision-' + 'a'*32 + '.json'}
    files = contract['members'][0]['files']
    ticket = Ticket(id='mid-infer-a', run_id='r', agent_id='inference', lane='optimization', iteration=1,
        payload={'mid_training_checkpoint': request, 'model_source': 'checkpoint', 'base_model': 'model',
                 'scoring_set': str(path.parent / files['questions']['path']),
                 'sample_submission': str(path.parent / files['sample_submission']['path'])},
        inputs={'device_info': {'path': '/device.json'}, 'checkpoint': {'path': request['checkpoint']},
                'inference_config': {'path': str(path.parent / files['inference_config']['path'])}})
    class Session:
        async def get(self, model, key): return train
    async def slurm(**kwargs): return SlurmStageJobContract()
    async def forbidden(**kwargs): raise AssertionError('must not rebuild frozen suite from live Run config')
    monkeypatch.setattr(runner, '_slurm_stage_job_contract', slurm)
    monkeypatch.setattr(runner, '_build_validation_suite_members', forbidden)
    inp = await build_checkpoint_inference_input(Session(), run, ticket, str(tmp_path), 'vllm', {})
    assert [member.name for member in inp.suite_members] == ['B', 'C']
    assert all(member.configuration_mode == 'reuse' and member.predictions_validation_command for member in inp.suite_members)
    assert inp.reusable_predict_script_path == str(path.parent / files['predict_script']['path'])
    assert inp.checkpoint_path == request['checkpoint']
    assert inp.configuration_mode == 'reuse'
    assert 'labels' not in inp.model_dump_json()


@pytest.mark.asyncio
async def test_checkpoint_wakeup_preserves_identity_when_normal_wake_is_already_queued():
    from zevo.engine.run.wakeup import queue_wakeup
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        db.add_all([Run(id='r', task_name='t', metric='accuracy'),
                    Ticket(id='orch', run_id='r', agent_id='orchestrator')])
        await db.commit()
        first = await queue_wakeup(db, agent_id='orchestrator', ticket_id='orch', source='cron')
        wake = await queue_wakeup(db, agent_id='orchestrator', ticket_id='orch', source='on_demand',
                                  payload={'adaptive_training_request_id': 'checkpoint'})
        assert wake.id == first.id and wake.payload['adaptive_training_request_id'] == 'checkpoint'
    await engine.dispose()
