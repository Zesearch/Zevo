from __future__ import annotations

import copy
import json
import runpy
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from zevo.engine.run import remote_jobs, training_completion

HELPER = Path(__file__).resolve().parents[1] / 'playbook/runners/train_checkpoint.py'


@pytest.mark.asyncio
@pytest.mark.parametrize('remote', [False, True])
@pytest.mark.parametrize('complete', [False, True])
async def test_backend_independently_checks_local_and_remote_checkpoint(tmp_path, monkeypatch, remote, complete):
    helper = runpy.run_path(str(HELPER))
    config = tmp_path/'train.yaml'
    config.write_text('training:\n  num_epochs: 1\n')
    training = {'num_epochs': 1, 'early_stopping': None}
    monkeypatch.setattr(training_completion, 'load_train_config', lambda _: SimpleNamespace(
        training=SimpleNamespace(model_dump=lambda **kw: training)))
    staging = tmp_path/'model.staging'
    staging.mkdir()
    (staging/'config.json').write_text('{}')
    (staging/'model.safetensors').write_bytes(b'weights')
    final = tmp_path/'model'
    if complete:
        plan = tmp_path/'plan.json'
        state = tmp_path/'state.json'
        helper['record_training_plan'](plan, ticket_id='train-1', config_path=config, planned_steps=774, planned_epochs=1)
        state.write_text(json.dumps({'global_step': 774, 'max_steps': 774}))
        helper['commit_training_checkpoint'](staging, final, plan_path=plan, trainer_state_path=state)
    else:
        helper['commit_checkpoint_directory'](staging, final)  # Valid intermediate weights.
    if remote:
        async def route(*args):
            return object()
        async def execute(info, command, **kwargs):
            args = shlex.split(command)
            result = subprocess.run([sys.executable, *args[1:]], capture_output=True, text=True)
            return {'ok': result.returncode == 0, 'stdout': result.stdout[:1000], 'error': result.stderr[:1000]}
        monkeypatch.setattr(remote_jobs, '_device_info_for_ticket', route)
        monkeypatch.setattr(remote_jobs, '_ssh', execute)
    result = SimpleNamespace(train_config_path=str(config), checkpoint_path=str(final), checkpoint_is_remote=remote)
    if complete:
        proof = await training_completion.verify_completed_training(None, SimpleNamespace(id='train-1'), result, HELPER)
        assert proof['completed_steps'] == 774
    else:
        with pytest.raises(ValueError, match='no training completion evidence'):
            await training_completion.verify_completed_training(None, SimpleNamespace(id='train-1'), result, HELPER)


@pytest.mark.parametrize('left,right,expected', [
    (0.6559077058470175, 0.6559077058470176, True),
    (0.1234501, 0.1234599, True),  # Ignore later digits, without rounding.
    (0.12344999, 0.12345001, False),  # Fifth decimal differs, even across a tiny gap.
    (-0.1234501, -0.1234599, True),
    (-0.12344999, -0.12345001, False),
    (1.12345, 2.12345, False),
    (1, 1.0, True),
    (0, 0.000009, True),
    (0, 0.00001, False),
    (1e100, 1e100, True),
    (True, 1, False),
    ('0.12345', 0.12345, False),
    (None, None, False),
    (float('nan'), float('nan'), False),
    (float('inf'), float('inf'), False),
    (float('-inf'), float('-inf'), False),
])
def test_metric_comparison_uses_first_five_decimal_places(left, right, expected):
    assert training_completion._same_metric_value(left, right) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize('change', [
    None, 'score', 'metric', 'missing_metric', 'extra_metric', 'checkpoint',
    'step', 'decision', 'contract', 'missing_measurement', 'extra_field',
])
async def test_adaptive_completion_tolerates_decimal_tails_but_preserves_evidence(change):
    approved = [{'step': 7091, 'checkpoint': '/train-1/checkpoint-7091',
                 'score': 0.6559077058470176, 'metrics': {'legal': 0.1234501}}]
    decision = {'request_id': 'r1', 'action': 'continue', 'next_step': 9454}
    lifecycle = {
        'adaptive_mid_training_evaluation': True,
        'adaptive_validation_contracts': {'train-1': 'frozen'},
        'adaptive_training_requests': {'r1': {
            'ticket_id': 'train-1', 'status': 'decided',
            'decision': decision, 'measurements': approved,
        }},
    }
    original = copy.deepcopy(lifecycle)
    report = {'contract_sha256': 'frozen', 'decisions': [copy.deepcopy(decision)],
              'measurements': [copy.deepcopy(approved[0]),
                               {'step': 11817, 'score': .66, 'metrics': {'legal': .66}}]}
    first = report['measurements'][0]
    first['score'] = 0.6559077058470175
    first['metrics']['legal'] = 0.1234599
    if change == 'score':
        first['score'] = 0.65591
    elif change == 'metric':
        first['metrics']['legal'] = 0.12346
    elif change == 'missing_metric':
        first['metrics'] = {}
    elif change == 'extra_metric':
        first['metrics']['other'] = .1
    elif change == 'checkpoint':
        first['checkpoint'] = '/train-1/checkpoint-9454'
    elif change == 'step':
        first['step'] = 7092
    elif change == 'decision':
        report['decisions'][0]['next_step'] = 9455
    elif change == 'contract':
        report['contract_sha256'] = 'changed'
    elif change == 'missing_measurement':
        report['measurements'] = []
    elif change == 'extra_field':
        first['other'] = 'unexpected'
    unchanged_report = copy.deepcopy(report)
    run = SimpleNamespace(lifecycle=lifecycle)
    db = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one=lambda: run)),
                         flush=AsyncMock())
    ticket = SimpleNamespace(id='train-1', run_id='run-1', payload={})
    if change:
        with pytest.raises(ValueError, match='adaptive completion'):
            await training_completion._verify_adaptive_decisions(db, ticket, {'adaptive_report': report})
        db.flush.assert_not_awaited()
        assert run.lifecycle == original
    else:
        await training_completion._verify_adaptive_decisions(db, ticket, {'adaptive_report': report})
        db.flush.assert_awaited_once()
        assert run.lifecycle['adaptive_training_requests'] == original['adaptive_training_requests']
        assert run.lifecycle['adaptive_training_reports']['train-1'] == unchanged_report
    assert report == unchanged_report
