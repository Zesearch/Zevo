from __future__ import annotations

import json
import runpy
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
