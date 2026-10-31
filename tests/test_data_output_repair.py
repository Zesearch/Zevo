from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from zevo.contracts.data import DataResult
from zevo.engine.run import runner
from zevo.engine.run.failure_policy import classify_failure


@pytest.mark.parametrize('field', ['remote_data_receipt_path', 'remote_training_package_path'])
def test_data_cannot_return_engine_finalization_fields(field):
    with pytest.raises(ValidationError, match='must not author engine-owned') as caught:
        DataResult(status='succeeded', ticket_id='data-r-001', notes='', error_message='',
                   operation='prepare_run_data', remote_dataset_path='/remote/data.jsonl',
                   remote_dataset_spec_path='/work/spec.json',
                   remote_data_profile_path='/remote/profile.json',
                   prepare_script_path='/work/prepare.py', data_recipe_path='/work/recipe.json',
                   **{field: '/work/preparation_receipt.json'})
    assert classify_failure(str(caught.value), agent_id='data').route == 'self'


@pytest.mark.asyncio
@pytest.mark.parametrize('mismatch', ['dataset', 'split', 'config', 'method', 'identity', 'fingerprint', 'missing_spec'])
async def test_remote_data_output_errors_schedule_repair(monkeypatch, mismatch):
    spec = SimpleNamespace(source=SimpleNamespace(dataset_id='org/data', split='train', config='default'), training_method='sft')
    recipe = SimpleNamespace(source_identity='source', source_fingerprint='fingerprint')
    inp = SimpleNamespace(dataset='org/data', dataset_split='train', dataset_config='default', training_method='sft', expected_source_identity='source')
    result = SimpleNamespace(remote_training_package_path='', remote_data_receipt_path='', remote_dataset_spec_path='/work/spec', data_recipe_path='/work/recipe')
    if mismatch == 'dataset': spec.source.dataset_id = 'org/wrong'
    if mismatch == 'split': spec.source.split = 'wrong'
    if mismatch == 'config': spec.source.config = 'wrong'
    if mismatch == 'method': spec.training_method = 'dpo'
    if mismatch == 'identity': recipe.source_identity = 'wrong'
    if mismatch == 'fingerprint': recipe.source_fingerprint = 'wrong'
    def load_spec(path):
        if mismatch == 'missing_spec': raise FileNotFoundError(path)
        return spec
    monkeypatch.setattr(runner, 'load_remote_dataset_spec', load_spec)
    monkeypatch.setattr(runner, 'load_data_recipe', lambda path: recipe)
    monkeypatch.setattr(runner, 'remote_source_fingerprint', lambda value: 'fingerprint')
    with pytest.raises(runner.DataOutputValidationError) as caught:
        runner._validate_remote_data_output(result, inp)
    ticket = SimpleNamespace(id='data-r-001', agent_id='data', status='running', repair_attempts=1)
    messages = []
    assert await runner._apply_ticket_failure_policy(SimpleNamespace(add=messages.append), ticket, error_message=str(caught.value))
    assert ticket.status == 'repairing'
    assert ticket.repair_attempts == 2
    assert len(messages) == 1


@pytest.mark.parametrize(('agent', 'error'), [
    ('train', 'TrainResult.tracking_url must exactly match the engine-owned Weights & Biases URL'),
    ('inference', 'trained-model inference changed baseline inference_config.yaml'),
    ('infrastructure', 'Invalid InfraResult: device_info_path missing'),
    ('registry', 'Invalid RegistryResult: model artifact missing'),
])
def test_other_stage_output_errors_remain_repairable(agent, error):
    assert classify_failure(error, agent_id=agent).route == 'self'
