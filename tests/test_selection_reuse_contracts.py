from copy import deepcopy
import json

import pytest
import yaml

from tests.test_sequential_yaml_workflow import _write_inference_config
from tests.test_method_branch_identity import case
from zevo.contracts.configuration import load_inference_config, inference_execution_contract, file_sha256
from zevo.contracts.inference import InferenceTaskInput, InferenceResult, InferenceSuiteMemberInput, InferenceSuiteMemberResult
from zevo.contracts.tickets import PipelineTrainRequestPayload, TrainPayload, specialist_request_payload_schemas
from zevo.engine.run.runner import _validate_specialist_yaml


@pytest.mark.parametrize('model', [PipelineTrainRequestPayload, TrainPayload])
def test_method_decision_is_separate_and_legacy_payloads_are_normalized(model):
    values = dict(operation='train', base_model='model', model_source='base_model', parent_selection_rationale='baseline')
    old = model(**values, configuration_suggestions={'training_method': ' DPO ', 'direction': 'Improve precision'})
    new = model(**values, selected_training_method='dpo', configuration_suggestions={'direction': 'Improve precision'})
    assert old.model_dump() == new.model_dump()
    assert old.configuration_suggestions == {'direction': 'Improve precision'}
    with pytest.raises(ValueError, match='conflicting'):
        model(**values, selected_training_method='sft', configuration_suggestions={'training_method': 'dpo'})
    with pytest.raises(ValueError, match='unsupported selected'):
        model(**values, selected_training_method='made_up')


def test_new_request_schema_does_not_advertise_method_as_advice():
    schema = specialist_request_payload_schemas()['train']['properties']
    assert 'selected_training_method' in schema
    assert 'training_method' not in schema['configuration_suggestions']['properties']


def test_train_work_order_migrates_old_selection_and_enforces_explicit_decision(tmp_path):
    inp, result = case(tmp_path)
    assert inp.selected_training_method == 'rft'
    assert 'training_method' not in inp.configuration_suggestions
    assert _validate_specialist_yaml(result, inp)['training_method'] == 'rft'
    changed_advice = inp.model_copy(update={'configuration_suggestions': {'direction': 'Alternative wording'}})
    assert _validate_specialist_yaml(result, changed_advice)['training_method'] == 'rft'
    changed_selection = inp.model_copy(update={'selected_training_method': 'sft'})
    with pytest.raises(ValueError, match='Orchestrator-selected method'):
        _validate_specialist_yaml(result, changed_selection)


@pytest.fixture
def reuse_case(tmp_path):
    baseline = _write_inference_config(tmp_path)
    original = yaml.safe_load(baseline.read_text())
    candidate = tmp_path/'candidate.yaml'
    candidate.write_text(yaml.safe_dump(original, sort_keys=False))
    diagnostics = tmp_path/'diagnostics.json'
    diagnostics.write_text(json.dumps({'records': [{'request_index': 0, 'row_index': 0,
        'turn': 1, 'finish_reason': 'stop', 'generated_tokens': 1}]}))
    inp = InferenceTaskInput.model_construct(configuration_mode='reuse', base_model=original['base_model'],
        generation_backend=original['generation_backend'], config_validation_command='', inference_config_path=str(baseline))
    result = InferenceResult(ticket_id='test', status='succeeded', error_message='', notes='',
        inference_config_path=str(candidate), predictions_path='/predictions.csv', generation_diagnostics_path=str(diagnostics), n_rows=1, n_requests=1)
    return inp, result, baseline, candidate, original


def test_reuse_accepts_formatting_and_advisory_audit_but_preserves_baseline(reuse_case):
    inp, result, baseline, candidate, original = reuse_case
    original_hash = file_sha256(baseline)
    body = deepcopy(original)
    body['suggestion_decisions'] = [dict(key='direction', decision='adjusted',
        suggested='Reduce cost', realized='Keep execution fixed', rationale='Reworded audit only')]
    candidate.write_text('# A presentation-only comment\n'+yaml.safe_dump(body, sort_keys=False, indent=4))
    assert file_sha256(candidate) != original_hash
    _validate_specialist_yaml(result, inp)
    assert file_sha256(baseline) == original_hash
    assert inference_execution_contract(load_inference_config(candidate)) == inference_execution_contract(load_inference_config(baseline))


@pytest.mark.parametrize('field,value', [('max_new_tokens', 999), ('seed', 999), ('top_p', .8)])
def test_reuse_still_rejects_changed_decoding(reuse_case, field, value):
    inp, result, _, candidate, original = reuse_case
    original['measurement'][field] = value
    candidate.write_text(yaml.safe_dump(original))
    with pytest.raises(ValueError, match='changed baseline'):
        _validate_specialist_yaml(result, inp)


def test_reuse_does_not_drop_nested_backend_notes(reuse_case):
    inp, result, _, candidate, original = reuse_case
    original['implementation_config']['notes'] = 'Unknown backend argument must remain binding'
    candidate.write_text(yaml.safe_dump(original))
    with pytest.raises(ValueError, match='changed baseline'):
        _validate_specialist_yaml(result, inp)


def test_suite_member_uses_the_same_execution_comparison(reuse_case):
    inp, result, baseline, candidate, original = reuse_case
    result.inference_config_path = str(baseline)
    inp.suite_members = [InferenceSuiteMemberInput.model_construct(name='second', test_set_name='second',
        configuration_mode='reuse', inference_config_path=str(baseline))]
    result.suite_members = [InferenceSuiteMemberResult(name='second', test_set_name='second',
        inference_config_path=str(candidate), predictions_path='/member.csv',
        generation_diagnostics_path=result.generation_diagnostics_path, n_rows=1, n_requests=1)]
    candidate.write_text('# Changed formatting\n'+yaml.safe_dump(original, sort_keys=False, indent=4))
    _validate_specialist_yaml(result, inp)
    original['measurement']['inference_config']['answer_regex'] = '(.*)'
    candidate.write_text(yaml.safe_dump(original))
    with pytest.raises(ValueError, match='baseline execution config'):
        _validate_specialist_yaml(result, inp)
