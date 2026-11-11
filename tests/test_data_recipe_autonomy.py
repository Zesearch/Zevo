import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from zevo.contracts.data import (DataRecipe, DataRecipeIntent, EngineDataCountDeficit,
    validate_recipe_intent, validate_prepared_row_target, validate_synthesis_authorization)
from zevo.contracts.tickets import DataPayload
from zevo.engine.remote_training_data import (_semantic_fingerprint, RemoteDatasetSpec,
    record_remote_preparation, validate_remote_preparation, build_remote_training_package,
    finalize_remote_training_data, remote_data_signature, recipe_sha256,
    remote_preparation_signature)


def recipe(**changes):
    return DataRecipe(**{**dict(dataset_name='fixture', source_identity='org/data',
        source_fingerprint='a' * 64, training_method='sft', method_format='messages',
        method_ids=['inline_transform']), **changes})


def test_iteration_three_delegation_allows_realized_selection_and_counts():
    # 0d6cc735 / Data iteration 3: coordinator delegates labels/counts while
    # describing filters. Each elaborated field formerly failed in sequence.
    intent = DataRecipeIntent(
        subset='Data chooses concrete label values and per-label quantities from observed distribution and records them in recipe.',
        filters=['Normalize standalone code into fenced blocks', 'Remove degenerate repetition', 'Retain complete assistant turns'],
        sampling={'strategy': 'stratified_by_retained_provenance_domain',
                  'reproducibility': 'Record realized per-stratum counts, selection expression and seed actually used in data recipe'})
    realized = recipe(subset="domain in ['chat', 'code', 'math']",
        filters=['fence standalone code with language label', 'repeat window threshold 0.8', 'require terminated assistant'],
        sampling={**intent.sampling, 'realized_strata': {'chat': 85000, 'code': 60000, 'math': 36281},
                  'selected_count': 181281})
    assert set(validate_recipe_intent(realized, intent)) == {'subset', 'filters', 'sampling'}


def test_explicit_constraints_are_partial_and_report_every_mismatch():
    intent = DataRecipeIntent(constraints={'sampling': {'strategy': 'uniform'}, 'seed': 0})
    assert validate_recipe_intent(recipe(sampling={'strategy': 'uniform', 'count': 20}), intent) == []
    with pytest.raises(ValueError) as caught:
        validate_recipe_intent(recipe(sampling={'strategy': 'biased'}, seed=2), intent)
    assert 'sampling.strategy' in str(caught.value) and 'seed' in str(caught.value)


def test_legacy_intent_serialization_and_signature_stay_compatible():
    payload = DataPayload(operation='prepare_run_data', dataset='org/data', training_method='sft')
    dumped = payload.model_dump()
    assert 'constraints' not in dumped['recipe_intent']
    assert DataPayload.model_validate(dumped).data_intent_signature == payload.data_intent_signature


def test_sequence_of_applied_methods_can_repeat():
    sequence = ['inline_transform', 'reformat_jsonl', 'inline_transform']
    assert recipe(method_ids=sequence).method_ids == sequence
    assert DataPayload(operation='prepare_run_data', dataset='org/data', training_method='sft',
                       configuration_pins={'method_ids': sequence}).configuration_pins['method_ids'] == sequence


def test_prepared_count_and_exact_final_count_have_distinct_ownership():
    validate_prepared_row_target({'target_size': 2, 'target_size_basis': 'prepared'}, final_rows=1, removed_rows=1)
    with pytest.raises(EngineDataCountDeficit):
        validate_prepared_row_target({'target_size': 2}, final_rows=1, removed_rows=1)
    with pytest.raises(ValueError, match='prepared row count'):
        validate_prepared_row_target({'target_size': 3, 'target_size_basis': 'prepared'}, final_rows=1, removed_rows=1)


@pytest.mark.parametrize('source', ['acquired', 'file', 'synthetic', 'huggingface'])
def test_generation_permission_applies_to_every_source(source):
    intent = DataRecipeIntent()
    with pytest.raises(ValueError, match='generation requires'):
        validate_synthesis_authorization(recipe(), intent, generated_rows=2,
                                         synthetic_source=source == 'synthetic')


def test_semantic_v3_preserves_code_case_whitespace_and_system_context():
    a = {'prompt': "What does print('A' == 'a') output?"}
    b = {'prompt': "What does print('a' == 'a') output?"}
    assert _semantic_fingerprint(a) != _semantic_fingerprint(b)
    assert _semantic_fingerprint(a, version=2) == _semantic_fingerprint(b, version=2)
    assert _semantic_fingerprint({'prompt': 'x\n y'}) != _semantic_fingerprint({'prompt': 'x y'})
    def chat(system):
        return {'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': 'hello'}]}
    assert _semantic_fingerprint(chat('Translate to French')) != _semantic_fingerprint(chat('Translate to German'))
    assert _semantic_fingerprint(chat('Translate to French'), version=2) == _semantic_fingerprint(chat('Translate to German'), version=2)
    assert _semantic_fingerprint({'prompt': 'hello'}) == _semantic_fingerprint({'messages': [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'answer'}]})


def preparation(tmp_path):
    spec = RemoteDatasetSpec(source={'kind': 'acquired', 'identity': 'org/data', 'revision': 'v1'}, training_method='sft', method_format='messages')
    values = {'spec': spec.model_dump(), 'recipe': recipe().model_dump(), 'profile': {'rows': 1}}
    for name, value in values.items():
        (tmp_path / (name + '.json')).write_text(json.dumps(value))
    (tmp_path / 'prepare.py').write_text('# deterministic preparation\n')
    (tmp_path / 'data.jsonl').write_text('{"prompt": "hello", "completion": "answer"}\n')
    args = dict(data_intent_signature='b' * 64, spec_path=tmp_path/'spec.json',
        recipe_path=tmp_path/'recipe.json', prepare_script_path=tmp_path/'prepare.py',
        dataset_path=tmp_path/'data.jsonl', profile_path=tmp_path/'profile.json', receipt_path=tmp_path/'prepared.json')
    return spec, args


def test_audit_only_edit_reuses_prepared_bytes_but_execution_edit_invalidates(tmp_path):
    spec, args = preparation(tmp_path)
    record_remote_preparation(**args)
    changed = recipe(audit_steps=['clarified provenance'], dataset_name='display name', direction='explanation')
    args['recipe_path'].write_text(changed.model_dump_json())
    validate_remote_preparation(**args)
    package = build_remote_training_package(spec=spec, recipe=changed.model_dump(), prepare_script_path=args['prepare_script_path'],
        blocked_semantic_fingerprints=[], data_intent_signature='b' * 64, recipe_path=args['recipe_path'])
    (tmp_path/'package.json').write_text(package.model_dump_json())
    result = finalize_remote_training_data(package_path=tmp_path/'package.json', prepared_path=args['dataset_path'],
        output_path=tmp_path/'final.jsonl', profile_path=args['profile_path'], receipt_path=tmp_path/'final-receipt.json',
        preparation_receipt_path=args['receipt_path'])
    assert result.n_rows == 1
    args['recipe_path'].write_text(recipe(sampling={'count': 2}).model_dump_json())
    with pytest.raises(ValueError, match='does not match'):
        validate_remote_preparation(**args)


def test_old_receipts_remain_valid_without_reinterpreting_their_hash(tmp_path):
    spec, args = preparation(tmp_path)
    receipt = record_remote_preparation(**args).model_dump()
    old_hash = recipe_sha256(recipe().model_dump(), version=1)
    receipt.pop('recipe_hash_version')
    receipt['data_recipe_sha256'] = old_hash
    receipt['preparation_signature'] = remote_preparation_signature(data_intent_signature='b' * 64, source=spec.source,
        prepare_script_sha256=receipt['prepare_script_sha256'], data_recipe_sha256=old_hash)
    args['receipt_path'].write_text(json.dumps(receipt))
    assert validate_remote_preparation(**args).recipe_hash_version == 1


def test_hub_default_resolution_resigns_normalized_input():
    from zevo.api.routers.shared.tickets import _resolve_data_source_selection
    payload = DataPayload(operation='prepare_run_data', dataset='org/data', training_method='sft').model_dump()
    with patch('zevo.engine.remote_datasets.lookup', return_value=SimpleNamespace(split='train', config='default')):
        resolved = _resolve_data_source_selection(payload)
    assert DataPayload.model_validate(resolved).dataset_split == 'train'
    assert resolved['data_intent_signature'] != payload['data_intent_signature']


def test_non_chat_system_conditioning_is_preserved():
    first = {'prompt': 'hello', 'system_prompt': 'Translate to French'}
    second = {**first, 'system_prompt': 'Translate to German'}
    assert _semantic_fingerprint(first) != _semantic_fingerprint(second)
    assert _semantic_fingerprint(first, version=2) == _semantic_fingerprint(second, version=2)
