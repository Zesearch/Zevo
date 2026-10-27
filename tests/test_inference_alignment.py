from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import json

import pytest

from zevo.contracts.configuration import load_inference_config
from zevo.contracts.inference_core import inference_core
from zevo.engine.run.inference_alignment import align_reused_config
from tests.test_sequential_yaml_workflow import _write_inference_config


@pytest.fixture
def configs(tmp_path):
    source = _write_inference_config(tmp_path)
    original = load_inference_config(source)
    shared = inference_core(original)
    shared['sampling'].update(skip_special_tokens=False, spaces_between_special_tokens=False)
    heldout = original.model_dump(mode='json')
    heldout['special_token_ids'] = {'eos_token': 2}
    heldout['measurement']['max_new_tokens'] = 2048
    heldout['measurement']['inference_config']['answer_regex'] = '[ABC]'
    heldout['implementation_config'].update(llm_kwargs={'max_model_len': 4096},
                                           sampling_extra={'max_tokens': 2048})
    source.write_text(json.dumps(heldout))
    return source, shared, tmp_path / 'ticket'


def test_aligns_sampling_and_token_aliases_without_mutating_source(configs):
    source, shared, work = configs
    before = source.read_bytes()
    aligned = Path(align_reused_config(str(source), shared, str(work), allow_create=True))
    config = load_inference_config(aligned)
    assert inference_core(config) == shared
    assert config.measurement.max_new_tokens == 2048
    assert config.measurement.inference_config['answer_regex'] == '[ABC]'
    assert config.implementation_config['llm_kwargs'] == {'max_model_len': 4096}
    assert source.read_bytes() == before
    assert aligned.with_suffix('.provenance.json').exists()
    assert align_reused_config(str(source), shared, str(work), allow_create=False) == str(aligned)


def test_does_not_rewrite_executed_configuration(configs):
    source, shared, work = configs
    with pytest.raises(ValueError, match='after a GPU execution'):
        align_reused_config(str(source), shared, str(work), allow_create=False)
    assert not work.exists()


def test_prompt_changes_require_explicit_regeneration(configs):
    source, shared, work = configs
    shared['chat_template_hash'] = 'b' * 64
    with pytest.raises(ValueError, match='regenerating its prompt example'):
        align_reused_config(str(source), shared, str(work), allow_create=True)
    assert not work.exists()


def test_alignment_rejects_tampered_output(configs):
    source, shared, work = configs
    path = Path(align_reused_config(str(source), shared, str(work), allow_create=True))
    path.write_text('changed')
    with pytest.raises(ValueError, match='modified after creation'):
        align_reused_config(str(source), shared, str(work), allow_create=True)


def test_matching_config_is_used_unchanged(tmp_path):
    path = _write_inference_config(tmp_path)
    core = inference_core(load_inference_config(path))
    assert align_reused_config(str(path), core, str(tmp_path/'new'), allow_create=False) == str(path)


@pytest.mark.asyncio
async def test_input_builder_aligns_primary_and_suite_before_slurm(configs, monkeypatch):
    from zevo.engine.run import runner
    from zevo.contracts.inference import InferenceSuiteMemberInput
    from zevo.contracts.infrastructure import SlurmStageJobContract
    from zevo.db import Ticket, Run
    source, shared, work = configs
    validation = load_inference_config(_write_inference_config(source.parent)).model_dump(mode='json')
    validation['implementation_config']['sampling_extra'] = shared['sampling']
    core_path = source.parent / 'validation.json'
    core_path.write_text(json.dumps(validation))
    # Reintroduce the mismatch on the separately bound held-out source.
    heldout = dict(validation)
    heldout['implementation_config'] = {'trust_remote_code': False}
    source.write_text(json.dumps(heldout))
    member = InferenceSuiteMemberInput(name='other', test_set_name='other',
        scoring_set='/other.csv', sample_submission='/sample.csv',
        inference_data_profile_path='', predictions_validation_command='validate',
        work_dir=str(work/'suite/001'), inference_config_path=str(source), configuration_mode='reuse')
    monkeypatch.setattr(runner, '_build_validation_suite_members', AsyncMock(return_value=[]))
    monkeypatch.setattr(runner, '_build_holdout_suite_members', AsyncMock(return_value=[member]))
    monkeypatch.setattr(runner, '_latest_slurm_resource_request', AsyncMock(return_value=None))
    async def job(**kwargs):
        assert inference_core(load_inference_config(kwargs['inference_config_path'])) == shared
        return SlurmStageJobContract()
    monkeypatch.setattr(runner, '_slurm_stage_job_contract', job)
    run = Run(id='run', metric='accuracy', holdout={})
    ticket = Ticket(id='holdout-infer-1', run_id='run', agent_id='inference', lane='held_out_test', iteration=2)
    inp = await runner._build_inference_input(ticket,
        {'operation':'run_inference','base_model':validation['base_model'], 'scoring_set':'/questions.csv',
         'sample_submission':'/sample.csv','model_source':'base_model'},
        {'device_info':{'path':'/device.json'},'inference_config':{'path':str(source)},
         'shared_inference_config':{'path':str(core_path)}}, str(work), 'vllm', run, SimpleNamespace())
    assert inp.inference_config_path != str(source)
    assert inference_core(load_inference_config(inp.suite_members[0].inference_config_path)) == shared
    assert inp.configuration_mode == 'reuse'


@pytest.mark.asyncio
async def test_first_heldout_keeps_test_selection_and_calibration_with_validation_script(tmp_path, monkeypatch):
    from zevo.engine.run import runner
    from zevo.contracts.infrastructure import SlurmStageJobContract
    from zevo.db import Ticket, Run
    config_path = _write_inference_config(tmp_path)
    config = load_inference_config(config_path)
    predict = tmp_path / 'validation-predict.py'
    predict.write_text('print("verified implementation")')
    monkeypatch.setattr(runner, '_build_validation_suite_members', AsyncMock(return_value=[]))
    monkeypatch.setattr(runner, '_build_holdout_suite_members', AsyncMock(return_value=[]))
    monkeypatch.setattr(runner, '_latest_slurm_resource_request', AsyncMock(return_value=None))
    monkeypatch.setattr(runner, '_slurm_stage_job_contract', AsyncMock(return_value=SlurmStageJobContract()))
    run = Run(id='run', metric='accuracy', holdout={})
    ticket = Ticket(id='heldout-baseline', run_id=run.id, agent_id='inference', lane='held_out_test', iteration=0)
    inp = await runner._build_inference_input(ticket,
        {'operation':'run_inference', 'base_model':config.base_model,
         'scoring_set':'/test/questions.jsonl', 'sample_submission':'/test/submission.csv',
         'model_source':'base_model'},
        {'device_info':{'path':'/device.json'},
         'shared_inference_config':{'path':str(config_path)},
         'predict_script':{'path':str(predict)},
         'inference_data_profile':{'path':'/test/profile.json'}},
        str(tmp_path / 'test-work'), 'vllm', run, SimpleNamespace())
    assert inp.reusable_predict_script_path == str(predict)
    assert inp.shared_inference_core == inference_core(config)
    assert inp.configuration_mode == 'select' and inp.inference_config_path == ''
    assert inp.length_calibration_helper_path
    assert inp.scoring_set == '/test/questions.jsonl'
    assert inp.inference_data_profile_path == '/test/profile.json'
