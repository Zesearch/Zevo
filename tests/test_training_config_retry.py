import hashlib
from types import SimpleNamespace

import pytest
import yaml

from zevo.engine.run.repair_progress import verify_repair_training_config


def config():
    return {'model': 'original', 'data_signature': 'original', 'training': {
        'learning_rate': 1e-5, 'implementation_config': {
            'planned_optimizer_steps': 1874,
            'output_dir': '/run/train/attempt-1/output',
            'launch_command': 'python train.py --inference-config /run/train/attempt-1/baseline_inference_config.yaml',
            'resolved_defaults_notes': 'original explanation',
            'startup_validation': {'checks': ['check loss'], 'rationale': 'initial'},
            'gpu_memory_preflight': {'per_gpu_vram_gib': 140, 'estimated_peak_per_rank_gib': 85,
                'components_gib': {}, 'note': 'original'},
        }}}


def registered(tmp_path):
    value = config()
    raw = yaml.safe_dump(value)
    path = tmp_path / 'train_config.yaml'
    path.write_text(raw)
    row = SimpleNamespace(meta={'training_config_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                               'training_config_snapshot': raw, 'execution_attempt': 1})
    return row, path, value


def test_diagnostic_retry_and_full_diff_audit(tmp_path):
    row, path, value = registered(tmp_path)
    impl = value['training']['implementation_config']
    impl['output_dir'] = '/run/train/attempt-2/output'
    impl['launch_command'] = impl['launch_command'].replace('attempt-1', 'attempt-2')
    impl['startup_max_first_batch_nll'] = 6.0
    impl['startup_max_first_logged_loss'] = 6.0
    impl['startup_max_reference_loss_mismatch'] = 0.25
    impl['resolved_defaults_notes'] = 'updated diagnosis'
    impl['startup_validation']['checks'].append('independent reference loss')
    impl['gpu_memory_preflight']['estimated_peak_per_rank_gib'] = 88
    impl['gpu_memory_preflight']['components_gib']['startup_gate_dense_reference_logits_no_grad'] = 3
    path.write_text(yaml.safe_dump(value))
    verify_repair_training_config(row, path)
    audit = row.meta['training_config_retry_audit']
    assert audit['original_sha256'] != audit['proposed_sha256']
    assert len(audit['changes']) == 9
    assert all(c['execution_only'] for c in audit['changes'])


@pytest.mark.parametrize('field,value', [
    ('planned_optimizer_steps', 1876), ('loss_type', 'nll'), ('unknown_notes', 'anything'),
    ('startup_max_first_batch_nll', -1), ('startup_max_first_logged_loss', float('inf')),
    ('launch_command', 'python other.py'), ('startup_validation', {'checks': ['x'], 'enabled': False}),
    ('gpu_memory_preflight', {'per_gpu_vram_gib': 140, 'estimated_peak_per_rank_gib': 145}),
])
def test_unknown_or_training_changes_require_review(tmp_path, field, value):
    row, path, data = registered(tmp_path)
    data['training']['implementation_config'][field] = value
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        verify_repair_training_config(row, path)


@pytest.mark.parametrize('change', ['model', 'learning_rate', 'data', 'duplicate', 'invalid'])
def test_identity_changes_and_invalid_yaml_rejected(tmp_path, change):
    row, path, data = registered(tmp_path)
    if change == 'model': data['model'] = 'new'
    if change == 'learning_rate': data['training']['learning_rate'] = 2e-5
    if change == 'data': data['data_signature'] = 'new'
    raw = yaml.safe_dump(data)
    if change == 'duplicate': raw += '\nmodel: original\n'
    if change == 'invalid': raw = 'model: [broken'
    path.write_text(raw)
    with pytest.raises(ValueError): verify_repair_training_config(row, path)


def test_comments_order_and_verified_legacy_snapshot(tmp_path):
    row, path, data = registered(tmp_path)
    original = tmp_path / 'train_config.attempt1.yaml'
    original.write_bytes(path.read_bytes())
    row.meta.pop('training_config_snapshot')
    path.write_text('# comment\n' + yaml.safe_dump(data, sort_keys=False))
    verify_repair_training_config(row, path)
    assert row.meta['training_config_retry_audit']['changes'] == []
    original.write_text('forged original')
    with pytest.raises(ValueError, match='snapshot unavailable'):
        verify_repair_training_config(row, path)


def test_guard_cannot_be_loosened(tmp_path):
    row, path, data = registered(tmp_path)
    data['training']['implementation_config']['startup_max_first_batch_nll'] = 6
    raw = yaml.safe_dump(data)
    row.meta.update(training_config_snapshot=raw, training_config_sha256=hashlib.sha256(raw.encode()).hexdigest())
    data['training']['implementation_config']['startup_max_first_batch_nll'] = 10
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError): verify_repair_training_config(row, path)


def fresh_repair(tmp_path):
    row, path, value = registered(tmp_path)
    value.update(training_method='sft', parent_model='Qwen/base',
                 loss_contract={'objective_config': {'loss_type': 'chunked_nll'}, 'mask': 'completion'})
    raw = yaml.safe_dump(value)
    row.meta.update(scheduler_state='FAILED', training_config_snapshot=raw,
                    training_config_sha256=hashlib.sha256(raw.encode()).hexdigest())
    value['training']['implementation_config']['output_dir'] = '/run/train/attempt-2/output'
    return row, path, value


@pytest.mark.parametrize('parameter,new', [
    ('learning_rate', 2e-5), ('batch_size', 2), ('gradient_accumulation_steps', 8),
    ('num_epochs', 3), ('optimizer', 'adamw_torch'), ('precision', 'bf16'),
    ('world_size', 2), ('distributed', {'backend': 'ddp', 'nodes': 1, 'gpus_per_node': 2}),
])
def test_train_can_tune_unpinned_parameters_on_fresh_failed_attempt(tmp_path, parameter, new):
    row, path, value = fresh_repair(tmp_path)
    value['training'][parameter] = new
    path.write_text(yaml.safe_dump(value))
    verify_repair_training_config(row, path, train_input=SimpleNamespace())
    audit = row.meta['training_config_retry_audit']
    assert audit['authority'] == 'train' and audit['restart_mode'] == 'restart_parent'
    assert any(c['path'] == 'training.' + parameter and c['train_owned'] for c in audit['changes'])
    assert audit['original_sha256'] != audit['proposed_sha256']


def test_train_can_repair_nll_implementation_and_memory_accounting(tmp_path):
    row, path, value = fresh_repair(tmp_path)
    value['loss_contract']['objective_config']['loss_type'] = 'nll'
    impl = value['training']['implementation_config']
    impl['startup_max_pre_step_grad_norm'] = 10000
    impl['gpu_memory_preflight']['components_gib']['dense_nll_logits'] = 13
    impl['gpu_memory_preflight']['estimated_peak_per_rank_gib'] = 104
    path.write_text(yaml.safe_dump(value))
    verify_repair_training_config(row, path, train_input=SimpleNamespace())
    assert row.meta['training_config_retry_audit']['authority'] == 'train'


@pytest.mark.parametrize('case', ['pin', 'loss_pin', 'model', 'data', 'method', 'mask',
                                 'data_selection', 'memory_capacity',
                                 'resume', 'unknown', 'different_objective'])
def test_train_tuning_cannot_cross_task_boundaries(tmp_path, case):
    row, path, value = fresh_repair(tmp_path)
    value['training']['learning_rate'] = 2e-5
    inp = SimpleNamespace()
    if case == 'pin': inp.configuration_pins = {'learning_rate': 1e-5}
    if case == 'loss_pin':
        inp.loss_objective_pins = {'loss_type': 'chunked_nll'}
        value['loss_contract']['objective_config']['loss_type'] = 'nll'
    if case == 'model': value['parent_model'] = 'other/model'
    if case == 'data': value['data_signature'] = 'other'
    if case == 'method': value['training_method'] = 'grpo'
    if case == 'mask': value['loss_contract']['mask'] = 'all'
    if case == 'data_selection': value['training']['data_selection'] = {'limit': 1}
    if case == 'memory_capacity': value['training']['implementation_config']['gpu_memory_preflight']['per_gpu_vram_gib'] = 200
    if case == 'resume': value['training']['implementation_config']['resume_from_checkpoint'] = '/old/checkpoint'
    if case == 'unknown': value['training']['implementation_config']['new_model_path'] = '/other/model'
    if case == 'different_objective': value['loss_contract']['objective_config']['loss_type'] = 'other'
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError):
        verify_repair_training_config(row, path, train_input=inp)
    assert 'training_config_retry_audit' not in row.meta


@pytest.mark.parametrize('case', ['completed', 'running', 'continuation', 'input_continuation', 'stored_continuation'])
def test_train_tuning_is_only_for_failed_fresh_attempts(tmp_path, case):
    row, path, value = fresh_repair(tmp_path)
    value['training']['learning_rate'] = 2e-5
    inp = SimpleNamespace()
    if case in {'completed', 'running'}: row.meta['scheduler_state'] = case.upper()
    if case == 'input_continuation': inp.continuation = {'checkpoint': '/checkpoint'}
    if case == 'stored_continuation': row.meta['training_continuation'] = {'checkpoint': '/checkpoint'}
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError):
        verify_repair_training_config(row, path, train_input=inp, continuation=case == 'continuation')


def test_completed_workload_needs_explicit_failed_collect_retry(tmp_path):
    row, path, value = fresh_repair(tmp_path)
    row.meta['scheduler_state'] = 'COMPLETED'
    row.meta['execution_repair'] = {'action': 'resubmit', 'execution_attempt': 1,
                                    'reason': 'Completed process produced invalid training; restart with corrected optimizer'}
    value['training']['learning_rate'] = 2e-5
    path.write_text(yaml.safe_dump(value))
    verify_repair_training_config(row, path, train_input=SimpleNamespace())
    assert row.meta['training_config_retry_audit']['reason'] == row.meta['execution_repair']['reason']


def test_continuation_rejects_even_execution_only_yaml_changes(tmp_path):
    row, path, value = fresh_repair(tmp_path)
    path.write_text(yaml.safe_dump(value))  # only attempt-local output path changed
    with pytest.raises(ValueError, match='byte-identical'):
        verify_repair_training_config(row, path, train_input=SimpleNamespace(), continuation=True)


@pytest.mark.parametrize('layout', ['unchanged', 'config_hash', 'trainer_output_dir'])
def test_train_repair_does_not_require_an_engine_prescribed_output_layout(tmp_path, layout):
    row, path, value = fresh_repair(tmp_path)
    baseline = yaml.safe_load(row.meta['training_config_snapshot'])
    impl = value['training']['implementation_config']
    if layout == 'unchanged':
        impl['output_dir'] = baseline['training']['implementation_config']['output_dir']
    elif layout == 'config_hash':
        impl['output_dir'] = '/run/train/runtime/new-config/trainer'
    else:
        for cfg in (baseline, value):
            cfg['training']['implementation_config'].pop('output_dir')
            cfg['training']['implementation_config']['trainer_output_dir'] = '/run/train/runtime/<config_sha12>/trainer'
        raw = yaml.safe_dump(baseline)
        row.meta.update(training_config_snapshot=raw,
                        training_config_sha256=hashlib.sha256(raw.encode()).hexdigest())
    value['training']['lora_target_modules'] = ['q_proj', 'k_proj', 'v_proj']
    path.write_text(yaml.safe_dump(value))
    verify_repair_training_config(row, path, train_input=SimpleNamespace())
    assert row.meta['training_config_retry_audit']['restart_mode'] == 'restart_parent'


def test_output_layout_change_does_not_bypass_user_pin(tmp_path):
    row, path, value = fresh_repair(tmp_path)
    path.write_text(yaml.safe_dump(value))
    pin = {'implementation_config': {'output_dir': '/run/train/attempt-1/output'}}
    with pytest.raises(ValueError, match='user pin'):
        verify_repair_training_config(row, path, train_input=SimpleNamespace(configuration_pins=pin))


@pytest.mark.parametrize('scope', ['all_tokens', 'answer_tokens'])
def test_supervision_scope_is_train_owned(tmp_path, scope):
    row, path, value = fresh_repair(tmp_path)
    value['loss_contract']['target_scope'] = scope
    value['training']['implementation_config']['assistant_only_loss'] = False
    path.write_text(yaml.safe_dump(value))
    verify_repair_training_config(row, path, train_input=SimpleNamespace())
    assert row.meta['training_config_retry_audit']['authority'] == 'train'


@pytest.mark.parametrize('pin', [{'world_size': 1}, {'loss_contract': {'target_scope': 'assistant_messages'}}])
def test_parallelism_and_scope_still_honor_user_pins(tmp_path, pin):
    row, path, value = fresh_repair(tmp_path)
    value['training']['world_size'] = 2
    value['loss_contract']['target_scope'] = 'all_tokens'
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match='user pin'):
        verify_repair_training_config(row, path, train_input=SimpleNamespace(configuration_pins=pin))


@pytest.mark.parametrize('operation', ['add', 'modify', 'remove'])
def test_new_implementation_evidence_does_not_need_orchestrator(tmp_path, operation):
    row, path, value = fresh_repair(tmp_path)
    evidence = {'lora_adapter_modules_expected': 224,
                'lora_adapter_tensors_expected': 448,
                'lora_target_modules_derivation': '32 layers x 7 projections',
                'custom_preflight_evidence': {'module_check': {'verified': True}}}
    impl = value['training']['implementation_config']
    if operation != 'add':
        baseline = yaml.safe_load(row.meta['training_config_snapshot'])
        baseline['training']['implementation_config'].update(evidence)
        raw = yaml.safe_dump(baseline)
        row.meta.update(training_config_snapshot=raw,
                        training_config_sha256=hashlib.sha256(raw.encode()).hexdigest())
    if operation != 'remove':
        impl.update(evidence)
        if operation == 'modify':
            impl['lora_target_modules_derivation'] += '; verified on meta device'
            impl['custom_preflight_evidence']['module_check']['verified'] = False
    path.write_text(yaml.safe_dump(value))
    verify_repair_training_config(row, path, train_input=SimpleNamespace())
    audit = row.meta['training_config_retry_audit']
    assert audit['authority'] == 'train'
    assert all(not c['requires_orchestrator_review'] for c in audit['changes'])


@pytest.mark.parametrize('field', ['prompt', 'tokenizer_source', 'inference_config_sha256',
                                  'method_config', 'parent_model'])
def test_explicit_review_boundaries_cannot_be_added_or_removed(tmp_path, field):
    row, path, value = fresh_repair(tmp_path)
    if field in value:
        del value[field]
    else:
        value[field] = {'changed': True}
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match='changed fields'):
        verify_repair_training_config(row, path, train_input=SimpleNamespace())


def test_new_implementation_field_still_honors_explicit_pin(tmp_path):
    row, path, value = fresh_repair(tmp_path)
    value['training']['implementation_config']['lora_adapter_modules_expected'] = 224
    path.write_text(yaml.safe_dump(value))
    inp = SimpleNamespace(configuration_pins={
        'implementation_config': {'lora_adapter_modules_expected': 112}})
    with pytest.raises(ValueError, match='user pin'):
        verify_repair_training_config(row, path, train_input=inp)
