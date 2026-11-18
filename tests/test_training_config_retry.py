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
    ('output_dir', '/other/run/attempt-2/output'), ('output_dir', '/run/train/attempt-2/../output'),
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
