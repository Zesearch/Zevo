from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


HELPER = Path(__file__).resolve().parents[1] / "playbook/runners/train_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("zevo_train_checkpoint_test", HELPER)
assert SPEC and SPEC.loader
checkpoint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checkpoint
SPEC.loader.exec_module(checkpoint)


def test_checkpoint_directory_is_validated_and_committed_atomically(tmp_path: Path) -> None:
    staging = tmp_path / "model.staging"
    final = tmp_path / "model"
    staging.mkdir()
    (staging / "config.json").write_text("{}", encoding="utf-8")
    (staging / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (staging / "model-00002-of-00002.safetensors").write_bytes(b"second")
    (staging / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
    }), encoding="utf-8")

    receipt = checkpoint.commit_checkpoint_directory(
        staging, final, metadata={"ticket_id": "train-1"},
    )

    assert not staging.exists()
    assert (final / checkpoint.MARKER_NAME).is_file()
    assert receipt["weight_bytes"] == len(b"firstsecond")
    assert receipt["metadata"] == {"ticket_id": "train-1"}
    assert checkpoint.commit_checkpoint_directory(staging, final) == receipt


def test_checkpoint_commit_rejects_missing_index_shard(tmp_path: Path) -> None:
    staging = tmp_path / "model.staging"
    staging.mkdir()
    (staging / "config.json").write_text("{}", encoding="utf-8")
    (staging / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (staging / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"b": "model-00002-of-00002.safetensors"}
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="missing shards"):
        checkpoint.commit_checkpoint_directory(staging, tmp_path / "model")


def test_checkpoint_commit_accepts_a_peft_adapter(tmp_path: Path) -> None:
    staging = tmp_path / "adapter.staging"
    staging.mkdir()
    (staging / "adapter_config.json").write_text("{}", encoding="utf-8")
    (staging / "adapter_model.safetensors").write_bytes(b"adapter")

    receipt = checkpoint.commit_checkpoint_directory(
        staging, tmp_path / "adapter",
    )

    assert receipt["weight_files"] == ["adapter_model.safetensors"]
    assert receipt["weight_bytes"] == len(b"adapter")


def test_checkpoint_commit_requires_sibling_directories(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(ValueError, match="must be siblings"):
        checkpoint.commit_checkpoint_directory(
            staging, tmp_path / "nested" / "model",
        )


def _training_files(tmp_path, *, step=774, early_stopping=None):
    import yaml
    config = tmp_path / 'train_config.yaml'
    training = {'num_epochs': 1}
    if early_stopping is not None:
        training['early_stopping'] = early_stopping
    config.write_text(yaml.safe_dump({'training': training}))
    plan = tmp_path / 'training_plan.json'
    checkpoint.record_training_plan(plan, ticket_id='train-1', config_path=config,
                                    planned_steps=774, planned_epochs=1)
    state = tmp_path / 'trainer_state.json'
    state.write_text(json.dumps({'global_step': step, 'max_steps': 774, 'log_history': []}))
    staging = tmp_path / 'model.staging'
    staging.mkdir()
    (staging / 'config.json').write_text('{}')
    (staging / 'model.safetensors').write_bytes(b'weights')
    return config, training, plan, state, staging


@pytest.mark.parametrize('step', [200, 247])
def test_partial_training_cannot_publish_as_complete(tmp_path, step):
    _, _, plan, state, staging = _training_files(tmp_path, step=step)
    with pytest.raises(ValueError, match='training incomplete'):
        checkpoint.commit_training_checkpoint(staging, tmp_path/'model', plan_path=plan, trainer_state_path=state)
    assert not (tmp_path/'model').exists()
    assert staging.exists()


def test_completed_training_save_can_be_repaired_without_retraining(tmp_path):
    import hashlib
    config, training, plan, state, staging = _training_files(tmp_path)
    (staging/'model.safetensors').write_bytes(b'')
    with pytest.raises(ValueError, match='empty files'):
        checkpoint.commit_training_checkpoint(staging, tmp_path/'model', plan_path=plan, trainer_state_path=state)
    assert json.loads(state.read_text())['global_step'] == 774
    (staging/'model.safetensors').write_bytes(b'recovered final weights')
    checkpoint.commit_training_checkpoint(staging, tmp_path/'model', plan_path=plan, trainer_state_path=state)
    proof = checkpoint.verify_training_completion(tmp_path/'model', ticket_id='train-1',
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), training=training)
    assert proof == {'reason': 'plan_completed', 'completed_steps': 774, 'planned_steps': 774}
    (tmp_path/'model'/'model.safetensors').write_bytes(b'truncated')
    with pytest.raises(ValueError, match='changed'):
        checkpoint.verify_training_completion(tmp_path/'model', ticket_id='train-1',
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), training=training)


def test_training_target_cannot_be_shortened_during_repair(tmp_path):
    config, _, plan, state, staging = _training_files(tmp_path, step=200)
    with pytest.raises(ValueError, match='different targets'):
        checkpoint.record_training_plan(plan, ticket_id='train-1', config_path=config,
                                        planned_steps=200, planned_epochs=1)
    state.write_text(json.dumps({'global_step': 200, 'max_steps': 200}))
    with pytest.raises(ValueError, match='frozen training plan'):
        checkpoint.commit_training_checkpoint(staging, tmp_path/'model', plan_path=plan, trainer_state_path=state)


@pytest.mark.parametrize('values,accepted', [([1., .9, .8], False), ([1., 1.1, 1.2], True)])
def test_early_stopping_requires_declared_rule_and_actual_evaluations(tmp_path, values, accepted):
    import hashlib
    policy = {'metric': 'eval_loss', 'mode': 'min', 'patience': 2, 'min_delta': 0.0}
    config, training, plan, state, staging = _training_files(tmp_path, step=200, early_stopping=policy)
    state.write_text(json.dumps({'global_step': 200, 'max_steps': 774,
        'log_history': [{'step': step, 'eval_loss': value} for step, value in zip([100,150,200], values)]}))
    if not accepted:
        with pytest.raises(ValueError, match='not reached'):
            checkpoint.commit_training_checkpoint(staging, tmp_path/'model', plan_path=plan, trainer_state_path=state)
    else:
        checkpoint.commit_training_checkpoint(staging, tmp_path/'model', plan_path=plan, trainer_state_path=state)
        assert checkpoint.verify_training_completion(tmp_path/'model', ticket_id='train-1',
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), training=training)['reason'] == 'early_stopping'
        with pytest.raises(ValueError, match='another Ticket or configuration'):
            checkpoint.verify_training_completion(tmp_path/'model', ticket_id='another-ticket',
                config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), training=training)


def test_valid_weights_alone_are_not_completed_training(tmp_path):
    import hashlib
    config, training, _, _, staging = _training_files(tmp_path)
    checkpoint.commit_checkpoint_directory(staging, tmp_path/'model')
    with pytest.raises(ValueError, match='no training completion evidence'):
        checkpoint.verify_training_completion(tmp_path/'model', ticket_id='train-1',
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), training=training)
