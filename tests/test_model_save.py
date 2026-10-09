"""Destination preservation, receipts, and Registry validation (no network)."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from zevo.contracts.model_save import ModelSavePolicy
from zevo.contracts.model_registry import RegisterResult
from zevo.engine.run import model_save_worker as worker
from zevo.engine.run.runner import _build_registry_input, _extract_summary_artifact_meta


def checkpoint(tmp_path):
    root = tmp_path / 'checkpoint'
    root.mkdir()
    (root / 'config.json').write_text('{}')
    (root / 'model.safetensors').write_bytes(b'model-bytes')
    (root / 'tokenizer.json').write_text('{}')
    return root


def test_remote_directory_copy_verified_idempotent_and_no_overwrite(tmp_path):
    source = checkpoint(tmp_path)
    request = {'source': str(source), 'tag': 'M-run123',
               'policy': {'weights': 'remote', 'remote_dir': str(tmp_path / 'saved')}}
    result = worker.save(request)
    dest = Path(result['path'])
    assert dest == tmp_path / 'saved' / 'M-run123'
    assert worker.manifest(source) == worker.manifest(dest)
    assert worker.save(request) == result
    (dest / 'model.safetensors').write_bytes(b'other-model')
    with pytest.raises(ValueError, match='different model'):
        worker.save(request)
    assert (dest / 'model.safetensors').read_bytes() == b'other-model'
    assert (source / 'model.safetensors').read_bytes() == b'model-bytes'


def test_remote_failed_copy_does_not_publish_partial_directory(tmp_path, monkeypatch):
    source = checkpoint(tmp_path)
    def corrupt(src, dst, **kwargs):
        (dst / 'config.json').write_text('{}')
        (dst / 'model.safetensors').write_bytes(b'partial')
    monkeypatch.setattr(worker.shutil, 'copytree', corrupt)
    with pytest.raises(ValueError, match='copied bytes'):
        worker.save({'source': str(source), 'tag': 'M-run123',
                     'policy': {'weights': 'remote', 'remote_dir': str(tmp_path / 'saved')}})
    assert list((tmp_path / 'saved').iterdir()) == []
    assert source.exists()


def test_missing_shard_rejected_before_save(tmp_path):
    source = checkpoint(tmp_path)
    (source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': 'missing.safetensors'}}))
    with pytest.raises(ValueError, match='incomplete'):
        worker.manifest(source)


@pytest.mark.parametrize('directory', ['', '/', '../saved', '/x/../saved', '/tmp/\nfoo'])
def test_invalid_gpu_destination(directory):
    with pytest.raises(ValidationError):
        ModelSavePolicy(weights='remote', remote_dir=directory)


def test_hf_is_default_and_requires_explicit_repo():
    assert ModelSavePolicy(hf_repo_id='me/model').weights == 'hf'
    with pytest.raises(ValidationError):
        ModelSavePolicy()


@pytest.mark.parametrize('private_matches,files_complete', [(True, True), (False, True), (True, False)])
def test_hf_upload_verifies_visibility_and_commit_files(tmp_path, monkeypatch, private_matches, files_complete):
    source = checkpoint(tmp_path)
    calls = []
    class Api:
        def __init__(self, token):
            assert token == 'secret'
        def create_repo(self, **kwargs):
            calls.append(('create', kwargs))
        def model_info(self, repo, **kwargs):
            if not kwargs:
                return SimpleNamespace(private=private_matches, siblings=[])
            assert kwargs['revision'] == 'commit123'
            siblings = [SimpleNamespace(rfilename=p.name, size=p.stat().st_size) for p in source.iterdir()]
            return SimpleNamespace(siblings=siblings if files_complete else [])
        def upload_folder(self, **kwargs):
            calls.append(('upload', kwargs))
            assert kwargs['folder_path'] == str(source)
            return SimpleNamespace(oid='commit123')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=Api))
    request = {'source': str(source), 'tag': 'M-run123', 'token': 'secret',
               'policy': {'weights': 'hf', 'hf_repo_id': 'me/model', 'hf_private': True}}
    if private_matches and files_complete:
        receipt = worker.save(request)
        assert receipt['path'] == 'https://huggingface.co/me/model/tree/commit123'
        assert receipt['revision'] == 'commit123'
    else:
        with pytest.raises(ValueError):
            worker.save(request)
    if not private_matches:
        assert not any(kind == 'upload' for kind, _ in calls)


@pytest.mark.parametrize('storage,path,revision', [
    ('hf', 'https://huggingface.co/me/model/tree/commit123', 'commit123'),
    ('remote', 'ssh://me@gpu:22/lustre/saved/M-run123', ''),
])
def test_registry_accepts_only_engine_verified_external_destination(tmp_path, storage, path, revision):
    metrics = tmp_path / 'metrics.json'
    metrics.write_text('{"score": 0.75}')
    ticket = SimpleNamespace(id='registry-run123-001', run_id='run123', iteration=1, customization={})
    provenance = dict(base_model='base/model', training_method='sft', dataset_source='dataset',
                      task_objective='task', metric='accuracy', metric_direction='max')
    inp = _build_registry_input(ticket, {**provenance, 'checkpoint_is_remote': True}, {
        'checkpoint': {'path': '/gpu/model'}, 'train_config': {'path': '/gpu/train.yaml'},
        'metrics': {'path': str(metrics)}, 'device_info': {'path': '/device.json'},
    }, str(tmp_path))
    inp.model_save_policy = {'weights': storage}
    inp.saved_model_path = path
    inp.saved_model_revision = revision
    entry = dict(run_id='run123', ticket_id=ticket.id, iteration=1, **provenance,
        model_path=path, storage=storage, revision=revision, retention='retained',
        eval={'score': .75}, registered_at=datetime.now(timezone.utc).isoformat())
    registry = tmp_path / 'registry.yaml'
    registry.write_text(yaml.safe_dump({'models': {'M-run123': entry}}))
    output = RegisterResult(ticket_id=ticket.id, status='succeeded', registry_path=str(registry),
                            register_script_path='', notes='', error_message='')
    def validate():
        return _extract_summary_artifact_meta(output, inp=inp, agent_id='registry', work_dir=str(tmp_path), run_id='run123')
    status, _, _, meta = validate()
    assert status == 'succeeded' and meta['model_path'] == path
    entry['model_path'] = 'ssh://other/forged'
    registry.write_text(yaml.safe_dump({'models': {'M-run123': entry}}))
    assert validate()[0] == 'failed'
    inp.model_save_policy = {}
    assert validate()[0] == 'failed'


@pytest.mark.asyncio
async def test_preservation_uses_selected_ssh_route_and_stdin_for_credentials(monkeypatch):
    from zevo.engine.run import model_save as saver, cancel_rescue
    calls = []
    class Process:
        returncode = 0
        async def communicate(self, payload):
            request = json.loads(payload)
            assert request['token'] == 'secret-token'
            assert request['source'] == '/gpu/model'
            return b'ZEVO_MODEL_SAVED={"storage":"hf","path":"https://huggingface.co/me/model/tree/abc","revision":"abc"}\n', b''
    async def spawn(*args, **kwargs):
        calls.append(args)
        assert 'secret-token' not in ' '.join(args)
        return Process()
    async def route(*args):
        return SimpleNamespace(ssh=SimpleNamespace(user='me', host='gpu', port=22),
                               cluster=SimpleNamespace(env_setup='source /site/env.sh'))
    async def commit(): pass
    async def refresh(*args, **kwargs): pass
    monkeypatch.setattr(cancel_rescue, 'hf_token', lambda: 'secret-token')
    monkeypatch.setattr(saver, '_device_info_for_ticket', route)
    monkeypatch.setattr(saver, 'ssh_base_args', lambda ssh: ['ssh', '-o', 'ProxyCommand=approved-route'])
    monkeypatch.setattr(saver.asyncio, 'create_subprocess_exec', spawn)
    run = SimpleNamespace(id='run123', lifecycle={})
    receipt = await saver.save_model(SimpleNamespace(commit=commit, refresh=refresh), run, SimpleNamespace(id='registry-run123'),
        source='/gpu/model', remote=True, policy=ModelSavePolicy(hf_repo_id='me/model'))
    assert calls[0][3] == 'me@gpu'
    assert calls[0][4].startswith('bash -lc ')
    assert 'ZEVO_TICKET_ID=registry-run123' in calls[0][4]
    assert run.lifecycle['model_storage'] == receipt


def test_saved_external_models_are_listed_without_local_files():
    from zevo.api.routers.ui.models import kept_models
    now = datetime.now(timezone.utc)
    rows = [SimpleNamespace(model_path=path, registered_at=now) for path in (
        'https://huggingface.co/me/model/tree/abc', 'ssh://me@gpu:22/saved/M-run123',
        '/missing/local/model')]
    assert kept_models(rows) == rows[:2]


@pytest.mark.asyncio
async def test_model_save_transport_failure_keeps_registry_repair_behavior():
    from zevo.engine.run.runner import _apply_ticket_failure_policy
    messages = []
    ticket = SimpleNamespace(id='registry-run123', agent_id='registry', repair_attempts=0, status='running')
    retry = await _apply_ticket_failure_policy(SimpleNamespace(add=messages.append), ticket,
        error_message='Model preservation transport exited 255')
    assert retry and ticket.status == 'repairing' and ticket.repair_attempts == 1
