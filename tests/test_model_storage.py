"""Storage display, upload authorization, credentials, and verified completion."""
import json
import sys
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.routers.ui import model_storage as storage
from zevo.contracts.model_save import ModelSavePolicy
from zevo.db.models import Base, RegistryModel, Run


@pytest.mark.parametrize('path,kind,machine,display', [
    ('ssh://alice@gpu.example:2222/lustre/my%20model', 'remote', 'alice@gpu.example:2222', '/lustre/my model'),
    ('https://huggingface.co/alice/model/tree/abc', 'hf', 'Hugging Face', 'https://huggingface.co/alice/model/tree/abc'),
    ('data/runs/r/models/M-r', 'local', 'Zevo server (container)', '/app/data/runs/r/models/M-r'),
    ('', 'missing', '', ''),
])
def test_location(path, kind, machine, display, monkeypatch):
    monkeypatch.delenv('ZEVO_HOST_REPO', raising=False)
    assert storage.location(path) == dict(kind=kind, machine=machine, path=display,
                                         url=path if kind == 'hf' else '')


def test_uploaded_link_does_not_follow_new_champion():
    state = dict(source='/old', iteration=1, status='succeeded', url='https://huggingface.co/u/m')
    run = NS(lifecycle={'model_hf_uploads': {'M-r': state}})
    assert not storage.upload_state(run, NS(version_tag='M-r', model_path='/old', iteration=2))
    assert not storage.upload_state(run, NS(version_tag='M-r', model_path='/new', iteration=1))


def test_interrupted_upload_can_retry():
    run = NS(lifecycle={'model_hf_uploads': {'M-r': dict(source='/old', iteration=1,
        status='uploading', started_at=0)}})
    assert storage.upload_state(run, NS(version_tag='M-r', model_path='/old', iteration=1))['status'] == 'failed'


@pytest.fixture
def upload_setup(monkeypatch):
    model = NS(version_tag='M-r', model_path='/saved', iteration=1)
    run = NS(id='r', lifecycle={})
    monkeypatch.setattr(storage, 'load_model', AsyncMock(return_value=(model, run)))
    token = Mock(return_value='private-token')
    monkeypatch.setattr(storage, 'hf_token', token)
    command = AsyncMock(return_value=(['worker'], '/saved'))
    monkeypatch.setattr(storage, 'upload_command', command)
    return model, run, token, command


@pytest.mark.asyncio
async def test_upload_uses_settings_without_persisting_token(upload_setup):
    model, run, token, command = upload_setup
    db = NS(commit=AsyncMock())
    background = BackgroundTasks()
    result = await storage.upload_hf('M-r', ModelSavePolicy(hf_repo_id='me/model'), background, db)
    token.assert_called_once_with()
    assert result['status'] == 'uploading'
    assert 'private-token' not in json.dumps(run.lifecycle)
    assert len(background.tasks) == 1
    assert background.tasks[0].args[-1]['token'] == 'private-token'
    assert model.model_path == '/saved'
    with pytest.raises(HTTPException) as error:
        await storage.upload_hf('M-r', ModelSavePolicy(hf_repo_id='me/model'), BackgroundTasks(), db)
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_missing_token_does_not_start_upload(upload_setup):
    _, _, token, command = upload_setup
    token.return_value = ''
    tasks = BackgroundTasks()
    with pytest.raises(HTTPException) as error:
        await storage.upload_hf('M-r', ModelSavePolicy(hf_repo_id='me/model'), tasks, NS())
    assert error.value.detail['code'] == 'hf_token_missing'
    command.assert_not_awaited()
    assert not tasks.tasks


@pytest.mark.asyncio
@pytest.mark.parametrize('success', [True, False])
async def test_upload_verification_and_safe_failure(monkeypatch, success):
    finish = AsyncMock()
    monkeypatch.setattr(storage, 'finish_upload', finish)
    receipt = dict(storage='hf', path='https://huggingface.co/me/model/tree/abc', revision='abc')
    code = (f'print("ZEVO_MODEL_SAVED=" + {json.dumps(receipt)!r})' if success
            else 'import sys;sys.stderr.write("secret-token");sys.exit(1)')
    await storage.perform_upload('r', 'M-r', 'job', [sys.executable, '-c', code], {'token': 'secret-token'})
    result = finish.await_args.args[-1]
    assert result['status'] == ('succeeded' if success else 'failed')
    assert 'secret-token' not in json.dumps(result)


@pytest.mark.asyncio
async def test_persisted_completion(monkeypatch):
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(storage, 'get_session_factory', lambda: sessions)
    run = Run(id='r', task_name='t', task_objective='o',
        agent_objective='o', metric='acc', metric_direction='max', lifecycle={'other': 'keep'})
    storage.set_state(run, 'M-r', dict(id='job', source='/saved', iteration=1, status='uploading'))
    async with sessions() as db:
        db.add(run)
        db.add(RegistryModel(version_tag='M-r', run_id='r', model_path='/saved',
            iteration=1, base_model='b', training_method='sft', metric='acc', metric_direction='max'))
        await db.commit()
    await storage.finish_upload('r', 'M-r', 'wrong-job', dict(status='succeeded', url='wrong'))
    await storage.finish_upload('r', 'M-r', 'job', dict(status='succeeded', url='https://huggingface.co/me/model'))
    async with sessions() as db:
        result = await storage.get_storage('M-r', db)
        assert result['hf_url'] == 'https://huggingface.co/me/model'
        assert result['location']['path'] == '/saved'
        saved = await db.get(Run, 'r')
        assert saved.lifecycle['other'] == 'keep'
    await engine.dispose()


@pytest.mark.asyncio
async def test_remote_upload_rejects_wrong_machine(monkeypatch):
    db = NS(execute=AsyncMock(return_value=NS(scalars=lambda: NS(all=lambda: [NS()]))))
    monkeypatch.setattr(storage, '_device_info_for_ticket', AsyncMock(return_value=NS(
        ssh=NS(host='other', port=22, user='alice'))))
    with pytest.raises(HTTPException) as error:
        await storage.upload_command(db, NS(model_path='ssh://alice@gpu:22/models/m'), NS(id='r'))
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_timeout_stops_upload_and_records_failure(monkeypatch):
    finish = AsyncMock()
    monkeypatch.setattr(storage, 'finish_upload', finish)
    monkeypatch.setattr(storage, 'UPLOAD_TIMEOUT', 0.05)
    await storage.perform_upload('r', 'M-r', 'job',
        [sys.executable, '-c', 'import time;time.sleep(10)'], {})
    assert finish.await_args.args[-1]['status'] == 'failed'


@pytest.mark.asyncio
async def test_remote_upload_uses_matching_route_and_original_path(monkeypatch):
    db = NS(execute=AsyncMock(return_value=NS(scalars=lambda: NS(all=lambda: [NS()]))))
    monkeypatch.setattr(storage, '_device_info_for_ticket', AsyncMock(return_value=NS(
        ssh=NS(host='gpu', port=2222, user='alice'), cluster=NS(env_setup='source /env/bin/activate'))))
    monkeypatch.setattr(storage, 'ssh_base_args', lambda ssh: ['ssh', '-p', str(ssh.port)])
    command, source = await storage.upload_command(db,
        NS(model_path='ssh://alice@gpu:2222/models/my%20model'), NS(id='r'))
    assert command[:4] == ['ssh', '-p', '2222', 'alice@gpu']
    assert 'source /env/bin/activate' in command[-1]
    assert source == '/models/my model'


def test_hf_token_reads_settings_saved_after_startup(tmp_path, monkeypatch):
    from zevo.api.routers.ui import settings
    path = tmp_path / '.env'
    monkeypatch.setattr(settings, 'ENV_PATH', path)
    monkeypatch.setenv('HF_TOKEN', 'old-process-token')
    path.write_text('HF_TOKEN=new-settings-token\n')
    assert storage.hf_token() == 'new-settings-token'
