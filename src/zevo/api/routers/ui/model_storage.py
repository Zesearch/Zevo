"""Saved model locations and user-requested Hub uploads, without copying weights here."""
from __future__ import annotations

import asyncio
import json
import shlex
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select

from zevo.api.artifacts import host_path_abs
from zevo.api.database import get_db
from zevo.contracts.model_save import ModelSavePolicy
from zevo.db import RegistryModel, Run, Ticket, get_session_factory
from zevo.engine.run.cancel_rescue import hf_token
from zevo.engine.remote_transfer import ssh_base_args
from zevo.engine.run.remote_jobs import _device_info_for_ticket

router = APIRouter()
UPLOAD_TIMEOUT = 3600


def location(raw: str) -> dict:
    parsed = urlsplit(raw)
    if parsed.scheme == 'https' and parsed.hostname == 'huggingface.co':
        return {'kind': 'hf', 'machine': 'Hugging Face', 'path': raw, 'url': raw}
    if parsed.scheme == 'ssh':
        machine = f'{unquote(parsed.username or "")}@{parsed.hostname}:{parsed.port or 22}'
        return {'kind': 'remote', 'machine': machine, 'path': unquote(parsed.path), 'url': ''}
    if not raw:
        return {'kind': 'missing', 'machine': '', 'path': '', 'url': ''}
    absolute = raw if Path(raw).is_absolute() else '/app/' + raw
    host = host_path_abs(absolute)
    return {'kind': 'local', 'machine': 'Zevo server' if host.startswith('/') else 'Zevo server (container)',
            'path': host if host.startswith('/') else absolute, 'url': ''}


def upload_state(run: Run, model: RegistryModel) -> dict:
    state = dict((run.lifecycle or {}).get('model_hf_uploads', {}).get(model.version_tag, {}))
    # A newer champion must never inherit an older checkpoint's upload link.
    if state.get('source') != model.model_path or state.get('iteration') != model.iteration:
        return {}
    if state.get('status') == 'uploading':
        age = datetime.now(timezone.utc).timestamp() - state.get('started_at', 0)
        if age > UPLOAD_TIMEOUT + 120:
            return {**state, 'status': 'failed', 'error': 'Upload timed out or was interrupted. You can retry.'}
    return state


async def load_model(db, tag, *, lock=False):
    model = (await db.execute(select(RegistryModel).where(
        RegistryModel.version_tag == tag,
    ))).scalar_one_or_none()
    if model is None:
        raise HTTPException(404, 'Saved model not found')
    if not model.run_id:
        return model, None
    query = select(Run).where(Run.id == model.run_id)
    if lock:
        query = query.with_for_update().execution_options(populate_existing=True)
    run = (await db.execute(query)).scalar_one()
    return model, run


def set_state(run, tag, state):
    run.lifecycle = {**(run.lifecycle or {}), 'model_hf_uploads': {
        **(run.lifecycle or {}).get('model_hf_uploads', {}), tag: state,
    }}


@router.get('/models/{tag}/storage')
async def get_storage(tag: str, db=Depends(get_db)):
    model, run = await load_model(db, tag)
    state = upload_state(run, model) if run else {}
    loc = location(model.model_path)
    return {'location': loc, 'upload': state,
            'hf_url': loc['url'] or (state.get('url', '') if state.get('status') == 'succeeded' else ''),
            'can_upload': run is not None and bool(model.model_path)}


async def upload_command(db, model, run):
    """Resolve only the recorded checkpoint's authenticated machine, never a caller-supplied host."""
    from zevo.engine.run import model_save_worker
    script = Path(model_save_worker.__file__).read_text()
    loc = location(model.model_path)
    if loc['kind'] == 'local':
        source = Path(model.model_path)
        if not source.is_absolute():
            source = Path('/app') / source if (Path('/app') / source).exists() else Path.cwd() / source
        if not source.is_dir():
            raise HTTPException(409, 'The saved model directory is no longer available on the Zevo server.')
        return [sys.executable, '-c', script], str(source)
    if loc['kind'] != 'remote':
        raise HTTPException(409, 'This model has no uploadable saved checkpoint.')
    uri = urlsplit(model.model_path)
    tickets = (await db.execute(select(Ticket).where(Ticket.run_id == run.id)
                               .order_by(Ticket.created_at.desc()))).scalars().all()
    for ticket in tickets:
        info = await _device_info_for_ticket(db, ticket)
        if info is None:
            continue
        ssh = info.ssh
        if (ssh.host, ssh.port, ssh.user) != (uri.hostname, uri.port or 22, unquote(uri.username or '')):
            continue
        setup = str(getattr(getattr(info, 'cluster', None), 'env_setup', '') or '').strip()
        command = ('set -e\n' + (setup + '\n' if setup else '')
                   + 'exec timeout 3600 python3 -c ' + shlex.quote(script))
        return [*ssh_base_args(ssh), f'{ssh.user}@{ssh.host}', 'bash -lc ' + shlex.quote(command)], loc['path']
    raise HTTPException(409, 'The saved machine has no matching SSH connection for this run.')


async def finish_upload(run_id, tag, job_id, result):
    async with get_session_factory()() as db:
        run = (await db.execute(select(Run).where(Run.id == run_id).with_for_update())).scalar_one()
        old = (run.lifecycle or {}).get('model_hf_uploads', {}).get(tag, {})
        if old.get('id') == job_id:
            set_state(run, tag, {**old, **result})
            await db.commit()


async def perform_upload(run_id, tag, job_id, command, request):
    proc = None
    result = {'status': 'failed', 'error': 'Upload failed. Check the saved machine connection and the Hugging Face write token in Settings, then retry.'}
    try:
        proc = await asyncio.create_subprocess_exec(*command, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, _ = await asyncio.wait_for(proc.communicate(json.dumps(request).encode()), UPLOAD_TIMEOUT)
        lines = [line for line in stdout.decode().splitlines() if line.startswith('ZEVO_MODEL_SAVED=')]
        if proc.returncode == 0 and len(lines) == 1:
            receipt = json.loads(lines[0].split('=', 1)[1])
            if receipt.get('storage') == 'hf' and receipt.get('revision'):
                result = {'status': 'succeeded', 'url': receipt['path'], 'revision': receipt['revision'], 'error': ''}
    except (Exception, asyncio.CancelledError):
        # Provider output may contain credentials. Never persist or return it.
        pass
    finally:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()
        await finish_upload(run_id, tag, job_id, result)


@router.post('/models/{tag}/upload-hf', status_code=202)
async def upload_hf(tag: str, body: ModelSavePolicy, background: BackgroundTasks,
                    db=Depends(get_db)):
    if body.weights != 'hf':
        raise HTTPException(422, 'Choose a Hugging Face repository.')
    model, run = await load_model(db, tag, lock=True)
    if run is None:
        raise HTTPException(404, 'Saved model not found')
    state = upload_state(run, model)
    if location(model.model_path)['kind'] == 'hf' or state.get('status') == 'succeeded':
        raise HTTPException(409, 'This model is already on Hugging Face.')
    if state.get('status') == 'uploading':
        raise HTTPException(409, 'An upload is already in progress.')
    token = hf_token()
    if not token:
        raise HTTPException(409, {'code': 'hf_token_missing', 'message': 'Set a Hugging Face write token in Settings before uploading.'})
    command, source = await upload_command(db, model, run)
    state = {'id': uuid.uuid4().hex, 'status': 'uploading', 'source': model.model_path,
             'iteration': model.iteration, 'repo_id': body.hf_repo_id, 'private': body.hf_private,
             'started_at': datetime.now(timezone.utc).timestamp()}
    set_state(run, tag, state)
    await db.commit()
    background.add_task(perform_upload, run.id, tag, state['id'], command,
                        {'source': source, 'tag': tag, 'policy': body.model_dump(), 'token': token})
    return state
