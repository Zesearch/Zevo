"""One destination implementation for normal Registry and cancel-and-save."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
import shlex
import sys
from urllib.parse import quote
from zevo.contracts.model_save import ModelSavePolicy
from zevo.contracts.model_registry import model_tag_for_run
from zevo.engine.remote_transfer import ssh_base_args
from zevo.engine.run import process_registry
from zevo.engine.run.remote_jobs import _device_info_for_ticket


async def save_model(session, run, ticket, *, source: str, remote: bool, policy: ModelSavePolicy) -> dict:
    from zevo.engine.run.cancel_rescue import hf_token
    request = {'source': source, 'policy': policy.model_dump(), 'tag': model_tag_for_run(run.id)}
    identity = json.dumps(request, sort_keys=True)
    if policy.weights == 'hf':
        request['token'] = hf_token()
        if not request['token']:
            raise ValueError('HF_TOKEN is not set in Settings')
    script = Path(__file__).with_name('model_save_worker.py').read_text()
    info = None
    if remote:
        info = await _device_info_for_ticket(session, ticket)
        if info is None:
            raise ValueError('No configured SSH route to the checkpoint host')
        setup = str(getattr(getattr(info, 'cluster', None), 'env_setup', '') or '').strip()
        command = ('set -e\n' + (setup + '\n' if setup else '')
                   + 'exec env ZEVO_TICKET_ID=' + shlex.quote(ticket.id)
                   + ' timeout 3600 python3 -c ' + shlex.quote(script))
        args = [*ssh_base_args(info.ssh), f'{info.ssh.user}@{info.ssh.host}', 'bash -lc ' + shlex.quote(command)]
    else:
        if policy.weights == 'remote':
            raise ValueError('GPU-machine storage requires a remote checkpoint and its SSH connection')
        args = [sys.executable, '-c', script]
    proc = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
    process_registry.register(ticket.id, proc)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(json.dumps(request).encode()), timeout=3600)
        if proc.returncode:
            # Never echo provider stderr or credentials into a shared run log.
            safe = [line for line in stderr.decode(errors='replace').splitlines()
                    if line.startswith('Model preservation failed:')]
            raise RuntimeError(safe[-1] if safe else f'Model preservation transport exited {proc.returncode}')
        lines = [line for line in stdout.decode().splitlines() if line.startswith('ZEVO_MODEL_SAVED=')]
        if len(lines) != 1:
            raise RuntimeError('Model preservation returned no verified receipt')
        receipt = json.loads(lines[0].split('=', 1)[1])
        receipt['identity'] = identity
        if receipt['storage'] == 'remote':
            ssh = info.ssh
            receipt['remote_path'] = receipt['path']
            receipt['instance_id'] = info.instance_id
            receipt['provider'] = info.provider
            receipt['path'] = f'ssh://{quote(ssh.user, safe="")}@{ssh.host}:{ssh.port}{quote(receipt["path"], safe="/")}'
        # Resource cleanup consults this receipt before releasing a cloud box.
        await session.refresh(run, attribute_names=['lifecycle'])
        run.lifecycle = {**(run.lifecycle or {}), 'model_storage': receipt}
        await session.commit()
        return receipt
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        process_registry.unregister(ticket.id, proc)
