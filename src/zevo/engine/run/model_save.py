"""One destination implementation for normal Registry and cancel-and-save."""
from __future__ import annotations
import asyncio
import hashlib
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


# The worker's HF branch imports huggingface_hub. On a cloud/instance host the
# login shell's python3 is the bare system interpreter (DLAMI: Python 3.10 with
# no Hub client), so the save died with ModuleNotFoundError after a complete
# optimisation (run 8ee1331e). Use the first interpreter that can import the
# client: the env_setup shell (clusters/instances), then the system-owned
# profiles that remote_env.sh built for Inference and Train on this host.
_REMOTE_INTERPRETER_PROBE = (
    'ZEVO_PY=python3\n'
    'for ZEVO_CANDIDATE in python3 "$HOME/zevo/env/infer/bin/python" "$HOME/zevo/env/train/bin/python"; do\n'
    '  if command -v "$ZEVO_CANDIDATE" >/dev/null 2>&1 && "$ZEVO_CANDIDATE" -c "import huggingface_hub" >/dev/null 2>&1; then\n'
    '    ZEVO_PY="$ZEVO_CANDIDATE"; break\n'
    '  fi\n'
    'done\n'
)


def save_identity(run, source: str, policy: ModelSavePolicy) -> str:
    return json.dumps({'source': source, 'policy': policy.model_dump(),
                       'tag': model_tag_for_run(run.id)}, sort_keys=True)


async def save_model(session, run, ticket, *, source: str, remote: bool, policy: ModelSavePolicy,
                     model_card: str | None = None) -> dict:
    from zevo.engine.run.cancel_rescue import hf_token
    request = {'source': source, 'policy': policy.model_dump(), 'tag': model_tag_for_run(run.id)}
    identity = save_identity(run, source, policy)
    if model_card is not None:
        request['model_card'] = model_card
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
                   + _REMOTE_INTERPRETER_PROBE
                   + 'exec env ZEVO_TICKET_ID=' + shlex.quote(ticket.id)
                   + ' timeout 3600 "$ZEVO_PY" -c ' + shlex.quote(script))
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
        receipt['ticket_id'] = ticket.id
        receipt['model_card_sha256'] = hashlib.sha256(model_card.encode()).hexdigest() if model_card is not None else ''
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


async def finalize_registry_model(session, run, ticket, *, inp, output, work_dir: str) -> dict:
    """Finalize a validated Registry result; failures use normal ticket repair."""
    import os
    import tempfile
    import yaml
    from zevo.contracts.tickets import TERMINAL_RUN_STATUSES

    await session.refresh(ticket, attribute_names=['status'])
    await session.refresh(run, attribute_names=['status', 'lifecycle'])
    if ticket.status == 'cancelled' or run.status in TERMINAL_RUN_STATUSES or run.status == 'paused':
        raise ValueError('Run or ticket is no longer active; model preservation was not started')
    policy = ModelSavePolicy.model_validate(inp.model_save_policy)
    model_card = None
    if output.model_card_path:
        if policy.weights != 'hf':
            raise ValueError('A model card replacement is only applicable to HF preservation')
        card = Path(output.model_card_path).resolve()
        if not card.is_relative_to(Path(work_dir).resolve()):
            raise ValueError('Corrected model card must be inside this ticket work_dir')
        # Bound the actual read as well as the nominal file size.
        with card.open('rb') as stream:
            content = stream.read(131073)
        if not content or len(content) > 131072:
            raise ValueError('Corrected model card must be nonempty and at most 128 KiB')
        model_card = content.decode('utf-8')
    identity = save_identity(run, inp.checkpoint_path, policy)
    card_hash = hashlib.sha256(model_card.encode()).hexdigest() if model_card is not None else ''
    receipt = (run.lifecycle or {}).get('model_storage') or {}
    if not (receipt.get('identity') == identity
            and receipt.get('ticket_id', ticket.id) == ticket.id
            and receipt.get('model_card_sha256', '') == card_hash):
        receipt = await save_model(session, run, ticket, source=inp.checkpoint_path,
            remote=inp.checkpoint_is_remote, policy=policy, model_card=model_card)
    # Only engine-owned destination fields change after preservation. All
    # provenance and evaluation facts were validated before this function.
    manifest = Path(output.registry_path)
    body = yaml.safe_load(manifest.read_text())
    entry = body['models'][inp.expected_version_tag]
    entry.update(model_path=receipt['path'], storage=receipt['storage'], revision=receipt['revision'])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=manifest.parent, prefix='.registry-', delete=False) as stream:
            temporary = Path(stream.name)
            yaml.safe_dump(body, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    inp.saved_model_path = receipt['path']
    inp.saved_model_revision = receipt['revision']
    return receipt
