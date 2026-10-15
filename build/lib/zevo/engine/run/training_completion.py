"""Backend gate between a Train result and any downstream work."""
from __future__ import annotations

import base64
import hashlib
import json
import runpy
import shlex
from pathlib import Path

from zevo.contracts.configuration import load_train_config


async def verify_completed_training(session, ticket, result, helper_path: Path) -> dict:
    config_path = Path(result.train_config_path)
    config = load_train_config(str(config_path))
    training = config.training.model_dump(mode="json")
    arguments = {
        "ticket_id": ticket.id,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "training": {"num_epochs": training["num_epochs"], "early_stopping": training.get("early_stopping")},
    }
    if not result.checkpoint_is_remote:
        helper = runpy.run_path(str(helper_path))
        return helper["verify_training_completion"](result.checkpoint_path, **arguments)
    from zevo.engine.run.remote_jobs import _device_info_for_ticket, _ssh
    info = await _device_info_for_ticket(session, ticket)
    if info is None:
        raise ValueError("cannot verify remote training completion without its device route")
    # Execute the backend's helper, not a user-supplied validator on the remote host.
    program = helper_path.read_text() + (
        "\ntry:\n result = verify_training_completion(" + repr(result.checkpoint_path)
        + ", **" + repr(arguments) + ")\n print(json.dumps({'ok': True, 'completion': result}))"
        + "\nexcept Exception as exc:\n print(json.dumps({'ok': False, 'error': str(exc)[:500]}))\n"
    )
    code = "import base64;exec(base64.b64decode(" + repr(base64.b64encode(program.encode()).decode()) + "))"
    response = await _ssh(info, "python3 -c " + shlex.quote(code), timeout_seconds=60)
    if not response.get("ok"):
        raise ValueError("remote training completion verification failed: " + str(response.get("error", "")))
    try:
        proof = json.loads(response.get("stdout", "").strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise ValueError("remote training completion verifier returned no valid evidence") from exc
    if proof.get("ok") is not True:
        raise ValueError(str(proof.get("error") or "remote training completion rejected"))
    return dict(proof["completion"])
