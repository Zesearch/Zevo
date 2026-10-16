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
        "training": {"num_epochs": training["num_epochs"], "early_stopping": training.get("early_stopping"), "adaptive_evaluation": training.get("adaptive_evaluation", False)},
    }
    if not result.checkpoint_is_remote:
        helper = runpy.run_path(str(helper_path))
        completion = helper["verify_training_completion"](result.checkpoint_path, **arguments)
        if training.get("adaptive_evaluation") or completion.get("adaptive_report"):
            await _verify_adaptive_decisions(session, ticket, completion)
        return completion
    from zevo.engine.run.remote_jobs import _device_info_for_ticket, _ssh
    info = await _device_info_for_ticket(session, ticket)
    if info is None:
        raise ValueError("cannot verify remote training completion without its device route")
    # Execute the backend's helper, not a user-supplied validator on the remote host.
    program = helper_path.read_text() + (
        "\ntry:\n result = verify_training_completion(" + repr(result.checkpoint_path)
        + ", **" + repr(arguments) + ")\n report = result.pop('adaptive_report', None)"
        + "\n if report: result['adaptive_report_sha256'] = __import__('hashlib').sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()"
        + "\n print(json.dumps({'ok': True, 'completion': result}))"
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
    completion = dict(proof["completion"])
    if completion.get("adaptive_report_sha256"):
        report = json.loads(Path(result.adaptive_training_report_path).read_text())
        digest = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
        if digest != completion["adaptive_report_sha256"]:
            raise ValueError("local adaptive report differs from published remote checkpoint")
        completion["adaptive_report"] = report
    if training.get("adaptive_evaluation") or completion.get("adaptive_report"):
        await _verify_adaptive_decisions(session, ticket, completion)
    return completion


async def _verify_adaptive_decisions(session, ticket, completion):
    from zevo.db import Run
    from sqlalchemy import select
    run = (await session.execute(select(Run).where(Run.id == ticket.run_id)
        .with_for_update().execution_options(populate_existing=True))).scalar_one()
    enabled = bool((run.lifecycle or {}).get("adaptive_mid_training_evaluation"))
    report = completion.get("adaptive_report")
    if not enabled:
        if report:
            raise ValueError("mid-training evaluation was not enabled for this Run")
        return
    if not report:
        raise ValueError("adaptive training completion lacks Validation evidence")
    lifecycle = run.lifecycle or {}
    if report.get("contract_sha256") != lifecycle.get("adaptive_validation_contracts", {}).get(ticket.id):
        raise ValueError("adaptive completion changed the frozen Validation contract")
    fraction = (ticket.payload or {}).get("first_validation_fraction")
    if fraction is not None:
        import math
        if (not report.get("measurements") or report["measurements"][0]["step"]
                != max(1, math.ceil(report["planned_steps"] * fraction))):
            raise ValueError("first Validation did not follow the Orchestrator plan")
    requests = lifecycle.get("adaptive_training_requests", {})
    for decision in report.get("decisions", []):
        request = requests.get(decision.get("request_id"), {})
        if (request.get("ticket_id") != ticket.id or request.get("status") != "decided"
                or request.get("decision") != decision):
            raise ValueError("adaptive completion contains a decision not approved by the Orchestrator")
        measured = report["measurements"][:len(request["measurements"])]
        if measured != request["measurements"]:
            raise ValueError("adaptive completion changed the measurements used for its decision")

    # Retain verified final telemetry even if the last worker log marker is lost.
    reports = dict(lifecycle.get("adaptive_training_reports") or {})
    reports[ticket.id] = json.loads(json.dumps(report))
    run.lifecycle = {**lifecycle, "adaptive_training_reports": reports}
    await session.flush()
