"""Replay successful Inference Slurm launches without asking an agent to rewrite them.

The snapshot belongs to a committed workload, not the latest mutable script on
disk. Rebinding is deliberately narrow: paths, model argument, ticket identity,
 lifecycle markers and scheduler resources. Everything else is byte-preserved.
"""
from __future__ import annotations

import hashlib
import re
import shlex
from pathlib import Path

from sqlalchemy import select


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_inputs(inp):
    paths = [inp.inference_config_path, inp.reusable_predict_script_path,
             *[m.inference_config_path for m in inp.suite_members]]
    try:
        return {str(Path(p).resolve()): digest(p) for p in paths if p}
    except OSError as exc:
        raise ValueError(f"reused Inference artifact unavailable: {exc}") from exc


def verify_frozen_inputs(expected):
    for path, sha in expected.items():
        try:
            actual = digest(path)
        except OSError as exc:
            raise ValueError(f"reused Inference artifact unavailable: {path}") from exc
        if actual != sha:
            raise ValueError(f"reused Inference artifact changed after input binding: {path}")


def snapshot(inp, body):
    contract = inp.slurm_job
    return {
        "version": 1, "body": body,
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "ticket_id": inp.ticket_id, "model": inp.checkpoint_path or inp.base_model,
        "work_dir": str(Path(inp.work_dir).resolve()),
        "contract": contract.model_dump(mode="json"),
        "inputs": {"questions": inp.scoring_set, "sample_submission": inp.sample_submission,
                   "inference_config": inp.inference_config_path},
        "predict_sha256": digest(Path(inp.work_dir) / "predict.py"),
    }


def _model_argument(body, old, new):
    """Change only the model argument (or its literal variable assignment).

    A global base-model replacement would also change tokenizer/template
    identity. Unknown shell expressions require repair, never a guess.
    """
    quoted = r"(?:" + "|".join(re.escape(v) for v in (shlex.quote(old), '"' + old + '"', "'" + old + "'")) + r")"
    pattern = r"(--model(?:=|\s+))" + quoted + r"(?=\s|\\|$)"
    body, count = re.subn(pattern, lambda m: m[1] + shlex.quote(new), body)
    if count:
        return body
    variables = re.findall(r'--model(?:=|\s+)["\']?\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?', body)
    for variable in set(variables):
        pattern = r"(?m)^(\s*(?:export\s+)?" + re.escape(variable) + r"=)" + quoted + r"(?=\s|$)"
        body, count = re.subn(pattern, lambda m: m[1] + shlex.quote(new), body)
        if count:
            return body
    raise ValueError("successful Slurm script has no safely rebindable --model argument; explicit launch repair required")


def render(source, inp):
    body = source["body"]
    if hashlib.sha256(body.encode()).hexdigest() != source["sha256"]:
        raise ValueError("successful Inference launch snapshot changed")
    old, new = source["contract"], inp.slurm_job
    # Replace the argument before paths so tokenizer references stay unchanged.
    body = _model_argument(body, source["model"], inp.checkpoint_path or inp.base_model)
    replacements = {
        old["lifecycle_prologue"]: new.lifecycle_prologue,
        old["remote_work_dir"]: new.remote_work_dir,
        source["work_dir"]: str(Path(inp.work_dir).resolve()),
        source["ticket_id"]: inp.ticket_id,
    }
    for role, current in (("questions", inp.scoring_set), ("sample_submission", inp.sample_submission),
                          ("inference_config", inp.inference_config_path)):
        previous = source["inputs"].get(role)
        if previous and current:
            replacements[previous] = current
    # Single pass: replacement values must not themselves be rewritten.
    keys = sorted((k for k in replacements if k), key=len, reverse=True)
    body = re.sub("|".join(re.escape(k) for k in keys), lambda m: replacements[m[0]], body)
    # Resource requests are engine-owned; preserve site/environment/container
    # directives and the executable body from the successful launch.
    resource_keys = {"job-name", "output", "error", "chdir", "nodes", "gpus", "gres", "gpus-per-node", "gpus-per-task"}
    lines = []
    for line in body.splitlines():
        directive = re.match(r"^#SBATCH\s+--([\w-]+)(?:=|\s|$)", line)
        if directive and directive[1] in resource_keys:
            continue
        if re.match(r"^#SBATCH\s+-(?:J|o|e|D|N|G)(?:=|\s|[^-])", line):
            continue
        lines.append(line)
    gpu_type = re.search(r"(?m)^#SBATCH\s+--(?:gres=gpu:|gpus-per-node=)([A-Za-z][\w-]*):\d+", body)
    gpu_request = f"{gpu_type[1]}:{new.gpus_per_node}" if gpu_type else str(new.gpus_per_node)
    headers = [f"#SBATCH --job-name={new.job_name}", f"#SBATCH --output={new.stdout_path}",
               f"#SBATCH --error={new.stderr_path}", f"#SBATCH --chdir={new.remote_work_dir}",
               f"#SBATCH --nodes={new.nodes}", f"#SBATCH --gpus-per-node={gpu_request}"]
    lines[1:1] = headers
    return "\n".join(lines) + "\n"


def effective_snapshot(source, controller_body):
    """Use the environment that actually ran, including retained allocations.

    A workload's SBATCH container directives are inert when an existing
    controller executes it with bash. Do not activate those directives on replay.
    """
    pattern = r"^#SBATCH\s+--container-[\w-]+(?:=|\s|$)"
    directives = [line for line in controller_body.splitlines() if re.match(pattern, line)]
    lines = [line for line in source["body"].splitlines() if not re.match(pattern, line)]
    lines[1:1] = directives
    body = "\n".join(lines) + "\n"
    return {**source, "body": body, "sha256": hashlib.sha256(body.encode()).hexdigest()}


async def verified_existing_launch(session, row, owner):
    """Import an existing successful launch only if both execution hashes match.

    This is evidence-based snapshotting, never a fallback to an arbitrary file
    named predict.sbatch left by a later failed activation.
    """
    from types import SimpleNamespace
    from zevo.db import HeartbeatResult
    from zevo.engine.run.runner import _slurm_stage_implementation_sha256
    meta = row.meta or {}
    results = (await session.execute(select(HeartbeatResult).where(
        HeartbeatResult.ticket_id == owner.id, HeartbeatResult.status == "succeeded")
        .order_by(HeartbeatResult.created_at.desc()))).scalars().all()
    for result in results:
        output = result.output or {}
        script = output.get("slurm_script_path")
        predict = output.get("predict_script_path")
        if not script or not predict or not meta.get("script_sha256") or not meta.get("implementation_sha256"):
            continue
        try:
            if digest(script) != meta["script_sha256"]:
                continue
            if _slurm_stage_implementation_sha256(SimpleNamespace(script_path=script)) != meta["implementation_sha256"]:
                continue
            from zevo.contracts.infrastructure import slurm_lifecycle_prologue
            from zevo.engine.run.scheduler.bindings import input_path
            body = Path(script).read_text()
            source = {"version": 1, "body": body, "sha256": digest(script), "ticket_id": owner.id,
                "model": input_path(owner.inputs or {}, "checkpoint", required=False) or owner.payload["base_model"],
                "work_dir": str(Path(script).resolve().parent),
                "contract": {"remote_work_dir": meta["remote_workdir"],
                             "lifecycle_prologue": slurm_lifecycle_prologue(meta["status_path"])},
                "predict_sha256": digest(predict),
                "inputs": {"questions": owner.payload.get("scoring_set", ""),
                           "sample_submission": owner.payload.get("sample_submission", ""),
                           "inference_config": output.get("inference_config_path", "")}}
            if not meta.get("controller_directory"):
                continue
            from zevo.engine.run.remote_jobs import _device_info_for_ticket, _ssh
            info = await _device_info_for_ticket(session, owner)
            if not info:
                continue
            response = await _ssh(info, "cat -- " + shlex.quote(meta["controller_directory"] + "/controller.sbatch"), timeout_seconds=30)
            if not response.get("ok"):
                continue
            return effective_snapshot(source, response["stdout"])
        except (OSError, ValueError, KeyError):
            continue
    return None


async def prepare_launch(session, run, ticket, inp):
    """Pin reuse inputs for all providers; issue Slurm replay when applicable."""
    if inp.configuration_mode != "reuse":
        return inp
    pins = frozen_inputs(inp)
    from zevo.db import Run
    from zevo.engine.run.remeasurement import active_attempt
    # Serialize lifecycle updates; repair must retain the original pins, while
    # an engine-authorized length remeasurement gets its own revision.
    await session.execute(select(Run).where(Run.id == run.id).with_for_update().execution_options(populate_existing=True))
    attempt = active_attempt(run, ticket.id)
    revision = str(attempt["revision"]) if attempt else "original"
    all_pins = dict((run.lifecycle or {}).get("inference_reuse_artifacts", {}))
    ticket_pins = dict(all_pins.get(ticket.id, {}))
    if revision in ticket_pins:
        original = ticket_pins[revision]
        if set(original) != set(pins):
            raise ValueError("reused Inference bindings changed without a measurement revision")
        verify_frozen_inputs(original)
        pins = original
    else:
        ticket_pins[revision] = pins
        all_pins[ticket.id] = ticket_pins
        run.lifecycle = {**(run.lifecycle or {}), "inference_reuse_artifacts": all_pins}
        await session.flush()
    contract = inp.slurm_job
    if not contract.enabled:
        return inp.model_copy(update={"frozen_reuse_artifacts": pins})
    from zevo.db import InfraInstance, Ticket
    # A failed actual workload can authorize a changed launch. A typo in a
    # collect Result alone cannot turn immutable replay into free-form codegen.
    prior = contract.previous_execution or {}
    repairing = contract.phase == "submit" and contract.attempt > 1 and prior.get("scheduler_state") in {
        "FAILED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL", "BOOT_FAIL", "CANCELLED", "PREEMPTED"}
    if contract.phase == "collect":
        row = await session.get(InfraInstance, contract.bookkeeping_row_id)
        source = (row.meta or {}).get("inference_launch") if row else None
        expected = (row.meta or {}).get("script_sha256", "") if row else ""
        pins = (row.meta or {}).get("frozen_reuse_artifacts", pins) if row else pins
        verify_frozen_inputs(pins)
        return inp.model_copy(update={"frozen_reuse_artifacts": pins,
                                      "slurm_launch_reuse": {"mode": "collect", "expected_sha256": expected}})
    rows = (await session.execute(select(InfraInstance, Ticket).join(Ticket, Ticket.id == InfraInstance.ticket_id)
        .where(InfraInstance.run_id == run.id, Ticket.agent_id == "inference",
               Ticket.status.in_(("succeeded", "degraded")))
        .order_by(InfraInstance.created_at.desc()))).all()
    expected_predict = digest(inp.reusable_predict_script_path) if inp.reusable_predict_script_path else ""
    for row, owner in rows:
        # Validation may never inherit held-out scripts or paths. Test may use
        # the successful Validation launcher, with its own bound data paths.
        if owner.lane != ticket.lane and not (ticket.lane == "held_out_test" and owner.lane == "optimization"):
            continue
        if (owner.payload or {}).get("base_model") != inp.base_model:
            continue
        if (row.meta or {}).get("scheduler_state") != "COMPLETED" or not (row.meta or {}).get("submission_committed"):
            continue
        source = (row.meta or {}).get("inference_launch")
        if not source:
            source = await verified_existing_launch(session, row, owner)
        if not source:
            continue
        if expected_predict and source.get("predict_sha256") != expected_predict:
            continue
        provenance = {"source_ticket_id": owner.id, "source_execution_id": row.id,
                      "source_sha256": source["sha256"], "mode": "repair" if repairing else "reuse"}
        source_path = Path(inp.work_dir) / "launch-reuse" / (source["sha256"] + ".sbatch")
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source["body"])
        provenance["source_script_path"] = str(source_path.resolve())
        if not repairing:
            try:
                body = render(source, inp)
            except ValueError as exc:
                # Unknown shell interpolation cannot be safely rewritten by
                # the engine. Explicit repair may adapt the launcher while
                # configuration and source implementation stay pinned.
                if "safely rebindable" not in str(exc):
                    raise
                provenance.update(mode="repair", reason=str(exc))
            else:
                Path(contract.script_path).parent.mkdir(parents=True, exist_ok=True)
                Path(contract.script_path).write_text(body)
                provenance["expected_sha256"] = digest(contract.script_path)
        return inp.model_copy(update={"frozen_reuse_artifacts": pins, "slurm_launch_reuse": provenance})
    # Missing evidence must not silently fall back to regenerating a launch.
    # Real failed workloads may still enter explicit, audited repair.
    return inp.model_copy(update={"frozen_reuse_artifacts": pins,
        "slurm_launch_reuse": {"mode": "repair" if repairing else "unavailable",
            "reason": "No successful compatible Slurm launch snapshot is available"}})
