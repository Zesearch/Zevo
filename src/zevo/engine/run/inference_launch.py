"""Seed inference launches from successful executions, preserving repaired drafts.

Configuration inputs stay immutable. Launch scripts may adapt the current work
order; committed execution snapshots remain immutable evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path

from sqlalchemy import select


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_inputs(inp):
    paths = [inp.inference_config_path,
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
    raise ValueError("successful Slurm script has no safely rebindable --model argument; task-specific launch adaptation required")


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


def effective_snapshot(source, controller_body, workload_launch="legacy_bash"):
    """Use the environment that actually ran, including retained allocations.

    A workload's SBATCH container directives are inert when an existing
    controller executes it with bash. Do not activate those directives on replay.
    """
    if workload_launch == "host_srun_v1":
        # New host controllers execute these directives on the workload step.
        return source
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
            if _slurm_stage_implementation_sha256(SimpleNamespace(stage="inference", script_path=script)) != meta["implementation_sha256"]:
                continue
            if digest(predict) != digest(Path(script).with_name("predict.py")):
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
            from zevo.engine.run.remote_jobs import _device_info_for_ticket
            from zevo.engine.run.gpu_controller import read_controller_headers
            info = await _device_info_for_ticket(session, owner)
            if not info:
                continue
            headers = await read_controller_headers(info, meta["controller_directory"])
            if headers is None:
                continue
            return effective_snapshot(source, headers, meta.get("controller_workload_launch", "legacy_bash"))
        except (OSError, ValueError, KeyError):
            continue
    return None


def launch_target_signature(inp, pins):
    """Identify the work order a mutable launch draft was prepared for."""
    job = inp.slurm_job
    target = {
        "model": inp.checkpoint_path or inp.base_model,
        "configuration_mode": inp.configuration_mode,
        "inputs": pins,
        "questions": inp.scoring_set,
        "sample_submission": inp.sample_submission,
        "members": [{key: getattr(member, key, "") for key in (
            "benchmark_id", "name", "scoring_set", "sample_submission",
            "inference_config_path", "configuration_mode", "work_dir",
        )} for member in inp.suite_members],
        "job": {key: getattr(job, key, "") for key in (
            "attempt", "script_path", "remote_script_path", "remote_work_dir",
            "job_name", "nodes", "num_gpus", "gpus_per_node", "stdout_path",
            "stderr_path", "status_path", "runtime_prologue", "lifecycle_prologue",
        )},
    }
    return hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()


async def prepare_launch(session, run, ticket, inp):
    """Pin reuse inputs for all providers; issue Slurm replay when applicable."""
    if inp.configuration_mode != "reuse" and not inp.reusable_predict_script_path:
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
        original = dict(ticket_pins[revision])
        # Older activations pinned the advisory predict source alongside configs.
        if inp.reusable_predict_script_path:
            original.pop(str(Path(inp.reusable_predict_script_path).resolve()), None)
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
    # Launch adaptation is independent of config reuse. Real GPU failures also
    # authorize repair of the environment; collect must preserve executed bytes.
    prior = contract.previous_execution or {}
    repairing = contract.phase == "submit" and contract.attempt > 1 and prior.get("scheduler_state") in {
        "FAILED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL", "BOOT_FAIL", "CANCELLED", "PREEMPTED"}
    if contract.phase == "collect":
        row = await session.get(InfraInstance, contract.bookkeeping_row_id)
        expected = (row.meta or {}).get("script_sha256", "") if row else ""
        pins = dict((row.meta or {}).get("frozen_reuse_artifacts", pins) if row else pins)
        if inp.reusable_predict_script_path:
            pins.pop(str(Path(inp.reusable_predict_script_path).resolve()), None)
        verify_frozen_inputs(pins)
        return inp.model_copy(update={"frozen_reuse_artifacts": pins,
                                      "slurm_launch_reuse": {"mode": "collect", "expected_sha256": expected}})
    signature = launch_target_signature(inp, pins)
    drafts = dict((run.lifecycle or {}).get("inference_launch_drafts", {}))
    ticket_drafts = dict(drafts.get(ticket.id, {}))
    draft = ticket_drafts.get(revision, {})
    if draft.get("target_sha256") == signature and Path(contract.script_path).is_file():
        provenance = dict(draft["provenance"])
        # A result-schema repair must not overwrite a corrected launcher.
        return inp.model_copy(update={"frozen_reuse_artifacts": pins,
                                      "slurm_launch_reuse": provenance})
    rows = (await session.execute(select(InfraInstance, Ticket).join(Ticket, Ticket.id == InfraInstance.ticket_id)
        .where(InfraInstance.run_id == run.id, Ticket.agent_id == "inference")
        # A completed execution remains usable when its ticket is running again
        # for remeasurement. Prefer that ticket's own latest successful launch.
        .order_by((Ticket.id == ticket.id).desc(), InfraInstance.created_at.desc()))).all()
    for row, owner in rows:
        # Validation may never inherit held-out scripts or paths. Test may use
        # the successful Validation launcher, with its own bound data paths.
        if owner.lane != ticket.lane and not (ticket.lane == "held_out_test" and owner.lane == "optimization"):
            continue
        if (owner.payload or {}).get("base_model") != inp.base_model:
            continue
        if not (row.meta or {}).get("submission_committed"):
            continue
        # A validated inference can survive controller cleanup failure. For a
        # non-COMPLETED scheduler row require the accepted result and both
        # execution hashes, never a submitted draft snapshot alone.
        source = ((row.meta or {}).get("inference_launch")
                  if (row.meta or {}).get("scheduler_state") == "COMPLETED" else None)
        if not source:
            source = await verified_existing_launch(session, row, owner)
        if not source:
            continue
        if (not isinstance(source.get("body"), str)
                or hashlib.sha256(source["body"].encode()).hexdigest() != source.get("sha256")):
            continue  # Corrupt reference evidence must not block fresh implementation.
        provenance = {"source_ticket_id": owner.id, "source_execution_id": row.id,
                      "source_sha256": source["sha256"], "mode": "repair" if repairing else "reuse"}
        adapting = not repairing and inp.configuration_mode == "select"
        if adapting:
            provenance.update(mode="adapt", reason=(
                "The target benchmarks require their own input mappings and length calibration. "
                "Use the successful launch and predict script as references when useful. "
                "Adapt or regenerate the implementation for this work order; keep its config contract."
            ))
        source_path = Path(inp.work_dir) / "launch-reuse" / (source["sha256"] + ".sbatch")
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source["body"])
        provenance["source_script_path"] = str(source_path.resolve())
        try:
            body = render(source, inp)
        except (ValueError, KeyError, TypeError) as exc:
            # Reference-only: arbitrary shell expressions need not be mechanically
            # rebindable. The agent can adapt or replace this seed.
            body = source["body"]
            provenance["reason"] = (provenance.get("reason", "") + " " + str(exc)).strip()
        Path(contract.script_path).parent.mkdir(parents=True, exist_ok=True)
        Path(contract.script_path).write_text(body)
        provenance["seed_sha256"] = digest(contract.script_path)
        ticket_drafts[revision] = {"target_sha256": signature, "provenance": provenance}
        drafts[ticket.id] = ticket_drafts
        run.lifecycle = {**(run.lifecycle or {}), "inference_launch_drafts": drafts}
        await session.flush()
        return inp.model_copy(update={"frozen_reuse_artifacts": pins, "slurm_launch_reuse": provenance})
    provenance = {"mode": "unavailable",
        "reason": "No verified successful Slurm reference is available; prepare a launcher for the current work order"}
    ticket_drafts[revision] = {"target_sha256": signature, "provenance": provenance}
    drafts[ticket.id] = ticket_drafts
    run.lifecycle = {**(run.lifecycle or {}), "inference_launch_drafts": drafts}
    await session.flush()
    return inp.model_copy(update={"frozen_reuse_artifacts": pins,
                                  "slurm_launch_reuse": provenance})
