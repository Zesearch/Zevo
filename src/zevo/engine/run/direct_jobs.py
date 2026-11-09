"""Direct-host adapter for the same finite workload protocol as Slurm.

Execution rows alias an existing allocation; they never rent a machine, hold a
second GPU lease, or accrue another infrastructure bill.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import select

from zevo.contracts.infrastructure import InfrastructureDeviceInfo, SlurmStageJobContract
from zevo.db import HeartbeatRun, InfraInstance, Run, Ticket
from zevo.engine.run.remote_jobs import _ssh


def direct(row):
    return bool((row.meta or {}).get("direct_workload"))


def contract_for(*, info, ticket, work_dir, stage, previous, retry, restart_instruction_id=""):
    meta = (previous.meta or {}) if previous else {}
    attempt = int(meta.get("execution_attempt", 1)) + bool(retry) if previous else 1
    root = PurePosixPath(info.instance.workdir) / ticket.id / f"attempt-{attempt}"
    result = SlurmStageJobContract(
        enabled=True, transport="direct", stage=stage,
        phase="submit" if previous is None or retry else "collect",
        script_path=str(Path(work_dir) / f"{stage}.sh"),
        remote_work_dir=str(root), remote_script_path=str(root / f"{stage}.sh"),
        status_path=str(root / "workload.json"), stdout_path=str(root / "workload.out"),
        stderr_path=str(root / "workload.err"), job_name=f"zevo-{ticket.id}",
        num_gpus=info.resource_plan.num_gpus, nodes=1, attempt=attempt,
        restart_instruction_id=restart_instruction_id,
        retry_of_bookkeeping_row_id=previous.id if retry else meta.get("retry_of_bookkeeping_row_id", ""),
        retry_of_job_id=previous.instance_id if retry else meta.get("retry_of_job_id", ""),
        bookkeeping_row_id=previous.id if previous and not retry else "",
        job_id=previous.instance_id if previous and not retry else "",
        scheduler_state=meta.get("scheduler_state", "") if previous and not retry else "",
        scheduler_exit_code=meta.get("scheduler_exit_code", "") if previous and not retry else "",
        scheduler_reason=meta.get("scheduler_reason", "") if previous and not retry else "",
        implementation_sha256=meta.get("implementation_sha256", "") if previous and not retry else "",
    )
    if retry:
        prior = dict(meta["execution_contract"])
        prior.update(phase="collect", bookkeeping_row_id=previous.id, job_id=previous.instance_id,
                     scheduler_state=meta.get("scheduler_state", ""),
                     scheduler_exit_code=meta.get("scheduler_exit_code", ""),
                     scheduler_reason=meta.get("scheduler_reason", ""), previous_execution={})
        prior["implementation_sha256"] = meta.get("implementation_sha256", "")
        result.previous_execution = prior
    return result


async def remote_call(info, manifest, action, *, helper_body=None):
    remote = str(PurePosixPath(manifest["script"]).parent / "zevo_direct_workload.py")
    if helper_body is not None:
        # The caller supplies engine-owned bytes, never an Agent implementation.
        encoded = base64.b64encode(helper_body).decode()
        program = ("from pathlib import Path;import base64,os;"
                   f"p=Path({remote!r});p.parent.mkdir(parents=True,exist_ok=True);"
                   f"body=base64.b64decode({encoded!r});"
                   "assert not p.exists() or p.read_bytes()==body,'supervisor changed';"
                   "p.write_bytes(body) if not p.exists() else None")
        result = await _ssh(info, "python3 -c " + shlex.quote(program), timeout_seconds=60)
        if not result.get("ok"):
            raise RuntimeError("cannot stage workload supervisor: " + result.get("error", ""))
    payload = json.dumps(manifest)
    command = ("printf %s " + shlex.quote(payload) + " | python3 " + shlex.quote(remote)
               + " " + shlex.quote(action))
    result = await _ssh(info, command, timeout_seconds=60, stdout_limit=16000)
    if not result.get("ok"):
        raise RuntimeError("workload observation/publication unavailable: " + result.get("error", ""))
    data = json.loads(result["stdout"].strip().splitlines()[-1])
    if data.get("execution_id") != manifest["execution_id"]:
        raise ValueError("remote receipt belongs to another execution")
    return data


async def submit(session, *, run, ticket, inp, heartbeat_id, remote_train_config_path="", prepared_config_paths=()):
    from zevo.engine.run.runner import _latest_slurm_resource_request, _slurm_stage_implementation_sha256
    from zevo.engine.run.inference_launch import snapshot
    from zevo.contracts.inference import InferenceTaskInput
    contract = inp.slurm_job
    if not contract.enabled or contract.transport != "direct" or contract.phase != "submit":
        raise ValueError("direct dispatch requires a prepared direct execution")
    # Serialize admission against cancellation and any other attempt publisher.
    await session.execute(select(Run).where(Run.id == run.id).with_for_update())
    await session.refresh(ticket)
    await session.refresh(run)
    if run.cancel_requested_at or run.status in {"cancelled", "failed", "success", "degraded"} or ticket.status == "cancelled":
        raise ValueError("workload cancelled before publication")
    previous = await _latest_slurm_resource_request(session, ticket)
    if ticket.agent_id == "train":
        from zevo.engine.run.repair_progress import verify_repair_training_config
        verify_repair_training_config(previous, Path(contract.script_path).with_name("train_config.yaml"))
    execution_id = str(uuid5(NAMESPACE_URL, f"zevo:direct:{ticket.id}:{contract.attempt}:{contract.remote_work_dir}"))
    existing = await session.get(InfraInstance, execution_id)
    if existing:
        if (existing.meta or {}).get("script_sha256") != hashlib.sha256(Path(contract.script_path).read_bytes()).hexdigest():
            raise ValueError("published execution cannot change its launch script")
        return existing
    if contract.attempt > 1:
        if (not previous or previous.id != contract.retry_of_bookkeeping_row_id
                or previous.instance_id != contract.retry_of_job_id
                or int(previous.meta.get("execution_attempt", 1)) + 1 != contract.attempt
                or previous.meta.get("scheduler_state") not in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}):
            raise ValueError("replacement must supersede the exact terminal execution")
    elif previous:
        raise ValueError("a workload already exists for this ticket")
    info = InfrastructureDeviceInfo.model_validate_json(Path(inp.device_info_path).read_text())
    if info.provider not in {"cloud", "instance"} or info.provider != run.gpu_provider or not info.instance:
        raise ValueError("direct workload requires the Run's existing host")
    owners = (await session.scalars(select(InfraInstance).where(
        InfraInstance.run_id == run.id, InfraInstance.provider == info.provider,
        InfraInstance.released_at.is_(None)))).all()
    owner = next((r for r in owners if not direct(r) and not (r.meta or {}).get("allocation_owner_row_id")
                  and (info.provider != "cloud" or r.instance_id == info.instance_id)), None)
    if info.provider == "cloud" and owner is None:
        raise ValueError("cloud allocation has been released; cannot publish a workload")
    script_hash = hashlib.sha256(Path(contract.script_path).read_bytes()).hexdigest()
    implementation = _slurm_stage_implementation_sha256(contract)
    if previous:
        from zevo.engine.run.remeasurement import restarts_workload
        from zevo.engine.run.adaptive_training import checkpoint_resume_authorized
        from zevo.engine.run.training_continuation import authorized as continuation_authorized
        from zevo.engine.run.runner import resumable_stage_interruption, execution_retry_authorized
        continuing = continuation_authorized(ticket, previous)
        restart = bool(contract.restart_instruction_id or restarts_workload(run, ticket.id, previous.id)
                       or checkpoint_resume_authorized(run, ticket, previous) or continuing
                       or resumable_stage_interruption(ticket, previous)
                       or execution_retry_authorized(previous.meta))
        if previous.meta.get("implementation_sha256") == implementation and not restart:
            raise ValueError("unchanged failed workload requires a verified continuation or authorized restart")
        if continuing:
            from types import SimpleNamespace
            from zevo.contracts.train import TrainingContinuation
            from zevo.engine.run.training_continuation import verify_request
            proof = previous.meta["training_continuation"]
            expected = {key: proof[key] for key in TrainingContinuation.model_fields}
            if inp.continuation is None or inp.continuation.model_dump() != expected:
                raise ValueError("direct continuation lost its full-state checkpoint binding")
            await verify_request(session, ticket,
                inp.model_copy(update={"slurm_job": SlurmStageJobContract.model_validate(contract.previous_execution)}),
                SimpleNamespace(continuation=inp.continuation,
                    train_config_path=str(Path(contract.script_path).with_name("train_config.yaml"))), previous)
    # Bind the generated program and every engine-owned helper staged beside it.
    files = {str(PurePosixPath(contract.remote_work_dir) / p.name): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in Path(contract.script_path).parent.glob("*.py")}
    config_paths = prepared_config_paths
    if contract.stage == "inference" and not config_paths:
        config_paths = (getattr(inp, "inference_config_path", "") or str(Path(contract.script_path).with_name("inference_config.yaml")),)
    if contract.stage == "train":
        config_paths = (str(Path(contract.script_path).with_name("train_config.yaml")),)
        remote_train_config_path = remote_train_config_path or str(PurePosixPath(contract.remote_work_dir) / "train_config.yaml")
        remote_config = PurePosixPath(remote_train_config_path)
        if (not remote_config.is_absolute() or ".." in remote_config.parts
                or not remote_config.is_relative_to(PurePosixPath(contract.remote_work_dir).parent)):
            raise ValueError("training configuration must remain inside the owned Ticket directory")
    for config_path in config_paths:
        if not config_path or not Path(config_path).is_file():
            if prepared_config_paths:
                raise ValueError("prepared workload configuration disappeared before publication")
            continue
        try:
            relative = Path(config_path).relative_to(Path(contract.script_path).parent)
        except ValueError:
            relative = None  # reuse sources outside work_dir are validated separately
        if relative is not None:
            remote_path = remote_train_config_path if contract.stage == "train" else str(PurePosixPath(contract.remote_work_dir) / relative.as_posix())
            files[remote_path] = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    manifest = {"execution_id": execution_id, "ticket_id": ticket.id,
                "script": contract.remote_script_path, "sha256": script_hash, "files": files,
                "allocation_lock": str(PurePosixPath(info.instance.workdir) / ".zevo-workload.lock"),
                "visible_devices": info.instance.visible_devices}
    row = InfraInstance(id=execution_id, instance_id=execution_id, provider=info.provider,
        status="provisioning", run_id=run.id, ticket_id=ticket.id,
        gpu_name=info.gpu.gpu_name, gpu_count=contract.num_gpus, vram_gb=info.gpu.vram_gb,
        dph=0, ssh_host=info.ssh.host, ssh_port=info.ssh.port, ssh_user=info.ssh.user,
        meta={"direct_workload": True, "resource_request": True, "auto_release": False,
              "allocation_owner_row_id": owner.id if owner else "direct-host-lease",
              "stage": contract.stage, "nodes": 1, "gpus_per_node": contract.num_gpus,
              "scheduler_state": "PENDING", "execution_attempt": contract.attempt,
              "execution_contract": contract.model_dump(mode="json"), "direct_manifest": manifest,
              "submission_heartbeat_id": heartbeat_id, "submission_committed": True,
              "submission_committed_at": datetime.now(timezone.utc).isoformat(),
              "retry_of_bookkeeping_row_id": contract.retry_of_bookkeeping_row_id,
              "retry_of_job_id": contract.retry_of_job_id,
              "remote_workdir": contract.remote_work_dir, "remote_script": contract.remote_script_path,
              "status_path": contract.status_path, "stdout_path": contract.stdout_path,
              "log_path": contract.stdout_path,
              "stderr_path": contract.stderr_path, "script_sha256": script_hash,
              "implementation_sha256": implementation,
              "device_info_path": inp.device_info_path})
    if isinstance(inp, InferenceTaskInput):
        row.meta = {**row.meta, "inference_launch": snapshot(inp, Path(contract.script_path).read_text()),
                    "frozen_reuse_artifacts": inp.frozen_reuse_artifacts}
    if contract.stage == "train":
        config = Path(contract.script_path).with_name("train_config.yaml")
        row.meta = {**row.meta, "training_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "training_config_remote_path": remote_train_config_path}
        if previous and previous.meta.get("training_continuation"):
            row.meta = {**row.meta, "resumed_from": previous.meta["training_continuation"]}
    from zevo.paths import REPO_ROOT
    row.meta = {**row.meta, "supervisor_source": (REPO_ROOT / "playbook/runners/direct_workload.py").read_text()}
    session.add(row)
    from zevo.engine.run.workload_execution import register_execution
    await register_execution(session, execution_id=row.id, ticket_id=ticket.id,
        heartbeat_id=heartbeat_id, runtime_key=f"{info.provider}:{run.id}", log_path=contract.stdout_path)
    # Durable launch intent precedes SSH. Reconciliation safely republishes the
    # same identity after a crash/ambiguous response; it never creates a retry.
    await session.commit()
    try:
        await remote_call(info, manifest, "publish", helper_body=row.meta["supervisor_source"].encode())
    except (OSError, ValueError, RuntimeError) as exc:
        row.meta = {**row.meta, "monitor_error": str(exc)[:1000]}
    await session.commit()
    return row


async def read_telemetry(session, row, ticket, info, meta):
    """Read bounded chunks with durable offsets; replay uses shared deduplication."""
    from zevo.engine.observe.markers import scan_text
    from zevo.engine.run.scheduler.reconciler import _persist_slurm_execution_marker
    hb = await session.get(HeartbeatRun, meta["submission_heartbeat_id"])
    if not hb:
        return
    offset = int(meta.get("telemetry_offset", 0))
    program = ("import json;from pathlib import Path;"
               f"p=Path({meta['stdout_path']!r});"
               f"f=p.open('rb') if p.exists() else None;"
               f"f.seek({offset}) if f else None;"
               "data=f.read(131072) if f else b'';"
               "print(json.dumps({'bytes':len(data),'text':data.decode('utf-8','replace')}))")
    response = await _ssh(info, "python3 -c " + shlex.quote(program), timeout_seconds=40, stdout_limit=800000)
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "telemetry unavailable"))
    chunk = json.loads(response["stdout"].strip().splitlines()[-1])
    buffered = str(meta.pop("telemetry_partial", "")) + chunk["text"]
    end = buffered.rfind("\n") + 1
    if not chunk["bytes"]:
        end = len(buffered)
    meta["telemetry_partial"] = buffered[end:][-1048576:]
    for kind, payload in scan_text(buffered[:end]):
        if payload.get("owner") != ticket.id:
            continue
        await _persist_slurm_execution_marker(session, ticket_id=ticket.id, heartbeat=hb,
            kind=kind, payload=payload, phase=str(payload.get("phase") or ticket.agent_id),
            attempt_id=row.id, execution_id=row.id, runtime_key=f"{row.provider}:{row.run_id}",
            log_path=meta["stdout_path"])
    meta["telemetry_offset"] = offset + chunk["bytes"]
    meta["telemetry_drained"] = chunk["bytes"] == 0


async def live_activation(session, ticket):
    """Recover orphaned control activations without killing their workload."""
    from zevo.db import AgentWakeupRequest
    from zevo.engine.run.scheduler.reconciler import _last_sign_of_life
    active = (await session.scalars(select(HeartbeatRun).where(
        HeartbeatRun.ticket_id == ticket.id, HeartbeatRun.finished_at.is_(None)))).all()
    if not active:
        return False
    wake = (await session.scalars(select(AgentWakeupRequest.id).where(
        AgentWakeupRequest.ticket_id == ticket.id, AgentWakeupRequest.status == "running"))).first()
    now = datetime.now(timezone.utc)
    live = False
    for heartbeat in active:
        if wake or (now - await _last_sign_of_life(session, heartbeat)).total_seconds() < 120:
            live = True
        else:
            heartbeat.finished_at = now
            heartbeat.exit_code = 1
            heartbeat.error_message = "Control activation disappeared; registered workload remains monitored"
    return live


async def reconcile(session):
    from zevo.contracts.tickets import TERMINAL_RUN_STATUSES, TERMINAL_TICKET_STATUSES
    from zevo.engine.run.scheduler.reconciler import finish_observed_workload, _mark_slurm_job_running
    rows = (await session.execute(select(InfraInstance, Run, Ticket)
        .join(Run, Run.id == InfraInstance.run_id).join(Ticket, Ticket.id == InfraInstance.ticket_id)
        .where(InfraInstance.provider.in_(["cloud", "instance"]),
               Run.status.notin_(list(TERMINAL_RUN_STATUSES)),
               Ticket.status.notin_(list(TERMINAL_TICKET_STATUSES)))
        .order_by(InfraInstance.created_at.desc()))).all()
    seen = set()
    checked = resumed = 0
    for row, run, ticket in rows:
        if not direct(row) or ticket.id in seen or run.cancel_requested_at:
            continue
        seen.add(ticket.id)
        # Match admission's Run -> Ticket lock order, including checkpoint
        # handoff, which also locks the Run. Skip another controller's claim.
        run = (await session.scalars(select(Run).where(Run.id == run.id)
            .with_for_update(skip_locked=True).execution_options(populate_existing=True))).one_or_none()
        if run is None:
            continue
        # Publication/cleanup and instruction replacement share Ticket ownership.
        # Refresh after locking so an old poll cannot cancel a newer execution
        # or overwrite an instruction's supersession marker.
        ticket = (await session.scalars(select(Ticket).where(Ticket.id == ticket.id)
            .with_for_update().execution_options(populate_existing=True))).one()
        await session.refresh(row)
        from zevo.engine.run.runner import _latest_slurm_resource_request
        latest = await _latest_slurm_resource_request(session, ticket)
        if (latest is None or latest.id != row.id or run.cancel_requested_at
                or run.status in TERMINAL_RUN_STATUSES or ticket.status in TERMINAL_TICKET_STATUSES):
            await session.commit()
            continue
        meta = dict(row.meta or {})
        if meta.get("superseded_by_instruction_id") or meta.get("collect_wakeup_queued"):
            await session.commit()
            continue
        try:
            info = InfrastructureDeviceInfo.model_validate_json(Path(meta["device_info_path"]).read_text())
            manifest = meta["direct_manifest"]
            # Idempotent publication also closes a crash before first SSH.
            receipt = await remote_call(info, manifest, "publish",
                helper_body=meta["supervisor_source"].encode())
            receipt = await remote_call(info, manifest, "observe")
            checked += 1
            state = receipt.get("state", "UNKNOWN")
            if state in {"UNKNOWN", "LOST"}:
                meta["monitor_error"] = receipt.get("error", "waiting for workload supervisor receipt")
                if state == "LOST":
                    from zevo.engine.run.remote_jobs import _direct_process_kill_command
                    cleanup = await _ssh(info, _direct_process_kill_command(ticket.id), timeout_seconds=40)
                    if cleanup.get("ok"):
                        state = "FAILED"
                        receipt.update(exit_code=1, error="supervisor lost; ticket workers stopped before repair")
                    else:
                        continue
                else:
                    continue
            meta.update(scheduler_state=state, scheduler_exit_code=str(receipt.get("exit_code", "")),
                        scheduler_reason=receipt.get("error", "workload exited"), monitor_last_at=datetime.now(timezone.utc).isoformat())
            meta.pop("monitor_error", None)
            if receipt.get("started_at"):
                row.ready_at = datetime.fromtimestamp(receipt["started_at"], timezone.utc)
            await read_telemetry(session, row, ticket, info, meta)
            now = datetime.now(timezone.utc)
            active = await live_activation(session, ticket)
            if state == "RUNNING" and not active:
                _mark_slurm_job_running(session=session, row=row, ticket=ticket, meta=meta, observed_at=now)
            if state in {"COMPLETED", "FAILED", "CANCELLED"} and not active and meta.get("telemetry_drained"):
                row.released_at = now  # this execution ended, the allocation remains owned
                row.status = "released" if state == "COMPLETED" else "failed"
                resumed += await finish_observed_workload(session, row, run, ticket, meta, state, set())
        except Exception as exc:
            meta["monitor_error"] = str(exc)[:1000]
        finally:
            row.meta = meta
            await session.commit()
    await session.commit()
    return checked, resumed
