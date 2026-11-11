"""Best-effort cancellation of Ticket-owned work on a remote GPU host.

Killing the local Agent or SSH client is insufficient: a remote Python process
may survive the disconnected session.  Every real Train command receives
``ZEVO_TICKET_ID`` and every Slurm job is required to use a Ticket-specific job
name. Cluster mode cancels its finite stage-owned job; cloud and instance modes
terminate only processes carrying the exact Ticket marker.
"""
from __future__ import annotations

import asyncio
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.contracts.infrastructure import InfrastructureDeviceInfo
from zevo.db import InfraInstance, Ticket, WorkProduct
from zevo.engine.remote_transfer import ssh_base_args


def _safe_ticket_id(value: str) -> str:
    if not value or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in value):
        raise ValueError(f"unsafe Ticket id for remote cancellation: {value!r}")
    return value


async def _device_info_for_ticket(
    db: AsyncSession, ticket: Ticket,
) -> InfrastructureDeviceInfo | None:
    binding = dict((ticket.inputs or {}).get("device_info") or {})
    path = str(binding.get("path") or "")
    if not path:
        work_product_id = str(binding.get("work_product_id") or "")
        source_ticket_id = str(binding.get("source_ticket_id") or "")
        row = None
        if work_product_id:
            row = await db.get(WorkProduct, work_product_id)
        elif source_ticket_id:
            row = (await db.execute(select(WorkProduct).where(
                WorkProduct.ticket_id == source_ticket_id,
                WorkProduct.role == "device_info",
            ).order_by(WorkProduct.created_at.desc()).limit(1))).scalar_one_or_none()
        path = str(row.path or "") if row is not None else ""
    if not path:
        return None
    try:
        return InfrastructureDeviceInfo.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None


async def _ssh(
    info: InfrastructureDeviceInfo,
    remote_command: str,
    *,
    timeout_seconds: float = 20,
    stdout_limit: int = 1000,
) -> dict[str, Any]:
    command = [
        *ssh_base_args(info.ssh),
        f"{info.ssh.user}@{info.ssh.host}",
        remote_command,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds,
        )
        return {
            "ok": proc.returncode == 0,
            "exit_code": int(proc.returncode or 0),
            "stdout": stdout.decode("utf-8", "replace")[:stdout_limit],
            "stdout_truncated": len(stdout.decode("utf-8", "replace")) > stdout_limit,
            "error": stderr.decode("utf-8", "replace")[-4000:],
        }
    except Exception as exc:  # cleanup must not mask cancellation itself
        return {"ok": False, "exit_code": -1, "stdout": "", "error": str(exc)[:1000]}


def _direct_process_kill_command(ticket_id: str) -> str:
    """Remote Python command matching an exact inherited environment value."""
    marker = f"ZEVO_TICKET_ID={_safe_ticket_id(ticket_id)}".encode().hex()
    script = f"""import os, signal, time
marker = bytes.fromhex('{marker}')
mine = os.getpid()
def owned(pid):
    try:
        root = '/proc/' + str(pid)
        values = open(root + '/environ', 'rb').read().split(b'\\0')
        fields = open(root + '/stat').read().rsplit(')', 1)[1].split()
        return fields[19] if marker in values and fields[0] != 'Z' else None
    except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError):
        return None
started = time.monotonic()
stopped = set()
while True:
    workers = [(int(p), owned(int(p))) for p in os.listdir('/proc') if p.isdigit() and int(p) != mine]
    workers = [(pid, start) for pid, start in workers if start is not None]
    if not workers:
        break
    elapsed = time.monotonic() - started
    if elapsed > 15:
        raise RuntimeError('owned ticket processes are still alive; cleanup not confirmed')
    for pid, start in workers:
        if owned(pid) == start:
            try:
                os.kill(pid, signal.SIGTERM if elapsed < 2 else signal.SIGKILL)
                stopped.add((pid, start))
            except ProcessLookupError:
                pass
    time.sleep(.1)
print('terminated', len(stopped), 'ticket processes')
"""
    return "python3 -c " + shlex.quote(script)


def _slurm_login_command(command: str) -> str:
    """Load the site's login environment before explicit setup and Slurm commands.

    Quote the entire body so the SSH account's outer shell cannot expand paths,
    variables or command substitutions before the login environment is loaded.
    """
    return "bash -lc " + shlex.quote(command)


def _slurm_cli_bootstrap_command(env_setup: str = "", *, required: tuple[str, ...] = ("squeue", "scancel")) -> str:
    """Expose Slurm in non-interactive shells and fail closed if unavailable."""
    if env_setup.strip():
        # Run trusted connection setup in the same shell as Slurm. Silence its
        # stdout so job ids and scheduler records remain machine-readable.
        bootstrap = "{\n" + env_setup.strip() + "\n} >/dev/null || { echo slurm-env-setup-failed >&2; exit 126; }; "
    else:
        bootstrap = (
            "if ! command -v squeue >/dev/null 2>&1 "
            "|| ! command -v scancel >/dev/null 2>&1; then "
            "if [ -r /etc/profile.d/modules.sh ]; then "
            ". /etc/profile.d/modules.sh >/dev/null 2>&1; "
            "if command -v module >/dev/null 2>&1; then "
            "module load default-environment >/dev/null 2>&1 || true; fi; "
            "fi; fi; "
        )
    for command in required:
        if command not in {"sbatch", "squeue", "sacct", "scancel", "scontrol"}:
            raise ValueError("unsupported Slurm command")
        bootstrap += (
            f"command -v {command} >/dev/null 2>&1 "
            f"|| {{ echo slurm-{command}-unavailable >&2; exit 127; }}; "
        )
    return bootstrap


def _slurm_job_kill_command(jobid_value: str, ticket_id: str, env_setup: str = "") -> str:
    """Cancel one exact stage-owned job after verifying its Ticket name."""
    jobid = _safe_ticket_id(jobid_value)
    name = f"zevo-{_safe_ticket_id(ticket_id)}"
    return _slurm_login_command(
        _slurm_cli_bootstrap_command(env_setup)
        + f"if ! record=$(squeue -h -j {shlex.quote(jobid)} -o '%i|%j'); then "
        "echo stage-job-query-failed >&2; exit 1; fi; "
        "if [ -z \"$record\" ]; then echo stage-job-already-terminal; exit 0; fi; "
        f"if ! printf '%s\\n' \"$record\" | awk -F'|' '$1 == \"{jobid}\" "
        f"&& $2 == \"{name}\" {{found=1}} END {{exit !found}}'; then "
        "echo stage-job-identity-mismatch >&2; exit 1; fi; "
        f"scancel {shlex.quote(jobid)}"
    )


async def _open_cluster_job(
    db: AsyncSession, ticket: Ticket,
) -> InfraInstance | None:
    return (await db.execute(select(InfraInstance).where(
        InfraInstance.ticket_id == ticket.id,
        InfraInstance.provider == "cluster",
        InfraInstance.instance_id != "",
        InfraInstance.released_at.is_(None),
    ).order_by(InfraInstance.created_at.desc()).limit(1))).scalar_one_or_none()


async def cancel_ticket_remote_job(
    db: AsyncSession, ticket: Ticket, *, preserve_allocation: bool = False,
    resize_allocation: bool = False,
) -> dict[str, Any]:
    """Stop remote work for one Ticket, never an unrelated device or job."""
    if ticket.agent_id not in {"data", "train", "inference", "evaluation", "registry"}:
        return {"ticket_id": ticket.id, "attempted": False, "reason": "no remote execution"}
    info = await _device_info_for_ticket(db, ticket)
    if info is None:
        return {"ticket_id": ticket.id, "attempted": False, "reason": "no resolved device route"}
    if info.provider == "cloud" and db is not None:
        owners = (await db.scalars(select(InfraInstance).where(
            InfraInstance.run_id == ticket.run_id, InfraInstance.provider == "cloud",
            InfraInstance.instance_id == info.instance_id))).all()
        if any(row.released_at is not None and (row.meta or {}).get("cloud_health", {}).get("state")
               in {"terminated", "deleted", "destroyed", "not-found"} for row in owners):
            return {"ticket_id": ticket.id, "attempted": False, "ok": True,
                    "reason": "cloud provider already confirmed this allocation lost"}
    direct_rows = []
    cluster_row = await _open_cluster_job(db, ticket) if info.provider == "cluster" else None
    if info.provider == "cluster" and resize_allocation:
        cluster_row = await allocation_for_resize(db, ticket) or cluster_row
    if ticket.agent_id == "evaluation":
        route = info.cluster or info.instance
        credentials = str(Path(route.workdir) / _safe_ticket_id(ticket.id) / "evaluation" / ".judge-env.json")
        # Queued workers cannot run their EXIT trap to remove private credentials.
        await _ssh(info, "rm -f -- " + shlex.quote(credentials))
    if info.provider == "cluster":
        if cluster_row is None:
            return {
                "ticket_id": ticket.id,
                "attempted": False,
                "reason": "no live stage-owned Slurm job",
            }
        meta = dict(cluster_row.meta or {})
        if preserve_allocation and meta.get("controller_directory"):
            from zevo.engine.run.gpu_controller import upload_text_command
            request_id = str(meta["controller_request_id"])
            directory = str(meta["controller_directory"])
            command = upload_text_command(f"{directory}/cancel/{request_id}", "cancel")
            # The controller publishes outcome only after its child group stops.
            outcome = shlex.quote(str(meta["controller_outcome_path"]))
            command += f" && for i in $(seq 1 10); do test -f {outcome} && exit 0; sleep 1; done; exit 1"
            return {"ticket_id": ticket.id, "attempted": True, "provider": "cluster",
                    **await _ssh(info, command, timeout_seconds=20)}
        command = _slurm_job_kill_command(
            cluster_row.instance_id, str(meta.get("allocation_owner_ticket_id") or ticket.id),
            env_setup=info.cluster.env_setup if info.cluster else "",
        )
    else:
        from zevo.engine.run.direct_jobs import direct, remote_call
        rows = (await db.scalars(select(InfraInstance).where(
            InfraInstance.ticket_id == ticket.id, InfraInstance.run_id == ticket.run_id,
            InfraInstance.released_at.is_(None)))).all() if db is not None else []
        direct_rows = [row for row in rows if direct(row)]
        for row in direct_rows:
            try:
                await remote_call(info, row.meta["direct_manifest"], "cancel",
                    helper_body=row.meta["supervisor_source"].encode())
            except (OSError, ValueError, RuntimeError) as exc:
                return {"ticket_id": ticket.id, "attempted": True, "ok": False, "error": str(exc)}
        command = _direct_process_kill_command(ticket.id)
    result = await _ssh(info, command)
    if cluster_row is not None and result.get("ok"):
        now = datetime.now(timezone.utc)
        # scancel accepted the request; the resource remains billable/owned
        # until the reconciler confirms the scheduler no longer holds it.
        cluster_row.meta = {**dict(cluster_row.meta or {}),
                            "cancellation_requested_at": now.isoformat()}
        cluster_row.release_reason = "cancellation requested; awaiting scheduler confirmation"
        await db.flush()
    if direct_rows and result.get("ok"):
        # Direct cleanup confirms the process tree is gone. Preserve that
        # terminal evidence even when an instruction superseded the watcher;
        # otherwise its replacement would remain bound to a RUNNING attempt.
        now = datetime.now(timezone.utc)
        for row in direct_rows:
            row.meta = {**dict(row.meta or {}), "scheduler_state": "CANCELLED",
                        "scheduler_exit_code": "143", "scheduler_reason": "owned workers stopped",
                        "cancellation_requested_at": now.isoformat(), "monitor_terminal": True}
            row.status = "released"
            row.released_at = now
            row.release_reason = "execution cancelled; allocation retained"
        await db.flush()
    return {
        "ticket_id": ticket.id,
        "attempted": True,
        "provider": info.provider,
        **result,
    }


async def allocation_for_resize(db: AsyncSession, ticket: Ticket) -> InfraInstance | None:
    """Resolve ownership even when the latest workload alias is already finished."""
    rows = (await db.execute(select(InfraInstance).where(
        InfraInstance.ticket_id == ticket.id, InfraInstance.provider == "cluster",
        InfraInstance.instance_id != "",
    ).order_by(InfraInstance.created_at.desc()))).scalars().all()
    latest = next((row for row in rows if (row.meta or {}).get("resource_request")
                   or (row.meta or {}).get("stage_job")), None)
    if latest is None:
        return None
    owner_id = (latest.meta or {}).get("allocation_owner_row_id")
    owner = await db.get(InfraInstance, owner_id) if owner_id else latest
    if owner is None or owner.run_id != ticket.run_id or owner.released_at is not None:
        return None
    siblings = (await db.execute(select(InfraInstance).where(
        InfraInstance.run_id == ticket.run_id, InfraInstance.instance_id == owner.instance_id,
        InfraInstance.provider == "cluster", InfraInstance.released_at.is_(None),
        InfraInstance.ticket_id != ticket.id,
    ))).scalars().all()
    if any(row.id != owner.id and not (row.meta or {}).get("monitor_terminal") for row in siblings):
        raise ValueError("cannot resize an allocation used by another stage")
    return owner


async def cancel_run_remote_jobs(
    db: AsyncSession, tickets: list[Ticket],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for ticket in tickets:
        result = await cancel_ticket_remote_job(db, ticket)
        if result.get("attempted"):
            results.append(result)
    return results
