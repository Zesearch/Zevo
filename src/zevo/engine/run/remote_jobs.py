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
            "stdout": stdout.decode("utf-8", "replace")[:1000],
            "error": stderr.decode("utf-8", "replace")[:1000],
        }
    except Exception as exc:  # cleanup must not mask cancellation itself
        return {"ok": False, "exit_code": -1, "stdout": "", "error": str(exc)[:1000]}


def _direct_process_kill_command(ticket_id: str) -> str:
    """Remote Python command matching an exact inherited environment value."""
    marker = f"ZEVO_TICKET_ID={_safe_ticket_id(ticket_id)}".encode().hex()
    script = f"""import os, signal, time
marker = bytes.fromhex('{marker}')
mine = os.getpid()
pids = []
for entry in os.listdir('/proc'):
    if not entry.isdigit() or int(entry) == mine:
        continue
    try:
        values = open('/proc/' + entry + '/environ', 'rb').read().split(b'\\0')
        if marker in values:
            pids.append(int(entry))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
for pid in pids:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
time.sleep(2)
for pid in pids:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
print('terminated', len(pids), 'ticket processes')
"""
    return "python3 -c " + shlex.quote(script)


def _slurm_cli_bootstrap_command() -> str:
    """Expose Slurm in non-interactive shells and fail closed if unavailable."""
    return (
        "if ! command -v squeue >/dev/null 2>&1 "
        "|| ! command -v scancel >/dev/null 2>&1; then "
        "if [ -r /etc/profile.d/modules.sh ]; then "
        ". /etc/profile.d/modules.sh >/dev/null 2>&1; "
        "if command -v module >/dev/null 2>&1; then "
        "module load default-environment >/dev/null 2>&1 || true; fi; "
        "fi; fi; "
        "command -v squeue >/dev/null 2>&1 "
        "|| { echo slurm-squeue-unavailable >&2; exit 127; }; "
        "command -v scancel >/dev/null 2>&1 "
        "|| { echo slurm-scancel-unavailable >&2; exit 127; }; "
    )


def _slurm_job_kill_command(jobid_value: str, ticket_id: str) -> str:
    """Cancel one exact stage-owned job after verifying its Ticket name."""
    jobid = _safe_ticket_id(jobid_value)
    name = f"zevo-{_safe_ticket_id(ticket_id)}"
    return (
        _slurm_cli_bootstrap_command()
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
    db: AsyncSession, ticket: Ticket,
) -> dict[str, Any]:
    """Stop remote work for one Ticket, never an unrelated device or job."""
    if ticket.agent_id not in {"train", "inference"}:
        return {"ticket_id": ticket.id, "attempted": False, "reason": "no remote execution"}
    info = await _device_info_for_ticket(db, ticket)
    if info is None:
        return {"ticket_id": ticket.id, "attempted": False, "reason": "no resolved device route"}
    cluster_row = await _open_cluster_job(db, ticket) if info.provider == "cluster" else None
    if info.provider == "cluster":
        if cluster_row is None:
            return {
                "ticket_id": ticket.id,
                "attempted": False,
                "reason": "no live stage-owned Slurm job",
            }
        command = _slurm_job_kill_command(cluster_row.instance_id, ticket.id)
    else:
        command = _direct_process_kill_command(ticket.id)
    result = await _ssh(info, command)
    if cluster_row is not None and result.get("ok"):
        now = datetime.now(timezone.utc)
        cluster_row.status = "released"
        cluster_row.released_at = now
        cluster_row.release_reason = "ticket cancelled"
        await db.flush()
    return {
        "ticket_id": ticket.id,
        "attempted": True,
        "provider": info.provider,
        **result,
    }


async def cancel_run_remote_jobs(
    db: AsyncSession, tickets: list[Ticket],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for ticket in tickets:
        result = await cancel_ticket_remote_job(db, ticket)
        if result.get("attempted"):
            results.append(result)
    return results
