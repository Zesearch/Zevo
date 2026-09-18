"""Owned resource cleanup with durable, retryable provider confirmation.

Provider handles come from this Run's bookkeeping or its registered device
artifacts. Remote work directories are deliberately retained: an agent-authored
path is not evidence that recursively deleting it is safe (including symlinks).
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select, update

from zevo.db import GpuLease, InfraInstance, Run, SshHost, Ticket, WorkProduct
from zevo.engine.ssh_auth import ssh_base_args
from zevo.providers import resolve_ssh_key


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


async def discover_resources(session, run: Run) -> list[InfraInstance]:
    rows = list((await session.execute(select(InfraInstance).where(
        InfraInstance.run_id == run.id,
    ))).scalars().all())
    identities = {(row.provider, row.instance_id) for row in rows}
    products = (await session.execute(select(WorkProduct).join(
        Ticket, Ticket.id == WorkProduct.ticket_id,
    ).where(Ticket.run_id == run.id, WorkProduct.role == "device_info"))).scalars().all()
    for product in products:
        info = dict(product.meta or {})
        try:
            document = json.loads(Path(product.path).read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                continue
            if document.get("run_id") not in (None, run.id):
                continue
            if document.get("ticket_id") not in (None, product.ticket_id):
                continue
            info = {**document, **info}
        except (OSError, ValueError):
            pass
        provider, handle = str(info.get("provider") or ""), str(info.get("instance_id") or "")
        if provider not in {"cloud", "cluster", "instance"} or not handle:
            continue
        if (provider, handle) in identities:
            continue
        # Persist discoveries BEFORE requesting destruction. A crash or failed
        # request must never leave an untracked, still-billing resource.
        row = InfraInstance(
            run_id=run.id, ticket_id=product.ticket_id, provider=provider,
            instance_id=handle, meta={
                "backend": info.get("cloud_backend") or run.cloud_backend or "",
                "auto_release": info.get("auto_release", True),
                "discovered_from_work_product": product.id,
            },
        )
        session.add(row)
        rows.append(row)
        identities.add((provider, handle))
    await session.commit()
    return rows


def _cloud_provider(backend: str):
    if backend == "vastai":
        from zevo.providers.vastai import VastAIProvider
        return VastAIProvider()
    if backend == "lambda":
        from zevo.providers.lambda_labs import LambdaCloudProvider
        return LambdaCloudProvider()
    raise ValueError("cloud backend is unknown; refusing cross-provider destruction")


async def release_cloud(row: InfraInstance, run: Run) -> bool:
    backend = str((row.meta or {}).get("backend") or (row.meta or {}).get("cloud_backend") or run.cloud_backend or "")
    provider = _cloud_provider(backend)

    async def gone() -> bool:
        instances = await provider.list_instances()
        matching = [item for item in instances if str(item.get("id", item.get("instance_id", ""))) == row.instance_id]
        # Lambda keeps a destroyed instance listed as "terminating" for a few
        # minutes; billing has already stopped, and re-sending terminate every
        # retry until it disappears only adds noise.
        return not matching or all(str(item.get("status", item.get("actual_status", ""))).lower() in {
            "terminated", "terminating", "deleted", "destroyed",
        } for item in matching)

    if await gone():
        return True
    await provider.destroy_instance(row.instance_id)
    return await gone()


async def release_cluster(session, row: InfraInstance, run: Run) -> bool:
    from zevo.engine.run.remote_jobs import _slurm_job_kill_command, _safe_ticket_id
    if not row.ticket_id:
        raise ValueError("Slurm cleanup requires the owning Ticket identity")
    ticket = await session.get(Ticket, row.ticket_id)
    if ticket is None or ticket.run_id != run.id:
        raise ValueError("Slurm cleanup Ticket does not belong to the Run")
    job_id = _safe_ticket_id(row.instance_id)
    command = _slurm_job_kill_command(job_id, row.ticket_id)
    # scancel accepting a request is not confirmation that the job released.
    command += (
        f" && remaining=$(squeue -h -j {job_id} -o '%i')"
        " && [ -z \"$remaining\" ]"
    )
    connection = await session.get(SshHost, run.ssh_host_id) if run.ssh_host_id else None
    host = connection.host if connection else os.environ.get("ZEVO_CLUSTER_SSH_HOST", "")
    if not host:
        raise ValueError("cluster SSH connection is unavailable")
    password = connection.password_path if connection else os.environ.get("ZEVO_CLUSTER_SSH_PASSWORD_FILE", "")
    key = connection.key_path if connection else resolve_ssh_key("ZEVO_CLUSTER_SSH_KEY")
    port = connection.port if connection else int(os.environ.get("ZEVO_CLUSTER_SSH_PORT", "22"))
    user = connection.username if connection else os.environ.get("ZEVO_CLUSTER_SSH_USER", "root")
    proc = await asyncio.create_subprocess_exec(
        *ssh_base_args(key_path="" if password else key, password_path=password, port=port),
        f"{user}@{host}", command,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        await asyncio.wait_for(proc.communicate(), timeout=30)
        return proc.returncode == 0
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


async def cleanup_run_resources(session, run: Run, *, force: bool = False, retry_now: bool = False) -> list[dict[str, Any]]:
    """Only confirmed releases stop accounting; failed rows retry with backoff."""
    rows = await discover_resources(session, run)
    results = []
    now = datetime.now(timezone.utc)
    for row in rows:
        if row.released_at is not None:
            continue
        meta = dict(row.meta or {})
        if not force and meta.get("auto_release") is False:
            continue
        next_at = meta.get("cleanup_next_at")
        if next_at and not retry_now:
            try:
                if _aware(datetime.fromisoformat(next_at)) > now:
                    continue
            except ValueError:
                pass
        confirmed, error = False, ""
        try:
            if row.provider == "instance":
                # The fixed host is operator-owned. Exact Ticket processes
                # are checked below before its leases are returned.
                continue
            if not row.instance_id:
                raise ValueError("provider allocation has no handle yet; cleanup pending")
            if row.provider == "cloud":
                confirmed = await asyncio.wait_for(release_cloud(row, run), timeout=60)
            elif row.provider == "cluster":
                confirmed = await release_cluster(session, row, run)
            else:
                raise ValueError("unknown provider; resource retained")
            if not confirmed:
                error = "provider release is not yet confirmed"
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        attempts = int(meta.get("cleanup_attempts") or 0) + 1
        meta.update(cleanup_attempts=attempts, cleanup_error=error[:1000], cleanup_last_at=now.isoformat())
        if confirmed:
            row.status = "released"
            row.released_at = now
            row.release_reason = "provider release confirmed"
            meta.pop("cleanup_next_at", None)
        else:
            meta["cleanup_next_at"] = (now + timedelta(seconds=min(300, 5 * 2 ** min(attempts, 6)))).isoformat()
        row.meta = meta
        results.append({"provider": row.provider, "instance_id": row.instance_id, "destroyed": confirmed, "error": error})
    live_lease = (await session.execute(select(GpuLease.id).where(
        GpuLease.run_id == run.id, GpuLease.released_at.is_(None),
    ).limit(1))).scalar_one_or_none()
    if run.gpu_provider == "instance" or live_lease is not None:
        from zevo.engine.run.remote_jobs import cancel_run_remote_jobs
        tickets = list((await session.execute(select(Ticket).where(Ticket.run_id == run.id))).scalars().all())
        stopped = await cancel_run_remote_jobs(session, tickets)
        if all(item.get("ok") for item in stopped):
            await session.execute(update(GpuLease).where(
                GpuLease.run_id == run.id, GpuLease.released_at.is_(None),
            ).values(released_at=now, release_reason="Ticket processes stopped"))
            for row in rows:
                if row.provider == "instance" and row.released_at is None:
                    row.status, row.released_at = "released", now
        else:
            results.append({"provider": "instance", "instance_id": "", "destroyed": False, "error": "Ticket process cleanup failed; GPU leases retained"})
    if run.cancel_outcome:
        run.cancel_outcome = {**dict(run.cancel_outcome), "compute_may_accrue": any(
            row.provider == "cloud" and row.released_at is None for row in rows
        )}
    await session.commit()
    return results


async def has_pending_resources(session, run_id: str) -> bool:
    instance = (await session.execute(select(InfraInstance.id).where(
        InfraInstance.run_id == run_id, InfraInstance.released_at.is_(None),
    ).limit(1))).scalar_one_or_none()
    lease = (await session.execute(select(GpuLease.id).where(
        GpuLease.run_id == run_id, GpuLease.released_at.is_(None),
    ).limit(1))).scalar_one_or_none()
    return instance is not None or lease is not None
