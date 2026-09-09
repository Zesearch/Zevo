"""Remote-compute tracking endpoints.

Infrastructure records cloud/instance resources; Train and Inference record
their finite cluster stage jobs. The submitting ticket owns the exact handle;
terminal lifecycle code provides automatic/forced cleanup.
The backend keeps a
single source of truth so:

  - leak detection (find instances active > N hours)
  - cumulative cost per run (sum dph * uptime)
  - preflight can warn when there are orphaned instances
  - UI / `hardware` can show what's currently running

Endpoints:
  POST   /api/infra/instances              create (agent on provision)
  PATCH  /api/infra/instances/{id}         update (agent on ready/release)
  GET    /api/infra/instances              list (active + filtered)
  GET    /api/infra/instances/{id}         detail
"""
from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from zevo.contracts._base import StrictBody
from zevo.contracts.infrastructure import (
    CreateInfraInstanceBody,
    InfraInstanceDTO,
    PatchInfraInstanceBody,
)
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.api.database import get_db
from zevo.db import InfraInstance


router = APIRouter()

# These fields are the deterministic runner/watcher ownership handshake. They
# are never Agent claims, even though provider-specific facts share `meta`.
_ENGINE_OWNED_SUBMISSION_META = {
    "log_path",
    "stderr_path",
    "submission_committed",
    "submission_heartbeat_id",
    "submission_committed_at",
}


def _reject_engine_owned_submission_meta(meta: dict) -> None:
    claimed = sorted(_ENGINE_OWNED_SUBMISSION_META.intersection(meta))
    if claimed:
        raise HTTPException(
            422,
            "engine-owned Slurm submission metadata cannot be supplied: "
            + ", ".join(claimed),
        )


# ─────────────────────────── DTOs ────────────────────────────────────────────


# ─────────────────────────── helpers ─────────────────────────────────────────


def _to_dto(r: InfraInstance) -> InfraInstanceDTO:
    now = datetime.now(timezone.utc)
    start = r.created_at if r.created_at else now
    end = r.released_at if r.released_at else now
    uptime = max(0, int((end - start).total_seconds()))
    # Cost = dph * uptime_hours (only meaningful for paid providers)
    cost = round((r.dph or 0.0) * (uptime / 3600.0), 6)
    return InfraInstanceDTO(
        id=r.id, instance_id=r.instance_id, provider=r.provider,
        status=r.status, run_id=r.run_id, ticket_id=r.ticket_id,
        gpu_name=r.gpu_name, gpu_count=r.gpu_count, vram_gb=r.vram_gb,
        dph=r.dph or 0.0, ssh_host=r.ssh_host, ssh_port=r.ssh_port,
        ssh_user=r.ssh_user, meta=r.meta or {},
        created_at=r.created_at.isoformat() if r.created_at else "",
        ready_at=r.ready_at.isoformat() if r.ready_at else None,
        released_at=r.released_at.isoformat() if r.released_at else None,
        release_reason=r.release_reason or "",
        uptime_seconds=uptime, estimated_cost_usd=cost,
    )


# ─────────────────────────── endpoints ───────────────────────────────────────


@router.post("/infra/instances", response_model=InfraInstanceDTO)
async def create_instance(
    body: CreateInfraInstanceBody, db: AsyncSession = Depends(get_db),
) -> InfraInstanceDTO:
    """Called right after a ticket requests an instance or submits/discovers a
    Slurm job. instance_id may
    be empty if provider is still spinning up — the agent calls PATCH
    later to fill it in."""
    _reject_engine_owned_submission_meta(body.meta)
    row = InfraInstance(
        instance_id=body.instance_id, provider=body.provider,
        status=body.status, run_id=body.run_id, ticket_id=body.ticket_id,
        gpu_name=body.gpu_name, gpu_count=body.gpu_count, vram_gb=body.vram_gb,
        dph=body.dph, ssh_host=body.ssh_host, ssh_port=body.ssh_port,
        ssh_user=body.ssh_user, meta=body.meta,
    )
    if body.status == "ready":
        row.ready_at = datetime.now(timezone.utc)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _to_dto(row)


@router.patch("/infra/instances/{row_id}", response_model=InfraInstanceDTO)
async def patch_instance(
    row_id: str, body: PatchInfraInstanceBody,
    db: AsyncSession = Depends(get_db),
) -> InfraInstanceDTO:
    """`row_id` is the bookkeeping row's own id (returned by POST), NOT the
    provider's `instance_id` — the DTO field of that name means the provider
    handle. Named apart so the two cannot be confused again."""
    r = (await db.execute(
        select(InfraInstance).where(InfraInstance.id == row_id)
    )).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"instance row {row_id!r} not found")

    for field in ("instance_id", "gpu_name", "gpu_count", "vram_gb", "dph",
                  "ssh_host", "ssh_port", "ssh_user", "meta", "release_reason"):
        v = getattr(body, field)
        if v is not None:
            if field == "meta":
                _reject_engine_owned_submission_meta(v)
                # An unrelated metadata update must not erase the engine's
                # already-committed submit/watcher ownership boundary.
                v = {
                    **v,
                    **{
                        key: (r.meta or {})[key]
                        for key in _ENGINE_OWNED_SUBMISSION_META
                        if key in (r.meta or {})
                    },
                }
            setattr(r, field, v)

    if body.status is not None:
        prev = r.status
        r.status = body.status
        now = datetime.now(timezone.utc)
        if body.status == "ready" and prev != "ready" and r.ready_at is None:
            r.ready_at = now
        if body.status == "released" and r.released_at is None:
            r.released_at = now

    await db.commit()
    await db.refresh(r)
    return _to_dto(r)


@router.get("/infra/instances", response_model=list[InfraInstanceDTO])
async def list_instances(
    active_only: bool = True,
    run_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[InfraInstanceDTO]:
    """List instances. Default = only those without a released_at
    (= currently costing money or holding a host)."""
    q = select(InfraInstance)
    if active_only:
        q = q.where(InfraInstance.released_at.is_(None))
    if run_id:
        q = q.where(InfraInstance.run_id == run_id)
    q = q.order_by(desc(InfraInstance.created_at))
    rows = (await db.execute(q)).scalars().all()
    return [_to_dto(r) for r in rows]


@router.get("/infra/instances/{row_id}", response_model=InfraInstanceDTO)
async def get_instance(
    row_id: str, db: AsyncSession = Depends(get_db),
) -> InfraInstanceDTO:
    """Fetch by the bookkeeping row id (see patch_instance's note)."""
    r = (await db.execute(
        select(InfraInstance).where(InfraInstance.id == row_id)
    )).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"instance row {row_id!r} not found")
    return _to_dto(r)


# ───────────────────────── SSH test tool (B.2) ───────────────────────────────


class SSHTestBody(StrictBody):
    host: str
    port: int = 22
    user: str = "root"
    # Either a path to a private key file (read by the agent's container)
    # OR an empty string meaning "use the user's mounted ~/.ssh/id_*"
    key_path: str = ""
    # Optional alternative: path to a mode-0600 one-line password file.
    password_path: str = ""
    # Soft cap so a hung remote can't block the API forever.
    timeout_seconds: int = 20


class SSHTestProbe(BaseModel):
    name: str
    ok: bool
    detail: str = ""    # the trimmed first-200-chars of stdout for the probe


class SSHTestResponse(BaseModel):
    reachable: bool                       # SSH handshake succeeded
    summary: str                          # one-line verdict
    probes: list[SSHTestProbe]
    elapsed_seconds: float


@router.post("/infra/ssh-test", response_model=SSHTestResponse)
async def ssh_test(body: SSHTestBody) -> SSHTestResponse:
    """Probe an SSH host: reachability + nvidia-smi + python + free disk.

    The infrastructure agent calls this before declaring a route ready. We
    never store credentials here; exactly one key/password path must point at
    a file already mounted into the backend container.

    All probes are joined into a single SSH session so the latency is
    one round-trip, not five. If the session can't be opened we return
    reachable=false and the list of probes is empty.
    """
    import asyncio
    import shlex
    import time

    start = time.monotonic()
    probes_script = " && echo ___PROBE_BREAK___ && ".join([
        # ssh-side: each probe is a short command whose output we
        # snip on ___PROBE_BREAK___ to attribute.
        "echo nvidia-smi:; (command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo 'nvidia-smi NOT FOUND')",
        "echo python:; (command -v python3 >/dev/null && python3 --version || echo 'python3 NOT FOUND')",
        "echo cuda:; (command -v nvcc >/dev/null && nvcc --version | tail -1 || echo 'nvcc NOT FOUND')",
        "echo disk:; df -BG --output=avail / | tail -1",
        "echo home:; pwd",
    ])
    from zevo.engine.ssh_auth import ssh_base_args, validate_ssh_target

    try:
        validate_ssh_target(host=body.host, user=body.user, port=body.port)
    except ValueError as exc:
        return SSHTestResponse(
            reachable=False,
            summary=str(exc),
            probes=[], elapsed_seconds=time.monotonic() - start,
        )

    try:
        ssh_args = ssh_base_args(
            key_path=body.key_path,
            password_path=body.password_path,
            port=body.port,
            strict_host_key_checking="accept-new",
            connect_timeout=max(3, min(body.timeout_seconds, 60)),
        )
    except ValueError as exc:
        return SSHTestResponse(
            reachable=False,
            summary=str(exc),
            probes=[], elapsed_seconds=time.monotonic() - start,
        )
    # `--` marks end-of-options so a host/user beginning with `-` can never be
    # parsed as an ssh flag (validate_ssh_target above already rejects those,
    # but the marker is defence-in-depth for the argv layout).
    ssh_args.append("--")
    ssh_args.append(f"{body.user}@{body.host}")
    ssh_args.append(probes_script)

    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *ssh_args,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            ),
            timeout=body.timeout_seconds,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=body.timeout_seconds,
        )
    except asyncio.TimeoutError:
        return SSHTestResponse(
            reachable=False,
            summary=f"timed out after {body.timeout_seconds}s connecting to {body.host}:{body.port}",
            probes=[], elapsed_seconds=time.monotonic() - start,
        )
    except FileNotFoundError:
        return SSHTestResponse(
            reachable=False,
            summary="ssh binary not found in the backend container",
            probes=[], elapsed_seconds=time.monotonic() - start,
        )

    elapsed = time.monotonic() - start
    if proc.returncode != 0:
        return SSHTestResponse(
            reachable=False,
            summary=(stderr.decode("utf-8", errors="replace")[:300] or "ssh failed").strip(),
            probes=[], elapsed_seconds=elapsed,
        )

    text = stdout.decode("utf-8", errors="replace")
    chunks = text.split("___PROBE_BREAK___")
    probes: list[SSHTestProbe] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        # First line is the label (e.g. "nvidia-smi:"), rest is the answer.
        head, _, body_text = c.partition("\n")
        name = head.rstrip(":").strip() or "?"
        body_text = body_text.strip()[:200]
        ok = "NOT FOUND" not in body_text
        probes.append(SSHTestProbe(name=name, ok=ok, detail=body_text))

    n_ok = sum(1 for p in probes if p.ok)
    return SSHTestResponse(
        reachable=True,
        summary=f"{n_ok}/{len(probes)} probes ok on {body.user}@{body.host}:{body.port}",
        probes=probes, elapsed_seconds=elapsed,
    )
