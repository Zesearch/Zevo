"""Resource lifetime decisions shared by provider-specific execution paths."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from zevo.db import InfraInstance, Run, Ticket

GPU_STAGES = {"train", "inference"}
TERMINAL = {"succeeded", "degraded", "failed", "cancelled", "skipped"}


def allocation_mode(run: Run) -> str:
    mode = getattr(run, "gpu_allocation_mode", None) or "per_stage"
    if mode not in {"per_stage", "per_run", "per_submission"}:
        raise ValueError(f"unsupported GPU allocation mode: {mode}")
    return mode


def needs_stage_release(run: Run, ticket: Ticket) -> bool:
    if allocation_mode(run) not in {"per_stage", "per_submission"} or ticket.status not in TERMINAL:
        return False
    if ticket.agent_id not in GPU_STAGES and not (
        getattr(run, "gpu_provider", "") == "cluster" and ticket.agent_id == "data"
    ):
        return False
    return not engine_chain_follows(run, ticket)


def engine_chain_follows(run: Run, ticket: Ticket) -> bool:
    """Whether the engine itself will spawn the next GPU consumer on this device.

    A successful optimization-lane Inference is always followed by Evaluation
    and then the engine-owned held-out measurement (prepare_holdout_data ->
    held-out Inference -> held-out Evaluation), bound to the same device_info.
    That chain has no Orchestrator wake and no Infrastructure ticket in it, so
    on a direct host (cloud/instance) releasing here strands the held-out
    Inference: every alarm-clock wake fails with "cloud instance has been
    released; provision a fresh device_info before the next stage" and the
    Run never ends (smoke run 83f87bfa, 2026-10-01). The held-out Inference's
    own terminal state is the release point; the Orchestrator provisions
    before any later stage it decides on, as the per_stage contract says.
    Cluster allocations are controller-owned and unaffected.
    """
    return (
        getattr(run, "gpu_provider", "") != "cluster"
        and ticket.agent_id == "inference"
        and getattr(ticket, "lane", "optimization") == "optimization"
        and ticket.status in {"succeeded", "degraded"}
    )


async def finish_stage(session, run: Run, ticket: Ticket) -> None:
    """Release only after terminal validation, never while repairing/collecting."""
    if not needs_stage_release(run, ticket):
        return
    from zevo.engine.run.resource_cleanup import release_cluster, cleanup_run_resources
    if run.gpu_provider == "cluster":
        rows = list((await session.execute(select(InfraInstance).where(
            InfraInstance.run_id == run.id, InfraInstance.ticket_id == ticket.id,
            InfraInstance.provider == "cluster", InfraInstance.released_at.is_(None),
        ))).scalars().all())
        for row in rows:
            if (row.meta or {}).get("allocation_owner_row_id"):
                continue
            row.meta = {**dict(row.meta or {}), "stage_release_pending": True}
            await session.commit()
            if await release_cluster(session, row, run):
                row.released_at = datetime.now(timezone.utc)
                row.status = "released"
                row.release_reason = "stage ended; allocation release confirmed"
            else:
                row.meta = {**dict(row.meta or {}), "stage_release_pending": True}
    else:
        # Direct-host stages are serialized by the pipeline. Fail closed if a
        # legacy/external caller has created another active consumer.
        active = (await session.execute(select(Ticket.id).where(
            Ticket.run_id == run.id, Ticket.id != ticket.id,
            Ticket.agent_id.in_(GPU_STAGES),
            Ticket.status.in_(["running", "repairing", "waiting_external"]),
        ).limit(1))).scalar_one_or_none()
        if active is not None:
            return
        rows = list((await session.execute(select(InfraInstance).where(
            InfraInstance.run_id == run.id, InfraInstance.released_at.is_(None),
        ))).scalars().all())
        for row in rows:
            row.meta = {**dict(row.meta or {}), "stage_release_pending": True,
                        "stage_consumer_ticket_id": ticket.id}
        await session.commit()
        await cleanup_run_resources(session, run, retry_now=True)
    await session.commit()


async def reconcile_stage_releases(session) -> int:
    """Retry a committed stage-release decision after a crash/provider failure."""
    rows = list((await session.execute(select(InfraInstance).where(
        InfraInstance.released_at.is_(None),
    ))).scalars().all())
    seen = set()
    count = 0
    for row in rows:
        meta = row.meta or {}
        if not meta.get("stage_release_pending"):
            continue
        ticket_id = meta.get("stage_consumer_ticket_id") or row.ticket_id
        if not ticket_id or ticket_id in seen:
            continue
        seen.add(ticket_id)
        run = await session.get(Run, row.run_id)
        ticket = await session.get(Ticket, ticket_id)
        if run is None or ticket is None or not needs_stage_release(run, ticket):
            continue
        try:
            await finish_stage(session, run, ticket)
            count += 1
        except (OSError, ValueError, RuntimeError):
            # The durable marker survives; provider confirmation is required.
            continue
    return count
