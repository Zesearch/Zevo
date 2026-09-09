"""Who gets which GPU on a shared fixed `instance` host.

The infrastructure agent holds the SSH credentials, probes configured fixed
hosts, and reports them here; this module holds
the state, so IT decides which cards the asking run gets. Neither side needs
what the other has: the backend never SSHes anywhere, and the agent never has
to know what other runs are doing.

Two rules shape the allocator, both from how the hardware actually works:

  * a slice never spans hosts. `CUDA_VISIBLE_DEVICES` addresses cards on one
    machine. Asking for 3 cards when two hosts have 2 free each is a REFUSAL,
    not a 2+1 split.
  * a slice is whole cards. VRAM and PCIe bandwidth are contended too, so
    fractional sharing would just move the collision somewhere less visible.

Refusing is a real outcome here, not an error to be smoothed over. There is no
scheduler queue: the host is already online, and a run that cannot be placed
fails saying exactly what is held and by whom, which is information they can
act on (free a host, add another host, or wait). Silently overlapping — what this
replaces — produced an OOM three stages later with no trace of the cause.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.api.database import get_db
from zevo.db import GpuLease, Run
from zevo.contracts.infrastructure import (
    GpuAllocationCandidate,
    GpuLeaseGrant,
    GpuLeaseRecord,
    GpuLeaseRequest,
)
from zevo.contracts.tickets import TERMINAL_RUN_STATUSES

router = APIRouter()

# A grant reads free cards and then writes them. Between those two steps a
# concurrent grant can take one, and the partial unique index turns that into
# an IntegrityError instead of a double-booked card. Losing the race is normal
# under load, so retry the whole selection — the second pass sees the winner's
# rows and picks elsewhere (or refuses).
_MAX_GRANT_ATTEMPTS = 6

# The index whose violation means "lost the race". Both Postgres and SQLite
# name it in the error text; nothing else we write can trip it.
_UNIQUE_LIVE_CARD = "ux_gpu_lease_live_card"
_SQLITE_LIVE_CARD_CONFLICT = (
    "UNIQUE constraint failed: gpu_leases.jobid, gpu_leases.gpu_index"
)


def _is_live_card_conflict(exc: IntegrityError) -> bool:
    """Whether the write lost the one expected allocation race.

    Postgres reports the partial index name. SQLite reports only its exact
    constrained columns, so both representations identify the same conflict
    without treating unrelated integrity errors as retryable.
    """
    detail = str(getattr(exc, "orig", exc))
    return _UNIQUE_LIVE_CARD in detail or _SQLITE_LIVE_CARD_CONFLICT in detail


def _grant(rows: list[GpuLease], *, reused: bool) -> GpuLeaseGrant:
    idx = sorted(r.gpu_index for r in rows)
    first = rows[0]
    return GpuLeaseGrant(
        jobid=first.jobid,
        node=first.node,
        gpu_indices=idx,
        gpu_count=len(idx),
        visible_devices=",".join(str(i) for i in idx),
        reused=reused,
    )


async def _reap(db: AsyncSession) -> None:
    """Release leases whose owning run is definitely terminal.

    The allocations supplied to one acquire request are placement candidates,
    not a global `squeue` snapshot. An omitted job may belong to another active
    run or may simply have missed a transient probe, so omission is never proof
    that its lease is dead. Allocation teardown uses the explicit release path;
    this opportunistic sweep handles only terminal runs recorded by Zevo.
    """
    dead_runs = select(Run.id).where(Run.status.in_(TERMINAL_RUN_STATUSES))
    await db.execute(
        update(GpuLease)
        .where(GpuLease.released_at.is_(None), GpuLease.run_id.in_(dead_runs))
        .values(
            released_at=datetime.now(timezone.utc),
            release_reason="run ended",
        )
    )
    await db.commit()


def _choose(
    allocs: list[GpuAllocationCandidate], taken: dict[str, set[int]], want: int
) -> tuple[GpuAllocationCandidate, list[int]] | None:
    """Pick one allocation and the cards to take from it.

    `want == 0` means "all of one allocation's free cards", so take the
    roomiest. Otherwise best-fit: the allocation with the FEWEST free cards
    that still holds the request. Filling a nearly-full box first keeps the
    emptier one whole, which is what lets a later 4-GPU request land at all;
    first-fit would scatter singles across both and refuse it.
    """
    free = {
        a.jobid: [
            i for i in sorted(a.idle_gpu_indices)
            if i not in taken.get(a.jobid, set())
        ]
        for a in allocs
    }
    if want == 0:
        best = max(allocs, key=lambda a: (len(free[a.jobid]), a.jobid), default=None)
        if best is None or not free[best.jobid]:
            return None
        return best, free[best.jobid]

    fits = [a for a in allocs if len(free[a.jobid]) >= want]
    if not fits:
        return None
    best = min(fits, key=lambda a: (len(free[a.jobid]), a.jobid))
    return best, free[best.jobid][:want]


def _refusal(
    allocs: list[GpuAllocationCandidate], usable: list[GpuAllocationCandidate],
    taken: dict[str, set[int]], want: int, min_vram_gb: int,
) -> str:
    """Why the request could not be placed.

    Two different situations reach here and the message says which, because
    they call for different actions. "Every card that would fit is leased"
    resolves when a run finishes, so relaunching later works. "No allocation
    is big enough" never resolves on its own, so relaunching is a waste — the
    user has to ask for fewer cards or start a bigger allocation.
    """
    if not allocs:
        return "no verified fixed instance host is available"

    parts = []
    for a in sorted(allocs, key=lambda x: x.jobid):
        idle_unleased = [
            i for i in a.idle_gpu_indices
            if i not in taken.get(a.jobid, set())
        ]
        parts.append(
            f"job {a.jobid} ({a.node or '?'}): "
            f"{len(idle_unleased)}/{a.gpu_count} idle and unleased"
        )
    state = "; ".join(parts)
    asked = f"{want} GPU(s)" if want else "at least 1 GPU"

    if not usable and any(a.gpu_count > 0 for a in allocs):
        return (
            f"no allocation has verified per-device VRAM meeting the "
            f"{min_vram_gb} GB floor required for {asked}. Re-probe VRAM or start an "
            "allocation that satisfies min_vram_gb."
        )

    # Capacity, not availability: could this request EVER fit on one of these
    # allocations, supposing every card on it were free?
    biggest = max((a.gpu_count for a in usable), default=0)
    if biggest < max(1, want):
        return (
            f"no allocation is big enough for {asked}: the largest is "
            f"{biggest} GPU(s). {state}. A slice never spans allocations, so "
            "free cards on different jobs do not add up. Lower num_gpus or "
            "start a larger allocation."
        )

    needed = max(1, want)
    idle_fit = [a for a in usable if len(a.idle_gpu_indices) >= needed]
    if idle_fit:
        # The live probe found enough idle cards on at least one allocation,
        # but the ledger removed them. This is specifically Zevo contention,
        # not foreign work on the cluster.
        return (
            f"cannot place {asked} right now: every idle card that would fit "
            f"is leased by another run in Zevo. {state}. Relaunch once one of "
            f"them finishes."
        )
    return (
        f"cannot place {asked} right now: the live probes found no allocation "
        f"with enough idle GPUs. {state}. Relaunch after the external GPU work "
        f"finishes or start another idle allocation."
    )


@router.post("/gpu/leases", response_model=GpuLeaseGrant)
async def acquire(body: GpuLeaseRequest, db: AsyncSession = Depends(get_db)) -> GpuLeaseGrant:
    """Lease GPUs on one of the reported allocations for `run_id`."""
    await _reap(db)

    # Already holding some? Hand back the same cards. The infra ticket can be
    # re-woken after a retry, and re-bidding would either double-book this run
    # or lose its cards to another while it was between attempts.
    mine = (await db.execute(
        select(GpuLease).where(
            GpuLease.run_id == body.run_id, GpuLease.released_at.is_(None)
        )
    )).scalars().all()
    if mine:
        return _grant(list(mine), reused=True)

    # A hard floor requires a measurement. Unknown VRAM (0) cannot prove that
    # the allocation satisfies the request and is therefore ineligible.
    usable = [
        a for a in body.allocations
        if a.gpu_count > 0
        and (not body.min_vram_gb or a.vram_gb >= body.min_vram_gb)
    ]

    for _ in range(_MAX_GRANT_ATTEMPTS):
        held = (await db.execute(
            select(GpuLease.jobid, GpuLease.gpu_index).where(GpuLease.released_at.is_(None))
        )).all()
        taken: dict[str, set[int]] = {}
        for jobid, gpu_index in held:
            taken.setdefault(jobid, set()).add(gpu_index)

        picked = _choose(usable, taken, body.num_gpus)
        if picked is None:
            raise HTTPException(
                409,
                _refusal(
                    body.allocations,
                    usable,
                    taken,
                    body.num_gpus,
                    body.min_vram_gb,
                ),
            )
        alloc, indices = picked

        rows = [
            GpuLease(
                jobid=alloc.jobid,
                node=alloc.node,
                gpu_index=i,
                run_id=body.run_id,
                ticket_id=body.ticket_id,
                gpu_name=alloc.gpu_name,
                vram_gb=alloc.vram_gb,
            )
            for i in indices
        ]
        db.add_all(rows)
        try:
            await db.commit()
        except IntegrityError as e:
            await db.rollback()
            # ONLY the live-card index means "someone beat us to it". Every
            # other constraint — a run_id that names no run, most of all — is a
            # caller error that retrying cannot fix, and retrying it six times
            # then reporting "the allocations kept changing" describes the
            # wrong problem entirely.
            if not _is_live_card_conflict(e):
                raise HTTPException(400, f"could not record the lease: {e.orig}") from e
            continue  # someone took one of these cards; re-read and pick again
        return _grant(rows, reused=False)

    raise HTTPException(
        409,
        "could not obtain a GPU lease: the allocations kept changing under us. "
        "Retry the ticket.",
    )


@router.delete("/gpu/leases")
async def release(
    run_id: str = Query(...),
    reason: str = Query(default="released by request"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, int]:
    """Give back every card `run_id` holds."""
    res = await db.execute(
        update(GpuLease)
        .where(GpuLease.run_id == run_id, GpuLease.released_at.is_(None))
        .values(released_at=datetime.now(timezone.utc), release_reason=reason)
    )
    await db.commit()
    return {"released": int(res.rowcount or 0)}


@router.get("/gpu/leases", response_model=list[GpuLeaseRecord])
async def list_leases(
    live_only: bool = Query(default=True),
    db: AsyncSession = Depends(get_db),
) -> list[GpuLeaseRecord]:
    q = select(GpuLease).order_by(GpuLease.jobid, GpuLease.gpu_index)
    if live_only:
        q = q.where(GpuLease.released_at.is_(None))
    rows = (await db.execute(q)).scalars().all()
    return [
        GpuLeaseRecord(
            jobid=r.jobid,
            node=r.node,
            gpu_index=r.gpu_index,
            run_id=r.run_id,
            gpu_name=r.gpu_name,
            acquired_at=r.acquired_at,
        )
        for r in rows
    ]
