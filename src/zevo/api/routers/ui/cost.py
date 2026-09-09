"""GET /api/cost/total -- the all-time / windowed spend rollup."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.api.database import get_db
from zevo.db import HeartbeatRun, InfraInstance, Run, Ticket
from zevo.engine.cost.budget import is_rented
from zevo.engine.cost.pricing import gpu_hourly


router = APIRouter()


class CostTotal(BaseModel):
    """Everything this installation has ever spent, split by what spent it.

    The two halves answer different questions — agent cost scales with how much
    the models talk, GPU cost with how long the hardware was held — and a single
    total hides which one is running away.
    """

    agent_usd: float      # LLM tokens across every heartbeat
    gpu_usd: float        # RENTED GPU time only; own hardware costs nothing
    total_usd: float
    # Every token the agents have read and written. The dollar figure already
    # depends on it, but the count is the thing that stays comparable across
    # model price changes.
    n_tokens: int = 0
    n_heartbeats: int
    n_runs: int


# How far back each window reaches. "all" is the absence of a cutoff rather
# than a very large one, so a clock skew cannot silently exclude old rows.
_WINDOWS: dict[str, timedelta | None] = {
    "1d": timedelta(days=1),
    "1w": timedelta(days=7),
    "1m": timedelta(days=30),
    "all": None,
}


@router.get("/cost/total", response_model=CostTotal)
async def cost_total(
    window: str = "all",
    db: AsyncSession = Depends(get_db),
) -> CostTotal:
    """Spend, optionally over a trailing window.

    An installation's all-time total stops being a useful number once it
    has been running a while: it only ever goes up, so it cannot answer
    "are we spending more than we were". The window cuts both halves the
    same way -- heartbeats by when they started, GPU time by the overlap
    between the instance's life and the window -- so the two still add up
    to the total shown beside them.
    """
    if window not in _WINDOWS:
        raise HTTPException(
            400, f"unknown window {window!r}; expected one of {sorted(_WINDOWS)}"
        )
    delta = _WINDOWS[window]
    since = datetime.now(timezone.utc) - delta if delta else None

    hb_where = [HeartbeatRun.started_at >= since] if since else []
    agent_usd, n_hb, n_tokens = (await db.execute(
        select(
            func.coalesce(func.sum(HeartbeatRun.estimated_cost_usd), 0.0),
            func.count(HeartbeatRun.id),
            func.coalesce(func.sum(
                HeartbeatRun.input_tokens
                + HeartbeatRun.output_tokens
                + HeartbeatRun.cached_input_tokens
                + HeartbeatRun.reasoning_output_tokens
            ), 0),
        ).where(*hb_where)
    )).one()

    # Priced the same way zevo.engine.cost.budget does it per run: only RENTED boxes
    # count, at the provider's own hourly rate. `cluster` and `instance` are the
    # user's own machines — holding them is not money spent.
    now = datetime.now(timezone.utc)
    ends = dict((await db.execute(select(Run.id, Run.finished_at))).all())
    rows = (await db.execute(select(InfraInstance))).scalars().all()
    gpu_usd = 0.0
    for r in rows:
        if not r.created_at:
            continue
        # A finished run stops the clock even if the instance row was never
        # closed — otherwise a cancelled run bills against `now` forever.
        end = r.released_at or ends.get(r.run_id) or now
        start = r.created_at
        if since is not None:
            # Only the part of this instance's life that falls inside the
            # window. An instance held across the boundary counts for the
            # hours on this side of it, not all of them and not none.
            if end <= since:
                continue
            start = max(start, since)
        uptime_h = max(0.0, (end - start).total_seconds() / 3600.0)
        if not is_rented(r.provider):
            continue
        if r.dph:
            gpu_usd += r.dph * uptime_h
        else:
            rate = gpu_hourly(r.gpu_name)
            if rate is not None:
                gpu_usd += rate * (r.gpu_count or 1) * uptime_h

    run_where = [Run.started_at >= since] if since else []
    n_runs = int((await db.execute(
        select(func.count(Run.id)).where(*run_where)
    )).scalar_one() or 0)
    return CostTotal(
        agent_usd=round(float(agent_usd or 0.0), 6),
        gpu_usd=round(gpu_usd, 6),
        total_usd=round(float(agent_usd or 0.0) + gpu_usd, 6),
        n_heartbeats=int(n_hb or 0),
        n_runs=n_runs,
        n_tokens=int(n_tokens or 0),
    )
