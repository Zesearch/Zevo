"""Per-run cost and wall-clock budget accounting.

Spent = (sum of heartbeat_runs.estimated_cost_usd for this run's tickets)
      + (GPU cost: per instance, pricing.gpu_hourly(gpu_name) * gpu_count *
         uptime_hours, or the provider's dph * uptime_hours if the GPU type
         isn't in the price table)

Used by:
  - backend ticket-spawn guard (refuse to create a new ticket if the
    run's max_cost_usd is set and spent >= cap)
  - preflight (compare est cost vs cap)
  - orchestrator payload (so the agent sees both remaining money and time
    before it decides whether another complete iteration fits)
  - UI cost meter
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.db import HeartbeatRun, InfraInstance, Run, Ticket
from zevo.engine.cost.pricing import gpu_hourly

# Providers that bill by the hour. `cluster` and `instance` are hardware the
# user already owns or has been allocated, so holding them costs nothing.
RENTED_PROVIDERS = {"cloud"}


def is_rented(provider: str | None) -> bool:
    return (provider or "").strip().lower() in RENTED_PROVIDERS


# The improvement loop's own order, so a per-agent breakdown reads like the
# pipeline it describes rather than alphabetically. Roles absent from a run are
# simply not present in the result; unknown ids sort after the known ones.
PIPELINE_ORDER = [
    "orchestrator", "data", "infrastructure", "train",
    "inference", "evaluation", "registry",
]


@dataclass
class AgentCostRow:
    """One role's LLM/agent spend on a run (frozen per-heartbeat cost summed)."""

    agent_id: str
    cost_usd: float
    heartbeats: int


@dataclass
class CostBreakdown:
    """Read-only per-agent cost view for a run.

    `agents` is the LLM/agent cost grouped by `agent_id` (orchestrator, data,
    infrastructure, train, inference, evaluation, registry) in pipeline order;
    `gpu_cost_usd` and the totals reuse `snapshot_for_run` exactly, so the same
    rented-only GPU rule applies and no new pricing is invented here.
    """

    run_id: str
    agents: list[AgentCostRow]
    agent_cost_usd: float     # sum of per-agent LLM cost (== snapshot llm_cost)
    gpu_cost_usd: float
    total_cost_usd: float


async def cost_breakdown_for_run(
    session: AsyncSession, run_id: str,
) -> CostBreakdown:
    """Group a run's recorded cost by agent role, plus GPU and totals.

    The per-agent numbers come from the same source the budget snapshot sums —
    `heartbeat_runs.estimated_cost_usd`, joined to the run through its tickets —
    so the agent rows add up to the snapshot's LLM cost, and GPU/total are taken
    straight from `snapshot_for_run`. Purely additive and read-only.
    """
    snap = await snapshot_for_run(session, run_id)

    rows = (await session.execute(
        select(
            HeartbeatRun.agent_id,
            func.coalesce(func.sum(HeartbeatRun.estimated_cost_usd), 0.0),
            func.count(),
        )
        .join(Ticket, HeartbeatRun.ticket_id == Ticket.id)
        .where(Ticket.run_id == run_id)
        .group_by(HeartbeatRun.agent_id)
    )).all()

    def order_key(agent_id: str) -> tuple[int, str]:
        return (
            PIPELINE_ORDER.index(agent_id)
            if agent_id in PIPELINE_ORDER else len(PIPELINE_ORDER),
            agent_id,
        )

    agents = [
        AgentCostRow(
            agent_id=str(agent_id or ""),
            cost_usd=round(float(cost or 0.0), 6),
            heartbeats=int(count or 0),
        )
        for agent_id, cost, count in rows
    ]
    agents.sort(key=lambda a: order_key(a.agent_id))

    return CostBreakdown(
        run_id=run_id,
        agents=agents,
        agent_cost_usd=snap.llm_cost_usd,
        gpu_cost_usd=snap.gpu_cost_usd,
        total_cost_usd=snap.spent_usd,
    )


@dataclass
class BudgetSnapshot:
    max_cost_usd: float       # 0.0 if no cap is set
    llm_cost_usd: float       # sum of heartbeat token costs
    gpu_cost_usd: float       # sum of infra dph * uptime
    spent_usd: float          # llm + gpu
    remaining_usd: float      # max_cost_usd - spent (negative if over)
    over_budget: bool         # max_cost_usd > 0 and spent >= max_cost_usd
    projected_next_iteration_usd: float
    can_afford_next_iteration: bool | None
    projection_basis: str
    max_runtime_hours: float
    max_queue_wait_hours: float
    queue_wait_hours: float
    elapsed_runtime_hours: float
    remaining_runtime_hours: float
    over_time_limit: bool
    projected_next_iteration_hours: float
    can_finish_next_iteration_in_time: bool | None
    runtime_projection_basis: str


async def snapshot_for_run(session: AsyncSession, run_id: str) -> BudgetSnapshot:
    """Cheap (~2 queries) snapshot of where a run stands cost-wise."""
    run = (await session.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if run is None:
        return BudgetSnapshot(
            max_cost_usd=0.0,
            llm_cost_usd=0.0,
            gpu_cost_usd=0.0,
            spent_usd=0.0,
            remaining_usd=0.0,
            over_budget=False,
            projected_next_iteration_usd=0.0,
            can_afford_next_iteration=None,
            projection_basis="run not found",
            max_runtime_hours=0.0,
            max_queue_wait_hours=0.0,
            queue_wait_hours=0.0,
            elapsed_runtime_hours=0.0,
            remaining_runtime_hours=0.0,
            over_time_limit=False,
            projected_next_iteration_hours=0.0,
            can_finish_next_iteration_in_time=None,
            runtime_projection_basis="run not found",
        )

    # LLM cost: heartbeats live on tickets that live on the run.
    llm_q = (
        select(func.coalesce(func.sum(HeartbeatRun.estimated_cost_usd), 0.0))
        .join(Ticket, HeartbeatRun.ticket_id == Ticket.id)
        .where(Ticket.run_id == run_id)
    )
    llm_cost = float((await session.execute(llm_q)).scalar_one() or 0.0)

    # GPU cost is MONEY SPENT, so only RENTED hardware counts. `cluster` and
    # `instance` are operator-supplied infrastructure (a Slurm site or fixed GPU
    # host) — Zevo has no provider bill for them, and pricing those at
    # market rates produced a number that dwarfed everything actually paid for.
    #
    # For a rented box the provider's own dph is the bill; the per-GPU-type
    # table is only a fallback for when the provider did not report one.
    def aware(value: datetime) -> datetime:
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    # A terminated run's GPU clock stops at the run's end — never keep billing
    # against `now`. This is the authoritative cap even if an instance's
    # `released_at` was never set (e.g. the run was cancelled outside the
    # reconciler's release path, so the InfraInstance row stayed open).
    run_end = aware(run.finished_at) if run.finished_at else now
    rows = (await session.execute(
        select(InfraInstance).where(InfraInstance.run_id == run_id)
    )).scalars().all()
    gpu_cost = 0.0
    for r in rows:
        if not r.created_at:
            continue
        if not is_rented(r.provider):
            continue
        end = aware(r.released_at) if r.released_at else run_end
        uptime_h = max(
            0.0, (end - aware(r.created_at)).total_seconds() / 3600.0,
        )
        if r.dph:
            gpu_cost += r.dph * uptime_h                        # dph = per instance
        else:
            rate = gpu_hourly(r.gpu_name)
            if rate is not None:
                gpu_cost += rate * (r.gpu_count or 1) * uptime_h  # table = per GPU

    spent = round(llm_cost + gpu_cost, 6)
    cap = float(run.max_cost_usd or 0.0)
    remaining = round(cap - spent, 6) if cap > 0 else 0.0
    # This is advisory reserve evidence for the Orchestrator, not a hard
    # reservation. Baseline/setup counts as one measured cycle; after trained
    # iterations exist, compare that all-in average with the most recent
    # iteration's LLM spend and keep the larger estimate. The backend's existing
    # over-budget guard remains the final emergency brake.
    cycle_count = max(1, int(run.iterations_completed or 0) + 1)
    average_cycle = spent / cycle_count if spent > 0 else 0.0
    iteration_cost_rows = (await session.execute(
        select(
            Ticket.iteration,
            func.coalesce(func.sum(HeartbeatRun.estimated_cost_usd), 0.0),
        )
        .join(HeartbeatRun, HeartbeatRun.ticket_id == Ticket.id)
        .where(Ticket.run_id == run_id, Ticket.iteration >= 1)
        .group_by(Ticket.iteration)
        .order_by(Ticket.iteration.desc())
    )).all()
    latest_iteration_llm = float(iteration_cost_rows[0][1]) if iteration_cost_rows else 0.0
    projected = round(max(average_cycle, latest_iteration_llm), 6)
    if projected <= 0:
        can_afford: bool | None = None
        basis = "no completed cost evidence; Orchestrator must estimate from planned work"
    elif cap <= 0:
        can_afford = True
        basis = "no cost cap; projection is informational"
    else:
        can_afford = remaining >= projected
        basis = (
            "max(all-in average measured cycle, latest trained-iteration LLM spend)"
        )

    # Wall-clock projection is intentionally evidence-based. A complete
    # trained iteration is the span from its first Specialist heartbeat to its
    # last; setup/Baseline and Orchestrator deliberation are not pretended to be
    # a Train→Inference→Evaluation cycle. Until one complete iteration exists,
    # the Orchestrator receives `None` and must estimate conservatively from the
    # planned work instead of trusting a made-up duration.
    started = aware(run.started_at) if run.started_at else now
    runtime_end = aware(run.finished_at) if run.finished_at else now
    raw_elapsed_h = max(0.0, (runtime_end - started).total_seconds() / 3600.0)
    # Slurm PENDING time is resource acquisition latency, not experiment
    # runtime. Merge intervals so retries/overlapping bookkeeping rows cannot
    # subtract the same wall-clock second twice.
    queue_intervals: list[tuple[datetime, datetime]] = []
    for instance in rows:
        if (instance.provider or "").strip().lower() != "cluster" or not instance.created_at:
            continue
        q_start = aware(instance.created_at)
        q_end_raw = instance.ready_at or instance.released_at or runtime_end
        q_end = min(aware(q_end_raw), runtime_end)
        q_start = max(q_start, started)
        if q_end > q_start:
            queue_intervals.append((q_start, q_end))
    queue_intervals.sort(key=lambda span: span[0])
    merged: list[tuple[datetime, datetime]] = []
    for q_start, q_end in queue_intervals:
        if not merged or q_start > merged[-1][1]:
            merged.append((q_start, q_end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], q_end))
    queue_wait_h = sum(
        (q_end - q_start).total_seconds() / 3600.0
        for q_start, q_end in merged
    )
    elapsed_h = max(0.0, raw_elapsed_h - queue_wait_h)
    runtime_cap = float(run.max_runtime_hours or 0.0)
    queue_cap = float(run.max_queue_wait_hours or 0.0)
    runtime_remaining = runtime_cap - elapsed_h if runtime_cap > 0 else 0.0

    completed = int(run.iterations_completed or 0)
    duration_rows = (await session.execute(
        select(Ticket.iteration, HeartbeatRun.started_at, HeartbeatRun.finished_at)
        .join(HeartbeatRun, HeartbeatRun.ticket_id == Ticket.id)
        .where(
            Ticket.run_id == run_id,
            Ticket.iteration >= 1,
            Ticket.iteration <= completed,
            Ticket.agent_id != "orchestrator",
            HeartbeatRun.finished_at.is_not(None),
        )
        .order_by(Ticket.iteration, HeartbeatRun.started_at)
    )).all() if completed > 0 else []
    spans: dict[int, tuple[datetime, datetime]] = {}
    for iteration, hb_start, hb_end in duration_rows:
        if hb_start is None or hb_end is None:
            continue
        start_at, end_at = aware(hb_start), aware(hb_end)
        current = spans.get(int(iteration))
        spans[int(iteration)] = (
            min(current[0], start_at) if current else start_at,
            max(current[1], end_at) if current else end_at,
        )
    durations = [
        max(0.0, (end_at - start_at).total_seconds() / 3600.0)
        for _, (start_at, end_at) in sorted(spans.items())
    ]
    if durations:
        projected_h = max(sum(durations) / len(durations), durations[-1])
        runtime_basis = "max(average completed iteration, latest completed iteration)"
        can_finish_in_time = (
            True if runtime_cap <= 0 else runtime_remaining >= projected_h
        )
    else:
        projected_h = 0.0
        runtime_basis = (
            "no completed iteration timing evidence; Orchestrator must estimate "
            "the planned complete loop"
        )
        can_finish_in_time = None

    return BudgetSnapshot(
        max_cost_usd=cap,
        llm_cost_usd=round(llm_cost, 6),
        gpu_cost_usd=round(gpu_cost, 6),
        spent_usd=spent,
        remaining_usd=remaining,
        over_budget=(cap > 0 and spent >= cap),
        projected_next_iteration_usd=projected,
        can_afford_next_iteration=can_afford,
        projection_basis=basis,
        max_runtime_hours=runtime_cap,
        max_queue_wait_hours=queue_cap,
        queue_wait_hours=round(queue_wait_h, 6),
        elapsed_runtime_hours=round(elapsed_h, 6),
        remaining_runtime_hours=round(runtime_remaining, 6),
        over_time_limit=(runtime_cap > 0 and elapsed_h >= runtime_cap),
        projected_next_iteration_hours=round(projected_h, 6),
        can_finish_next_iteration_in_time=can_finish_in_time,
        runtime_projection_basis=runtime_basis,
    )
