"""GET/POST /runs, GET /runs/{id}.

POST /runs accepts the named Run envelope and either a complete UserRequest or
a catalogue task_name. It queues the stable Orchestrator Ticket in the
background and returns the run_id immediately; LLM and GPU work never blocks
the HTTP response.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import asc, delete as sa_delete, desc, func, or_, select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.api.config import settings
from zevo.api.compute_defaults import (
    ComputeProvider,
    ComputeTarget,
    DefaultComputeError,
    resolve_default_compute,
)
from zevo.contracts.cancel import CancelWeightsPolicy
from zevo.providers import resolve_ssh_key
from zevo.api.database import get_db
from zevo.api.ui_access import is_trusted_ui_request
from zevo.db import (
    HeartbeatResult,
    HeartbeatRun,
    RegistryModel,
    Run,
    ScoreEvent,
    SshHost,
    Ticket,
)
from zevo.contracts.tickets import TERMINAL_RUN_STATUSES
from zevo.engine.observe.run_metrics import (
    baseline_and_best,
    baseline_and_best_test,
    improvement_test,
    validation_best_score,
)
from zevo.engine.ssh_auth import ssh_base_args
from zevo.contracts.customizations import RunCustomizations
from zevo.contracts.orchestrator import (
    AutoUserRequest,
    PatchRunRequest,
    UserRequest,
    inherit_test_validation_contract,
    scoring_asset_errors,
)
from zevo.contracts.tickets import RunMode, RunStatus


router = APIRouter()


def _normalize_gpu_provider(value: str) -> str:
    """Valid values are cluster|cloud|instance; anything else -> '' (auto)."""
    v = (value or "").strip().lower()
    return v if v in ("cluster", "cloud", "instance") else ""


async def _resolve_run_compute(
    db: AsyncSession, body: "CreateRunRequest",
) -> ComputeTarget:
    """Resolve an explicit picker value or the configured concrete default."""
    provider = _normalize_gpu_provider(body.gpu_provider or "")
    if provider:
        return ComputeTarget(
            value="explicit",
            provider=cast(ComputeProvider, provider),
            cloud_backend=(body.cloud_backend or "").strip().lower(),
            ssh_host_id=(body.ssh_host_id or "").strip(),
        )
    try:
        return await resolve_default_compute(db)
    except DefaultComputeError as exc:
        raise HTTPException(422, str(exc)) from exc


def _dataset_profile_for(dataset_path: str) -> dict | None:
    """Look up the cached DatasetProfile for the user's dataset, if any.

    Returns a SMALL dict (the orchestrator only needs the structured
    signals — task type, format, row count, recommended split, issues).
    We deliberately don't include sample values or column dtypes; the
    data agent reads the raw file directly when it needs that.

    Returns None for:
      - dataset path not set
      - dataset path not under data/files/
      - no profile on disk yet (no auto-profile happened)
      - profile load failed (corrupted cache)
    """
    if not dataset_path:
        return None
    from pathlib import Path as _P
    from zevo.engine.dataset_profiler import load_cached, is_cache_valid

    # Map user's path (which might be relative or live anywhere) back to
    # a file-set dir under data/files/<name>/.
    # The cached profile lives at <dataset_dir>/.profile.json.
    p = _P(dataset_path)
    if not p.exists():
        return None
    # The file-set dir is the parent of the file when uploaded via the Files page.
    dataset_dir = p.parent
    if not (dataset_dir / ".profile.json").exists():
        return None
    cached = load_cached(dataset_dir)
    if cached is None:
        return None
    # Validate cache freshness against the actual file the user pointed at.
    try:
        if not is_cache_valid(dataset_dir, p, cached):
            return None
    except Exception:  # noqa: BLE001
        return None

    # Trim to the structured signals the orchestrator and Data Agent use.
    return {
        "task_type":           cached.task_type.label,
        "task_type_confidence": cached.task_type.confidence,
        "task_type_rationale":  cached.task_type.rationale,
        "expected_format":     cached.expected_format,
        "n_rows":              cached.n_rows,
        "n_columns":           cached.n_columns,
        "recommended_split":   cached.split_recommendation.model_dump(),
        "duplicates":          cached.duplicates.model_dump(),
        "issues":              [i.model_dump() for i in cached.issues],
        "ready_for_data":      cached.ready_for_data,
        "profiler_version":    cached.profiler_version,
    }


class RunSummary(BaseModel):
    id: str
    task_name: str
    # What the user called this execution. Every current launch requires it.
    run_name: str = ""
    setting_id: str = ""
    setting_name: str = ""
    # Run-level GPU maximum; selected/measured devices remain Infrastructure data.
    num_gpus: int = 0
    task_objective: str
    agent_objective: str
    status: RunStatus
    is_terminal: bool
    summary: str
    registry_version_tag: str
    halted_reason: str
    # Cancel asked for, weights still being copied off the box. The run reads
    # `running` until the copy is done, so this is the flag the UI shows.
    cancelling: bool = False
    cancel_policy: dict[str, Any] = Field(default_factory=dict)
    cancel_outcome: dict[str, Any] = Field(default_factory=dict)
    started_at: str
    finished_at: str | None = None
    # Zevo model-improvement iteration fields.
    iteration_budget: int = 0
    iterations_completed: int = 0
    stop_threshold: float | None = None
    metric: str
    metric_direction: Literal["max", "min"]
    validation_metric: str
    validation_metric_direction: Literal["max", "min"]
    best_validation_score: float | None = None
    # G.1 — hard $ cap. 0.0 = no cap.
    max_cost_usd: float = 0.0
    # Run-only wall-clock cap in hours. 0.0 = unlimited.
    max_runtime_hours: float = 0.0
    # Slurm PENDING time has its own cap and is excluded from duration_s.
    max_queue_wait_hours: float = 24.0
    queue_wait_seconds: int = 0
    queue_waiting: bool = False
    # Actual spend (LLM + GPU) and elapsed wall-clock (finished-started, or
    # now-started while running).
    cost_usd: float = 0.0
    duration_s: int | None = None
    mode: RunMode = "full_pipeline"
    customizations: dict[str, Any] = Field(default_factory=dict)
    decision_pins: dict[str, Any] = Field(default_factory=dict)
    model_lineages: dict[str, Any] = Field(default_factory=dict)
    # The run's own iteration-0 reference point, lifted out of `history` so the
    # dashboard can chart lift-over-baseline across many runs from the LIST
    # endpoint alone. None = the run never recorded a baseline probe.
    #
    # `best_trained_validation_score` is the best score, in `metric_direction`, over
    # NON-baseline history entries. It differs from best_validation_score, which folds
    # the baseline into that comparison --
    # a run whose training only ever did worse than the untuned model reports
    # best_validation_score == baseline, which would read as "no change" instead of
    # "regressed". Free to compute: list_runs already loads the whole Run row.
    baseline_validation_score: float | None = None
    baseline_test_score: float | None = None
    best_trained_validation_score: float | None = None
    # ── the held-out goal ────────────────────────────────────────────────────
    # Everything above is measured on VALIDATION — the set the run tuned
    # against. `champion_test_score` belongs to whichever iteration wins on
    # validation; the complete chronology lives in ScoreEvent. Agent callers
    # receive null while active;
    # trusted dashboard requests can observe the live measurement.
    champion_test_score: float | None = None
    # How this run came by its validation set: user/Hugging Face supplied, or a
    # deterministic Test split created by Run setup.
    validation_source: str = ""
    # Whether the scoring contract is on the Run yet. Always true except for an
    # `auto` Run still in its scoping stage, whose metric/direction/validation
    # fields above are placeholders until the Data agent's ScopingResult settles.
    scoring_settled: bool = True
    # Auto mode: how the held-out was obtained — "public_benchmark" or
    # "synthesized". Empty for a user-supplied held-out.
    eval_source: str = ""


class RunDetail(RunSummary):
    tickets: list[dict[str, Any]]
    # Per-iteration facts from Evaluation, enriched with orchestrator decisions.
    history: list[dict[str, Any]] = Field(default_factory=list)
    # The same total, split by what spent it. Agent cost scales with how much
    # the models talk; GPU cost with how long RENTED hardware was held, which is
    # why it is legitimately $0 on a run that used the user's own machine.
    agent_cost_usd: float = 0.0
    gpu_cost_usd: float = 0.0
    # Which model DROVE the agents on this run — the harness, not the model it
    # trained. Same definition the harness leaderboard ranks by, so the strip
    # and the board cannot name a run's harness differently.
    harness_model: str = ""


async def _settle_splits(
    run: Run, user_request: UserRequest,
) -> tuple[UserRequest, dict[str, Any], str]:
    """Settle scoring populations without exposing them to selection.

    The logic lives in `zevo.engine.run.split_settlement.settle_splits` so an
    `auto` Run can settle AFTER creation (once scoping has derived its scoring
    contract); this wrapper keeps the HTTP contract of the creation path: every
    settlement problem is a 400 carrying the engine's complete message.

    The returned Agent request contains neither Validation nor Test assets.
    Those populations remain in the engine-owned holdout snapshot; the runner
    prepares the answer-free Validation view only after Data fixes Training.
    """
    from zevo.engine.run.split_settlement import SplitSettlementError, settle_splits

    try:
        return await settle_splits(
            run, user_request, work_dir_root=str(settings.work_dir_root),
        )
    except SplitSettlementError as e:
        raise HTTPException(400, str(e))


def _baseline_and_best(r: Run) -> tuple[float | None, float | None]:
    """Champion model-lineage baseline and best trained validation result."""
    return baseline_and_best(r.history, r.validation_metric_direction)


def _summary(
    r: Run, *, cost_usd: float = 0.0, duration_s: int | None = None,
    holdout: bool = False, queue_wait_seconds: int = 0,
    queue_waiting: bool = False,
) -> RunSummary:
    baseline, best_trained = _baseline_and_best(r)
    derived_best_validation = validation_best_score(
        r.history, r.validation_metric_direction,
    )
    baseline_test, _ = baseline_and_best_test(
        r.history, r.validation_metric_direction,
    )
    return RunSummary(
        id=r.id, task_name=r.task_name,
        run_name=r.run_name,
        setting_id=r.setting_id or "",
        setting_name=r.setting_name or "",
        task_objective=r.task_objective or "",
        agent_objective=r.agent_objective or "",
        status=r.status, is_terminal=r.status in TERMINAL_RUN_STATUSES,
        summary=r.summary,
        registry_version_tag=r.registry_version_tag,
        halted_reason=r.halted_reason,
        cancelling=r.cancel_requested_at is not None and r.status not in TERMINAL_RUN_STATUSES,
        cancel_policy=dict(r.cancel_policy or {}),
        cancel_outcome=dict(r.cancel_outcome or {}),
        started_at=r.started_at.isoformat() if r.started_at else "",
        finished_at=r.finished_at.isoformat() if r.finished_at else None,
        iteration_budget=r.iteration_budget,
        iterations_completed=r.iterations_completed,
        stop_threshold=r.stop_threshold,
        metric=r.metric,
        metric_direction=r.metric_direction,
        validation_metric=r.validation_metric,
        validation_metric_direction=r.validation_metric_direction,
        # Derive the headline from Validation journal facts. An older in-flight
        # Run may have had this materialized column contaminated by a held-out
        # Test event; never repeat that value to the dashboard.
        best_validation_score=(
            derived_best_validation
            if derived_best_validation is not None
            else r.best_validation_score
        ),
        max_cost_usd=float(r.max_cost_usd or 0.0),
        max_runtime_hours=float(r.max_runtime_hours or 0.0),
        max_queue_wait_hours=float(r.max_queue_wait_hours or 0.0),
        queue_wait_seconds=max(0, int(queue_wait_seconds)),
        queue_waiting=bool(queue_waiting),
        cost_usd=cost_usd,
        duration_s=duration_s,
        mode=r.mode,
        customizations=r.customizations,
        decision_pins=dict(r.decision_pins or {}),
        model_lineages=dict(r.model_lineages or {}),
        baseline_validation_score=baseline,
        best_trained_validation_score=best_trained,
        baseline_test_score=baseline_test,
        champion_test_score=(
            float(r.champion_test_score) if holdout and r.champion_test_score is not None else None
        ),
        validation_source=str((r.holdout or {}).get("validation_source") or ""),
        scoring_settled=bool(r.scoring_settled),
        eval_source=str((r.holdout or {}).get("eval_source") or ""),
        num_gpus=r.num_gpus,
    )


async def _costs_for_runs(db: AsyncSession, runs: list[Run]) -> dict[str, float]:
    """Spend per run, for many runs, in two queries.

    Mirrors zevo.engine.cost.budget.snapshot_for_run exactly — same rented-only GPU rule,
    same dph-then-price-table order — but folded over a set of runs instead of
    one, because the list needs a cost for every row and `sort=cost` needs one
    for every MATCH, not just the page. Calling snapshot_for_run per run cost
    two queries a row.
    """
    from datetime import datetime, timezone
    from zevo.engine.cost.budget import is_rented
    from zevo.engine.cost.pricing import gpu_hourly
    from zevo.db import HeartbeatRun, InfraInstance, Ticket

    ids = [r.id for r in runs]
    if not ids:
        return {}

    llm_rows = (await db.execute(
        select(Ticket.run_id, func.coalesce(func.sum(HeartbeatRun.estimated_cost_usd), 0.0))
        .join(HeartbeatRun, HeartbeatRun.ticket_id == Ticket.id)
        .where(Ticket.run_id.in_(ids))
        .group_by(Ticket.run_id)
    )).all()
    out = {rid: float(c or 0.0) for rid, c in llm_rows}

    now = datetime.now(timezone.utc)
    ends = {r.id: (r.finished_at or now) for r in runs}
    infra = (await db.execute(
        select(InfraInstance).where(InfraInstance.run_id.in_(ids))
    )).scalars().all()
    for i in infra:
        if not i.created_at or not is_rented(i.provider):
            continue
        end = i.released_at or ends.get(i.run_id, now)
        hours = max(0.0, (end - i.created_at).total_seconds() / 3600.0)
        if i.dph:
            out[i.run_id] = out.get(i.run_id, 0.0) + i.dph * hours
        else:
            rate = gpu_hourly(i.gpu_name)
            if rate is not None:
                out[i.run_id] = out.get(i.run_id, 0.0) + rate * (i.gpu_count or 1) * hours
    return {rid: round(v, 2) for rid, v in out.items()}


def _duration_s(r: Run) -> int | None:
    """Elapsed seconds: finished-started, or now-started while still running."""
    from datetime import datetime, timezone
    if not r.started_at:
        return None
    def aware(value: datetime) -> datetime:
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    end = aware(r.finished_at) if r.finished_at else datetime.now(timezone.utc)
    return max(0, int((end - aware(r.started_at)).total_seconds()))


async def _queue_wait_state_for_runs(
    db: AsyncSession, runs: list[Run],
) -> tuple[dict[str, int], set[str]]:
    """Merged cluster provision→ready seconds and actively queued Runs."""
    from datetime import datetime, timezone
    from zevo.db import InfraInstance
    ids = [run.id for run in runs]
    if not ids:
        return {}, set()
    rows = (await db.execute(
        select(InfraInstance).where(
            InfraInstance.run_id.in_(ids), InfraInstance.provider == "cluster",
        )
    )).scalars().all()
    now = datetime.now(timezone.utc)
    def aware(value: datetime) -> datetime:
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

    bounds = {
        run.id: (
            aware(run.started_at) if run.started_at else now,
            aware(run.finished_at) if run.finished_at else now,
        )
        for run in runs
    }
    grouped: dict[str, list[tuple[datetime, datetime]]] = {}
    waiting: set[str] = set()
    for item in rows:
        start_bound, end_bound = bounds.get(item.run_id, (None, now))
        if not item.created_at or not start_bound:
            continue
        start = max(aware(item.created_at), start_bound)
        end = min(aware(item.ready_at or item.released_at or end_bound), end_bound)
        if end > start:
            grouped.setdefault(str(item.run_id), []).append((start, end))
        if item.ready_at is None and item.released_at is None:
            waiting.add(str(item.run_id))
    out: dict[str, int] = {}
    for run_id, spans in grouped.items():
        merged: list[tuple[datetime, datetime]] = []
        for start, end in sorted(spans):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        out[run_id] = max(0, int(sum((b - a).total_seconds() for a, b in merged)))
    return out, waiting


async def _cost_and_duration(db: AsyncSession, r: Run) -> tuple[float, int | None]:
    """Actual spend (LLM + GPU) + elapsed seconds (finished-started, or
    now-started while running)."""
    from datetime import datetime, timezone
    from zevo.engine.cost.budget import snapshot_for_run
    queue_wait_s = 0
    try:
        snap = await snapshot_for_run(db, r.id)
        cost = round(float(snap.spent_usd), 2)
        queue_wait_s = max(0, int(snap.queue_wait_hours * 3600))
    except Exception:
        cost = 0.0
    dur: int | None = None
    if r.started_at:
        started = r.started_at if r.started_at.tzinfo else r.started_at.replace(tzinfo=timezone.utc)
        end = r.finished_at or datetime.now(timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        dur = max(0, int((end - started).total_seconds()) - queue_wait_s)
    return cost, dur


MAX_RUN_PAGE = 500


# Sortable columns, by the key the UI sends. Only real columns: cost and
# duration are derived per row after the query, so ordering by them would mean
# loading every run just to sort a page of 25.
_RUN_SORTS = {
    "started": Run.started_at,
    "task": Run.task_name,
    # The column the list calls "Best score" is the HELD-OUT one: sorting runs
    # by what they scored on their own validation set would rank them by how
    # hard each tuned against a set only it ever saw.
    "score": Run.champion_test_score,
    "status": Run.status,
    "iterations": Run.iterations_completed,
    # A run still going is measured against now, the same as the Time column.
    "time": func.coalesce(Run.finished_at, func.now()) - Run.started_at,
}

# Cost is not a column — it is heartbeat spend plus rented-GPU uptime, and the
# GPU half falls back to a price table that has no SQL equivalent. Sorting on it
# therefore prices every match and orders in Python (see list_runs), which is
# why it is not in the dict above rather than being approximated in SQL.
_COST_SORT = "cost"
_IMPROVEMENT_SORT = "improvement"


@router.get("/runs", response_model=list[RunSummary])
async def list_runs(
    response: Response,
    db: AsyncSession = Depends(get_db),
    limit: int = 100,
    offset: int = 0,
    q: str = "",
    sort: str = "started",
    order: str = "desc",
    group: str = "",
) -> list[RunSummary]:
    """Paginated, and searchable/sortable server-side.

    Search and sort have to run in the database: filtering the page the client
    already holds would only ever search 25 of 10,000 rows.

    The body stays a bare array -- the UI, the CLI and the orchestrator's own
    curl calls all index into it -- so the total row count rides along in the
    X-Total-Count header instead of wrapping the payload in an envelope.
    """
    limit = max(1, min(limit, MAX_RUN_PAGE))
    offset = max(0, offset)

    # One box searches the id, the run's own name and the task name: users type
    # whichever of the three they remember.
    where = []
    term = (q or "").strip()
    if term:
        like = f"%{term.lower()}%"
        where.append(or_(
            func.lower(Run.task_name).like(like),
            func.lower(Run.run_name).like(like),
            func.lower(Run.id).like(like),
        ))

    col = _RUN_SORTS.get(sort, Run.started_at)
    direction = asc if (order or "").lower() == "asc" else desc
    # started_at is the tiebreaker so equal keys keep a stable, meaningful order
    # rather than whatever the planner returns.
    order_by = [direction(col)] + ([] if col is Run.started_at else [desc(Run.started_at)])
    # Grouping is a FIRST key, not a replacement for the sort: `group=task` with
    # `sort=score&order=desc` answers "for each task, best score first", which
    # neither one alone can.
    grouped = (group or "").lower() == "task"
    if grouped:
        order_by = [asc(Run.task_name)] + order_by

    count_q = select(func.count()).select_from(Run)
    rows_q = select(Run)
    for w in where:
        count_q = count_q.where(w)
        rows_q = rows_q.where(w)

    total = (await db.execute(count_q)).scalar_one()

    if sort == _COST_SORT:
        # Price every match, order, then take the page. Sorting the page alone
        # would order 25 rows out of the whole result and call it sorted.
        matches = (await db.execute(rows_q.order_by(desc(Run.started_at)))).scalars().all()
        costs = await _costs_for_runs(db, list(matches))
        asc_ = (order or "").lower() == "asc"
        # Sort descending by negating, so the task key can stay ascending in the
        # same pass — Python sorts are stable but not per-key directional.
        matches = sorted(
            matches,
            key=lambda r: ((r.task_name if grouped else ""),
                           (costs.get(r.id, 0.0) if asc_ else -costs.get(r.id, 0.0))),
        )
        rows = matches[offset:offset + limit]
    elif sort == _IMPROVEMENT_SORT:
        # Improvement is derived from the Run's held-out history rather than a
        # database column. As with Cost, sort the full matching set before
        # pagination; sorting only the visible page would make the arrow lie.
        matches = (await db.execute(rows_q.order_by(desc(Run.started_at)))).scalars().all()
        gains = {
            run.id: improvement_test(
                run.history,
                run.validation_metric_direction,
                run.metric_direction,
            )
            for run in matches
        }
        asc_ = (order or "").lower() == "asc"
        matches = sorted(
            matches,
            key=lambda run: (
                run.task_name if grouped else "",
                gains[run.id] is None,
                (
                    gains[run.id]
                    if asc_ else -gains[run.id]
                ) if gains[run.id] is not None else 0.0,
            ),
        )
        rows = matches[offset:offset + limit]
        costs = await _costs_for_runs(db, list(rows))
    else:
        rows = (
            await db.execute(rows_q.order_by(*order_by).limit(limit).offset(offset))
        ).scalars().all()
        costs = await _costs_for_runs(db, list(rows))

    response.headers["X-Total-Count"] = str(total)
    # Without this the browser cannot read the header on a same-origin XHR that
    # went through a proxy, nor on any cross-origin call.
    response.headers["Access-Control-Expose-Headers"] = "X-Total-Count"

    queue_waits, queue_waiting = await _queue_wait_state_for_runs(db, list(rows))
    return [
        _summary(
            r, cost_usd=costs.get(r.id, 0.0),
            duration_s=(
                None if _duration_s(r) is None
                else max(0, int(_duration_s(r) or 0) - queue_waits.get(r.id, 0))
            ),
            queue_wait_seconds=queue_waits.get(r.id, 0),
            queue_waiting=r.id in queue_waiting,
            holdout=r.status in TERMINAL_RUN_STATUSES,
        )
        for r in rows
    ]


@router.get("/runs/task-counts")
async def run_task_counts(
    db: AsyncSession = Depends(get_db), q: str = "",
) -> dict[str, int]:
    """{task_name: how many runs it has}, over the same filter as the list.

    The list is paginated, so a group header on it can only ever count the rows
    of the page it is on — "3 on this page" out of eighteen. This is the number
    that header wants, and it is one grouped query rather than a column on every
    row. Declared above /runs/{run_id} so the literal path wins the match.
    """
    stmt = (
        select(Run.task_name, func.count().label("n"))
        .where(Run.task_name != "")
        .group_by(Run.task_name)
    )
    term = (q or "").strip()
    if term:
        like = f"%{term.lower()}%"
        stmt = stmt.where(or_(
            func.lower(Run.task_name).like(like),
            func.lower(Run.run_name).like(like),
            func.lower(Run.id).like(like),
        ))
    return {r.task_name: int(r.n or 0) for r in (await db.execute(stmt)).all()}


@router.get("/runs/{run_id}", response_model=RunDetail)
async def get_run(
    run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> RunDetail:
    """Return the Run through the caller's fixed visibility boundary.

    Trusted dashboard requests can observe the private held-out lane live.
    Agent-shaped requests see validation only until the Run is terminal. There
    is no caller-controlled reveal flag, so the optimization loop cannot opt
    itself into held-out state.
    """
    r = (await db.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"run {run_id} not found")
    reveal_holdout = (
        r.status in TERMINAL_RUN_STATUSES or is_trusted_ui_request(request)
    )
    stmt = select(Ticket).where(Ticket.run_id == run_id)
    if not reveal_holdout:
        stmt = stmt.where(Ticket.lane != "held_out_test")
    tickets = (await db.execute(stmt.order_by(Ticket.created_at))).scalars().all()
    _cost, _dur = await _cost_and_duration(db, r)
    from zevo.engine.cost.budget import snapshot_for_run
    snap = await snapshot_for_run(db, run_id)
    queue_waits, queue_waiting = await _queue_wait_state_for_runs(db, [r])
    queue_wait_seconds = queue_waits.get(run_id, 0)
    base = _summary(
        r, cost_usd=_cost, duration_s=_dur,
        queue_wait_seconds=queue_wait_seconds,
        queue_waiting=run_id in queue_waiting,
        holdout=reveal_holdout,
    )
    from zevo.engine.run.runner import _agent_history
    history = (
        list(r.history or []) if reveal_holdout
        else _agent_history(r.history or [])
    )
    from zevo.api.routers.ui.leaderboard import _harness_by_run
    return RunDetail(
        **base.model_dump(),
        agent_cost_usd=round(snap.llm_cost_usd, 6),
        gpu_cost_usd=round(snap.gpu_cost_usd, 6),
        harness_model=(await _harness_by_run(db, [run_id])).get(run_id, ""),
        history=history,
        tickets=[
            {
                "id": t.id,
                "agent_id": t.agent_id,
                "input_format": t.input_format,
                "lane": t.lane,
                "iteration": t.iteration,
                "operation": str((t.payload or {}).get("operation") or ""),
                "model_source": str((t.payload or {}).get("model_source") or ""),
                "status": t.status,
                "summary": t.summary or "",
                "error_message": t.error_message or "",
                "created_at": t.created_at.isoformat() if t.created_at else "",
            }
            for t in tickets
        ],
    )


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Every launch names both the reusable problem and this execution. A
    # UserRequest is optional only when task_name resolves to the catalogue.
    # In `auto` mode the body carries an AutoUserRequest instead: the objective
    # plus optional hints, with every scoring field and pin absent (the Data
    # agent's scoping stage derives the scoring contract). The two shapes are
    # disjoint — a complete UserRequest is rejected by AutoUserRequest's closed
    # schema and an objective-only body lacks UserRequest's required fields —
    # so the union parses exactly one of them; the validator below then pins
    # which one `mode` permits.
    user_request: UserRequest | AutoUserRequest | None = None
    task_name: str = Field(min_length=1, max_length=64)
    # task_name names the work; many runs share one, so this tells executions
    # apart in every list and detail view.
    run_name: str = Field(min_length=1, max_length=128)
    # Run/Setting fields. None resolves to the system default; Task owns no
    # experiment configuration and is therefore never a hidden fallback.
    iteration_budget: int | None = Field(None, ge=0)
    stop_threshold: float | None = Field(
        None,
        allow_inf_nan=False,
        description=(
            "Optional validation-score stopping threshold. None disables "
            "target-hit stopping. Zero remains a valid threshold, especially "
            "for a metric minimized toward zero. It is never inferred "
            "from validation or held-out test results."
        ),
    )
    # G.1 — hard $ cap. 0.0 = no cap.
    max_cost_usd: float | None = Field(None, ge=0.0)
    # Run-only deadline. It is intentionally absent from TaskSetting.
    max_runtime_hours: float | None = Field(
        None, ge=0.0,
        description="Maximum wall-clock hours for this Run; None or 0 means unlimited.",
    )
    max_queue_wait_hours: float | None = Field(
        None, gt=0.0, le=168.0,
        description=(
            "Maximum hours an automatically submitted cluster job may remain "
            "queued (up to seven days). Queue time does not consume max_runtime_hours."
        ),
    )
    # Runtime execution choices live here exactly once. They are not part of
    # UserRequest, which defines the task/data/scoring problem.
    # None -> vllm.
    generation_backend: Literal["hf", "vllm"] | None = None
    # Maximum GPUs the Run may use at once. None/0 means no upper bound;
    # Infrastructure selects a concrete positive count for every provider.
    num_gpus: int | None = Field(
        None,
        ge=0,
        description=(
            "Maximum GPUs the Run may use at once. Infrastructure chooses an "
            "actual positive count at or below it. None or zero means unlimited."
        ),
    )
    gpu_provider: Literal["cluster", "cloud", "instance"] | None = None
    # Optional verified SSH profile for a cluster/instance run.
    ssh_host_id: str = ""
    # Which cloud to rent on when gpu_provider == "cloud" (pins Vast.ai vs Lambda
    # for this run instead of the deployment default). Empty = deployment default.
    cloud_backend: Literal["", "vastai", "lambda"] = ""
    mode: Literal["full_pipeline", "customized_pipeline", "auto"] = "full_pipeline"
    customizations: RunCustomizations | None = None
    # Whether this run's configuration is written down as a reusable setting,
    # and under what name.
    #
    save_setting: bool = False
    setting_name: str = Field("", max_length=32)
    setting_id: str = ""

    @model_validator(mode="after")
    def request_shape_matches_mode(self) -> "CreateRunRequest":
        """`auto` derives scoring; every other mode takes a complete UserRequest.

        Existing modes are unchanged: a body that would previously have failed
        UserRequest validation still fails here, it just fails naming the mode.
        """
        if self.mode == "auto":
            if self.user_request is None:
                raise ValueError(
                    "auto mode requires user_request.task_objective (the objective is "
                    "the only required input)"
                )
            if not isinstance(self.user_request, AutoUserRequest):
                raise ValueError(
                    "auto mode takes an objective plus optional training-side "
                    "data/model/method pins and hints; Test, metric, answer, "
                    "submission, and Validation fields are derived by scoping"
                )
            if self.customizations is not None:
                raise ValueError("customizations are valid only with mode=customized_pipeline")
            if self.save_setting or self.setting_id:
                raise ValueError(
                    "auto mode does not save or select a Setting: its scoring "
                    "contract is derived per run, not reusable configuration"
                )
        elif isinstance(self.user_request, AutoUserRequest):
            raise ValueError(
                f"{self.mode} requires a complete user_request (metric, "
                "metric_direction, test_set, test_answer_fields, "
                "test_sample_submission, training_method, dataset, base_model, "
                "constraints); use mode=auto to have Zevo derive the Test contract"
            )
        return self

    @model_validator(mode="after")
    def full_pipeline_keeps_experiment_details_system_owned(self) -> "CreateRunRequest":
        if self.mode != "full_pipeline" or not isinstance(self.user_request, UserRequest):
            return self
        request = self.user_request
        supplied = [
            name
            for name, value in {
                "prompt_framing": request.prompt_framing,
                "system_prompt": request.system_prompt,
                "loss_objective_config": request.loss_objective_config,
                "inference_config": request.inference_config,
                "decoding_config": request.decoding_config,
            }.items()
            if value not in ("", {}, None)
        ]
        if supplied:
            raise ValueError(
                "Full Pipeline does not accept user-owned prompt, loss, or "
                "inference details; use Customized Pipeline for: "
                + ", ".join(supplied)
            )
        return self

class CreateRunResponse(BaseModel):
    run_id: str
    status: Literal["running"]


@router.patch("/runs/{run_id}", response_model=RunSummary)
async def patch_run(
    run_id: str,
    body: PatchRunRequest,
    db: AsyncSession = Depends(get_db),
) -> RunSummary:
    """Used by the reactive orchestrator to mark a run done/failed.

    Guards against false-positive `success`: when the orchestrator
    requests status="success", the server verifies that the run has no
    failed tickets and no still-running tickets. If either check fails,
    we reject with 409 + a structured error listing the blockers, so
    the orchestrator either retries the failed steps or marks the run
    failed/halted honestly.
    """
    # Journal facts, held-out scores, and the final supervisor decision all
    # update the same Run row. Lock it so a JSON history merge cannot overwrite
    # a score that committed at the same moment.
    r = (await db.execute(
        select(Run).where(Run.id == run_id).with_for_update()
    )).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"run {run_id} not found")

    # Apply a Journal narrative before terminal validation, allowing the final
    # narrative, consolidated summary, and success transition to be one atomic
    # PATCH instead of three race-prone calls.
    if body.history_entry is not None:
        # Idempotent narrative merge: a row is keyed by (iteration, source).
        # Evaluation must have created the factual row before the Orchestrator
        # can enrich it, and a re-wake updates that row rather than duplicating it.
        entry = body.history_entry.model_dump()
        key = (entry.get("iteration"), entry.get("source"))
        history = [dict(h) if isinstance(h, dict) else h for h in (r.history or [])]
        for h in history:
            if isinstance(h, dict):
                h.pop("notes", None)
        for i, h in enumerate(history):
            if (h.get("iteration"), h.get("source")) == key:
                # Evaluation owns factual scores; the supervisor owns the
                # decision prose. Copy only those fields: the request model's
                # defaults must never erase base_model/methods, and neither an
                # Agent-supplied validation value nor an invisible held-out
                # value may replace the engine's measurements.
                merged = dict(h)
                for field in ("action", "result", "analysis", "next"):
                    merged[field] = entry[field]
                history[i] = merged
                break
        else:
            raise HTTPException(
                409,
                {
                    "error": "cannot write Journal narrative",
                    "reason": (
                        "Evaluation has not created the factual history row for "
                        f"iteration={key[0]}, source={key[1]!r}."
                    ),
                },
            )
        r.history = history

    if body.summary is not None:
        r.summary = body.summary.strip()
    if body.halted_reason is not None:
        r.halted_reason = body.halted_reason.strip()

    if body.status is not None:
        if body.status in TERMINAL_RUN_STATUSES:
            from zevo.engine.observe.run_metrics import incomplete_journal_entries
            incomplete = incomplete_journal_entries(r.history)
            if incomplete:
                raise HTTPException(
                    409,
                    {
                        "error": "cannot finish run with an incomplete Journal",
                        "entries": [
                            {"iteration": iteration, "source": source}
                            for iteration, source in incomplete
                        ],
                        "required_fields": ["action", "result", "analysis", "next"],
                    },
                )
        if body.status == "success":
            tickets = (
                await db.execute(select(Ticket).where(Ticket.run_id == run_id))
            ).scalars().all()
            failed = sorted(t.id for t in tickets if t.status == "failed")
            # The supervisor performs this PATCH during its own final heartbeat,
            # so that one Ticket is necessarily `running`. Every child must be
            # terminal; no other unfinished Ticket is exempt.
            still_running = sorted(
                t.id for t in tickets
                if t.id != r.supervisor_ticket_id
                and t.status in ("queued", "running", "repairing", "awaiting_input", "waiting_external")
            )
            blockers: list[str] = []
            if not (r.summary or "").strip():
                blockers.append("summary is empty")
            if not (r.registry_version_tag or "").strip():
                blockers.append("no retained registry model")
            if int(r.iterations_completed or 0) < 1:
                blockers.append("no trained iteration completed Validation evaluation")
            if failed or still_running or blockers:
                raise HTTPException(
                    409,
                    {
                        "error": "cannot mark run success",
                        "reason": (
                            "A clean success requires a consolidated summary, "
                            "a retained final champion, at least one Validation-"
                            "measured trained iteration, and no failed or unfinished "
                            "child Tickets."
                        ),
                        "failed_tickets": failed,
                        "unfinished_tickets": still_running,
                        "blockers": blockers,
                    },
                )
        r.status = body.status
        if body.status in TERMINAL_RUN_STATUSES:
            r.finished_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(r)
    return _summary(r)


async def _destroy_one_cloud_instance(iid: str, backend_hint: str) -> dict:
    """Destroy one rented cloud instance, trying the hinted backend first and
    then the other only as terminal leak cleanup. A successful device contract
    always records `cloud_backend`; an empty hint can occur only when a failed
    acquisition persisted its handle before the final artifact. Returns
    {instance_id, destroyed, backend, error}."""
    hint = (backend_hint or "").strip().lower()
    order = ["lambda", "vastai"] if hint == "lambda" else ["vastai", "lambda"]

    def _provider(name: str):
        if name == "lambda":
            from zevo.providers.lambda_labs.provider import LambdaCloudProvider
            return LambdaCloudProvider()
        from zevo.providers.vastai.provider import VastAIProvider
        return VastAIProvider()

    last_err = ""
    for name in order:
        try:
            if await _provider(name).destroy_instance(iid):
                return {"instance_id": iid, "destroyed": True, "backend": name, "error": ""}
        except Exception as e:  # missing key / not-found on this backend — try the next
            last_err = str(e)[:120]
    return {"instance_id": iid, "destroyed": False, "backend": "", "error": last_err}


async def _destroy_run_gpu_instances(db: AsyncSession, run_id: str) -> list[dict]:
    """Best-effort: destroy every cloud GPU (Vast.ai OR Lambda) this run rented.

    The infra agent records each box's provider, instance_id, and cloud_backend
    in the `device_info` work_product's `meta`. Cancelling a run
    kills the in-flight subprocesses but does NOT run the agent's `release_remote`
    step, so without this the rented GPU keeps billing. We read the ids straight
    from the DB and destroy each on the recorded backend (falling back only for
    an acquisition that failed before its final contract). Never raises — cleanup must not break the
    cancel response; leaked-instance detection is the backstop.
    """
    from zevo.db import WorkProduct

    rows = (await db.execute(
        select(WorkProduct)
        .join(Ticket, WorkProduct.ticket_id == Ticket.id)
        .where(Ticket.run_id == run_id)
    )).scalars().all()

    # Unique cloud instance ids (+ backend hint) across all infra tickets.
    seen: dict[str, str] = {}
    for wp in rows:
        meta = wp.meta or {}
        if meta.get("provider") == "cloud":
            iid = str(meta.get("instance_id") or "").strip()
            if iid and iid not in seen:
                seen[iid] = str(meta.get("cloud_backend") or "").strip().lower()

    if not seen:
        return []
    return [await _destroy_one_cloud_instance(iid, hint) for iid, hint in seen.items()]


async def _ssh_login_node(
    prefix: str,
    remote_cmd: str,
    connection: SshHost | None = None,
) -> tuple[bool, str]:
    """Run one command on a Slurm login node. Never raises.

    `prefix` picks the credential set -- CLUSTER for allocations Zevo made,
    INSTANCE for the one the user started. The backend mounts the same
    `~/.ssh` as the scheduler, so the same keys work here.
    """
    host = (
        connection.host if connection is not None
        else os.environ.get(f"ZEVO_{prefix}_SSH_HOST", "").strip()
    )
    if not host:
        return False, f"ZEVO_{prefix}_SSH_HOST unset"
    key_path = (
        connection.key_path if connection is not None
        else resolve_ssh_key(f"ZEVO_{prefix}_SSH_KEY")
    )
    password_path = (
        connection.password_path if connection is not None
        else os.environ.get(f"ZEVO_{prefix}_SSH_PASSWORD_FILE", "").strip()
    )
    if password_path:
        key_path = ""
    port = (
        connection.port if connection is not None
        else os.environ.get(f"ZEVO_{prefix}_SSH_PORT", "22") or "22"
    )
    user = (
        connection.username if connection is not None
        else os.environ.get(f"ZEVO_{prefix}_SSH_USER", "root")
    )
    cmd = [
        *ssh_base_args(
            key_path=key_path,
            password_path=password_path,
            port=int(port),
        ),
        f"{user}@{host}", remote_cmd,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode == 0:
            return True, out.decode("utf-8", "ignore").strip()
        return False, err.decode("utf-8", "ignore")[:160]
    except Exception as e:
        return False, str(e)[:160]


async def _scancel_run_slurm_jobs(db: AsyncSession, run_id: str) -> list[dict]:
    """Best-effort backstop for finite Zevo-owned cluster Slurm jobs.

    Direct cloud/instance processes are handled by ``cancel_run_remote_jobs``.
    Never raises.
    """
    from zevo.db import InfraInstance, WorkProduct
    from zevo.engine.run.remote_jobs import (
        _slurm_cli_bootstrap_command,
        _slurm_job_kill_command,
    )

    rows = (await db.execute(
        select(WorkProduct)
        .join(Ticket, WorkProduct.ticket_id == Ticket.id)
        .where(Ticket.run_id == run_id)
    )).scalars().all()
    run_row = await db.get(Run, run_id)
    selected_connection = None
    if run_row is not None and run_row.ssh_host_id:
        selected_connection = await db.get(SshHost, run_row.ssh_host_id)

    owned: dict[str, str] = {}  # finite JOBID -> exact Train/Inference ticket

    stage_rows = (await db.execute(select(InfraInstance).where(
        InfraInstance.run_id == run_id,
        InfraInstance.provider == "cluster",
        InfraInstance.instance_id != "",
    ))).scalars().all()
    for row in stage_rows:
        if row.ticket_id:
            owned[row.instance_id] = row.ticket_id
    for wp in rows:
        meta = wp.meta or {}
        jid = str(meta.get("instance_id") or "").strip()
        if not jid:
            continue
        provider = meta.get("provider")
        if provider == "cluster" and jid not in owned:
            owned[jid] = str(wp.ticket_id or "")

    results: list[dict] = []
    for j, ticket_id in owned.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", j):
            results.append({
                "jobid": j, "scancelled": False,
                "error": "unsafe Slurm job id in bookkeeping",
            })
            continue
        if ticket_id and not re.fullmatch(r"[A-Za-z0-9_-]+", ticket_id):
            results.append({
                "jobid": j, "scancelled": False,
                "error": "unsafe Ticket id in Slurm bookkeeping",
            })
            continue
        expected_name = f"zevo-{ticket_id}" if ticket_id else ""
        if expected_name:
            remote_command = _slurm_job_kill_command(j, ticket_id)
        else:
            remote_command = _slurm_cli_bootstrap_command() + f"scancel {j}"
        ok, err = await _ssh_login_node(
            "CLUSTER", remote_command, selected_connection,
        )
        results.append({"jobid": j, "scancelled": ok, "error": "" if ok else err})

    return results


async def _mark_run_instances_released(db: AsyncSession, run_id: str) -> int:
    """Mark every open InfraInstance row for the run as released, so the cost
    accounting clock stops and `hardware` / leak-detection are accurate.
    Provider-agnostic: this only closes OUR bookkeeping — the actual Vast.ai
    destroy / Slurm scancel is done separately (cloud/cluster only); a fixed
    `instance` host is never shut down, just clock-stopped.

    The GPU LEASES go back here too. A fixed `instance` host is never torn down,
    but the cards this run held on it must return to the pool or the next
    run cannot be placed on hardware that is now idle. This helper is the one
    exit every cancel and delete path already shares, so putting it here covers
    all of them; the reconciler handles runs that end on their own."""
    from zevo.db import GpuLease, InfraInstance
    await db.execute(
        sa_update(GpuLease)
        .where(GpuLease.run_id == run_id, GpuLease.released_at.is_(None))
        .values(
            released_at=datetime.now(timezone.utc),
            release_reason="run cancelled",
        )
    )
    rows = (await db.execute(
        select(InfraInstance).where(
            InfraInstance.run_id == run_id, InfraInstance.released_at.is_(None)
        )
    )).scalars().all()
    now = datetime.now(timezone.utc)
    for inst in rows:
        inst.status = "released"
        inst.released_at = now
        if not inst.release_reason:
            inst.release_reason = "run cancelled"
    # Commit unconditionally: the lease release above is real work even when
    # this run never registered an InfraInstance row.
    await db.commit()
    return len(rows)


async def _finish_run_heartbeats(
    db: AsyncSession,
    run_id: str,
    *,
    reason: str,
    exit_code: int = 130,
) -> int:
    """Close unfinished local activations when their Run is cancelled.

    A runner normally closes its own heartbeat after observing the Ticket
    cancellation. If the runner itself was killed or restarted, that final
    write never arrives and the UI otherwise reports a permanently live
    activation on a terminal Run. This update is idempotent and scoped through
    the Run's Ticket ids.
    """
    ticket_ids = select(Ticket.id).where(Ticket.run_id == run_id)
    result = await db.execute(
        sa_update(HeartbeatRun)
        .where(
            HeartbeatRun.ticket_id.in_(ticket_ids),
            HeartbeatRun.finished_at.is_(None),
        )
        .values(
            finished_at=datetime.now(timezone.utc),
            exit_code=exit_code,
            error_message=reason,
        )
    )
    return int(result.rowcount or 0)


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    body: CancelWeightsPolicy | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Cancel an active Run, regardless of launch mode.

    Marks every non-terminal Ticket on it as `cancelled`. The scheduler's
    runner flusher polls ticket.status every ~3s and SIGTERMs the agent
    subprocess for any ticket that flipped to `cancelled`, so in-flight
    heartbeats die within a few seconds. The orchestrator's supervisor ticket
    is also marked cancelled, preventing it from emitting more children.

    The body says what happens to the trained weights. `discard` closes the
    Run and destroys its GPU here and now. `download` / `hf` only record the
    request: the Run stays `running` until the daemon has copied the champion
    checkpoint off the box (engine/run/cancel_rescue.py), then it is closed
    and the box reaped like any other terminal run.
    """
    policy = body or CancelWeightsPolicy()
    r = (await db.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"run {run_id} not found")
    if r.status in TERMINAL_RUN_STATUSES:
        # Already terminal can still mean a remote trainer survived the local
        # SSH/Agent process.  Re-run exact Ticket cleanup before provider
        # lifecycle cleanup; this makes Cancel a useful leak-recovery action.
        tickets = (await db.execute(
            select(Ticket).where(Ticket.run_id == run_id)
        )).scalars().all()
        from zevo.engine.run.remote_jobs import cancel_run_remote_jobs
        remote_jobs_cancelled = await cancel_run_remote_jobs(db, tickets)
        gpus_destroyed = await _destroy_run_gpu_instances(db, run_id)
        slurm_scancelled = await _scancel_run_slurm_jobs(db, run_id)
        heartbeats_finished = await _finish_run_heartbeats(
            db,
            run_id,
            reason="cancelled by user (terminal Run cleanup)",
        )
        await _mark_run_instances_released(db, run_id)
        return {
            "status": r.status,
            "run_id": run_id,
            "note": "already terminal; cleaned any leaked Ticket task and provider resource",
            "remote_jobs_cancelled": remote_jobs_cancelled,
            "gpus_destroyed": gpus_destroyed,
            "slurm_scancelled": slurm_scancelled,
            "heartbeats_finished": heartbeats_finished,
        }

    if r.cancel_requested_at is not None:
        raise HTTPException(409, "cancel already requested; weights are being rescued")

    # Everything that can refuse does so BEFORE any work is killed.
    if policy.weights != "discard":
        from zevo.engine.run.cancel_rescue import hf_token, prepare_rescue_dir
        try:
            rescue_dir = prepare_rescue_dir(r, policy)
        except OSError as e:
            raise HTTPException(400, f"cannot write the rescue folder: {e}")
        if policy.weights == "hf" and not hf_token():
            raise HTTPException(400, "HF_TOKEN is not set in Settings")

    # Cancel every non-terminal ticket on this run. The runner's flusher
    # picks up `cancelled` status within ~3s and kills the subprocess.
    tickets = (await db.execute(
        select(Ticket).where(
            Ticket.run_id == run_id,
            Ticket.status.in_(["queued", "running", "repairing", "awaiting_input", "waiting_external"]),
        )
    )).scalars().all()
    cancelled_ticket_ids = []
    for t in tickets:
        t.status = "cancelled"
        t.error_message = "cancelled by user (run cancelled)"
        cancelled_ticket_ids.append(t.id)

    heartbeats_finished = await _finish_run_heartbeats(
        db,
        run_id,
        reason="cancelled by user (run cancelled)",
    )

    if policy.weights != "discard":
        # Leave the Run open: the reconciler skips cancel-requested runs, so the
        # box survives until the rescue task has the checkpoint and closes it.
        r.cancel_requested_at = datetime.now(timezone.utc)
        r.cancel_policy = policy.model_dump(mode="json")
        r.halted_reason = "cancel requested; keeping the champion checkpoint first"
        await db.commit()
        from zevo.engine.run.remote_jobs import cancel_run_remote_jobs
        remote_jobs_cancelled = await cancel_run_remote_jobs(db, tickets)
        from zevo.engine.observe.audit import audit
        await audit(
            db, event_type="run.cancel", target_type="run", target_id=run_id,
            summary=f"cancel requested ({policy.weights}); {len(cancelled_ticket_ids)} tickets cancelled",
            after={"tickets_cancelled": cancelled_ticket_ids,
                   "remote_jobs_cancelled": remote_jobs_cancelled,
                   "heartbeats_finished": heartbeats_finished,
                   "cancel_policy": r.cancel_policy},
        )
        return {
            "status": "cancelling",
            "run_id": run_id,
            "tickets_cancelled": cancelled_ticket_ids,
            "remote_jobs_cancelled": remote_jobs_cancelled,
            "heartbeats_finished": heartbeats_finished,
            "rescue_dir": str(rescue_dir),
            "note": "weights are being copied off the box; the run closes and the GPU is released when that finishes",
        }

    r.status = "cancelled"
    r.halted_reason = "cancelled by user"
    r.finished_at = datetime.now(timezone.utc)
    r.cancel_policy = policy.model_dump(mode="json")
    r.cancel_outcome = {"weights": "discard", "finished_at": r.finished_at.isoformat()}
    await db.commit()

    # First terminate Ticket-owned work on the remote host. Killing only the
    # local Agent/SSH client can orphan a Python process that keeps consuming
    # the assigned GPU. This does not destroy the device or a user allocation.
    from zevo.engine.run.remote_jobs import cancel_run_remote_jobs
    remote_jobs_cancelled = await cancel_run_remote_jobs(db, tickets)

    # Tear down any GPU this run was holding so cancelling never leaves a Vast.ai
    # instance billing or a Slurm job squatting on cluster GPUs. Best-effort.
    gpus_destroyed = await _destroy_run_gpu_instances(db, run_id)
    slurm_scancelled = await _scancel_run_slurm_jobs(db, run_id)
    await _mark_run_instances_released(db, run_id)

    from zevo.engine.observe.audit import audit
    await audit(
        db, event_type="run.cancel", target_type="run", target_id=run_id,
        summary=f"cancelled {len(cancelled_ticket_ids)} non-terminal tickets",
        after={"tickets_cancelled": cancelled_ticket_ids,
               "gpus_destroyed": gpus_destroyed,
               "remote_jobs_cancelled": remote_jobs_cancelled,
               "slurm_scancelled": slurm_scancelled,
               "heartbeats_finished": heartbeats_finished},
    )

    return {
        "status": "cancelled",
        "run_id": run_id,
        "tickets_cancelled": cancelled_ticket_ids,
        "gpus_destroyed": gpus_destroyed,
        "remote_jobs_cancelled": remote_jobs_cancelled,
        "slurm_scancelled": slurm_scancelled,
        "heartbeats_finished": heartbeats_finished,
        "note": "scheduler and Ticket-scoped remote tasks are stopped; infrastructure lifecycle cleanup follows provider policy",
    }


@router.delete("/runs/{run_id}")
async def delete_run(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete a run and everything attached to it (tickets, heartbeats,
    work products, messages, results, and execution events — all cascade at
    the DB level).

    If the run is still active, its non-terminal tickets are flipped to
    `cancelled` first (so the scheduler SIGTERMs any in-flight subprocess) and
    any rented GPU is torn down, so deleting never leaves a Vast.ai instance
    billing. This is destructive and irreversible.
    """
    r = (await db.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"run {run_id} not found")

    if r.status not in TERMINAL_RUN_STATUSES:
        tickets = (await db.execute(
            select(Ticket).where(
                Ticket.run_id == run_id,
                Ticket.status.in_(["queued", "running", "repairing", "awaiting_input", "waiting_external"]),
            )
        )).scalars().all()
        for t in tickets:
            t.status = "cancelled"
            t.error_message = "cancelled by user (run deleted)"
        await db.commit()
    else:
        tickets = (await db.execute(
            select(Ticket).where(Ticket.run_id == run_id)
        )).scalars().all()

    # Tear down any GPU this run was holding before the rows vanish (reads ids
    # from work_products.meta). Best-effort; never blocks the delete.
    from zevo.engine.run.remote_jobs import cancel_run_remote_jobs
    remote_jobs_cancelled = await cancel_run_remote_jobs(db, tickets)
    gpus_destroyed = await _destroy_run_gpu_instances(db, run_id)
    slurm_scancelled = await _scancel_run_slurm_jobs(db, run_id)
    await _mark_run_instances_released(db, run_id)

    from zevo.engine.observe.audit import audit
    await audit(
        db, event_type="run.delete", target_type="run", target_id=run_id,
        summary=f"deleted run {run_id[:8]}",
        after={"gpus_destroyed": gpus_destroyed,
               "remote_jobs_cancelled": remote_jobs_cancelled,
               "slurm_scancelled": slurm_scancelled},
    )

    # The models this run registered go with it. The FK is ON DELETE SET NULL,
    # which would leave rows naming a run that no longer exists: invisible on
    # the Models page (it filters to live runs) but still counted by
    # `registry list` and anything reading the table directly. A model
    # whose run, tickets and weights are gone is not a model.
    registry_deleted = (await db.execute(
        sa_delete(RegistryModel).where(RegistryModel.run_id == run_id)
    )).rowcount or 0
    # DB-level ondelete=CASCADE on tickets.run_id (and the ticket children)
    # removes the whole tree.
    await db.execute(sa_delete(Run).where(Run.id == run_id))
    await db.commit()
    return {"deleted": True, "run_id": run_id, "registry_deleted": registry_deleted,
            "gpus_destroyed": gpus_destroyed,
            "remote_jobs_cancelled": remote_jobs_cancelled,
            "slurm_scancelled": slurm_scancelled}


async def _check_selected_ssh_host(
    db: AsyncSession, ssh_host_id: str, resolved_gpu_provider: str,
) -> None:
    """A selected Cluster/Instance connection must exist, be verified, and have
    the same category as the Run. The runner re-checks this defense-in-depth."""
    if resolved_gpu_provider in ("cluster", "instance") and ssh_host_id.strip():
        _sid = ssh_host_id.strip()
        _box = None
        if _sid:
            _box = (await db.execute(
                select(SshHost).where(SshHost.id == _sid)
            )).scalar_one_or_none()
        if _box is None:
            raise HTTPException(422, "Selected SSH connection was not found.")
        if _box.status != "verified":
            raise HTTPException(
                422,
                f"SSH connection {_box.label or _box.host!r} isn't verified yet — "
                "verify it under Settings before running on it.",
            )
        if _box.category != resolved_gpu_provider:
            raise HTTPException(
                422,
                f"SSH connection {_box.label or _box.host!r} is {_box.category}, "
                f"not {resolved_gpu_provider}.",
            )


async def _create_auto_run(
    db: AsyncSession, body: CreateRunRequest, *, task_name: str, run_name: str,
    task_row: Any,
) -> CreateRunResponse:
    """`mode="auto"`: create the Run from the objective and queue scoping.

    What this deliberately does NOT do, compared with the pipeline path:

      * no `scoring_asset_errors` — there is no scoring contract yet;
      * no `_settle_splits` — there is no Test set to split yet;
      * no supervisor Ticket — the Orchestrator has nothing to plan until the
        contract exists, and Data payloads cannot be stamped without a
        settled Validation set.

    Instead the Run is persisted (`mode="auto"`, `scoring_settled=False`, empty
    metric and holdout) together with ONE engine-owned Data Ticket,
    `scope-<run8>-001` (`operation="scope_problem"`), in a single commit, and
    that Ticket is woken. When it succeeds the engine settles the contract,
    creates the supervisor with a `run_created` trigger and wakes it — from
    which point the Run proceeds exactly as a full_pipeline Run would.
    """
    from zevo.api.routers.ui.tasks import compose_objective
    from zevo.engine.persistence import create_run as persist_create
    from zevo.engine.run.scoping import new_scoping_ticket
    from zevo.engine.run.wakeup import queue_wakeup

    auto_request = body.user_request
    if not isinstance(auto_request, AutoUserRequest):  # pragma: no cover - pinned by the request validator
        raise HTTPException(400, "auto mode requires user_request.task_objective")
    if task_row is not None:
        raise HTTPException(
            400,
            f"task {task_name!r} is predefined and already owns a scoring contract; "
            "auto mode is for a new problem stated as an objective. Launch the "
            "predefined task with full_pipeline, or choose a new task name.",
        )
    task_objective = auto_request.task_objective.strip()
    if not task_objective:
        raise HTTPException(400, "task_objective is required and must describe the task.")

    compute = await _resolve_run_compute(db, body)
    resolved_gpu_provider = compute.provider
    await _check_selected_ssh_host(db, compute.ssh_host_id, resolved_gpu_provider)

    from zevo.contracts.training_methods import (
        method_config_errors,
        normalize_method_config,
    )
    method_config = normalize_method_config(auto_request.method_config)
    method_errors = method_config_errors(
        auto_request.training_method,
        method_config,
        require_dependencies=True,
    )
    if method_errors:
        raise HTTPException(400, "; ".join(method_errors))

    # Auto replaces only Test setup. The optimization side keeps the same
    # optional data/model/method ownership as Standard.
    agent_objective = compose_objective(
        task_objective,
        dataset=auto_request.dataset,
        base_model=auto_request.base_model,
        training_method=auto_request.training_method,
    )
    # commit=False: the Run and its scoping Ticket land in ONE transaction, as
    # the pipeline path does for the Run + supervisor. Empty metric and the
    # column default direction are placeholders behind `scoring_settled=False`.
    run = await persist_create(
        db,
        task_name=task_name,
        run_name=run_name,
        task_objective=task_objective,
        agent_objective=agent_objective,
        metric="",
        metric_direction="max",
        validation_metric="",
        validation_metric_direction="max",
        commit=False,
    )
    run.scoring_settled = False
    run.mode = "auto"
    run.iteration_budget = body.iteration_budget if body.iteration_budget is not None else 0
    run.stop_threshold = body.stop_threshold
    run.max_cost_usd = float(body.max_cost_usd or 0.0)
    run.max_runtime_hours = float(body.max_runtime_hours or 0.0)
    run.max_queue_wait_hours = (
        24.0 if body.max_queue_wait_hours is None else float(body.max_queue_wait_hours)
    )
    run.generation_backend = body.generation_backend or "vllm"
    run.num_gpus = max(0, int(body.num_gpus or 0))
    run.gpu_provider = resolved_gpu_provider
    run.ssh_host_id = (
        (compute.ssh_host_id or None)
        if resolved_gpu_provider in ("cluster", "instance") else None
    )
    run.cloud_backend = (
        compute.cloud_backend if resolved_gpu_provider == "cloud" else ""
    )
    run.setting_id = None
    run.setting_name = ""
    # This is the Auto request's durable optimization snapshot while scoping
    # owns only the evaluation contract. Exact values are pins; query values
    # remain advisory when their corresponding exact value is blank.
    run.decision_pins = {
        key: value
        for key, value in {
            "dataset": auto_request.dataset,
            "dataset_split": auto_request.dataset_split,
            "dataset_config": auto_request.dataset_config,
            "data_query": auto_request.data_query,
            "base_model": auto_request.base_model,
            "model_query": auto_request.model_query,
            "training_method": auto_request.training_method,
            "method_query": auto_request.method_query,
            "method_config": method_config,
        }.items()
        if value not in ("", {}, None)
    }
    run.model_lineages = {}
    run.customizations = {}
    run.holdout = {}
    run.supervisor_ticket_id = ""
    await db.flush()

    scope = new_scoping_ticket(run, auto_request)
    db.add(scope)
    run.status = "running"
    await db.commit()
    await db.refresh(scope)

    await queue_wakeup(
        db, agent_id="data", ticket_id=scope.id,
        source="assignment",
        reason=f"auto run created (task_name={task_name!r}): scope the scoring contract",
    )

    from zevo.engine.observe.audit import audit
    await audit(
        db, event_type="run.create", target_type="run", target_id=run.id,
        summary=run.task_objective[:200],
        after={
            "task_name": task_name,
            "run_name": run.run_name or "",
            "mode": "auto",
            "scoping_ticket_id": scope.id,
            "max_cost_usd": float(run.max_cost_usd or 0.0),
            "max_runtime_hours": float(run.max_runtime_hours or 0.0),
            "iteration_budget": run.iteration_budget,
            "stop_threshold": run.stop_threshold,
            "gpu_provider": run.gpu_provider,
        },
    )
    return CreateRunResponse(run_id=run.id, status="running")


@router.post("/runs", response_model=CreateRunResponse)
async def create_run(
    body: CreateRunRequest,
    db: AsyncSession = Depends(get_db),
) -> CreateRunResponse:
    """Create the Run and its one reactive supervisor Ticket.

    The Ticket auto-enqueues a wakeup. The Orchestrator normally advances one
    semantic transition per heartbeat; the DAG expansion endpoint is the sole
    multi-node transition and independently queues the Run Setup Infrastructure
    and Data prerequisites. POST returns immediately with the run_id; the UI subscribes to
    /ws/runs/{id} or polls /runs/{id} for progress.
    """
    from zevo.api.routers.ui.tasks import task_to_user_request
    from zevo.engine.persistence import create_run as persist_create
    from zevo.db import Task
    from zevo.engine.run.wakeup import queue_wakeup
    from zevo.holdout_storage import protect_assets
    from fastapi.concurrency import run_in_threadpool

    # Every run is named. A predefined task_name executes that package as-is; an
    # unrecognised one is a custom task and must carry its own user_request.
    # Without this, custom runs landed with task_name="" and showed up as
    # "(unnamed)" everywhere they were listed or grouped.
    task_name = (body.task_name or "").strip()
    if not task_name:
        raise HTTPException(400, "task_name is required. Use a predefined task, or name your own.")
    run_name = (body.run_name or "").strip()
    if not run_name:
        raise HTTPException(400, "run_name is required and must name this execution.")

    # The catalogue is a table now, so a predefined name is resolved with a
    # lookup rather than a dict in code — which is what lets the user add to and
    # delete from it at runtime.
    task_row = await db.get(Task, task_name)
    if body.mode == "auto":
        # Objective-only launch: no scoring contract to check or settle here.
        # The Data agent's scoping Ticket derives it and the engine settles it
        # afterwards (zevo.engine.run.scoping); everything below this line is
        # the full/customized pipeline creation path, unchanged.
        return await _create_auto_run(
            db, body, task_name=task_name, run_name=run_name, task_row=task_row,
        )
    user_request = body.user_request
    if user_request is None:
        if task_row is None:
            raise HTTPException(
                404,
                f"task {task_name!r} is not predefined — send a user_request to define it, "
                f"or pick one of the predefined tasks from GET /tasks.",
            )
        user_request = task_to_user_request(task_row)
    # A Task's semantic inference protocol is authoritative regardless of
    # whether the launch came from its saved Setting or a complete request
    # assembled by another client.  Resolve it once, then project it into the
    # existing inference mapping so every baseline/checkpoint reuses the exact
    # instruction, output format and answer parser without another form.
    from zevo.contracts.task_protocol import (
        TaskInferenceProtocol,
        default_task_inference_protocol,
    )

    requested_protocol = user_request.inference_protocol
    if task_row is not None:
        protocol = TaskInferenceProtocol.model_validate(
            task_row.inference_protocol
            or default_task_inference_protocol(
                task_row.task_objective or "", task_row.metric,
            ).model_dump()
        )
        if requested_protocol is not None and requested_protocol != protocol:
            raise HTTPException(
                409,
                "the launch inference protocol conflicts with the predefined "
                "Task's frozen protocol",
            )
    else:
        protocol = requested_protocol or default_task_inference_protocol(
            user_request.task_objective, user_request.metric,
        )
    protocol_mapping = protocol.inference_mapping()
    supplied_mapping = dict(user_request.inference_config or {})
    conflicts = sorted(
        key for key, value in protocol_mapping.items()
        if key in supplied_mapping and supplied_mapping[key] != value
    )
    if conflicts:
        raise HTTPException(
            409,
            "the launch inference mapping conflicts with the Task's frozen "
            f"inference protocol: {', '.join(conflicts)}",
        )
    # Keep Setting-owned inference knobs separate. The Task protocol is merged
    # only when an Inference work order is stamped; otherwise saving a Setting
    # before its first Run and matching it at launch would produce two
    # identities for the same experiment.
    user_request = user_request.model_copy(update={"inference_protocol": protocol})
    from zevo.contracts.training_methods import (
        method_config_errors,
        normalize_method_config,
    )
    user_request = user_request.model_copy(update={
        "method_config": normalize_method_config(user_request.method_config),
    })
    derived_validation = not user_request.validation_set.strip()
    user_request = inherit_test_validation_contract(user_request)

    # Test bytes must not share a path with anything on the optimization lane.
    # After this check, managed Test assets are moved beneath the private root;
    # the logical paths stay unchanged for Task/UI compatibility.
    test_assets = {
        str(v).strip() for v in (
            user_request.test_set,
            user_request.test_sample_submission,
        ) if str(v).strip()
    }
    optimization_assets = {
        str(v).strip() for v in (user_request.dataset,) if str(v).strip()
    }
    if not derived_validation:
        optimization_assets.update({
            str(v).strip() for v in (
                user_request.validation_set,
                user_request.validation_sample_submission,
                user_request.validation_evaluation_script,
            ) if str(v).strip()
        })
    overlap = sorted(test_assets & optimization_assets)
    if overlap:
        raise HTTPException(
            400,
            "Test assets must be separate from Training and Validation assets; "
            f"shared path: {overlap[0]}",
        )
    protected_test, protected_sample = await run_in_threadpool(
        protect_assets,
        user_request.test_set,
        user_request.test_sample_submission,
    )
    user_request = user_request.model_copy(update={
        "test_set": protected_test,
        "test_sample_submission": protected_sample,
    })
    if derived_validation:
        user_request = inherit_test_validation_contract(user_request)
    if user_request.metric_type == "custom":
        from zevo.evaluator_storage import freeze_evaluator
        try:
            frozen_evaluator, evaluator_sha256 = await run_in_threadpool(
                freeze_evaluator, user_request.evaluation_script,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        user_request = user_request.model_copy(update={
            "evaluation_script": frozen_evaluator,
            "evaluator_sha256": evaluator_sha256,
        })
    else:
        user_request = user_request.model_copy(update={
            "evaluation_script": "", "evaluator_sha256": "",
        })
    if derived_validation:
        # Re-resolve after Test's custom evaluator has been frozen so both
        # lanes point at exactly the same immutable bytes and digest.
        user_request = inherit_test_validation_contract(user_request)
    elif user_request.validation_metric_type == "custom":
        from zevo.evaluator_storage import freeze_evaluator
        try:
            validation_evaluator, validation_evaluator_sha256 = await run_in_threadpool(
                freeze_evaluator, user_request.validation_evaluation_script,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        user_request = user_request.model_copy(update={
            "validation_evaluation_script": validation_evaluator,
            "validation_evaluator_sha256": validation_evaluator_sha256,
        })
    else:
        user_request = user_request.model_copy(update={
            "validation_evaluation_script": "",
            "validation_evaluator_sha256": "",
        })
    method_errors = method_config_errors(
        user_request.training_method,
        user_request.method_config,
        require_dependencies=True,
    )
    if method_errors:
        raise HTTPException(400, "; ".join(method_errors))
    scoring_errors = scoring_asset_errors(user_request)
    if scoring_errors:
        raise HTTPException(400, "; ".join(scoring_errors))
    compute = await _resolve_run_compute(db, body)
    resolved_gpu_provider = compute.provider
    await _check_selected_ssh_host(db, compute.ssh_host_id, resolved_gpu_provider)

    # Saving the first setting of a newly named task also writes down the task
    # itself. Previously the UI offered "save this setting" for a custom task,
    # but this branch silently skipped the setting because there was no Task
    # row to own it; after the run, neither reusable object existed. A saved
    # custom launch already carries the complete problem definition, so create
    # the Task and its first Setting in the same transaction.
    if task_row is None and body.save_setting:
        missing_task_assets = []
        if not user_request.test_set.strip():
            missing_task_assets.append("test_set")
        if not user_request.test_answer_fields:
            missing_task_assets.append("test_answer_fields")
        if not user_request.test_sample_submission.strip():
            missing_task_assets.append("test_sample_submission")
        if missing_task_assets:
            raise HTTPException(
                400,
                "saving a new task requires " + ", ".join(missing_task_assets),
            )
        task_row = Task(
            name=task_name,
            task_objective=user_request.task_objective.strip(),
            test_set=user_request.test_set.strip(),
            test_answer_fields=[
                str(c).strip() for c in user_request.test_answer_fields if str(c).strip()
            ],
            test_sample_submission=user_request.test_sample_submission.strip(),
            metric_type=user_request.metric_type,
            evaluation_script=user_request.evaluation_script.strip(),
            evaluator_sha256=user_request.evaluator_sha256,
            metric=user_request.metric.strip(),
            metric_direction=user_request.metric_direction,
        )
        db.add(task_row)

    # A task states the PROBLEM; the sentences that say which model, which
    # method and whether the training data is handed over describe the SETTING,
    # and this run's setting is the one in front of us — not the one whichever
    # task row was named happens to default to. Composing here covers both ways
    # in: launching by task name alone, and launching with fields edited.
    task_objective = (
        task_row.task_objective if task_row is not None
        else user_request.task_objective
    ).strip()
    if not task_objective:
        raise HTTPException(400, "task_objective is required and must describe the task.")
    # A catalogue Task supplies Test defaults, while the submitted UserRequest
    # is the exact scoring contract for this Run. This lets one fixed held-out
    # dataset be reported through a different built-in metric or a frozen
    # custom evaluator without editing the reusable Task. Historical Runs keep
    # their effective metric and direction on the Run/holdout snapshot.
    metric = user_request.metric
    metric_direction = user_request.metric_direction
    validation_metric = user_request.validation_metric
    validation_metric_direction = user_request.validation_metric_direction
    # One clean objective everywhere the task problem is represented. The
    # separately composed Agent objective is the only place Setting clauses
    # belong. This applies equally to catalogue and newly named custom tasks.
    user_request = user_request.model_copy(update={
        "task_objective": task_objective,
        "metric": metric,
        "metric_direction": metric_direction,
    })
    from zevo.api.routers.ui.tasks import compose_objective
    agent_objective = compose_objective(
        task_objective,
        dataset=user_request.dataset,
        base_model=user_request.base_model,
        training_method=user_request.training_method,
    )

    # Resolve run-level values once. The same effective setting is persisted in
    # TaskSetting and on Run; computing it before either row exists avoids
    # reading an uninitialized Run or persisting different effective limits.
    resolved_iteration_budget = (
        body.iteration_budget if body.iteration_budget is not None else 0
    )
    resolved_stop_threshold = body.stop_threshold
    resolved_max_cost_usd = (
        body.max_cost_usd
        if body.max_cost_usd is not None
        else 0.0
    )
    # A deadline belongs only to this execution. Do not include it in Setting
    # identity or persistence below: relaunching the same experiment tomorrow
    # may have a different amount of time available.
    resolved_max_runtime_hours = float(body.max_runtime_hours or 0.0)
    resolved_max_queue_wait_hours = (
        24.0 if body.max_queue_wait_hours is None
        else float(body.max_queue_wait_hours)
    )
    resolved_generation_backend = body.generation_backend or "vllm"

    # Resolve one exact Setting identity for this immutable launch snapshot.
    # Runs point at that row for naming provenance; the full request remains on
    # the Run/supervisor Ticket, so editing or deleting a Setting never rewrites
    # history.
    selected_setting = None
    if body.setting_id and task_row is None:
        raise HTTPException(
            400,
            f"setting {body.setting_id!r} cannot belong to unknown task {task_name!r}.",
        )
    if task_row is not None:
        from zevo.api.routers.ui.tasks import setting_identity
        from zevo.db import TaskSetting

        # The caps are on the RUN, not on `user_request`, and they are part of
        # the identity — launching this configuration under a different budget
        # is a different setting to save, not the same one relaunched.
        key = setting_identity({
            **user_request.model_dump(),
            "iteration_budget": resolved_iteration_budget,
            "max_cost_usd": resolved_max_cost_usd,
            "stop_threshold": resolved_stop_threshold,
        })
        known = (await db.execute(
            select(TaskSetting).where(TaskSetting.task_name == task_name)
        )).scalars().all()
        exact = next((s for s in known if setting_identity(s) == key), None)
        if body.setting_id:
            chosen = next((s for s in known if s.id == body.setting_id), None)
            if chosen is None:
                raise HTTPException(
                    400, f"setting {body.setting_id!r} does not belong to task {task_name!r}.",
                )
            if setting_identity(chosen) != key:
                raise HTTPException(
                    409,
                    "the selected setting no longer matches the launch fields; "
                    "reselect it or save the edited configuration as a new setting.",
                )
            selected_setting = chosen
        elif exact is not None:
            selected_setting = exact
        elif body.save_setting:
            from zevo.api.routers.ui.tasks import (
                _clean_setting_name, _name_taken, _next_setting_name,
            )
            wanted = _clean_setting_name(body.setting_name)
            if _name_taken(wanted, known):
                raise HTTPException(
                    409,
                    f"another setting on {task_name!r} is already called {wanted!r}.",
                )
            selected_setting = TaskSetting(
                task_name=task_name,
                name=wanted or _next_setting_name(known),
                dataset=user_request.dataset or "",
                dataset_split=user_request.dataset_split or "",
                dataset_config=user_request.dataset_config or "",
                validation_set=user_request.validation_set or "",
                validation_split=user_request.validation_split or "",
                validation_config=user_request.validation_config or "",
                validation_answer_fields=list(user_request.validation_answer_fields or []),
                validation_sample_submission=user_request.validation_sample_submission or "",
                validation_metric_type=user_request.validation_metric_type,
                validation_metric=user_request.validation_metric,
                validation_metric_direction=user_request.validation_metric_direction,
                validation_evaluation_script=(
                    user_request.validation_evaluation_script or ""
                ),
                validation_evaluator_sha256=(
                    user_request.validation_evaluator_sha256 or ""
                ),
                base_model=user_request.base_model or "",
                training_method=user_request.training_method or "",
                method_config=dict(user_request.method_config or {}),
                prompt_framing=user_request.prompt_framing or "",
                system_prompt=user_request.system_prompt or "",
                loss_objective_config=dict(user_request.loss_objective_config or {}),
                inference_config=dict(user_request.inference_config or {}),
                decoding_config=dict(user_request.decoding_config or {}),
                data_query=getattr(user_request, "data_query", "") or "",
                model_query=user_request.model_query or "",
                method_query=user_request.method_query or "",
                iteration_budget=max(0, resolved_iteration_budget),
                max_cost_usd=max(0.0, resolved_max_cost_usd),
                stop_threshold=resolved_stop_threshold,
            )
            db.add(selected_setting)
            await db.flush()

    # commit=False: the Run is flushed (id assigned) but NOT committed, so the
    # rest of creation — split settlement, the supervisor Ticket — runs in the
    # same transaction and commits once at the end. If any later step raises
    # (e.g. a recoverable 400 deriving the validation split), get_db's session
    # context rolls the whole thing back and no orphan `planning` Run survives.
    run = await persist_create(
        db,
        task_name=task_name,
        run_name=run_name,
        task_objective=task_objective,
        agent_objective=agent_objective,
        metric=metric,
        metric_direction=metric_direction,
        validation_metric=validation_metric,
        validation_metric_direction=validation_metric_direction,
        commit=False,
    )

    # Stamp the already-resolved budget and target so every consumer sees the
    # exact values that identified the saved setting.
    run.iteration_budget = resolved_iteration_budget
    run.stop_threshold = resolved_stop_threshold
    run.metric = metric
    run.metric_direction = metric_direction
    run.validation_metric = validation_metric
    run.validation_metric_direction = validation_metric_direction
    run.setting_id = selected_setting.id if selected_setting is not None else None
    run.setting_name = selected_setting.name if selected_setting is not None else ""
    run.max_cost_usd = float(resolved_max_cost_usd or 0.0)
    run.max_runtime_hours = resolved_max_runtime_hours
    run.max_queue_wait_hours = resolved_max_queue_wait_hours
    # Generation backend: the explicit run-envelope value wins, otherwise
    # vllm. Tasks and Settings do not own runtime. The runner stamps it onto
    # inference/train ticket inputs, so workers do not choose independently.
    run.generation_backend = resolved_generation_backend
    run.num_gpus = max(0, int(body.num_gpus or 0))
    run.gpu_provider = resolved_gpu_provider
    # NULL (not '') when no profile is selected: ssh_host_id is a real FK to
    # ssh_hosts now, and '' could never satisfy it. All readers already treat
    # falsy/None as "use the deployment-level fallback".
    run.ssh_host_id = (
        (compute.ssh_host_id or None)
        if resolved_gpu_provider in ("cluster", "instance") else None
    )
    run.cloud_backend = compute.cloud_backend if resolved_gpu_provider == "cloud" else ""
    run.mode = body.mode
    # Ownership is determined only by what the user supplied at launch. These
    # pins do not encode an autonomy level and do not acquire defaults later.
    # Any omitted field remains Specialist-owned and is recorded in its YAML.
    run.decision_pins = {
        key: value
        for key, value in {
            "dataset": user_request.dataset,
            "dataset_split": user_request.dataset_split,
            "dataset_config": user_request.dataset_config,
            "data_query": user_request.data_query,
            "base_model": user_request.base_model,
            "training_method": user_request.training_method,
            "method_config": dict(user_request.method_config or {}),
            "prompt_framing": user_request.prompt_framing,
            "system_prompt": user_request.system_prompt,
            "loss_objective_config": dict(user_request.loss_objective_config or {}),
            "inference_config": dict(user_request.inference_config or {}),
            "inference_protocol": protocol.model_dump(mode="json"),
            "decoding_config": dict(user_request.decoding_config or {}),
        }.items()
        if value not in ("", {}, None)
    }
    run.model_lineages = {}
    if run.mode == "customized_pipeline":
        run.customizations = (body.customizations or RunCustomizations()).model_dump()
    else:
        run.customizations = {}
    # No commit here: the Run stays pending so split settlement + the supervisor
    # Ticket join the same transaction. A single commit lands them all (below).
    await db.flush()

    # Create the ONE supervisor ticket for this run. The orchestrator is a
    # single agent with a single work order: this same orchestrate-...-001 is
    # re-woken on every child completion, across every Zevo model-improvement loop, so
    # all its heartbeats + emitted children stay under one ticket (see
    # _maybe_wake_supervisor, which reuses this id and just refreshes the
    # payload's trigger/history each wake; iteration belongs to the Ticket).
    sup_id = f"orchestrate-{run.id[:8]}-001"

    # Settle the three-way split before the orchestrator sees anything. The
    # run tunes on validation and is judged on the remaining Test rows. When
    # Validation is absent, both are settled together by carving 20% from Test.
    agent_request, holdout, _split_note = await _settle_splits(run, user_request)
    run.holdout = holdout
    if user_request.dataset:
        # The user owns the source, while split settlement owns the exact file
        # workers may train on.  Preserve both meanings instead of comparing a
        # generated train-minus-validation path with the original upload.
        pins = dict(run.decision_pins or {})
        pins["dataset_source"] = user_request.dataset
        pins["dataset"] = agent_request.dataset
        pins["dataset_source_split"] = user_request.dataset_split
        pins["dataset_source_config"] = user_request.dataset_config
        pins["dataset_split"] = agent_request.dataset_split
        pins["dataset_config"] = agent_request.dataset_config
        run.decision_pins = pins

    # F.4 — embed the cached dataset profile (if any) so the orchestrator's
    # very first turn has structured context about the data instead of
    # having to guess from the path. It uses the profile to choose compatible
    # high-level method/data semantics; Data still inspects its raw source.
    # Profiled from the ORIGINAL dataset path. A carve writes the trimmed
    # training file into the run's own directory, where no cached profile
    # exists — and the profile describes the data's shape, which dropping a
    # tenth of the rows does not change.
    profile = _dataset_profile_for(user_request.dataset)
    # The payload (incl. the G.1 budget snapshot the orchestrator sees from
    # turn 0) is assembled in zevo.engine.run.supervisor, shared with the
    # post-scoping settlement of an `auto` Run so both start identically.
    from zevo.engine.run.supervisor import initial_supervisor_payload, new_supervisor_ticket
    sup_payload = initial_supervisor_payload(
        run, agent_request=agent_request, holdout=holdout, dataset_profile=profile,
    )
    sup = new_supervisor_ticket(run, sup_payload)
    db.add(sup)
    run.supervisor_ticket_id = sup_id
    run.status = "running"
    await db.commit()
    await db.refresh(sup)

    # Auto-enqueue a wakeup; the daemon will pick it up.
    await queue_wakeup(
        db, agent_id="orchestrator", ticket_id=sup_id,
        source="assignment",
        reason=f"new run created (task_name={task_name!r})",
    )

    # I.4 — audit: who started the run + caps + objective
    from zevo.engine.observe.audit import audit
    await audit(
        db, event_type="run.create", target_type="run", target_id=run.id,
        summary=run.task_objective[:200],
        after={
            "task_name": task_name,
            "run_name": run.run_name or "",
            "max_cost_usd": float(run.max_cost_usd or 0.0),
            "max_runtime_hours": float(run.max_runtime_hours or 0.0),
            "iteration_budget": run.iteration_budget,
            "stop_threshold": run.stop_threshold,
            "gpu_provider": run.gpu_provider,
        },
    )

    return CreateRunResponse(run_id=run.id, status="running")




async def _require_work_product(
    db: AsyncSession, *, run_id: str, ticket_id: str, agent_id: str, role: str,
) -> tuple[Ticket, Any]:
    """Resolve one exact, usable artifact without trusting a path in a request."""
    from zevo.db import WorkProduct

    ticket = (await db.execute(select(Ticket).where(
        Ticket.id == ticket_id,
        Ticket.run_id == run_id,
        Ticket.agent_id == agent_id,
        Ticket.lane == "optimization",
        Ticket.status.in_(("succeeded", "degraded")),
    ))).scalar_one_or_none()
    if ticket is None:
        raise HTTPException(
            409, f"{ticket_id!r} is not a usable {agent_id} Ticket in this Run",
        )
    products = (await db.execute(select(WorkProduct).where(
        WorkProduct.ticket_id == ticket.id,
        WorkProduct.role == role,
    ).order_by(WorkProduct.created_at.asc()))).scalars().all()
    product = next((
        candidate for candidate in products
        if role == "checkpoint"
        and (candidate.meta or {}).get("checkpoint_kind") == "final"
    ), products[0] if products else None)
    if product is None or not str(product.path or "").strip():
        raise HTTPException(
            409, f"Ticket {ticket.id!r} has no registered {role} WorkProduct",
        )
    return ticket, product


# ─────────────────────────── budget endpoint (G.1) ───────────────────────────


class BudgetDTO(BaseModel):
    run_id: str
    max_cost_usd: float
    llm_cost_usd: float
    gpu_cost_usd: float
    spent_usd: float
    remaining_usd: float
    over_budget: bool
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


class RunRequestDTO(BaseModel):
    """What a run was actually given, as recorded when it was created."""

    task_name: str
    task_predefined: bool          # does a catalogue task by that name exist now?
    task_objective: str
    base_model: str
    training_method: str
    method_config: dict[str, Any]
    iteration_budget: int
    stop_threshold: float | None
    metric: str
    metric_direction: Literal["max", "min"]
    validation_metric: str
    validation_metric_direction: Literal["max", "min"]
    max_cost_usd: float
    max_runtime_hours: float
    max_queue_wait_hours: float
    gpu_provider: Literal["cluster", "cloud", "instance"]
    num_gpus: int
    generation_backend: Literal["hf", "vllm"]
    # Each file, classified the same way the Tasks list classifies a task's
    # data: registered / huggingface / path / none.
    files: list[dict]
    # Which driver/model each role actually ran on, read back from the
    # heartbeats. Config is per-agent and editable between runs, so the only
    # honest answer for THIS run is what its own heartbeats recorded.
    agents: list[dict] = Field(default_factory=list)


async def _agent_config_of_run(db: AsyncSession, run_id: str) -> list[dict]:
    """Per role: the driver/model its heartbeats ran on, in pipeline order.

    An agent is normally one driver/model for a whole run, but it can be
    reconfigured mid-flight, so every distinct pair is kept and the busiest
    leads. Roles that never woke in this run are absent rather than guessed at.
    """
    from zevo.db import HeartbeatRun

    rows = (await db.execute(
        select(HeartbeatRun.agent_id, HeartbeatRun.driver, HeartbeatRun.model,
               func.count().label("n"))
        .join(Ticket, Ticket.id == HeartbeatRun.ticket_id)
        .where(Ticket.run_id == run_id)
        .group_by(HeartbeatRun.agent_id, HeartbeatRun.driver, HeartbeatRun.model)
    )).all()

    # The pipeline's own order, so the list reads like the loop it describes.
    ORDER = ["orchestrator", "data", "infrastructure", "train",
             "inference", "evaluation", "registry"]
    by_agent: dict[str, list[dict]] = {}
    for agent_id, driver, model, n in rows:
        if agent_id == "evaluation":
            continue  # deterministic runner, not an Agent configuration
        by_agent.setdefault(agent_id, []).append(
            {"driver": driver or "", "model": model or "", "heartbeats": int(n)})

    out: list[dict] = []
    for agent_id in sorted(by_agent, key=lambda a: (ORDER.index(a) if a in ORDER else len(ORDER), a)):
        variants = sorted(by_agent[agent_id], key=lambda v: -v["heartbeats"])
        out.append({"agent_id": agent_id, "variants": variants,
                    "heartbeats": sum(v["heartbeats"] for v in variants)})
    return out


@router.get("/runs/{run_id}/request", response_model=RunRequestDTO)
async def get_run_request(
    run_id: str, request: Request, db: AsyncSession = Depends(get_db),
) -> RunRequestDTO:
    """The run's own inputs — not the task's.

    A run keeps `task_name` as plain text and the request it was launched with
    lives in the supervisor ticket's payload, so this is the only record of what
    a particular run was handed: which files, which model, which caps. Reading
    the catalogue instead would show what the task says TODAY, which is a
    different question and can differ from what actually ran.
    """
    from zevo.api.routers.ui.tasks import data_source
    from zevo.db.models import Task

    run = (await db.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if run is None:
        raise HTTPException(404, f"run {run_id} not found")

    req: dict = {}
    if run.supervisor_ticket_id:
        sup = (await db.execute(
            select(Ticket).where(Ticket.id == run.supervisor_ticket_id)
        )).scalar_one_or_none()
        if sup is not None and isinstance(sup.payload, dict):
            req = sup.payload.get("user_request") or {}

    def _training_data() -> dict:
        """Training data is not always a path.

        A task can name it as a hub id, or describe it as an acquisition query
        for the data agent to satisfy — the capybara task does the latter, and calling
        that "not given" would be wrong: it IS specified, just not as a file.
        """
        src = data_source(str(req.get("dataset") or ""))
        if src["kind"] != "none":
            return src
        query = str(req.get("data_query") or "").strip()
        if not query:
            return src
        # A hub id inside the query is the concrete thing to show.
        hub = re.search(r"`?\b([\w.-]+/[\w.-]+)\b`?", query)
        if hub:
            # Route it through the same classifier as everything else, so a hub
            # id the catalogue declares is reported as being in the catalogue.
            return data_source(hub.group(1))
        return {"kind": "query", "name": query[:60], "detail": query,
                "remote": False, "url": ""}

    # The test set is deliberately absent from the orchestrator's copy of the
    # request, so read it off the run — this view is for the user, who is not
    # the party being kept honest.
    reveal_holdout = (
        run.status in TERMINAL_RUN_STATUSES or is_trusted_ui_request(request)
    )
    holdout = dict(run.holdout or {}) if reveal_holdout else {}

    def _from(key: str) -> dict:
        return data_source(str(req.get(key) or holdout.get(key) or ""))

    # The questions-only copies are not inputs any more — the data agent derives
    # them from these fields, so the fields are what there is to show.
    def _fields(key: str) -> dict:
        cols = list(req.get(key) or holdout.get(key) or [])
        return {"kind": "columns" if cols else "none", "name": ", ".join(cols),
                "detail": ", ".join(cols), "remote": False, "url": ""}

    # Training data, then each lane's independent metric/data binding.
    files = [{"role": "training data", **_training_data()}]
    for lane, keys in (
        ("validation", ("validation_set", "validation_answer_fields",
                        "validation_sample_submission",
                        "validation_evaluation_script")),
        ("test", ("test_set", "test_answer_fields", "test_sample_submission",
                  "test_evaluation_script")),
    ):
        set_key, fields_key, submission_key, evaluator_key = keys
        files += [
            {"role": f"{lane} set", **_from(set_key)},
            {"role": f"{lane} answer fields", **_fields(fields_key)},
            {"role": f"{lane} sample submission", **_from(submission_key)},
            {"role": f"{lane} evaluation script", **_from(evaluator_key)},
        ]
    predefined = (await db.get(Task, run.task_name)) is not None if run.task_name else False

    return RunRequestDTO(
        task_name=run.task_name or "",
        task_predefined=bool(predefined),
        # The problem only. The three sentences `compose_objective` appends say
        # which model, which method and whether data was provided — and this
        # panel renders exactly those three under `Settings`, immediately below.
        # Printing them twice made the objective look like it had run on.
        task_objective=run.task_objective,
        base_model=str(req.get("base_model") or ""),
        training_method=str(req.get("training_method") or ""),
        method_config=dict(req.get("method_config") or {}),
        iteration_budget=int(run.iteration_budget or 0),
        stop_threshold=run.stop_threshold,
        metric=run.metric,
        metric_direction=run.metric_direction,
        validation_metric=run.validation_metric,
        validation_metric_direction=run.validation_metric_direction,
        max_cost_usd=float(run.max_cost_usd or 0.0),
        max_runtime_hours=float(run.max_runtime_hours or 0.0),
        max_queue_wait_hours=float(run.max_queue_wait_hours or 0.0),
        gpu_provider=run.gpu_provider,
        num_gpus=max(0, int(run.num_gpus or 0)),
        generation_backend=run.generation_backend,
        files=files,
        agents=await _agent_config_of_run(db, run_id),
    )


@router.get("/runs/{run_id}/budget", response_model=BudgetDTO)
async def get_run_budget(
    run_id: str, db: AsyncSession = Depends(get_db),
) -> BudgetDTO:
    """Live cost and wall-clock snapshot. Orchestrator GETs this before
    another iteration; the backend also enforces already-exhausted hard caps."""
    from zevo.engine.cost.budget import snapshot_for_run
    r = (await db.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"run {run_id} not found")
    snap = await snapshot_for_run(db, run_id)
    return BudgetDTO(
        run_id=run_id,
        max_cost_usd=snap.max_cost_usd,
        llm_cost_usd=snap.llm_cost_usd,
        gpu_cost_usd=snap.gpu_cost_usd,
        spent_usd=snap.spent_usd,
        remaining_usd=snap.remaining_usd,
        over_budget=snap.over_budget,
        projected_next_iteration_usd=snap.projected_next_iteration_usd,
        can_afford_next_iteration=snap.can_afford_next_iteration,
        projection_basis=snap.projection_basis,
        max_runtime_hours=snap.max_runtime_hours,
        max_queue_wait_hours=snap.max_queue_wait_hours,
        queue_wait_hours=snap.queue_wait_hours,
        elapsed_runtime_hours=snap.elapsed_runtime_hours,
        remaining_runtime_hours=snap.remaining_runtime_hours,
        over_time_limit=snap.over_time_limit,
        projected_next_iteration_hours=snap.projected_next_iteration_hours,
        can_finish_next_iteration_in_time=snap.can_finish_next_iteration_in_time,
        runtime_projection_basis=snap.runtime_projection_basis,
    )


# ─────────────────────── per-agent cost breakdown ────────────────────────────


class AgentCostDTO(BaseModel):
    agent_id: str
    cost_usd: float
    heartbeats: int


class CostBreakdownDTO(BaseModel):
    """Read-only per-agent cost split for a run.

    `agents` is the LLM/agent cost grouped by role (orchestrator, data,
    infrastructure, train, inference, evaluation, registry). The numbers are
    sourced from the same per-heartbeat cost the budget snapshot sums, and GPU
    plus the totals mirror `snapshot_for_run`, so nothing here re-prices work.
    """

    run_id: str
    agents: list[AgentCostDTO]
    agent_cost_usd: float     # total LLM/agent cost (sum of agents[].cost_usd)
    gpu_cost_usd: float       # rented GPU time; legitimately 0 on owned hardware
    total_cost_usd: float     # agent_cost_usd + gpu_cost_usd


@router.get("/runs/{run_id}/cost-breakdown", response_model=CostBreakdownDTO)
async def get_run_cost_breakdown(
    run_id: str, db: AsyncSession = Depends(get_db),
) -> CostBreakdownDTO:
    """Cost grouped by agent role, plus GPU and totals — surfacing only.

    The run detail already exposes `agent_cost_usd`/`gpu_cost_usd` totals; this
    breaks the agent half down by which role spent it, so an operator can see
    that (for example) the orchestrator's deliberation dwarfed the GPU bill.
    """
    from zevo.engine.cost.budget import cost_breakdown_for_run

    r = (await db.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"run {run_id} not found")
    breakdown = await cost_breakdown_for_run(db, run_id)
    return CostBreakdownDTO(
        run_id=run_id,
        agents=[
            AgentCostDTO(
                agent_id=a.agent_id, cost_usd=a.cost_usd, heartbeats=a.heartbeats,
            )
            for a in breakdown.agents
        ],
        agent_cost_usd=breakdown.agent_cost_usd,
        gpu_cost_usd=breakdown.gpu_cost_usd,
        total_cost_usd=breakdown.total_cost_usd,
    )


# ─────────────────────────── score events (G.3) ──────────────────────────────


class ScoreEventDTO(BaseModel):
    id: str
    iteration: int
    split: str           # validation | test
    source: str          # baseline | trained | validation
    score: float
    metric_name: str
    extras: dict
    notes: str
    ts: str


class ScoreSeries(BaseModel):
    run_id: str
    # The VALIDATION series — what the run optimized. This is the default view
    # and the only one an agent-shaped request gets back.
    events: list[ScoreEventDTO]
    baseline_validation_score: float | None
    latest_validation_score: float | None
    best_validation_score: float | None
    delta_vs_baseline: float | None
    # Trusted dashboard requests can observe this series live. Agent callers
    # receive it only after the Run becomes terminal.
    test_events: list[ScoreEventDTO] = Field(default_factory=list)
    baseline_test_score: float | None = None
    latest_test_score: float | None = None
    champion_test_score: float | None = None


def _score_event_dto(e: ScoreEvent) -> ScoreEventDTO:
    return ScoreEventDTO(
        id=e.id, iteration=e.iteration, split=e.split or "validation", source=e.source,
        score=float(e.score),
        metric_name=e.metric_name,
        extras=e.extras or {}, notes=e.notes or "",
        ts=e.ts.isoformat() if e.ts else "",
    )


@router.get("/runs/{run_id}/scores", response_model=ScoreSeries)
async def list_run_scores(
    run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> ScoreSeries:
    """Return validation scores while active and test scores after completion."""
    from zevo.db import ScoreEvent
    r = (await db.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"run {run_id} not found")
    rows = (await db.execute(
        select(ScoreEvent).where(ScoreEvent.run_id == run_id)
        .order_by(ScoreEvent.ts)
    )).scalars().all()
    every = [_score_event_dto(e) for e in rows]
    events = [e for e in every if (e.split or "validation") != "test"]
    trained = [e for e in events if e.source == "trained"]
    baseline = next((e for e in events if e.source == "baseline"), None)
    latest = trained[-1].score if trained else None
    from zevo.engine.method.score_direction import best_score, improvement
    direction = r.validation_metric_direction
    best = best_score((e.score for e in trained), direction)
    delta = (
        round(improvement(latest, baseline.score, direction), 6)
        if (baseline is not None and latest is not None) else None
    )
    series = ScoreSeries(
        run_id=run_id, events=events,
        baseline_validation_score=baseline.score if baseline else None,
        latest_validation_score=latest, best_validation_score=best,
        delta_vs_baseline=delta,
    )
    if r.status in TERMINAL_RUN_STATUSES or is_trusted_ui_request(request):
        test = [e for e in every if e.split == "test"]
        test_trained = [e for e in test if e.source == "trained"]
        test_base = next((e for e in test if e.source == "baseline"), None)
        series.test_events = test
        series.baseline_test_score = test_base.score if test_base else None
        series.latest_test_score = test_trained[-1].score if test_trained else None
        # Not max(test): the reported number is whatever the validation-picked
        # champion scored. See Run.champion_test_score.
        series.champion_test_score = (
            None if r.champion_test_score is None else float(r.champion_test_score)
        )
    return series


# ──────────────────────────── artifacts (I.2) ────────────────────────────────


# Path/existence helpers live in zevo.api.artifacts so runs / tickets /
# registry all report artifacts identically (files AND directory models).
from zevo.api.artifacts import host_path as _host_path, artifact_stat as _artifact_stat


class ArtifactDTO(BaseModel):
    id: str
    ticket_id: str
    role: str
    path: str             # in-container path (used to read the file)
    local_path: str = ""  # where it lives on the host: ./runs/<run-id>/<ticket>/…
    exists: bool
    # `exists` describes the WorkProduct's original path. Checkpoints are born
    # on a remote GPU, so that path is not locally probeable even when Registry
    # has retained the champion under the Run's local models directory.
    availability: Literal["local", "remote", "saved_model", "missing"]
    size_bytes: int
    meta: dict
    created_at: str


class ArtifactDetail(ArtifactDTO):
    """Same as ArtifactDTO + preview (first ~8KB of text artifacts)."""
    preview_kind: str = "text"      # "text" | "jsonl" | "binary" | "missing"
    preview: str = ""
    preview_columns: list[str] = Field(default_factory=list)
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)
    preview_page: int = 1
    preview_page_size: int = 10


def _clip_preview_value(value: Any) -> Any:
    """Bound a table cell without destroying its JSON shape."""
    if isinstance(value, str):
        return value if len(value) <= 4000 else value[:4000] + "…"
    if isinstance(value, list):
        clipped = [_clip_preview_value(item) for item in value[:50]]
        if len(value) > 50:
            clipped.append(f"… {len(value) - 50} more items")
        return clipped
    if isinstance(value, dict):
        return {
            str(key): _clip_preview_value(item)
            for key, item in list(value.items())[:50]
        }
    return value


def _preview_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def _preview_records(
    path: Path, *, limit: int = 10, offset: int = 0,
) -> list[dict[str, Any]]:
    """Read only enough tabular rows for an artifact drawer preview.

    Prediction previews are deliberately bounded: opening a 200k-row output in
    the UI must not make the API deserialize the entire dataset. JSON arrays
    are the one non-streaming format, but their preview is still truncated
    immediately after parsing.
    """
    suffix = path.suffix.lower()
    rows: list[dict[str, Any]] = []
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                if index < offset:
                    continue
                rows.append(dict(row))
                if len(rows) >= limit:
                    break
    elif suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            logical_index = 0
            for line in handle:
                if not line.strip():
                    continue
                if logical_index < offset:
                    logical_index += 1
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    return []
                rows.append(row)
                logical_index += 1
                if len(rows) >= limit:
                    break
    elif suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            return []
        rows = [
            row for row in loaded[offset:offset + limit]
            if isinstance(row, dict)
        ]
    return [
        {str(key): _clip_preview_value(value) for key, value in row.items()}
        for row in rows
    ]


def _prediction_with_questions_rows(
    predictions: Path,
    questions: Path,
    *,
    ground_truth: Path | None = None,
    answer_fields: list[str] | None = None,
    limit: int = 10,
    offset: int = 0,
) -> list[dict[str, Any]]:
    prediction_rows = _preview_records(predictions, limit=limit, offset=offset)
    question_rows = _preview_records(questions, limit=limit, offset=offset)
    if not prediction_rows or not question_rows:
        return []

    truth_rows = (
        _preview_records(ground_truth, limit=limit, offset=offset)
        if ground_truth is not None and ground_truth.is_file()
        else []
    )
    fields = list(answer_fields or [])

    by_id = {
        str(row["id"]): row for row in question_rows
        if row.get("id") not in (None, "")
    }
    truth_by_id = {
        str(row["id"]): row for row in truth_rows
        if row.get("id") not in (None, "")
    }
    combined: list[dict[str, Any]] = []
    for index, prediction in enumerate(prediction_rows):
        prediction_id = prediction.get("id")
        question = (
            by_id.get(str(prediction_id))
            if prediction_id not in (None, "")
            else None
        )
        if question is None and index < len(question_rows):
            question = question_rows[index]
        row = dict(question or {})
        truth = (
            truth_by_id.get(str(prediction_id))
            if prediction_id not in (None, "")
            else None
        )
        if truth is None and index < len(truth_rows):
            truth = truth_rows[index]
        present_truth = [field for field in fields if truth and field in truth]
        for field in present_truth:
            key = "ground_truth" if len(present_truth) == 1 else f"ground_truth · {field}"
            row[key] = truth[field]

        prediction_values = [
            (key, value) for key, value in prediction.items()
            if key not in row or row.get(key) != value
        ]
        for key, value in prediction_values:
            output_key = "prediction" if len(prediction_values) == 1 else f"prediction · {key}"
            row[output_key] = value
        combined.append(row)
    return combined


def _prediction_with_questions_preview(
    predictions: Path, questions: Path, *, limit: int = 10, offset: int = 0,
) -> str:
    """Join answer-only predictions to their readable question records.

    The inference artifact validator already guarantees equal row count/order
    and stable IDs when present. Rechecking the ID here makes the preview safe
    for older artifacts too; a mismatch falls back to positional pairing
    rather than silently attaching a different question.
    """
    combined = _prediction_with_questions_rows(
        predictions, questions, limit=limit, offset=offset,
    )
    if not combined:
        return ""
    body = json.dumps(combined, ensure_ascii=False, indent=2)
    return (
        f"Question + prediction preview (first {len(combined)} rows)\n\n"
        + body
    )[:8000]


def _registry_storage_path(raw: str) -> Path | None:
    """Resolve Registry's host-relative path inside the API container."""
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    in_container = Path("/app") / path
    return in_container if Path("/app").exists() else Path.cwd() / path


async def _retained_model_for_run(
    db: AsyncSession, run_id: str,
) -> tuple[RegistryModel | None, Path | None, bool, int]:
    model = (await db.execute(
        select(RegistryModel)
        .where(RegistryModel.run_id == run_id)
        .order_by(desc(RegistryModel.registered_at))
        .limit(1)
    )).scalar_one_or_none()
    path = _registry_storage_path(model.model_path) if model else None
    exists, size = _artifact_stat(path)
    return model, path, exists, size


def _artifact_availability(
    *, role: str, meta: dict, ticket_iteration: int,
    original_exists: bool, retained_model: RegistryModel | None,
    retained_exists: bool,
) -> Literal["local", "remote", "saved_model", "missing"]:
    if role == "checkpoint" and meta.get("location") == "remote":
        if (
            retained_model is not None
            and retained_exists
            and ticket_iteration == retained_model.iteration
        ):
            return "saved_model"
        return "remote"
    return "local" if original_exists else "missing"


@router.get("/runs/{run_id}/artifacts", response_model=list[ArtifactDTO])
async def list_run_artifacts(
    run_id: str, request: Request, db: AsyncSession = Depends(get_db),
) -> list[ArtifactDTO]:
    """All WorkProducts for tickets on this run, ordered newest-first.
    Includes size + existence flag so the UI can spot deleted artifacts."""
    from zevo.db import WorkProduct
    from pathlib import Path as _P
    r = (await db.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"run {run_id} not found")
    artifacts_stmt = (
        select(WorkProduct, Ticket)
        .join(Ticket, WorkProduct.ticket_id == Ticket.id)
        .where(Ticket.run_id == run_id)
        .order_by(desc(WorkProduct.created_at))
    )
    if r.status not in TERMINAL_RUN_STATUSES and not is_trusted_ui_request(request):
        artifacts_stmt = artifacts_stmt.where(Ticket.lane != "held_out_test")
    rows = (await db.execute(artifacts_stmt)).all()

    retained, retained_path, retained_exists, retained_size = (
        await _retained_model_for_run(db, run_id)
    )

    out: list[ArtifactDTO] = []
    for wp, tk in rows:
        p = _P(wp.path) if wp.path else None
        exists, size = _artifact_stat(p)
        meta = dict(wp.meta or {})
        availability = _artifact_availability(
            role=wp.role, meta=meta, ticket_iteration=int(tk.iteration or 0),
            original_exists=exists, retained_model=retained,
            retained_exists=retained_exists,
        )
        if availability == "saved_model" and retained_path is not None:
            meta["saved_model_path"] = _host_path(str(retained_path))
            meta["version_tag"] = retained.version_tag if retained else ""
            size = retained_size
        out.append(ArtifactDTO(
            id=wp.id, ticket_id=wp.ticket_id, role=wp.role, path=wp.path,
            local_path=_host_path(wp.path or ""),
            exists=exists, availability=availability, size_bytes=size, meta=meta,
            created_at=wp.created_at.isoformat() if wp.created_at else "",
        ))
    return out


@router.get("/runs/{run_id}/artifacts/{artifact_id}", response_model=ArtifactDetail)
async def get_artifact_detail(
    run_id: str, artifact_id: str, request: Request,
    preview_page: int = Query(1, ge=1, le=10),
    preview_page_size: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> ArtifactDetail:
    """Artifact detail + a cheap preview. Text/JSONL files: first 8KB.
    Binary: just size + extension. Missing on disk: explicit flag."""
    from zevo.db import WorkProduct
    from pathlib import Path as _P
    row = (await db.execute(
        select(WorkProduct, Ticket)
        .join(Ticket, WorkProduct.ticket_id == Ticket.id)
        .where(Ticket.run_id == run_id, WorkProduct.id == artifact_id)
    )).first()
    if row is None:
        raise HTTPException(404, f"artifact {artifact_id} not found in run {run_id}")
    wp, tk = row
    if tk.lane == "held_out_test" and not is_trusted_ui_request(request):
        run_status = (await db.execute(
            select(Run.status).where(Run.id == run_id)
        )).scalar_one_or_none()
        if run_status not in TERMINAL_RUN_STATUSES:
            raise HTTPException(404, f"artifact {artifact_id} not found in run {run_id}")
    p = _P(wp.path) if wp.path else None
    exists, size = _artifact_stat(p)
    meta = dict(wp.meta or {})
    retained, retained_path, retained_exists, retained_size = (
        await _retained_model_for_run(db, run_id)
    )
    availability = _artifact_availability(
        role=wp.role, meta=meta, ticket_iteration=int(tk.iteration or 0),
        original_exists=exists, retained_model=retained,
        retained_exists=retained_exists,
    )
    if availability == "saved_model" and retained_path is not None:
        meta["saved_model_path"] = _host_path(str(retained_path))
        meta["version_tag"] = retained.version_tag if retained else ""
        size = retained_size

    preview_kind = "missing"
    preview = ""
    preview_columns: list[str] = []
    preview_rows: list[dict[str, Any]] = []
    preview_offset = (preview_page - 1) * preview_page_size
    if availability == "saved_model":
        preview_kind = "text"
        preview = (
            "This checkpoint was produced on the remote GPU. Registry retained "
            "the selected model locally, so it is available from the Models page.\n\n"
            f"remote checkpoint: {wp.path}\n"
            f"saved model: {meta.get('saved_model_path', '')}"
        )
    elif availability == "remote":
        preview_kind = "text"
        preview = (
            "This checkpoint was produced at the remote GPU path. It was not "
            "selected as the Run's retained model, and remote availability is "
            "not continuously probed.\n\n"
            f"remote checkpoint: {wp.path}"
        )
    elif not exists:
        preview_kind = "missing"
        preview = f"path does not exist on disk: {wp.path}"
    elif p.is_dir():
        # Directory artifact (e.g. a LoRA `model/` — config + safetensors +
        # tokenizer). Not "missing"; preview a listing of what's inside.
        preview_kind = "text"
        entries = sorted(f for f in p.rglob("*"))
        lines = []
        for f in entries[:200]:
            rel = f.relative_to(p)
            try:
                lines.append(f"{f.stat().st_size:>12,}  {rel}" if f.is_file()
                             else f"{'<dir>':>12}  {rel}/")
            except OSError:
                lines.append(f"{'?':>12}  {rel}")
        more = f"\n… and {len(entries) - 200} more" if len(entries) > 200 else ""
        preview = (
            f"directory: {wp.path}\n"
            f"({sum(1 for f in entries if f.is_file())} files, {size:,} bytes total)\n\n"
            + "\n".join(lines) + more
        )
    elif wp.role == "predictions":
        # A submission-shaped prediction often contains only `id` and the
        # answer column. That is correct for Evaluation but poor for a human
        # preview, so pair it with the exact questions-only file assigned to
        # this Inference ticket.
        question_path = Path(str((tk.payload or {}).get("scoring_set") or ""))
        truth_path: Path | None = None
        answer_fields: list[str] = []
        if is_trusted_ui_request(request):
            run_row = (await db.execute(
                select(Run).where(Run.id == run_id)
            )).scalar_one_or_none()
            holdout = dict(run_row.holdout or {}) if run_row is not None else {}
            scoring_lane = "test" if tk.lane == "held_out_test" else "validation"
            candidate = Path(str(holdout.get(f"{scoring_lane}_set") or ""))
            if not candidate.is_file():
                try:
                    legacy_relative = candidate.relative_to("/app/data")
                except ValueError:
                    legacy_relative = None
                if legacy_relative is not None:
                    private_candidate = Path("/run/zevo-holdout") / legacy_relative
                    if private_candidate.is_file():
                        candidate = private_candidate
            if candidate.is_file():
                truth_path = candidate
                answer_fields = [
                    str(field) for field in holdout.get(f"{scoring_lane}_answer_fields", [])
                    if str(field)
                ]
        try:
            preview_rows = (
                _prediction_with_questions_rows(
                    p, question_path,
                    ground_truth=truth_path,
                    answer_fields=answer_fields,
                    limit=preview_page_size, offset=preview_offset,
                )
                if question_path.is_file() else []
            )
        except (OSError, ValueError, json.JSONDecodeError):
            preview_rows = []
        if preview_rows:
            preview_columns = _preview_columns(preview_rows)
            preview_kind = "jsonl"
            body = json.dumps(preview_rows, ensure_ascii=False, indent=2)
            preview = (
                f"Question + prediction preview (page {preview_page})\n\n"
                + body
            )[:8000]
        else:
            try:
                preview_kind = "text"
                preview = p.read_text(encoding="utf-8", errors="replace")[:8000]
            except OSError as e:
                preview_kind = "missing"
                preview = f"failed to read: {e}"
    else:
        suffix = p.suffix.lower()
        try:
            if suffix in (".jsonl", ".json", ".txt", ".md", ".csv", ".yaml", ".yml", ".py"):
                preview_kind = "jsonl" if suffix == ".jsonl" else "text"
                preview = p.read_text(encoding="utf-8", errors="replace")[:8000]
            else:
                preview_kind = "binary"
                preview = (
                    f"{suffix or 'no extension'} file, {size:,} bytes. "
                    f"No text preview available."
                )
        except OSError as e:
            preview_kind = "missing"
            preview = f"failed to read: {e}"

    if (
        not preview_rows
        and availability == "local"
        and p is not None
        and p.is_file()
        and wp.role in {"training_dataset", "validation_dataset", "scoring_questions"}
    ):
        try:
            preview_rows = _preview_records(
                p, limit=preview_page_size, offset=preview_offset,
            )
            preview_columns = _preview_columns(preview_rows)
        except (OSError, ValueError, json.JSONDecodeError):
            preview_rows = []
            preview_columns = []

    return ArtifactDetail(
        id=wp.id, ticket_id=wp.ticket_id, role=wp.role, path=wp.path,
        local_path=_host_path(wp.path or ""),
        exists=exists, availability=availability, size_bytes=size, meta=meta,
        created_at=wp.created_at.isoformat() if wp.created_at else "",
        preview_kind=preview_kind, preview=preview,
        preview_columns=preview_columns, preview_rows=preview_rows,
        preview_page=preview_page, preview_page_size=preview_page_size,
    )


# ───────────────────────── stop-policy (#7) ──────────────────────────────────


class StopDecisionDTO(BaseModel):
    should_stop: bool
    trigger: str
    reason: str
    recommended_action: str       # mark_done | mark_failed | continue
    inputs: dict                  # echoes what we evaluated against


class PolicyPatchBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration_budget: int | None = Field(None, ge=0)
    stop_threshold: float | None = Field(None, allow_inf_nan=False)
    max_cost_usd: float | None = Field(None, ge=0.0)
    max_runtime_hours: float | None = Field(None, ge=0.0)
    max_queue_wait_hours: float | None = Field(None, gt=0.0, le=168.0)
    min_delta_per_iter: float | None = Field(None, ge=0.0)
    regression_tolerance: float | None = Field(None, ge=0.0)


@router.patch("/runs/{run_id}/policy", response_model=RunSummary)
async def patch_run_policy(
    run_id: str, body: PolicyPatchBody,
    db: AsyncSession = Depends(get_db),
) -> RunSummary:
    """Update the Zevo model-improvement loop policy mid-run. Useful when
    the operator decides "actually, let it run 2 more iterations" or
    "regress tolerance is too tight, bump it"."""
    r = (await db.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"run {run_id} not found")

    before = {
        "iteration_budget": r.iteration_budget,
        "stop_threshold": r.stop_threshold,
        "max_cost_usd": float(r.max_cost_usd or 0.0),
        "max_runtime_hours": float(r.max_runtime_hours or 0.0),
        "max_queue_wait_hours": float(r.max_queue_wait_hours or 0.0),
        "min_delta_per_iter": float(r.min_delta_per_iter or 0.0),
        "regression_tolerance": float(r.regression_tolerance or 0.0),
    }
    if body.iteration_budget is not None:
        r.iteration_budget = body.iteration_budget
    if "stop_threshold" in body.model_fields_set:
        r.stop_threshold = body.stop_threshold
    if body.max_cost_usd is not None:
        r.max_cost_usd = float(body.max_cost_usd)
    if body.max_runtime_hours is not None:
        r.max_runtime_hours = float(body.max_runtime_hours)
    if body.max_queue_wait_hours is not None:
        r.max_queue_wait_hours = float(body.max_queue_wait_hours)
    if body.min_delta_per_iter is not None:
        r.min_delta_per_iter = float(body.min_delta_per_iter)
    if body.regression_tolerance is not None:
        r.regression_tolerance = float(body.regression_tolerance)
    await db.commit()
    await db.refresh(r)

    from zevo.engine.observe.audit import audit
    after = {
        "iteration_budget": r.iteration_budget,
        "stop_threshold": r.stop_threshold,
        "max_cost_usd": float(r.max_cost_usd or 0.0),
        "max_runtime_hours": float(r.max_runtime_hours or 0.0),
        "max_queue_wait_hours": float(r.max_queue_wait_hours or 0.0),
        "min_delta_per_iter": float(r.min_delta_per_iter or 0.0),
        "regression_tolerance": float(r.regression_tolerance or 0.0),
    }
    # Compare only keys present in BOTH snapshots so a future field added to one
    # dict but not the other can never KeyError the whole endpoint again.
    changed = sorted(k for k in before if k in after and before[k] != after[k])
    await audit(
        db, event_type="run.policy_patch", target_type="run", target_id=run_id,
        summary=f"changed: {changed}",
        before={k: before[k] for k in changed},
        after={k: after[k] for k in changed},
    )
    return _summary(r)


@router.get("/runs/{run_id}/should-stop", response_model=StopDecisionDTO)
async def should_stop(
    run_id: str, db: AsyncSession = Depends(get_db),
) -> StopDecisionDTO:
    """Orchestrator GETs this each iteration to make the stop decision
    consistently. Pure function over score_events + run policy +
    current budget — no LLM call needed."""
    from zevo.engine.cost.budget import snapshot_for_run
    from zevo.db import ScoreEvent
    from zevo.engine.method.loop_policy import decide

    r = (await db.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"run {run_id} not found")

    # Validation only. The stop decision — target hit, plateaued, regressed —
    # is a decision about the run, and making it on the held-out numbers would
    # be tuning the stopping point to the test set.
    score_rows = (await db.execute(
        select(ScoreEvent).where(
            ScoreEvent.run_id == run_id,
            ScoreEvent.source == "trained",
            ScoreEvent.split != "test",
        ).order_by(ScoreEvent.ts)
    )).scalars().all()
    scores = [float(s.score) for s in score_rows]

    snap = await snapshot_for_run(db, run_id)

    decision = decide(
        scores=scores,
        iterations_completed=int(r.iterations_completed or 0),
        iteration_budget=int(r.iteration_budget or 0),
        stop_threshold=r.stop_threshold,
        metric_direction=r.validation_metric_direction,
        min_delta_per_iter=float(r.min_delta_per_iter or 0.0),
        regression_tolerance=float(r.regression_tolerance or 0.0),
        over_budget=bool(snap.over_budget),
        over_time_limit=bool(snap.over_time_limit),
    )

    return StopDecisionDTO(
        should_stop=decision.should_stop,
        trigger=decision.trigger,
        reason=decision.reason,
        recommended_action=decision.recommended_action,
        inputs={
            "scores": scores,
            "iterations_completed": r.iterations_completed,
            "iteration_budget": r.iteration_budget,
            "stop_threshold": r.stop_threshold,
            "min_delta_per_iter": float(r.min_delta_per_iter or 0.0),
            "regression_tolerance": float(r.regression_tolerance or 0.0),
            "over_budget": snap.over_budget,
            "over_time_limit": snap.over_time_limit,
            "remaining_runtime_hours": snap.remaining_runtime_hours,
            "projected_next_iteration_hours": snap.projected_next_iteration_hours,
            "can_finish_next_iteration_in_time": snap.can_finish_next_iteration_in_time,
        },
    )
