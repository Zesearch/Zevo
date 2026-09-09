"""Leaderboard — for one task, which entrant did best. Two boards, two entrants.

  GET /leaderboard?by=harness  — entrant = the model DRIVING the agents, for
                                 Runs that retained a model. Answers "which
                                 harness produced the best saved model".
  GET /leaderboard?by=base     — every MODEL the task has produced, each with
                                 the run and the setting behind it. Answers
                                 "what has this task produced, and how".

The two boards answer their questions at different grains, but neither silently
collapses observations. The model board contains every saved model. The harness
board contains every Run with a saved model and groups them by Setting, because
comparing two harnesses only means something when the work they were given is
held fixed. A measured Run that retained no model is not an entrant.

They are separate boards because a run pairs one harness with one model, so a
single entrant ranking would confound the two. Every row carries two co-equal
held-out facts from the SAME validation-selected champion: `champion_test_score` and the
direction-normalized `improvement` over its held-out baseline. The API ranks
both independently and never folds them into a composite.

Every number this module reports is measured on the HELD-OUT set, on both boards
and for both ends of the improvement figure. A run tunes on its validation set,
so a validation score says how well it fitted its own yardstick; only the
held-out one is comparable across runs that carved different yardsticks. Runs
without a real held-out measurement are excluded outright, by
`was_measured_on_heldout`.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.api.database import get_db
from zevo.engine.cost.budget import snapshot_for_run
from zevo.engine.method.score_direction import improvement, is_better
from zevo.db.models import HeartbeatRun, RegistryModel, Run, ScoreEvent, Ticket
# The one definition of "same setting", shared with the tasks router so a
# leaderboard row and a task's saved settings agree on what a setting is.
from zevo.engine.observe.run_metrics import (
    baseline_and_best_test,
    was_measured_on_heldout,
)


router = APIRouter()



async def _harness_by_run(db: AsyncSession, run_ids: list[str]) -> dict[str, str]:
    """run_id -> the most common non-empty heartbeat model for that run.

    Agents in one run normally share a model, so the mode is that model; it also
    survives a run where one agent was reconfigured mid-flight.
    """
    if not run_ids:
        return {}
    rows = (await db.execute(
        select(Ticket.run_id, HeartbeatRun.model)
        .join(HeartbeatRun, HeartbeatRun.ticket_id == Ticket.id)
        .where(Ticket.run_id.in_(run_ids), HeartbeatRun.model != "")
    )).all()
    counts: dict[str, Counter] = defaultdict(Counter)
    for run_id, model in rows:
        counts[run_id][model] += 1
    return {rid: c.most_common(1)[0][0] for rid, c in counts.items()}


async def _setting_by_run(db: AsyncSession, run_ids: list[str]) -> dict[str, dict]:
    """run_id -> the setting it was launched with.

    A run records its request in its supervisor ticket, so this is what was
    actually handed over rather than what the task says today.
    """
    if not run_ids:
        return {}
    rows = (await db.execute(
        select(Run.id, Ticket.payload)
        .join(Ticket, Ticket.id == Run.supervisor_ticket_id)
        .where(Run.id.in_(run_ids))
    )).all()
    out: dict[str, dict] = {}
    for run_id, payload in rows:
        req = (payload or {}).get("user_request") if isinstance(payload, dict) else None
        if isinstance(req, dict):
            out[run_id] = req
    return out


async def _saved_settings(db: AsyncSession) -> dict[str, dict]:
    """{setting_id: the saved row} — exact name, level and caps.

    The whole row, not just the name: a leaderboard row's setting popup has to
    say exactly what the task's own settings table says, down to the iteration
    and budget caps, or the two disagree about the same thing.
    """
    from zevo.api.routers.ui.tasks import autonomy_level
    from zevo.db import TaskSetting

    rows = (await db.execute(select(TaskSetting))).scalars().all()
    return {
        s.id: {
            "setting": s.name or "",
            "level": autonomy_level(s.dataset, s.base_model, s.training_method),
            "iteration_budget": int(s.iteration_budget or 0),
            "max_cost_usd": float(s.max_cost_usd or 0.0),
        }
        for s in rows
    }


def _short(model: str) -> str:
    """`Qwen/Qwen3-4B` -> `Qwen3-4B`. The org is the same on every row."""
    return (model or "").rsplit("/", 1)[-1]


def _data_label(dataset: str) -> str:
    """What the training data reads as: the dataset it lives in, or the hub id."""
    d = (dataset or "").strip()
    if not d:
        return ""
    if d.startswith("/"):
        parts = d.rstrip("/").split("/")
        return parts[-2] if len(parts) > 1 else parts[-1]
    return d


def _runs_with_saved_models(runs: list[Any], tag_of: dict[str, str]) -> list[Any]:
    """Harness entrants are Runs with a concrete retained model identity."""
    return [run for run in runs if run.id in tag_of and tag_of[run.id]]


def _base_model_of(history: list[Any] | None, direction: str = "max") -> str:
    """The base model that produced the run's best score.

    A run may try more than one base model whenever that field was not pinned, so the run is
    credited to the model behind its best iteration rather than to whatever it
    started with. Falls back to the last entry that names one.
    """
    best_value, best_model, fallback = None, "", ""
    for h in history or []:
        if not isinstance(h, dict):
            continue
        model = (h.get("base_model") or "").strip()
        if model:
            fallback = model
        value = h.get("score")
        if model and isinstance(value, (int, float)) and not isinstance(value, bool) and is_better(float(value), best_value, direction):
            best_value, best_model = float(value), model
    return best_model or fallback


def _rank_cells(cells: list[dict], *, by: str) -> list[dict]:
    """Attach independent dense ranks for the two held-out measurements.

    Model rows compete within one Task. Harness-run rows compete within one
    Task + Setting so the assigned work is held fixed. Test-score direction
    follows the Task contract; improvement is direction-normalized, so larger
    is always better. Missing improvement remains unranked.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for cell in cells:
        scope = (cell["task_name"],)
        if by == "harness":
            scope += (tuple(cell.get("setting_key") or []),)
        groups[scope].append(cell)

    for group in groups.values():
        direction = group[0].get("metric_direction", "max")
        score_values = sorted(
            {float(cell["champion_test_score"]) for cell in group},
            reverse=direction != "min",
        )
        score_rank = {value: rank for rank, value in enumerate(score_values, 1)}

        improvement_values = sorted(
            {
                float(cell["improvement"])
                for cell in group
                if cell.get("improvement") is not None
            },
            reverse=True,
        )
        improvement_rank = {
            value: rank for rank, value in enumerate(improvement_values, 1)
        }

        for cell in group:
            cell["champion_test_score_rank"] = score_rank[float(cell["champion_test_score"])]
            improvement = cell.get("improvement")
            cell["improvement_rank"] = (
                improvement_rank[float(improvement)]
                if improvement is not None
                else None
            )
    return cells


@router.get("/leaderboard")
async def leaderboard(
    by: str = Query("harness", pattern="^(harness|base)$"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """One task's results. `by` chooses what a row IS — see the module docstring."""
    runs = (await db.execute(
        # history comes along so each cell can report how much the run gained
        # over its own baseline, not just where it landed.
        # champion_test_score, not best_validation_score: a board is a comparison, and
        # each run's validation set is its own — several were carved per-run.
        # The held-out score is the one number every run on a task was measured
        # by without having been able to tune on it.
        select(Run.id, Run.task_name, Run.run_name, Run.setting_id, Run.setting_name,
               Run.champion_test_score, Run.started_at,
               Run.finished_at, Run.history, Run.metric, Run.metric_direction,
               Run.validation_metric_direction,
               Run.iteration_budget, Run.max_cost_usd, Run.stop_threshold)
        .where(
            Run.champion_test_score.is_not(None),
            Run.task_name != "",
            was_measured_on_heldout(),
        )
    )).all()
    if not runs:
        return {"by": by, "cells": [], "models": []}
    by_id = {r.id: r for r in runs}

    settings = await _setting_by_run(db, [r.id for r in runs])
    saved = await _saved_settings(db)

    def _setting_of(run_id: str) -> dict:
        """How to name and describe the setting a run used."""
        from zevo.api.routers.ui.tasks import autonomy_level, setting_identity

        req = settings.get(run_id) or {}
        run = by_id[run_id]
        model = str(req.get("base_model") or "") or _base_model_of(
            run.history, run.validation_metric_direction
        )
        method = str(req.get("training_method") or "")
        data = _data_label(str(req.get("dataset") or ""))
        key = setting_identity({
            **req,
            "iteration_budget": run.iteration_budget,
            "max_cost_usd": run.max_cost_usd,
            "stop_threshold": run.stop_threshold,
        })
        return {
            "setting_key": key,
            "base_model": model, "training_method": method, "data": data,
            # A name is resolved only through the exact Setting row persisted
            # on the Run. Similar-looking paths or partial keys never borrow a
            # different Setting's label.
            "setting": run.setting_name or "Unsaved configuration",
            "level": autonomy_level(
                str(req.get("dataset") or ""), model, method,
            ),
            "iteration_budget": int(run.iteration_budget or 0),
            "max_cost_usd": float(run.max_cost_usd or 0.0),
            "stop_threshold": run.stop_threshold,
            **saved.get(run.setting_id or "", {}),
        }

    async def _cell(run_id: str, **extra) -> dict:
        r = by_id[run_id]
        snap = await snapshot_for_run(db, r.id)
        duration_s = None
        if r.started_at and r.finished_at:
            duration_s = max(0, int((r.finished_at - r.started_at).total_seconds()))
        # The HELD-OUT baseline, to sit under a HELD-OUT headline. `champion_test_score`
        # here is already `Run.champion_test_score` (see the query above), so
        # pairing it with the validation baseline would have reported a gain
        # measured between two different sets — a number belonging to neither.
        baseline, _ = baseline_and_best_test(
            r.history, r.validation_metric_direction,
        )
        return {
            "task_name": r.task_name, "run_id": r.id, "run_name": r.run_name or "",
            "metric": r.metric,
            "metric_direction": r.metric_direction,
            "champion_test_score": r.champion_test_score,
            # the untuned probe this run measured itself against, on the test
            # set; null when the run never took one or it was never measured.
            "baseline_test_score": baseline,
            "cost_usd": round(snap.spent_usd, 2), "duration_s": duration_s,
            # Direction-normalized gain over this run's own baseline, both ends
            # on the held-out set; null when either end is missing.
            "improvement": (lambda v: round(v, 6) if v is not None else None)(
                improvement(r.champion_test_score, baseline, r.metric_direction)
                if baseline is not None else None
            ),
            **_setting_of(r.id),
            # Last, so a caller with a truer number for this row can say so.
            **extra,
        }

    # The model each run left behind, so a row can name what it produced. One
    # per run — the superseded iterations are not on disk (see kept_models).
    from zevo.api.routers.ui.models import kept_models

    registry = (await db.execute(
        select(RegistryModel).where(RegistryModel.run_id.in_(list(by_id)))
        .order_by(RegistryModel.registered_at.desc())
    )).scalars().all()
    kept = kept_models(list(registry))
    tag_of = {m.run_id: m.version_tag for m in kept if m.run_id}

    cells = []
    if by == "harness":
        # One row per harness Run that actually retained a model. A completed
        # measurement with no saved model is not a model outcome and must not
        # appear as a synthetic "No model" entrant. Collapsing repeated Runs requires choosing a
        # winner by one metric before displaying the other, which makes the two
        # supposedly paired measurements unequal. Grouping by Setting in the
        # UI keeps the comparison like-for-like without discarding observations.
        saved_runs = _runs_with_saved_models(runs, tag_of)
        harness_of = await _harness_by_run(db, [r.id for r in saved_runs])
        for r in saved_runs:
            h = harness_of.get(r.id)
            if not h:
                continue
            cells.append(await _cell(r.id, model=h, version_tag=tag_of[r.id]))
    else:
        # Every model the task has saved, newest first — the same set the Models
        # page lists, which is one per run: the superseded iterations are not on
        # disk any more, so a board that ranked them would rank files that do not
        # exist. A model IS a thing, though, so two on one setting are two rows.
        models = kept
        # One stable Registry row is the validation-selected champion of its
        # Run. `_cell` therefore takes both held-out outcomes from that same
        # Run selection; looking the score up again by Registry iteration while
        # deriving improvement by Run would create two candidate identities.
        for m in models:
            cells.append(await _cell(
                m.run_id, model=m.base_model or "", version_tag=m.version_tag,
            ))

    cells = _rank_cells(cells, by=by)
    return {"by": by, "cells": cells, "models": sorted({c["model"] for c in cells if c["model"]})}
