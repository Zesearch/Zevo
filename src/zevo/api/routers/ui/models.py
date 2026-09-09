"""The trained-MODELS endpoints, as the UI's Models page reads them.

  GET    /models                   list all
  GET    /models/compare?a=&b=     side-by-side metric/config diff
  GET    /models/{tag}             single row
  GET    /models/{tag}/card        auto-generated Markdown card

`/models/compare` is declared BEFORE `/models/{tag}` on purpose: routes match in
declaration order, so the wildcard would otherwise swallow it and look up a
model whose version tag is the literal string "compare".

Named for the page, not for the storage behind it. The rows come from the
`registry_models` table, mirrored from each Registry Ticket's local manifest —
that agent keeps its name because it is a role in the pipeline rather than a
screen. What a person sees is Models, so that is what the endpoints are called.

A model-lifecycle feature used to live here too: a `stage` per model
(experimental -> staging -> production), a `promote` endpoint with score
gates and a unique production slot, and a `rollback` that undid the last
promotion. Its UI was removed from the Models page, leaving the endpoints
with no caller in the app, the CLI or anything else — the audit log records the
feature being used exactly once, the day before the buttons went. A `lineage`
endpoint went the same way. Both are deleted rather than kept warm: an endpoint
nothing calls is not a feature, it is a second definition of the data model
that no one is checking.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.api.artifacts import host_path_abs
from zevo.api.database import get_db
from zevo.db import AuditEvent, RegistryModel, Run, ScoreEvent
from zevo.engine.method.score_direction import improvement
from zevo.engine.observe.run_metrics import baseline_and_best_test


router = APIRouter()


# ─────────────────────────────── DTOs ────────────────────────────────────────


class ModelDTO(BaseModel):
    version_tag: str
    iteration: int
    base_model: str
    training_method: str
    dataset_source: str
    model_path: str
    task_objective: str
    metric: str
    metric_direction: str
    eval: dict[str, Any]
    registered_at: str
    run_id: str | None = None
    # The task the producing run was launched under, when it had one. A run id
    # says which execution made this model; the task name says what it was for,
    # which is the part a person recognises.
    task_name: str = ""
    # The HELD-OUT score of this Run's validation-selected champion. `eval.score`
    # is the VALIDATION number — the registry agent is handed the loop's own
    # eval metrics and has no path to the test set, by design.
    champion_test_score: float | None = None
    # The paired held-out outcome for this saved champion. Both ends are on the
    # same test set; improvement is normalized so positive is always better.
    baseline_test_score: float | None = None
    improvement: float | None = None
    # `model_path`, absolute on the HOST. The container only knows /app, so the
    # stored path is repo-relative and cannot be pasted anywhere; compose passes
    # the host's own view of this directory as ZEVO_HOST_REPO. Empty when that
    # is unset, or when this version kept no model.
    model_path_abs: str = ""


class CompareCell(BaseModel):
    a: Any
    b: Any
    differs: bool


class CompareResponse(BaseModel):
    a_tag: str
    b_tag: str
    fields: dict[str, CompareCell]
    champion_test_score_delta: float | None  # direction-normalized held-out difference: b vs a
    headline_summary: str          # one-line for the UI


# ─────────────────────────── helpers ────────────────────────────────────────


def _to_dto(
    m: RegistryModel,
    task_name: str = "",
    champion_test_score: float | None = None,
    baseline_test_score: float | None = None,
    model_improvement: float | None = None,
) -> ModelDTO:
    return ModelDTO(
        version_tag=m.version_tag, iteration=m.iteration, base_model=m.base_model,
        training_method=m.training_method,
        dataset_source=m.dataset_source, model_path=m.model_path,
        task_objective=m.task_objective, metric=m.metric,
        metric_direction=m.metric_direction, eval=m.eval or {},
        registered_at=m.registered_at.isoformat() if m.registered_at else "",
        run_id=m.run_id, task_name=task_name, champion_test_score=champion_test_score,
        baseline_test_score=baseline_test_score,
        improvement=model_improvement,
        model_path_abs=(
            host_path_abs("/app/" + m.model_path.lstrip("/"))
            if m.model_path and not m.model_path.startswith("/")
            else m.model_path
        ),
    )


async def _run_outcomes(db: AsyncSession, run_ids: list[str]) -> dict[str, dict[str, float | None]]:
    """Paired held-out baseline and improvement for each producing Run."""
    ids = [r for r in set(run_ids) if r]
    if not ids:
        return {}
    rows = (await db.execute(select(Run).where(Run.id.in_(ids)))).scalars().all()
    out: dict[str, dict[str, float | None]] = {}
    for run in rows:
        baseline, _ = baseline_and_best_test(
            run.history, run.validation_metric_direction,
        )
        best = (
            float(run.champion_test_score)
            if run.champion_test_score is not None else None
        )
        # A model rescued on cancel can exist before its run's first held-out
        # eval, so either side may be missing; the other call sites guard too.
        gain = (
            improvement(best, baseline, run.metric_direction)
            if best is not None and baseline is not None else None
        )
        out[run.id] = {
            "champion_test_score": best,
            "baseline_test_score": baseline,
            "improvement": round(gain, 6) if gain is not None else None,
        }
    return out


async def _task_names(db: AsyncSession, run_ids: list[str]) -> dict[str, str]:
    """{run_id: task_name} for the runs that still exist and were named.

    One query for the whole page rather than one per card.
    """
    ids = [r for r in run_ids if r]
    if not ids:
        return {}
    rows = (await db.execute(
        select(Run.id, Run.task_name).where(Run.id.in_(ids))
    )).all()
    return {rid: name for rid, name in rows if name}


# ─────────────────────────── list + detail ──────────────────────────────────


def kept_models(rows: list[RegistryModel]) -> list[RegistryModel]:
    """The models that still EXIST, newest first.

    A run owns one stable row. A winning iteration replaces that row; a losing
    iteration never creates another model. A non-empty model_path confirms the
    champion artifact is still retained.

    Shared with the leaderboard, so the two pages cannot disagree about how many
    models a task has.
    """
    def exists(model: RegistryModel) -> bool:
        raw = (model.model_path or "").strip()
        if not raw:
            return False
        path = Path(raw)
        if path.is_absolute():
            return path.exists()
        # Production runs with cwd=/app. The second path keeps local tests and
        # non-container development faithful to the same repo-relative value.
        return (Path("/app") / path).exists() or (Path.cwd() / path).exists()

    keep = [r for r in rows if exists(r)]
    keep.sort(key=lambda m: m.registered_at, reverse=True)
    return keep


@router.get("/models", response_model=list[ModelDTO])
async def list_models(db: AsyncSession = Depends(get_db)) -> list[ModelDTO]:
    rows = (
        await db.execute(select(RegistryModel).order_by(desc(RegistryModel.registered_at)))
    ).scalars().all()
    keep = kept_models(list(rows))
    tasks = await _task_names(db, [m.run_id for m in keep])
    outcomes = await _run_outcomes(db, [r.run_id or "" for r in keep])
    return [
        _to_dto(
            r,
            tasks.get(r.run_id or "", ""),
            outcomes.get(r.run_id or "", {}).get("champion_test_score"),
            outcomes.get(r.run_id or "", {}).get("baseline_test_score"),
            outcomes.get(r.run_id or "", {}).get("improvement"),
        )
        for r in keep
    ]

# ─────────────────────────── compare ────────────────────────────────────────


@router.get("/models/compare", response_model=CompareResponse)
async def compare(
    a: str, b: str, db: AsyncSession = Depends(get_db),
) -> CompareResponse:
    """Side-by-side metric + config diff for two registry entries."""
    if a == b:
        raise HTTPException(400, "a and b must be different version_tags")
    rows = (await db.execute(
        select(RegistryModel).where(RegistryModel.version_tag.in_([a, b]))
    )).scalars().all()
    by_tag = {r.version_tag: r for r in rows}
    if a not in by_tag:
        raise HTTPException(404, f"a ({a!r}) not found")
    if b not in by_tag:
        raise HTTPException(404, f"b ({b!r}) not found")
    ra, rb = by_tag[a], by_tag[b]

    # The rows a person compares, under the names the Models page uses for
    # them. `model_path` is left out: it is not shown anywhere in the UI, and a
    # diff row nobody can act on is noise.
    fields: dict[str, CompareCell] = {}
    for attr, label in (("base_model", "base"), ("training_method", "training method"),
                        ("dataset_source", "files"), ("task_objective", "objective"),
                        ("metric", "metric"), ("metric_direction", "target"),
                        ("run_id", "run")):
        va, vb = getattr(ra, attr, None), getattr(rb, attr, None)
        fields[label] = CompareCell(a=va, b=vb, differs=(va != vb))
    outcomes = await _run_outcomes(db, [ra.run_id or "", rb.run_id or ""])
    oa = outcomes.get(ra.run_id or "", {})
    ob = outcomes.get(rb.run_id or "", {})
    for label, key in (("test score", "champion_test_score"), ("improvement", "improvement")):
        va, vb = oa.get(key), ob.get(key)
        fields[label] = CompareCell(a=va, b=vb, differs=(va != vb))
    # Eval fields union
    eval_keys = set((ra.eval or {}).keys()) | set((rb.eval or {}).keys())
    for k in sorted(eval_keys):
        va = (ra.eval or {}).get(k)
        vb = (rb.eval or {}).get(k)
        fields[f"eval.{k}"] = CompareCell(a=va, b=vb, differs=(va != vb))

    score_a = oa.get("champion_test_score")
    score_b = ob.get("champion_test_score")
    comparable = ra.metric == rb.metric and ra.metric_direction == rb.metric_direction
    delta = round(improvement(score_b, score_a, ra.metric_direction), 6) if (
        comparable and score_a is not None and score_b is not None
    ) else None

    # The stable Registry tag is also the model's user-facing name.
    def _name(m: RegistryModel) -> str:
        return m.version_tag

    if delta is not None:
        summary = f"{_name(rb)} {'+' if delta >= 0 else ''}{delta:g} held-out test difference vs {_name(ra)}"
    elif not comparable:
        summary = f"{_name(ra)} and {_name(rb)} use different evaluator targets; structural diff only"
    else:
        summary = f"{_name(ra)} or {_name(rb)} missing score; structural diff only"

    return CompareResponse(
        a_tag=a, b_tag=b, fields=fields,
        champion_test_score_delta=delta, headline_summary=summary,
    )




@router.get("/models/{version_tag}", response_model=ModelDTO)
async def get_model(version_tag: str, db: AsyncSession = Depends(get_db)) -> ModelDTO:
    r = (await db.execute(
        select(RegistryModel).where(RegistryModel.version_tag == version_tag)
    )).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"model {version_tag} not found")
    outcomes = await _run_outcomes(db, [r.run_id or ""])
    outcome = outcomes.get(r.run_id or "", {})
    return _to_dto(
        r,
        (await _task_names(db, [r.run_id or ""])).get(r.run_id or "", ""),
        outcome.get("champion_test_score"),
        outcome.get("baseline_test_score"),
        outcome.get("improvement"),
    )


# ─────────────────────────── model card ─────────────────────────────────────


@router.get("/models/{version_tag}/card", response_class=Response)
async def get_model_card(
    version_tag: str, db: AsyncSession = Depends(get_db),
) -> Response:
    """Return text/markdown — viewable in the UI or downloadable."""
    from zevo.engine.observe.model_card import render_model_card

    r = (await db.execute(
        select(RegistryModel).where(RegistryModel.version_tag == version_tag)
    )).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"model {version_tag} not found")

    reg_dict = _to_dto(r).model_dump()

    # Load run + score history + audit if we have a run_id.
    run_dict: dict | None = None
    scores: list[dict] = []
    budget: dict | None = None
    if r.run_id:
        run_row = (await db.execute(
            select(Run).where(Run.id == r.run_id)
        )).scalar_one_or_none()
        if run_row is not None:
            baseline_test, _ = baseline_and_best_test(
                run_row.history, run_row.validation_metric_direction
            )
            gain = improvement(
                run_row.champion_test_score,
                baseline_test,
                run_row.metric_direction,
            ) if (
                run_row.champion_test_score is not None
                and baseline_test is not None
            ) else None
            run_dict = {
                "id": run_row.id, "status": run_row.status,
                # The model card reports the HELD-OUT numbers: they are what
                # this checkpoint is worth on data nothing in the run tuned on.
                # The validation figures are kept alongside so the card can
                # show the gap, which is the honest measure of how much the
                # loop fitted itself to its own set.
                "champion_test_score": run_row.champion_test_score,
                "baseline_test_score": baseline_test,
                "improvement": gain,
                "best_validation_score": run_row.best_validation_score,
                "metric": run_row.metric,
                "iterations_completed": run_row.iterations_completed,
                "iteration_budget": run_row.iteration_budget,
                "started_at": run_row.started_at.isoformat() if run_row.started_at else "",
                "finished_at": run_row.finished_at.isoformat() if run_row.finished_at else "",
                "task_objective": run_row.task_objective,
                "agent_objective": run_row.agent_objective,
            }
            # Score history
            score_rows = (await db.execute(
                select(ScoreEvent).where(ScoreEvent.run_id == r.run_id)
                .order_by(ScoreEvent.ts)
            )).scalars().all()
            scores = [
                {"iteration": s.iteration, "source": s.source,
                 "split": s.split or "validation",
                 "score": s.score, "notes": s.notes}
                for s in score_rows
            ]
            # Budget
            try:
                from zevo.engine.cost.budget import snapshot_for_run
                snap = await snapshot_for_run(db, r.run_id)
                budget = {
                    "max_cost_usd": snap.max_cost_usd,
                    "spent_usd": snap.spent_usd,
                    "llm_cost_usd": snap.llm_cost_usd,
                    "gpu_cost_usd": snap.gpu_cost_usd,
                    "max_runtime_hours": snap.max_runtime_hours,
                    "elapsed_runtime_hours": snap.elapsed_runtime_hours,
                }
            except Exception:  # noqa: BLE001
                pass

    # Audit history for this registry tag
    audit_rows = (await db.execute(
        select(AuditEvent).where(
            AuditEvent.target_type == "registry_model",
            AuditEvent.target_id == version_tag,
        ).order_by(desc(AuditEvent.ts)).limit(20)
    )).scalars().all()
    audit_list = [
        {"ts": a.ts.isoformat() if a.ts else "",
         "event_type": a.event_type, "actor": a.actor, "summary": a.summary}
        for a in audit_rows
    ]

    md = render_model_card(
        registry=reg_dict, run=run_dict, score_history=scores,
        audit=audit_list, budget=budget,
    )
    return Response(content=md, media_type="text/markdown; charset=utf-8")
