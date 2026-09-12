"""GET/POST/PATCH/DELETE /tasks — the catalogue of named, runnable tasks.

Tasks are rows, not code. The shipped catalogue was seeded into the table by
migration b2c3d4e5f6a8, so a task the product came with and one the user typed
in are the same kind of thing: both listed here, both deletable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from zevo.paths import files_root, uploads_root

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import case, delete as sa_delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.api.database import get_db
from zevo.engine.observe.run_metrics import was_measured_on_heldout
from zevo.engine.method.score_direction import is_better
from zevo.db import Task, TaskSetting
from zevo.db.models import Run
from zevo.contracts.orchestrator import (
    TaskTestSet,
    UserRequest,
    validate_test_suite,
)
from zevo.contracts.training_methods import (
    method_config_errors,
    normalize_method_config,
)
from zevo.contracts.prompting import (
    canonical_prompt_contract,
    normalize_prompt_framing,
    validate_decoding_config,
    validate_inference_config,
    validate_loss_objective_config,
)
from zevo.holdout_storage import protect_assets


router = APIRouter()


_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_DATASETS_DIR = Path(files_root())
_UPLOAD_ROOT = Path(uploads_root())

def hub_url(ident: str) -> str:
    """Where a remote data reference actually lives, for a browser."""
    ident = (ident or "").strip()
    if not ident:
        return ""
    return ident if ident.startswith("http") else f"https://huggingface.co/datasets/{ident}"


def _dataset_declaring(hub_id: str) -> tuple[str, str]:
    """The catalogue entry that lists `hub_id` as one of its remote files,
    and the URL that entry recorded for it.

    A hub id in a task is still catalogue data — the dataset that owns it just
    keeps it remote instead of on disk. Finding it here means the Tasks list can
    name the dataset rather than showing a bare hub id with nothing behind it.
    """
    if not _DATASETS_DIR.is_dir():
        return "", ""
    for d in sorted(p for p in _DATASETS_DIR.iterdir() if p.is_dir()):
        f = d / "source.json"
        if not f.is_file():
            continue
        try:
            src = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for r in src.get("remote") or []:
            if isinstance(r, dict) and r.get("id") == hub_id:
                # The recorded url wins: a `url` entry is not a hub id.
                return d.name, str(r.get("url") or "") or hub_url(hub_id)
    return "", ""


def data_source(dataset: str) -> dict:
    """Where a task's training data comes from, as the UI needs to say it.

    Five answers, derived from the string itself rather than a column, so a task
    typed in today is classified the same way as one the migration seeded:

      registered  a dataset in the catalogue -> carries its NAME
      huggingface a hub id (`owner/name`, nothing on disk) -> say so, it is fetched
      uploaded    a file dropped into the launch dialog, still in staging
      path        some other file the caller pointed at
      none        nothing given; Zevo acquires it (this is what makes L3/L4)

    `detail` is the repo-relative path where that reads better than the absolute
    container one — "data/files/medqa-usmle/test.csv" says where a file came
    from; "/app/data/files/medqa-usmle/test.csv" says it with more noise.
    """
    d = (dataset or "").strip()
    if not d:
        return {"kind": "none", "name": "", "detail": "", "remote": False, "url": ""}
    p = Path(d)
    try:
        rel = p.relative_to(_DATASETS_DIR)
        # data/files/<name>/<path…> -> the catalogue entry is <name> and
        # the detail is the rest of the path, NOT just the filename. A bundle
        # keeps its files in train/ validation/ test/ now, so `test_eval.py`
        # alone named three different files and pointed at none of them.
        return {"kind": "registered", "name": rel.parts[0],
                "detail": "/".join(rel.parts[1:]) or p.name,
                "remote": False, "url": ""}
    except ValueError:
        pass
    # A hub id is `owner/name`: two segments, no leading slash, no suffix.
    if not d.startswith("/") and d.count("/") == 1 and not Path(d).suffix:
        owner, url = _dataset_declaring(d)
        if owner:
            # In the catalogue AND fetched from the hub — both are true, and the
            # UI wants to say both.
            return {"kind": "registered", "name": owner, "detail": d,
                    "remote": True, "url": url}
        return {"kind": "huggingface", "name": d, "detail": "fetched from HuggingFace",
                "remote": True, "url": hub_url(d)}
    rel = d[len("/app/"):] if d.startswith("/app/") else d
    try:
        p.relative_to(_UPLOAD_ROOT)
        # data/uploads/<uuid>/<file> — staged by the launch dialog, never
        # named or catalogued.
        return {"kind": "uploaded", "name": p.name, "detail": rel, "remote": False, "url": ""}
    except ValueError:
        pass
    return {"kind": "path", "name": p.name, "detail": rel, "remote": False, "url": ""}


def autonomy_level(dataset: str, base_model: str, training_method: str) -> str:
    """Return the canonical ownership rung for a Setting.

    The ladder releases one specific decision at a time:

        L1  user pins data + model + method
        L2  Zevo decides data; user pins model + method
        L3  Zevo decides data + method; user pins model
        L4  Zevo decides data + model + method

    The UI still permits an expert to pin a non-ladder combination. Calling
    that L2 merely because two boxes are filled would misstate who owns which
    decision, so those configurations are labelled Custom.
    """
    pins = tuple(bool((v or "").strip()) for v in (dataset, base_model, training_method))
    return {
        (True, True, True): "L1",
        (False, True, True): "L2",
        (False, True, False): "L3",
        (False, False, False): "L4",
    }.get(pins, "Custom")


# ───────────────────────────── task vs. setting ──────────────────────────────
#
# A task is the PROBLEM: an objective plus the files that define and score it.
# A setting is one way of attacking it — which base model, which training
# method, whether the training data is handed over — and the autonomy level is
# just how much of that setting is pinned down. One task therefore has many
# settings, and the catalogue used to spell that out by shipping l1-med,
# l2-med, l3-med and l4-med as four tasks that differed in nothing else.
#
# Because the three decisions used to be baked into each task's objective PROSE
# ("Train on the PROVIDED training set."), a task's stored objective is now the
# problem only, and the sentences that describe the setting are appended when a
# run is created. Same sentences the catalogue always used, so the agents read
# what they have always read — they just follow the setting the run picked
# instead of the one frozen into the task's name.

def setting_clauses(*, dataset: str, base_model: str, training_method: str) -> list[str]:
    """The three sentences that tell the agents what this run pins down."""
    return [
        "Train on the PROVIDED training set."
        if (dataset or "").strip() else
        "NO training set is provided — acquire and curate the training data yourself.",

        f"Use the {training_method.strip()} training method."
        if (training_method or "").strip() else
        "Choose the training method(s) yourself.",

        f"Fine-tune {base_model.strip()}."
        if (base_model or "").strip() else
        "Pick the base model yourself.",
    ]


def compose_objective(base_objective: str, *, dataset: str, base_model: str,
                      training_method: str) -> str:
    """A task's problem statement, plus what THIS run's setting decides."""
    base = (base_objective or "").strip()
    return " ".join([base, *setting_clauses(
        dataset=dataset, base_model=base_model, training_method=training_method,
    )]).strip()


class TaskBody(BaseModel):
    """A reusable problem definition, independent of any experiment setting."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    task_objective: str = ""
    test_sets: list[TaskTestSet] = Field(min_length=1)

    @field_validator("test_sets")
    @classmethod
    def unique_test_set_names(cls, value: list[TaskTestSet]) -> list[TaskTestSet]:
        return validate_test_suite(value)


class TaskPatch(BaseModel):
    """Edit payload. Every field is optional and only what was SENT is written,
    so an omitted scoring asset is not mistaken for a request to erase it.

    `name` is absent on purpose: it is the primary key, and runs record
    `task_name` as plain text, so renaming would detach a task from its own
    history rather than carrying it along.
    """

    model_config = ConfigDict(extra="forbid")

    task_objective: str | None = None
    test_sets: list[TaskTestSet] | None = Field(None, min_length=1)

    @field_validator("test_sets")
    @classmethod
    def unique_optional_test_set_names(
        cls, value: list[TaskTestSet] | None,
    ) -> list[TaskTestSet] | None:
        return None if value is None else validate_test_suite(value)


class TaskDTO(BaseModel):
    name: str
    task_objective: str
    test_sets: list[TaskTestSet]
    # Primary projection retained on the API while Run/Setting clients move to
    # the suite contract. The Tasks UI renders ``test_sets`` directly.
    test_set: str
    test_answer_fields: list[str]
    test_sample_submission: str
    metric_type: Literal["builtin", "custom"]
    evaluation_script: str
    evaluator_sha256: str
    metric: str
    metric_direction: Literal["max", "min"]
    run_count: int = 0
    last_run_at: str = ""
    best_test_score: float | None = None
    created_at: str = ""


def _stored_test_sets(t: Task) -> list[TaskTestSet]:
    values = list(t.test_sets or [])
    if values:
        return validate_test_suite([
            TaskTestSet.model_validate(value) for value in values
        ])
    # Shipped catalogue rows predate suites. Treat their one Test contract as
    # a one-item suite until they are edited; no Run ever receives two shapes.
    return [TaskTestSet(
        name=t.name,
        test_set=t.test_set or "",
        inference_query=t.task_objective or "Answer the input.",
        sample_submission=t.test_sample_submission or "",
        metric_type=t.metric_type,
        metric=t.metric,
        answer_fields=list(t.test_answer_fields or []),
        metric_direction=t.metric_direction,
        evaluation_script=t.evaluation_script or "",
        evaluator_sha256=t.evaluator_sha256 or "",
    )]


def _suite_headline(items: list[TaskTestSet]) -> tuple[str, Literal["max", "min"]]:
    if len(items) == 1:
        return items[0].metric, items[0].metric_direction
    return "suite_average", items[0].metric_direction


async def _protect_test_suite(items: list[TaskTestSet]) -> list[TaskTestSet]:
    """Freeze every Test asset, including a custom scorer when selected."""
    from zevo.evaluator_storage import freeze_evaluator

    protected: list[TaskTestSet] = []
    for item in items:
        test_set, sample = await run_in_threadpool(
            protect_assets, item.test_set, item.sample_submission,
        )
        evaluation_script = ""
        evaluator_sha256 = ""
        if item.metric_type == "custom":
            try:
                evaluation_script, evaluator_sha256 = await run_in_threadpool(
                    freeze_evaluator, item.evaluation_script,
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        protected.append(item.model_copy(update={
            "test_set": test_set,
            "sample_submission": sample,
            "evaluation_script": evaluation_script,
            "evaluator_sha256": evaluator_sha256,
        }))
    return protected


def _dto(t: Task, stats: dict | None = None) -> TaskDTO:
    st = stats or {}
    test_sets = _stored_test_sets(t)
    best_key = "best_min" if t.metric_direction == "min" else "best_max"
    return TaskDTO(
        name=t.name,
        # Run history for this task. The list sorts on these, so they travel
        # with the task rather than being stitched together in the browser.
        run_count=st.get("run_count", 0),
        last_run_at=st.get("last_run_at", ""),
        best_test_score=st.get(best_key),
        # When the task entered the catalogue. Distinct from last_run_at: a task
        # defined months ago and never run has one and not the other.
        created_at=t.created_at.isoformat() if t.created_at else "",
        task_objective=t.task_objective or "",
        test_sets=test_sets,
        test_set=t.test_set or "",
        test_answer_fields=list(t.test_answer_fields or []),
        test_sample_submission=t.test_sample_submission or "",
        metric_type=t.metric_type,
        evaluation_script=t.evaluation_script or "",
        evaluator_sha256=t.evaluator_sha256 or "",
        metric=t.metric,
        metric_direction=t.metric_direction,
    )


def task_to_user_request(t: Task) -> UserRequest:
    """The row as the orchestrator's input contract."""
    test_sets = _stored_test_sets(t)
    primary = test_sets[0]
    return UserRequest(
        task_objective=t.task_objective or "",
        test_sets=test_sets,
        metric=primary.metric,
        metric_direction=primary.metric_direction,
        metric_type=primary.metric_type,
        evaluation_script=primary.evaluation_script,
        evaluator_sha256=primary.evaluator_sha256,
        # A Task-only launch has no saved Setting to supply Validation. Start
        # with an explicit independent contract equal in value to Test; the UI
        # and CLI replace it when a Setting or per-Run Validation choice exists.
        validation_metric=primary.metric,
        validation_metric_direction=primary.metric_direction,
        validation_metric_type=primary.metric_type,
        validation_evaluation_script=primary.evaluation_script,
        validation_evaluator_sha256=primary.evaluator_sha256,
        training_method="",
        dataset="",
        data_query="",
        base_model="",
        test_set=primary.test_set,
        test_answer_fields=list(primary.answer_fields),
        test_sample_submission=primary.sample_submission,
        constraints=[],
    )


@router.get("/tasks", response_model=list[TaskDTO])
async def list_tasks(db: AsyncSession = Depends(get_db)) -> list[TaskDTO]:
    rows = (await db.execute(select(Task).order_by(Task.name))).scalars().all()
    # One grouped pass over runs instead of a query per task. Runs keep
    # task_name as plain text, so a run whose task was deleted simply has no
    # task row to attach to — it is skipped here, not lost.
    agg = (await db.execute(
        select(Run.task_name,
               func.count().label("n"),
               func.max(Run.started_at).label("last"),
               # The reported number is the held-out one — see leaderboard.
               # Conditional, not a WHERE: a run that predates the held-out
               # lane is still a run of this task and belongs in `n` and in
               # `last`. What it cannot do is supply `best`, because its score
               # was measured on the set it tuned on.
               func.max(
                   case((was_measured_on_heldout(), Run.champion_test_score))
               ).label("best_max"),
               func.min(
                   case((was_measured_on_heldout(), Run.champion_test_score))
               ).label("best_min"))
        .where(Run.task_name != "")
        .group_by(Run.task_name)
    )).all()
    stats = {
        r.task_name: {
            "run_count": int(r.n or 0),
            "last_run_at": r.last.isoformat() if r.last else "",
            "best_max": float(r.best_max) if r.best_max is not None else None,
            "best_min": float(r.best_min) if r.best_min is not None else None,
        }
        for r in agg
    }
    return [_dto(t, stats.get(t.name)) for t in rows]


@router.post("/tasks", status_code=201, response_model=TaskDTO)
async def create_task(body: TaskBody, db: AsyncSession = Depends(get_db)) -> TaskDTO:
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(400, "name is required.")
    if await db.get(Task, name):
        raise HTTPException(409, f"task {name!r} already exists — delete it first, or pick another name.")
    if not (body.task_objective or "").strip():
        raise HTTPException(400, "task_objective is required — it is what the agents are asked to achieve.")
    protected = await _protect_test_suite(body.test_sets)
    primary = protected[0]
    metric, metric_direction = _suite_headline(protected)
    single = len(protected) == 1

    t = Task(
        name=name,
        task_objective=body.task_objective.strip(),
        test_sets=[item.model_dump(mode="json") for item in protected],
        test_set=primary.test_set,
        test_answer_fields=list(primary.answer_fields),
        test_sample_submission=primary.sample_submission,
        metric_type=primary.metric_type if single else "builtin",
        evaluation_script=primary.evaluation_script if single else "",
        evaluator_sha256=primary.evaluator_sha256 if single else "",
        metric=metric,
        metric_direction=metric_direction,
    )
    db.add(t)
    await db.commit()
    await db.refresh(t)
    return _dto(t)


@router.patch("/tasks/{name}", response_model=TaskDTO)
async def update_task(name: str, body: TaskPatch, db: AsyncSession = Depends(get_db)) -> TaskDTO:
    """Edit an existing task in place.

    Runs already started keep the definition they were launched with (they copy
    it into the Run), so editing a task changes only future launches.
    """
    t = await db.get(Task, name)
    if t is None:
        raise HTTPException(404, f"task {name!r} not found.")

    changed = body.model_dump(exclude_unset=True)
    if "task_objective" in changed and not (changed["task_objective"] or "").strip():
        raise HTTPException(400, "task_objective cannot be emptied.")
    if not changed:
        return _dto(t)

    evaluation_fields = ("test_sets",)

    def normalized(field_name: str, value: Any) -> Any:
        if field_name == "test_sets":
            return tuple(
                tuple(sorted(TaskTestSet.model_validate(item).model_dump().items()))
                for item in (value or [])
            )
        return value.strip() if isinstance(value, str) else value

    changed_evaluation_fields = [
        field_name
        for field_name in evaluation_fields
        if field_name in changed
        and changed[field_name] is not None
        and normalized(field_name, changed[field_name])
        != normalized(field_name, getattr(t, field_name))
    ]
    if changed_evaluation_fields:
        run_count = int((await db.execute(
            select(func.count()).select_from(Run).where(Run.task_name == name)
        )).scalar_one())
        if run_count:
            raise HTTPException(
                409,
                "a task's evaluation contract is immutable after it has runs; "
                f"cannot change {', '.join(changed_evaluation_fields)}. Create "
                "a new task for a different test setup, metric, or target.",
            )

    if "test_sets" in changed and changed["test_sets"] is not None:
        supplied = [TaskTestSet.model_validate(item) for item in changed["test_sets"]]
        protected = await _protect_test_suite(supplied)
        primary = protected[0]
        single = len(protected) == 1
        t.test_sets = [item.model_dump(mode="json") for item in protected]
        t.test_set = primary.test_set
        t.test_answer_fields = list(primary.answer_fields)
        t.test_sample_submission = primary.sample_submission
        t.metric, t.metric_direction = _suite_headline(protected)
        t.metric_type = primary.metric_type if single else "builtin"
        t.evaluation_script = primary.evaluation_script if single else ""
        t.evaluator_sha256 = primary.evaluator_sha256 if single else ""

    for field_name, value in changed.items():
        if field_name == "test_sets":
            continue
        if value is None:
            continue  # sent as null = "leave it alone", same as omitting it
        if isinstance(value, list):
            value = [str(c).strip() for c in value if str(c).strip()]
        elif isinstance(value, str):
            value = value.strip()
        setattr(t, field_name, value)
    await db.commit()
    await db.refresh(t)
    return _dto(t)


@router.delete("/tasks/{name}")
async def delete_task(name: str, db: AsyncSession = Depends(get_db)) -> dict:
    if not await db.get(Task, name):
        raise HTTPException(404, f"task {name!r} not found.")
    # Runs store task_name as a plain string, so deleting a definition never
    # orphans or rewrites run history.
    await db.execute(sa_delete(Task).where(Task.name == name))
    await db.commit()
    return {"deleted": True, "name": name}


class SettingBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    """A setting the user writes down, before or without ever running it."""

    # What to call it. "" keeps the name it has, or takes the next `sN` when
    # there is none — `sN` is a default, not the scheme. It looks like a
    # sequence number and is not one: names are never reused after a delete, so
    # the list can read s1, s2, s5 and leave the reader counting five. A name
    # the user chose says which line of attack this is; `s5` only says it was
    # the fifth ever written down.
    name: str = Field("", max_length=32)
    dataset: str = ""
    # Only meaningful when dataset is a hub id.
    dataset_split: str = ""
    dataset_config: str = ""
    # What runs on this setting TUNE against. "" = each run carves 20% from
    # Test, with a minimum of 200 rows (zevo.engine.method.validation_split).
    validation_set: str = ""
    # Only meaningful when validation_set is a hub id: which slice, and which
    # named subset. Run creation fetches that slice to a file.
    validation_split: str = ""
    validation_config: str = ""
    validation_answer_fields: list[str] = Field(default_factory=list)
    validation_sample_submission: str = ""
    validation_metric_type: Literal["builtin", "custom"] = "builtin"
    validation_metric: str = Field("token_f1", min_length=1, max_length=64)
    validation_metric_direction: Literal["max", "min"] = "max"
    validation_evaluation_script: str = ""
    base_model: str = ""
    training_method: str = ""
    method_config: dict[str, Any] = Field(default_factory=dict)
    prompt_framing: str = ""
    system_prompt: str = ""
    loss_objective_config: dict[str, Any] = Field(default_factory=dict)
    inference_config: dict[str, Any] = Field(default_factory=dict)
    decoding_config: dict[str, Any] = Field(default_factory=dict)
    data_query: str = ""
    model_query: str = ""
    method_query: str = ""
    iteration_budget: int = Field(0, ge=0)
    max_cost_usd: float = Field(0.0, ge=0.0)
    stop_threshold: float | None = Field(None, allow_inf_nan=False)


class SettingDTO(BaseModel):
    id: str
    name: str
    created_at: str
    dataset: str
    dataset_split: str
    dataset_config: str
    data_source: dict[str, Any]
    validation_set: str
    validation_data_source: dict[str, Any]
    validation_split: str
    validation_config: str
    validation_answer_fields: list[str]
    validation_sample_submission: str
    validation_metric_type: Literal["builtin", "custom"]
    validation_metric: str
    validation_metric_direction: Literal["max", "min"]
    validation_evaluation_script: str
    validation_evaluator_sha256: str
    base_model: str
    training_method: str
    method_config: dict[str, Any]
    prompt_framing: str
    system_prompt: str
    loss_objective_config: dict[str, Any]
    inference_config: dict[str, Any]
    decoding_config: dict[str, Any]
    data_query: str
    model_query: str
    method_query: str
    iteration_budget: int
    max_cost_usd: float
    stop_threshold: float | None
    run_count: int
    best_test_score: float | None
    last_run_at: str
    last_run_id: str
    last_run_name: str
    level: str


class SettingMatchDTO(BaseModel):
    match: SettingDTO | None = None


def _setting_dto(s: TaskSetting, *, run_count: int = 0, best_test_score: float | None = None,
                 last_run_at: str = "", last_run_id: str = "", last_run_name: str = "") -> dict:
    return {
        "id": s.id,
        "name": s.name or "",
        "created_at": s.created_at.isoformat() if s.created_at else "",
        "dataset": s.dataset or "",
        "dataset_split": s.dataset_split or "",
        "dataset_config": s.dataset_config or "",
        "data_source": data_source(s.dataset or ""),
        "validation_set": s.validation_set or "",
        # Classified the same way as the training data, so the UI can tell
        # whether the two came out of one dataset folder or two.
        "validation_data_source": data_source(s.validation_set or ""),
        "validation_split": s.validation_split or "",
        "validation_config": s.validation_config or "",
        "validation_answer_fields": list(s.validation_answer_fields or []),
        "validation_sample_submission": s.validation_sample_submission or "",
        "validation_metric_type": s.validation_metric_type,
        "validation_metric": s.validation_metric,
        "validation_metric_direction": s.validation_metric_direction,
        "validation_evaluation_script": s.validation_evaluation_script or "",
        "validation_evaluator_sha256": s.validation_evaluator_sha256 or "",
        "base_model": s.base_model or "",
        "training_method": s.training_method or "",
        "method_config": dict(s.method_config or {}),
        "prompt_framing": s.prompt_framing or "",
        "system_prompt": s.system_prompt or "",
        "loss_objective_config": dict(s.loss_objective_config or {}),
        "inference_config": dict(s.inference_config or {}),
        "decoding_config": dict(s.decoding_config or {}),
        "data_query": s.data_query or "",
        "model_query": s.model_query or "",
        "method_query": s.method_query or "",
        "iteration_budget": int(s.iteration_budget or 0),
        "max_cost_usd": float(s.max_cost_usd or 0.0),
        "stop_threshold": s.stop_threshold,
        # Not stored: three fields decide it, so deriving keeps it honest when
        # one of them is edited.
        "level": autonomy_level(s.dataset or "", s.base_model or "", s.training_method or ""),
        # How it has done so far. The table does not show these, but ordering
        # does — "frequent" is what puts the setting you keep returning to on top.
        "run_count": run_count,
        "best_test_score": best_test_score,
        "last_run_at": last_run_at,
        "last_run_id": last_run_id,
        "last_run_name": last_run_name,
    }


def _clean_setting_name(raw: str) -> str:
    """A name, or "" for none. One line, trimmed, and short enough to sit in a
    table cell — the leaderboard prints it beside every score."""
    n = " ".join((raw or "").split())
    return n[:32]


def _name_taken(name: str, others: list) -> bool:
    """Two settings of one task sharing a name is two rows the leaderboard
    cannot tell apart, which is the whole reason a setting has a name."""
    n = name.strip().lower()
    return bool(n) and any((s.name or "").strip().lower() == n for s in others)


def _next_setting_name(existing: list) -> str:
    """`s1`, `s2`, … — one past the highest ever handed out for this task.

    Counting rows instead would reuse a name after a delete, and two runs of
    "s2" that meant different things is worse than a gap in the sequence.
    """
    highest = 0
    for s in existing:
        n = (s.name or "").strip()
        if n.startswith("s") and n[1:].isdigit():
            highest = max(highest, int(n[1:]))
    return f"s{highest + 1}"


# Every user-owned field a setting stores, and therefore every field that
# distinguishes one from another. The model-derived reasoning type is not a
# Setting field. `id`, `name` and `created_at` are not here: a label and a
# timestamp do not make a different experiment. `task_name` is not here either
# — settings are only ever compared within one task.
#
# Budgets used to be excluded, on the reasoning that "capping a rerun at $50
# does not make it a different line of attack". But a setting stores its
# budget, so leaving it out made changing the cap say "same as the saved
# setting, so nothing new is saved" — while the stored 0 stayed on the row,
# describing runs that never honoured it. A field the row keeps is a field that
# tells two rows apart; anything else keeps a number nobody maintains.
_SETTING_IDENTITY_FIELDS = (
    "dataset", "dataset_split", "dataset_config", "data_query",
    "base_model", "model_query", "training_method", "method_query", "method_config",
    "prompt_framing", "system_prompt",
    "loss_objective_config", "inference_config", "decoding_config",
    "validation_set", "validation_split", "validation_config",
    "validation_answer_fields",
    "validation_sample_submission",
    "validation_metric_type", "validation_metric",
    "validation_metric_direction", "validation_evaluation_script",
    "iteration_budget", "max_cost_usd", "stop_threshold",
)


# The two numeric ones. They need their own default because for them "absent"
# and `0` are the SAME statement — no cap — while for a text field absent and
# empty are both "", and a list's is (). Without this, a body that omits
# `max_cost_usd` and one that sends 0 describe one setting and hash to two.
_SETTING_NUMERIC_FIELDS = frozenset({"iteration_budget", "max_cost_usd"})
_SETTING_OPTIONAL_NUMERIC_FIELDS = frozenset({"stop_threshold"})

# Stored as a list, but a form sends it as the comma-separated string the user
# typed. Both have to reduce to the same tuple or a launch never matches the
# row it came from.
_SETTING_LIST_FIELDS = frozenset({"validation_answer_fields"})

# With no independent Validation set, every field below is resolved from the
# Task's Test contract when a Run starts. Older rows persisted some of those
# inherited values while newer rows keep the empty derived-value sentinel.
# They describe the same experiment and must therefore have one identity.
_DERIVED_VALIDATION_IDENTITY_FIELDS = frozenset({
    "validation_split",
    "validation_config",
    "validation_answer_fields",
    "validation_sample_submission",
    "validation_metric_type",
    "validation_metric",
    "validation_metric_direction",
    "validation_evaluation_script",
})


def _norm_setting_value(field: str, v) -> object:
    """One shape per value, so equal configurations compare equal.

    The same field arrives as a str from a form, a list from a stored row and
    None from a request that omitted it, and `("a",) != ["a"]` would split a
    setting from itself.
    """
    if field in _SETTING_NUMERIC_FIELDS:
        try:
            return float(v or 0)
        except (TypeError, ValueError):
            return 0.0
    if field in _SETTING_OPTIONAL_NUMERIC_FIELDS:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    if field in _SETTING_LIST_FIELDS:
        items = v.split(",") if isinstance(v, str) else list(v or ())
        return tuple(str(x).strip() for x in items if str(x).strip())
    if field in {
        "method_config", "loss_objective_config", "inference_config",
        "decoding_config",
    }:
        if isinstance(v, str):
            try:
                v = json.loads(v) if v.strip() else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                return v.strip()
        return json.dumps(v or {}, sort_keys=True, separators=(",", ":"))
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return tuple(str(x).strip() for x in v if str(x).strip())
    return str(v).strip()


def setting_identity(src) -> tuple:
    """What makes two settings the same experiment: all of it.

    `src` is anything carrying the fields — a `TaskSetting` row, a request
    body, or a plain dict — so the one rule serves the save path, the match
    endpoint and the run-creation check without any of them restating it.

    The GPU provider and the generation backend are NOT in it, and are not
    stored on a setting at all: renting a box today and using your own cluster
    tomorrow does not make two experiments out of one.

    The leaderboard uses this same identity when it groups harness runs by
    Setting, so saved settings and comparison groups cannot silently disagree.
    """
    get = src.get if isinstance(src, dict) else lambda k, d=None: getattr(src, k, d)
    derived_validation = not str(get("validation_set") or "").strip()
    return tuple(
        _norm_setting_value(
            field,
            "" if derived_validation and field in _DERIVED_VALIDATION_IDENTITY_FIELDS
            else get(field),
        )
        for field in _SETTING_IDENTITY_FIELDS
    )


async def _run_stats(db: AsyncSession, name: str) -> dict[tuple, dict]:
    """{setting key: how its runs went}, read from the runs themselves.

    A run records the request it was launched with in its supervisor ticket, so
    this is what actually happened rather than what the row says today.
    """
    from zevo.db.models import Ticket

    runs = (await db.execute(
        select(Run).where(Run.task_name == name).order_by(Run.started_at.desc())
    )).scalars().all()
    # Which of them the reported score may come from. The rest still COUNT as
    # runs of the setting; they just cannot supply `best_test_score`, because the
    # number they carry was measured on the set they tuned on. Same rule as the
    # leaderboard, so the two pages cannot disagree about one task.
    measured = set((await db.execute(
        select(Run.id).where(Run.task_name == name, was_measured_on_heldout())
    )).scalars().all())
    if not runs:
        return {}
    sup_ids = [r.supervisor_ticket_id for r in runs if r.supervisor_ticket_id]
    payloads: dict[str, dict] = {}
    if sup_ids:
        for t in (await db.execute(select(Ticket).where(Ticket.id.in_(sup_ids)))).scalars().all():
            if isinstance(t.payload, dict):
                payloads[t.id] = t.payload.get("user_request") or {}

    out: dict[tuple, dict] = {}
    for r in runs:
        req = payloads.get(r.supervisor_ticket_id or "", {})
        # The caps come off the RUN, not the request: `UserRequest` does not
        # carry them, the Run row is where they landed. Everything else is the
        # request the run was actually launched with.
        key = setting_identity({
            **req,
            "iteration_budget": r.iteration_budget,
            "max_cost_usd": r.max_cost_usd,
            "stop_threshold": r.stop_threshold,
        })
        st = out.setdefault(key, {
            "run_count": 0, "best_test_score": None,
            # Runs are newest first, so the first one seen is the latest.
            "last_run_at": r.started_at.isoformat() if r.started_at else "",
            "last_run_id": r.id, "last_run_name": r.run_name or "",
        })
        st["run_count"] += 1
        best = r.champion_test_score if r.id in measured else None
        if best is not None and is_better(
            float(best), st["best_test_score"], r.metric_direction
        ):
            st["best_test_score"] = float(best)
    return out


@router.get("/tasks/{name}/settings", response_model=list[SettingDTO])
async def task_settings(name: str, db: AsyncSession = Depends(get_db)) -> list[dict]:
    """The settings this task can be attacked with, most-run first.

    Rows, not a derivation: a run records its setting here, and the user can
    write one down that has never been run. Ordering still comes from the run
    history, so the one you keep returning to stays at the top.

    Tasks that no longer exist still answer — the settings of a deleted or
    renamed task are still its settings.
    """
    rows = (await db.execute(
        select(TaskSetting).where(TaskSetting.task_name == name)
    )).scalars().all()
    stats = await _run_stats(db, name)
    out = [_setting_dto(s, **stats.get(setting_identity(s), {})) for s in rows]
    # Oldest first — the order they were added, which is also the order the
    # names were handed out in, so the list reads s1, s2, s3. They used to be
    # sorted most-run-first, from back when the block was called "frequent
    # settings" and the rows had no names; once a row is called `s2`, a list
    # that puts it above s1 is one you have to read twice to trust.
    def _seq(d: dict) -> tuple:
        n = (d.get("name") or "").strip()
        rank = int(n[1:]) if n.startswith("s") and n[1:].isdigit() else 1 << 30
        return (d.get("created_at") or "", rank)

    out.sort(key=_seq)
    return out


@router.get("/tasks/{name}/settings/match", response_model=SettingMatchDTO)
async def match_task_setting(
    name: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """The setting this configuration already IS, or null.

    Answered here rather than compared in the browser so the rule lives in one
    place: `setting_identity` decides what makes two settings the same experiment,
    and a second copy of that decision in the UI is one that drifts.
    """
    # Read straight off the query string rather than declaring each field: the
    # identity is a list in one place, and a signature restating it is a second
    # copy that goes stale the next time a field joins the row. A caller that
    # omits a field is saying it is empty, which is what the normalizer means
    # by absent.
    candidate = {
        k: v for k, v in request.query_params.items()
        if k in _SETTING_IDENTITY_FIELDS
    }
    if not str(candidate.get("validation_set") or "").strip():
        task = await db.get(Task, name)
        if task is not None:
            candidate.update({
                "validation_answer_fields": [],
                "validation_sample_submission": "",
                "validation_metric_type": task.metric_type,
                "validation_metric": task.metric,
                "validation_metric_direction": task.metric_direction,
                "validation_evaluation_script": task.evaluation_script or "",
            })
    key = setting_identity(candidate)
    rows = (await db.execute(
        select(TaskSetting).where(TaskSetting.task_name == name)
    )).scalars().all()
    hit = next((s for s in rows if setting_identity(s) == key), None)
    return {"match": _setting_dto(hit) if hit is not None else None}


def _validate_named_validation_assets(body: SettingBody) -> None:
    if not body.validation_set.strip():
        return
    missing = []
    if not body.validation_answer_fields:
        missing.append("validation_answer_fields")
    if not body.validation_sample_submission.strip():
        missing.append("validation_sample_submission")
    if missing:
        raise HTTPException(
            400,
            "validation_set requires " + ", ".join(missing),
        )


async def _freeze_validation_metric(
    body: SettingBody,
) -> tuple[SettingBody, str]:
    """Validate one Setting-owned metric and freeze custom evaluator bytes."""
    metric = body.validation_metric.strip().lower()
    if body.validation_metric_type == "builtin":
        if metric not in BUILTIN_METRICS:
            raise HTTPException(
                400,
                f"unknown built-in Validation metric {body.validation_metric!r}; "
                "built-ins are " + ", ".join(sorted(BUILTIN_METRICS)),
            )
        if body.validation_evaluation_script.strip():
            raise HTTPException(
                400, "built-in Validation metric cannot include an evaluator script",
            )
        return body.model_copy(update={
            "validation_metric": metric,
            "validation_evaluation_script": "",
        }), ""

    if not body.validation_evaluation_script.strip():
        raise HTTPException(
            400, "custom Validation metric requires an evaluator script",
        )
    from zevo.evaluator_storage import freeze_evaluator
    try:
        script, digest = await run_in_threadpool(
            freeze_evaluator, body.validation_evaluation_script.strip(),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return body.model_copy(update={
        "validation_metric": body.validation_metric.strip(),
        "validation_evaluation_script": script,
    }), digest


async def _resolve_setting_validation_contract(
    body: SettingBody, task: Task,
) -> tuple[SettingBody, str]:
    """Inherit Task scoring unless this Setting supplies Validation data."""
    if body.validation_set.strip():
        return await _freeze_validation_metric(body)
    return body.model_copy(update={
        "validation_answer_fields": [],
        "validation_sample_submission": "",
        "validation_metric_type": task.metric_type,
        "validation_metric": task.metric,
        "validation_metric_direction": task.metric_direction,
        "validation_evaluation_script": task.evaluation_script or "",
    }), task.evaluator_sha256 or ""


def _validate_method_config(body: SettingBody) -> None:
    errors = method_config_errors(
        body.training_method, body.method_config, require_dependencies=True,
    )
    if errors:
        raise HTTPException(400, "; ".join(errors))


def _validate_setting_decision_inputs(body: SettingBody) -> dict[str, Any]:
    """Validate and normalize the Setting-owned experiment preferences."""
    try:
        framing = normalize_prompt_framing(body.prompt_framing) if body.prompt_framing else ""
        system = body.system_prompt.strip()
        if framing:
            framing, _reasoning_type, system = canonical_prompt_contract(
                prompt_framing=framing,
                model_reasoning_type="non_thinking",
                system_prompt=system,
            )
        elif system:
            raise ValueError("system_prompt requires prompt_framing")
        loss = validate_loss_objective_config(
            body.training_method, body.loss_objective_config,
        )
        if loss and not body.training_method.strip():
            raise ValueError("loss_objective_config requires training_method")
        return {
            "prompt_framing": framing,
            "system_prompt": system,
            "loss_objective_config": loss,
            "inference_config": validate_inference_config(body.inference_config),
            "decoding_config": validate_decoding_config(body.decoding_config),
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/tasks/{name}/settings", status_code=201, response_model=SettingDTO)
async def create_task_setting(
    name: str, body: SettingBody, db: AsyncSession = Depends(get_db),
) -> dict:
    """Write a setting down without running it.

    An identical one is returned rather than duplicated: the list is a set of
    ways to attack the problem, and the same way twice is one way.
    """
    task = await db.get(Task, name)
    if task is None:
        raise HTTPException(404, f"task {name!r} not found.")
    _validate_named_validation_assets(body)
    body, validation_evaluator_sha256 = await _resolve_setting_validation_contract(
        body, task,
    )
    _validate_method_config(body)
    contracts = _validate_setting_decision_inputs(body)
    key = setting_identity({**body.model_dump(), **contracts})
    existing = (await db.execute(
        select(TaskSetting).where(TaskSetting.task_name == name)
    )).scalars().all()
    for s in existing:
        if setting_identity(s) == key:
            return _setting_dto(s)

    wanted = _clean_setting_name(body.name)
    if _name_taken(wanted, existing):
        raise HTTPException(409, f"another setting on {name!r} is already called {wanted!r}.")

    row = TaskSetting(
        task_name=name,
        name=wanted or _next_setting_name(existing),
        dataset=body.dataset.strip(),
        dataset_split=body.dataset_split.strip(),
        dataset_config=body.dataset_config.strip(),
        validation_set=body.validation_set.strip(),
        validation_split=body.validation_split.strip(),
        validation_config=body.validation_config.strip(),
        validation_answer_fields=[c.strip() for c in (body.validation_answer_fields or []) if c.strip()],
        validation_sample_submission=body.validation_sample_submission.strip(),
        validation_metric_type=body.validation_metric_type,
        validation_metric=body.validation_metric,
        validation_metric_direction=body.validation_metric_direction,
        validation_evaluation_script=body.validation_evaluation_script,
        validation_evaluator_sha256=validation_evaluator_sha256,
        base_model=body.base_model.strip(),
        training_method=body.training_method.strip(),
        method_config=normalize_method_config(body.method_config),
        **contracts,
        data_query=body.data_query.strip(),
        model_query=body.model_query.strip(),
        method_query=body.method_query.strip(),
        iteration_budget=max(0, body.iteration_budget),
        max_cost_usd=max(0.0, body.max_cost_usd),
        stop_threshold=body.stop_threshold,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _setting_dto(row)


@router.patch("/tasks/{name}/settings/{setting_id}", response_model=SettingDTO)
async def update_task_setting(
    name: str, setting_id: str, body: SettingBody, db: AsyncSession = Depends(get_db),
) -> dict:
    """Change what a setting says, in place.

    Editing rather than delete-and-re-add keeps the row's NAME: `s2` is what
    the runs on this task's board are labelled by, and handing the same line of
    attack a new name after a typo fix orphans every reading of it.

    Runs already made are untouched. Each carries its own copy of the request it
    was launched with, so this is a change to what the NEXT run gets, not a
    rewrite of what earlier ones did.
    """
    task = await db.get(Task, name)
    if task is None:
        raise HTTPException(404, f"task {name!r} not found.")
    _validate_named_validation_assets(body)
    body, validation_evaluator_sha256 = await _resolve_setting_validation_contract(
        body, task,
    )
    _validate_method_config(body)
    contracts = _validate_setting_decision_inputs(body)
    row = await db.get(TaskSetting, setting_id)
    if row is None or row.task_name != name:
        raise HTTPException(404, f"setting {setting_id!r} not found on task {name!r}.")

    key = setting_identity({**body.model_dump(), **contracts})
    others = (await db.execute(
        select(TaskSetting).where(
            TaskSetting.task_name == name, TaskSetting.id != setting_id)
    )).scalars().all()
    clash = next((s for s in others if setting_identity(s) == key), None)
    if clash is not None:
        # The list is a set of ways to attack the problem; two rows saying the
        # same thing would split one line of attack's history across both.
        raise HTTPException(
            409, f"that is the same setting as {clash.name or clash.id}.")

    wanted = _clean_setting_name(body.name)
    if wanted and _name_taken(wanted, others):
        raise HTTPException(409, f"another setting on {name!r} is already called {wanted!r}.")
    if wanted:
        row.name = wanted
    row.dataset = body.dataset.strip()
    row.dataset_split = body.dataset_split.strip()
    row.dataset_config = body.dataset_config.strip()
    row.validation_set = body.validation_set.strip()
    row.validation_split = body.validation_split.strip()
    row.validation_config = body.validation_config.strip()
    row.validation_answer_fields = [
        c.strip() for c in (body.validation_answer_fields or []) if c.strip()
    ]
    row.validation_sample_submission = body.validation_sample_submission.strip()
    row.validation_metric_type = body.validation_metric_type
    row.validation_metric = body.validation_metric
    row.validation_metric_direction = body.validation_metric_direction
    row.validation_evaluation_script = body.validation_evaluation_script
    row.validation_evaluator_sha256 = validation_evaluator_sha256
    row.base_model = body.base_model.strip()
    row.training_method = body.training_method.strip()
    row.method_config = normalize_method_config(body.method_config)
    row.prompt_framing = contracts["prompt_framing"]
    row.system_prompt = contracts["system_prompt"]
    row.loss_objective_config = contracts["loss_objective_config"]
    row.inference_config = contracts["inference_config"]
    row.decoding_config = contracts["decoding_config"]
    row.data_query = body.data_query.strip()
    row.model_query = body.model_query.strip()
    row.method_query = body.method_query.strip()
    row.iteration_budget = max(0, body.iteration_budget)
    row.max_cost_usd = max(0.0, body.max_cost_usd)
    row.stop_threshold = body.stop_threshold
    await db.commit()
    await db.refresh(row)
    return _setting_dto(row)


@router.delete("/tasks/{name}/settings/{setting_id}")
async def delete_task_setting(
    name: str, setting_id: str, db: AsyncSession = Depends(get_db),
) -> dict:
    """Take a setting off the list. Runs that used it are untouched: each keeps
    its own copy of the request it was launched with."""
    row = await db.get(TaskSetting, setting_id)
    if row is None or row.task_name != name:
        raise HTTPException(404, f"setting {setting_id!r} not found on task {name!r}.")
    await db.execute(sa_delete(TaskSetting).where(TaskSetting.id == setting_id))
    await db.commit()
    return {"deleted": True, "id": setting_id}


@router.get("/tasks/{name}", response_model=TaskDTO)
async def get_task(name: str, db: AsyncSession = Depends(get_db)) -> TaskDTO:
    """Return the same Task resource shape used by the collection endpoint."""
    rows = await list_tasks(db)
    task = next((row for row in rows if row.name == name), None)
    if task is None:
        raise HTTPException(404, f"task {name!r} not found.")
    return task
