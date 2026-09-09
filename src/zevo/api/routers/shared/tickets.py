"""Ticket creation, inspection, conversation, and execution controls."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.paths import work_dir_root
from zevo.api.database import get_db
from zevo.api.ui_access import is_trusted_ui_request
from zevo.db import (
    Agent as AgentRow,
    ExecutionEvent,
    HeartbeatResult,
    HeartbeatRun,
    InfraInstance,
    RegistryModel,
    Run,
    ScoreEvent,
    Ticket,
    TicketMessage,
    TicketNotice,
    WorkProduct,
    get_session_factory,
)
from zevo.contracts.customizations import RunCustomizations
from zevo.contracts.data import is_sha256
from zevo.contracts.model_registry import model_tag_for_run
from zevo.engine.method.score_direction import is_better
from zevo.contracts._base import StrictBody
from zevo.contracts.tickets import (
    TERMINAL_TICKET_STATUSES,
    CreateTicketBody,
    CreateTicketResponse,
    TicketInputFormat,
    TicketLane,
    TicketStatus,
    PipelineRegistryRequestPayload,
    PIPELINE_REQUEST_PAYLOAD_BY_AGENT,
    RegistryPayload,
    TrainPayload,
    validate_bindings,
    validate_stored_payload,
)


def _resolve_data_source_selection(payload: dict[str, Any]) -> dict[str, Any]:
    """Materialize a Hub split before a Data Ticket is stored.

    Pipeline Runs normally arrive with this already resolved. This also covers
    standalone Data Tickets.
    """
    out = dict(payload)
    dataset = str(out.get("dataset") or "").strip()
    if not dataset or str(out.get("dataset_split") or "").strip():
        return out
    from zevo.engine.remote_datasets import looks_like_hub_id, lookup
    if not looks_like_hub_id(dataset):
        return out
    spec = lookup(dataset, role="train")
    out["dataset_split"] = str(spec.split if spec is not None else "train")
    if not str(out.get("dataset_config") or "").strip() and spec is not None:
        out["dataset_config"] = str(spec.config or "")
    return out


from zevo.contracts.tickets import TERMINAL_RUN_STATUSES


router = APIRouter()


class TicketDTO(CreateTicketResponse):
    """The canonical Ticket row returned by list/detail APIs."""


class MessageDTO(BaseModel):
    id: str
    author: str
    body: str
    created_at: str
    triggered_wakeup_id: str = ""


class NoticeDTO(BaseModel):
    id: str
    code: str
    severity: str
    body: str
    created_at: str


class ResultDTO(BaseModel):
    id: str
    heartbeat_id: str
    agent_id: str
    status: Literal["succeeded", "degraded", "deferred", "failed"]
    output: dict[str, Any]
    created_at: str


from zevo.api.artifacts import host_path as _host_path, host_paths_in_text as _host_text


class WorkProductDTO(BaseModel):
    id: str
    role: str
    path: str
    local_path: str = ""  # where it lives on the host: runs/<run-id>/<ticket>/…
    meta: dict[str, Any]
    created_at: str


class ExecutionEventDTO(BaseModel):
    # Agent activation that received the event.
    heartbeat_id: str
    # Concrete process launch within the activation. Retries deliberately get
    # a new value so their step series never overwrite each other.
    attempt_id: str
    event_type: Literal["attempt", "phase", "progress"]
    phase: str
    current_step: int
    total_steps: int
    loss: float
    extras: dict[str, Any]
    ts: str


class TicketDetail(TicketDTO):
    messages: list[MessageDTO]
    notices: list[NoticeDTO]
    results: list[ResultDTO]
    work_products: list[WorkProductDTO]
    execution_events: list[ExecutionEventDTO]
    resolved_configs: dict[str, dict[str, Any]]


def _to_dto(t: Ticket) -> TicketDTO:
    # Rewrite in-container paths (/app/..., /app/data/runs/...) to HOST paths for
    # display. Display-only: the STORED payload keeps container paths that agents
    # read to open/scp files — the runner reads the DB row directly, not this DTO.
    from zevo.api.artifacts import host_paths_in
    return TicketDTO(
        id=t.id, run_id=t.run_id, agent_id=t.agent_id, status=t.status,
        input_format=t.input_format, lane=t.lane,
        iteration=int(t.iteration or 0), payload=host_paths_in(t.payload or {}),
        customization=host_paths_in(t.customization or {}),
        inputs=host_paths_in(t.inputs or {}),
        summary=t.summary or "", error_message=t.error_message or "",
        repair_attempts=int(t.repair_attempts or 0),
        repair_route=str(t.repair_route or ""),
        created_at=t.created_at.isoformat() if t.created_at else "",
        updated_at=t.updated_at.isoformat() if t.updated_at else "",
    )


@router.get("/tickets", response_model=list[TicketDTO])
async def list_tickets(
    request: Request,
    run_id: str = "",
    agent_id: str = "",
    status: str = "",
    limit: int = Query(200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
) -> list[TicketDTO]:
    """List Tickets through the caller's fixed held-out visibility boundary.

    The trusted dashboard can observe them live; Agent callers cannot until the
    Run is terminal. Visibility is never selected by a query flag.
    """
    q = select(Ticket).join(Run, Run.id == Ticket.run_id)
    if not is_trusted_ui_request(request):
        q = q.where(or_(
            Ticket.lane != "held_out_test",
            Run.status.in_(TERMINAL_RUN_STATUSES),
        ))
    q = q.order_by(desc(Ticket.created_at)).limit(limit)
    if run_id:
        q = q.where(Ticket.run_id == run_id)
    if agent_id:
        q = q.where(Ticket.agent_id == agent_id)
    if status:
        q = q.where(Ticket.status == status)
    rows = (await db.execute(q)).scalars().all()
    return [_to_dto(r) for r in rows]


# ---- POST /tickets ------------------------------------------------------

_TICKET_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# Map agent_id -> default ticket-id prefix (e.g. data -> "data")
# The id prefix per agent. These are the strings a reader sees in every ticket
# id, so they are part of the product's vocabulary, not an internal detail —
# Prefixes are compact identifiers only; Agent ownership comes from agent_id.
#
# `orchestrator` was `plan`, which no orchestrator ticket has ever carried: the
# supervisor id is built by hand as `orchestrate-<run8>-001` in two places
# (backend.runs and zevo.engine.run.runner), and this docstring below has said
# "orchestrate-<run8>-001 all along" while the table said otherwise. A ticket
# created through THIS endpoint would have been the one `plan-…` id in a
# database of `orchestrate-…` ones.
_AGENT_PREFIX = {
    "data":           "data",
    "infrastructure": "infra",
    "train":          "train",
    "inference":      "infer",
    "evaluation":     "eval",
    "registry":       "registry",
    "orchestrator":   "orchestrate",
}

def _stamp_pipeline_payload(
    *, run: Run, agent_id: str, payload: dict[str, Any],
) -> dict[str, Any]:
    """Make every persisted pipeline Specialist payload self-contained.

    The Orchestrator chooses stages and high-level advice; the API stamps exact
    user pins and the immutable Validation specification before validation and
    persistence. The stored Ticket is therefore the work order that actually
    runs. Held-out Tickets are engine-created elsewhere.
    """
    if run.mode == "single_stage":
        return payload
    spec = dict(run.holdout or {})
    from zevo.engine.run.runner import _validation_files
    evaluation_script, sample_submission = _validation_files(spec)
    stamped = dict(payload)
    pins = dict(run.decision_pins or {})

    def stamp_exact(name: str, value: Any) -> None:
        if value in (None, "", {}):
            return
        supplied = stamped.get(name)
        if supplied not in (None, "", {}) and supplied != value:
            raise ValueError(
                f"pipeline payload {name!r} conflicts with the user-owned Run pin"
            )
        stamped[name] = value

    if agent_id == "train":
        stamp_exact("base_model", pins.get("base_model"))
        stamp_exact("training_method_pin", pins.get("training_method"))
        stamp_exact("method_config_pins", dict(pins.get("method_config") or {}))
        stamp_exact(
            "loss_objective_pins",
            dict(pins.get("loss_objective_config") or {}),
        )
        return stamped
    if agent_id == "inference":
        stamp_exact("base_model", pins.get("base_model"))
        run_configuration_pins = {
            key: pins[key]
            for key in (
                "prompt_framing", "system_prompt",
                "inference_config", "decoding_config",
            )
            if pins.get(key) not in (None, "", {})
        }
        supplied = dict(stamped.get("configuration_pins") or {})
        conflicts = sorted(
            key for key, value in run_configuration_pins.items()
            if key in supplied and supplied[key] != value
        )
        if conflicts:
            raise ValueError(
                "pipeline Inference configuration conflicts with user-owned Run "
                f"pins: {conflicts}"
            )
        stamped["configuration_pins"] = {
            **supplied,
            **run_configuration_pins,
        }
    if agent_id == "data":
        effective_training = str(spec.get("training_dataset") or "")
        for name in (
            "dataset_source", "dataset", "dataset_split", "dataset_config",
            "data_query", "training_method",
        ):
            if effective_training and name == "dataset":
                # Every revision starts from the last prepared Training
                # artifact. Validation was already settled before Data ran;
                # a Test-derived set inherited Test's scoring contract there.
                stamped[name] = effective_training
            elif effective_training and name in {"dataset_split", "dataset_config"}:
                stamped[name] = ""
            else:
                stamp_exact(name, pins.get(name))
        operation = str(payload.get("operation") or "")
        if operation != "prepare_run_data":
            raise ValueError("public pipeline Data supports prepare_run_data only")
        # Training-data selection is a sealed capability boundary. Even if an
        # old or adversarial child request includes Validation facts, discard
        # them before validating/storing the Data work order. The engine binds
        # the frozen scoring package only after Data has finished.
        for name in (
            "scoring_set", "answer_fields", "evaluation_script",
            "evaluator_sha256", "metric", "sample_submission",
        ):
            stamped.pop(name, None)
        stamped["validation_policy"] = "supplied"
        stamped["validation_fraction"] = 0.0
        stamped["metric_type"] = "builtin"
    elif agent_id == "inference":
        scoring_set = str(spec.get("validation_public") or "")
        if not scoring_set:
            raise ValueError(
                "pipeline Inference requires the Run's settled validation questions; "
                "complete the Data ticket first"
            )
        stamped["scoring_set"] = scoring_set
        stamped["sample_submission"] = sample_submission
    elif agent_id == "evaluation":
        scoring_set = str(spec.get("validation_set") or "")
        if not scoring_set:
            raise ValueError("pipeline Evaluation requires a validation_set")
        stamped["scoring_set"] = scoring_set
        stamped["evaluation_script"] = evaluation_script
        stamped["evaluator_sha256"] = str(
            spec.get("validation_evaluator_sha256") or ""
        )
        stamped["answer_fields"] = list(spec.get("validation_answer_fields") or [])
        stamped["sample_submission"] = sample_submission
        stamped["metric"] = run.validation_metric
    return stamped


async def _next_ticket_id(db: AsyncSession, agent_id: str, run_id: str) -> str:
    """`data-<run8>-001` — the run's first data ticket, not the system's 23rd.

    The number used to be global: the allocator scanned `data-%` across the
    whole table with no run filter, so `data-023` meant "the twenty-third data
    ticket this installation ever made". Inside one run that number said
    nothing a reader could use, and it cost a scan of every ticket ever created
    on every emit.

    Scoping it to the run fixes both, and follows a shape already in the system:
    the supervisor has been `orchestrate-<run8>-001` all along, and the held-out
    held-out lane uses the same id convention and is distinguished by `lane`.

    Ticket ids are identifiers only; stage semantics come from ``agent_id``.
    """
    prefix = _AGENT_PREFIX.get(agent_id, agent_id.split("-")[0])
    if not run_id:
        # No run to scope to (a standalone ticket). Keep the global counter: it
        # is what guarantees such an id cannot collide with a run-scoped one.
        pattern, fmt = rf"^{re.escape(prefix)}-(\d+)$", prefix + "-%s"
        rows = (await db.execute(
            select(Ticket.id).where(Ticket.id.like(f"{prefix}-%"))
        )).scalars().all()
    else:
        run8 = run_id[:8]
        pattern = rf"^{re.escape(prefix)}-{re.escape(run8)}-(\d+)$"
        fmt = f"{prefix}-{run8}-%s"
        rows = (await db.execute(
            select(Ticket.id).where(
                Ticket.run_id == run_id,
                Ticket.id.like(f"{prefix}-{run8}-%"),
            )
        )).scalars().all()
    used = {int(m.group(1)) for m in (re.match(pattern, r) for r in rows) if m}
    n = 1
    while n in used:
        n += 1
    return fmt % f"{n:03d}"


async def _wakeup_waits_for_supervisor(
    db: AsyncSession,
    *,
    run: Run | None,
    assignee: str,
) -> bool:
    """Whether a newly-created child must wait for its creator to finish.

    Orchestrator creates a child through the Ticket API while its own driver
    process is still running.  Waking that child from the POST handler lets the
    scheduler start it before the Orchestrator has returned (and before its
    ``SupervisorAction`` has even been validated).  A pipeline child created
    during an active supervisor heartbeat is therefore only *declared* here;
    the runner releases its wakeup after the supervisor heartbeat commits.

    Standalone Tickets and calls made while no supervisor heartbeat is active
    retain the normal immediate-assignment behavior.
    """
    if (
        run is None
        or run.mode == "single_stage"
        or assignee == "orchestrator"
        or not run.supervisor_ticket_id
    ):
        return False
    active = (await db.execute(
        select(HeartbeatRun.id).where(
            HeartbeatRun.ticket_id == run.supervisor_ticket_id,
            HeartbeatRun.finished_at.is_(None),
        ).limit(1)
    )).scalar_one_or_none()
    return active is not None


async def _optimization_data_tickets(db: AsyncSession, run_id: str) -> list[Ticket]:
    """Live optimization-lane Data Tickets that count toward the iteration ladder.

    Excludes Auto mode's `scope_problem` work order: it is Run Setup, not the
    Data stage. It is a `data`-agent Ticket on the optimization lane, so without
    this exclusion its succeeded row counted as an existing iteration-0 Data
    Ticket -- the orchestrator could never author `prepare_run_data` (409
    "iteration 0 Data is already settled or active") and baseline inference
    deadlocked on the missing inference_data_profile.
    """
    from zevo.engine.run.scoping import is_scoping_ticket

    rows = (await db.execute(select(Ticket).where(
        Ticket.run_id == run_id,
        Ticket.agent_id == "data",
        Ticket.lane == "optimization",
        Ticket.status.notin_(("failed", "cancelled", "skipped")),
    ).order_by(Ticket.iteration, Ticket.created_at))).scalars().all()
    return [t for t in rows if not is_scoping_ticket(t)]


@router.post("/tickets", response_model=CreateTicketResponse)
async def create_ticket(
    body: CreateTicketBody,
    db: AsyncSession = Depends(get_db),
) -> CreateTicketResponse:
    # Validate agent exists
    a = (await db.execute(select(AgentRow).where(AgentRow.id == body.agent_id))).scalar_one_or_none()
    if a is None:
        raise HTTPException(400, f"agent {body.agent_id!r} does not exist")
    if body.agent_id == "evaluation":
        if body.input_format != "typed":
            raise HTTPException(
                400,
                "evaluation is a deterministic system stage and does not accept freeform tickets",
            )
        if body.customization is not None and not body.customization.is_empty():
            raise HTTPException(
                400,
                "evaluation uses the immutable lane-owned scorer and cannot be customized",
            )

    # Resolve run_id -- create a standalone run if omitted
    run_id = body.run_id
    owning_run: Run | None = None
    if not run_id:
        if not (body.run_name or "").strip():
            raise HTTPException(
                400, "run_name is required when creating a standalone single-stage Run.",
            )
        if not (body.task_name or "").strip():
            raise HTTPException(
                400, "task_name is required when creating a standalone single-stage Run.",
            )
        # A single-agent call still produces a run, and every run is listed by
        # name — hard-coding "standalone" made them all indistinguishable.
        objective = (
            str((body.payload or {}).get("request") or "").strip()
            if body.input_format == "freeform"
            else f"Single-stage task for {body.agent_id}"
        )
        if not objective:
            raise HTTPException(400, "freeform single-stage request must not be empty")
        run = Run(
            task_objective=objective,
            agent_objective=objective,
            task_name=(body.task_name or "").strip(),
            run_name=(body.run_name or "").strip(),
            mode="single_stage",
            gpu_provider=body.gpu_provider or "instance",
            num_gpus=max(0, int(body.num_gpus or 0)),
            generation_backend=body.generation_backend or "vllm",
            metric=body.metric.strip(),
            metric_direction=body.metric_direction,
            status="running",
            summary="Standalone ticket created via /api/tickets",
        )
        db.add(run)
        await db.flush()
        run_id = run.id
        owning_run = run
    else:
        if any(value is not None for value in (
            body.gpu_provider, body.num_gpus, body.generation_backend,
        )):
            raise HTTPException(
                422,
                "runtime fields are allowed only when creating a standalone "
                "single-stage Run; an existing Run already owns its runtime",
            )
        # validate run exists
        r = (await db.execute(
            select(Run).where(Run.id == run_id).with_for_update()
        )).scalar_one_or_none()
        if r is None:
            raise HTTPException(400, f"run {run_id!r} does not exist")
        owning_run = r
        if r.mode != "single_stage" and body.agent_id != "orchestrator":
            final_registry = (await db.execute(select(Ticket).where(
                Ticket.run_id == run_id,
                Ticket.agent_id == "registry",
            ).order_by(Ticket.created_at.asc()).limit(1))).scalar_one_or_none()
            if final_registry is not None:
                raise HTTPException(
                    409,
                    {
                        "error": "the Run has entered final Registry",
                        "ticket_id": final_registry.id,
                        "advice": (
                            "No new optimization stage may start after final "
                            "champion registration. Repair that Registry Ticket "
                            "or finish the Run."
                        ),
                    },
                )
            live_specialist = (await db.execute(select(Ticket).where(
                Ticket.run_id == run_id,
                Ticket.lane == "optimization",
                Ticket.agent_id != "orchestrator",
                Ticket.status.in_(("queued", "running", "repairing", "awaiting_input", "waiting_external")),
            ).order_by(Ticket.created_at.asc()).limit(1))).scalar_one_or_none()
            if live_specialist is not None:
                raise HTTPException(
                    409,
                    {
                        "error": "the Run already has an active Specialist",
                        "ticket_id": live_specialist.id,
                        "agent_id": live_specialist.agent_id,
                        "advice": "Wait for that Ticket to finish before emitting the next stage.",
                    },
                )
        # A factual score row is not a complete iteration record. The next
        # transition is blocked until the Orchestrator supplies all four Journal
        # fields. Final Registry is not an exception: it runs only after every
        # measured iteration has a complete decision record.
        from zevo.engine.observe.run_metrics import incomplete_journal_entries
        pending_journal = incomplete_journal_entries(r.history)
        if pending_journal:
            iteration, source = pending_journal[-1]
            raise HTTPException(
                409,
                {
                    "error": "iteration Journal is incomplete",
                    "iteration": iteration,
                    "source": source,
                    "required_fields": ["action", "result", "analysis", "next"],
                    "advice": (
                        "PATCH the existing Run history row before creating "
                        "another pipeline Ticket."
                    ),
                },
            )
        # G.1 — budget guard. If the run has a max_cost_usd set and it's
        # already exhausted, refuse to spawn another expensive ticket.
        # This is the last-line safety so a buggy orchestrator cannot keep
        # spending. Exhaustion is a normal stop when a retained model exists;
        # it is a failure only when the run has no usable outcome.
        # Final Registry is the bounded, necessary retention step after the
        # Orchestrator decides another full iteration is unaffordable.  It does
        # not launch training/inference and must remain available at the cap.
        if body.agent_id != "registry" and (
            (r.max_cost_usd and r.max_cost_usd > 0)
            or (r.max_runtime_hours and r.max_runtime_hours > 0)
        ):
            from zevo.engine.cost.budget import snapshot_for_run
            snap = await snapshot_for_run(db, run_id)
            if snap.over_budget:
                raise HTTPException(
                    402,  # Payment Required — closest semantic match
                    {
                        "error": "run is over budget",
                        "max_cost_usd": snap.max_cost_usd,
                        "spent_usd": snap.spent_usd,
                        "remaining_usd": snap.remaining_usd,
                        "advice": (
                            "Stop spawning Tickets. PATCH status='success' with a "
                            "consolidated summary when a retained model exists; "
                            "otherwise PATCH status='failed' with "
                            "halted_reason='budget exhausted before a usable result'."
                        ),
                    },
                )
            if snap.over_time_limit:
                raise HTTPException(
                    409,
                    {
                        "error": "run reached its time limit",
                        "max_runtime_hours": snap.max_runtime_hours,
                        "elapsed_runtime_hours": snap.elapsed_runtime_hours,
                        "remaining_runtime_hours": snap.remaining_runtime_hours,
                        "advice": (
                            "Do not start another Ticket. Retain the best usable "
                            "model and finalize the Run; fail only when no usable "
                            "result exists."
                        ),
                    },
                )

    # Resolve ticket_id
    tid = body.ticket_id
    if tid:
        if not _TICKET_ID_RE.match(tid):
            raise HTTPException(400, f"invalid ticket_id {tid!r}; must match {_TICKET_ID_RE.pattern}")
        if (await db.execute(select(Ticket).where(Ticket.id == tid))).scalar_one_or_none():
            raise HTTPException(409, f"ticket {tid!r} already exists")
    else:
        tid = await _next_ticket_id(db, body.agent_id, body.run_id or "")

    effective_customization = body.customization
    if (
        body.run_id
        and owning_run is not None
        and owning_run.mode != "single_stage"
        and body.agent_id != "orchestrator"
    ):
        try:
            run_customizations = RunCustomizations.model_validate(
                owning_run.customizations or {}
            )
        except ValueError as exc:
            raise HTTPException(422, f"invalid stored Run customizations: {exc}") from exc
        run_owned = run_customizations.for_agent(body.agent_id)
        requested = (
            body.customization
            if body.customization is not None and not body.customization.is_empty()
            else None
        )
        if requested is not None:
            raise HTTPException(
                422,
                "pipeline customization must be declared once at Run creation "
                "and cannot be supplied again or changed by a Ticket caller",
            )
        effective_customization = run_owned

    try:
        raw_payload = dict(body.payload or {})
        if body.agent_id == "data" and raw_payload.get("operation") == "prepare_holdout_data":
            raise ValueError(
                "prepare_holdout_data is engine-private; public Tickets always use "
                "the optimization lane"
            )
        is_pipeline_request = bool(
            body.input_format == "typed"
            and body.run_id
            and owning_run is not None
            and owning_run.mode != "single_stage"
            and body.agent_id != "orchestrator"
        )
        if is_pipeline_request:
            request_model = PIPELINE_REQUEST_PAYLOAD_BY_AGENT.get(body.agent_id)
            if request_model is None:
                raise ValueError(
                    f"no pipeline request payload contract for {body.agent_id!r}"
                )
            raw_payload = request_model.model_validate(raw_payload).model_dump()
        if effective_customization and effective_customization.parameters:
            if body.agent_id not in {"data", "train", "inference"}:
                raise ValueError(
                    f"{body.agent_id} does not own configurable execution values"
                )
            target = (
                "configuration_pins"
                if effective_customization.enforcement == "strict"
                else "configuration_suggestions"
            )
            values = dict(raw_payload.get(target) or {})
            overlap = sorted(set(values) & set(effective_customization.parameters))
            if overlap:
                raise ValueError(
                    f"customization duplicates {target} keys: {overlap}"
                )
            raw_payload[target] = {
                **values,
                **dict(effective_customization.parameters),
            }
        if body.input_format == "typed" and owning_run is not None:
            raw_payload = _stamp_pipeline_payload(
                run=owning_run, agent_id=body.agent_id, payload=raw_payload,
            )
        pipeline_registry_request = bool(
            is_pipeline_request and body.agent_id == "registry"
        )
        if pipeline_registry_request:
            PipelineRegistryRequestPayload.model_validate(raw_payload)
            payload: dict[str, Any] = {}
        else:
            payload = validate_stored_payload(
                agent_id=body.agent_id,
                input_format=body.input_format,
                payload=raw_payload,
            )
        if body.input_format == "typed" and body.agent_id == "data":
            payload = _resolve_data_source_selection(payload)
        if body.input_format == "typed" and owning_run is not None:
            operation = str(raw_payload.get("operation") or "")
            if body.agent_id == "data" and operation == "prepare_run_data":
                iteration = int(body.iteration or 0)
                existing_data = await _optimization_data_tickets(db, run_id)
                if not existing_data and iteration != 0:
                    raise HTTPException(409, "the first optimization Data ticket is iteration 0")
                if existing_data and iteration == 0:
                    raise HTTPException(409, "iteration 0 Data is already settled or active")
                if existing_data:
                    previous_payload = dict(existing_data[-1].payload or {})
                    source_fields = (
                        "dataset_source", "dataset", "dataset_split",
                        "dataset_config", "data_query",
                    )
                    source_changed = any(
                        str(previous_payload.get(name) or "")
                        != str(payload.get(name) or "")
                        for name in source_fields
                    )
                    method_changed = str(
                        previous_payload.get("training_method") or ""
                    ) != str(payload.get("training_method") or "")
                    transition = dict(payload.get("branch_transition") or {})
                    transition_level = str(transition.get("level") or "none")
                    allowed_levels: set[str] = set()
                    if method_changed:
                        allowed_levels = {"method", "base_model"}
                    elif source_changed:
                        allowed_levels = {"data", "method", "base_model"}
                    if allowed_levels and transition_level not in allowed_levels:
                        raise HTTPException(
                            409,
                            {
                                "error": "search branch transition is not authorized",
                                "required_levels": sorted(allowed_levels),
                                "advice": (
                                    "Keep the active branch, or provide a complete "
                                    "branch_transition naming the exhausted branch, "
                                    "its Validation evidence, and the next branch."
                                ),
                            },
                        )
                    if not allowed_levels and transition_level != "none":
                        raise HTTPException(
                            409,
                            "branch_transition was supplied but the Data source and "
                            "training method remain in the active branch",
                        )
                same_iteration = [
                    ticket for ticket in existing_data
                    if int(ticket.iteration or 0) == iteration
                ]
                if same_iteration:
                    raise HTTPException(
                        409,
                        "one coherent Data recipe is allowed per iteration; "
                        f"reuse {same_iteration[-1].id} or move the revision to the next iteration",
                    )
                requested_signature = str(payload.get("data_intent_signature") or "")
                reusable = next((
                    ticket for ticket in reversed(existing_data)
                    if str((ticket.payload or {}).get("data_intent_signature") or "")
                    == requested_signature
                ), None)
                if reusable is not None:
                    raise HTTPException(
                        409,
                        {
                            "error": "training data recipe is unchanged",
                            "reusable_ticket_id": reusable.id,
                            "data_intent_signature": requested_signature,
                            "advice": (
                                "Do not run Data again. Bind this Ticket's training_dataset "
                                "and validation_dataset to Train."
                            ),
                        },
                    )
            if body.agent_id == "train" and payload.get("operation") == "train":
                base_model = str(payload.get("base_model") or "")
                previous_method = next((
                    str(row.get("training_method") or "").strip().lower()
                    for row in reversed(owning_run.history or [])
                    if isinstance(row, dict)
                    and row.get("source") == "trained"
                    and str(row.get("training_method") or "").strip()
                ), "")
                suggested_method = str(
                    (payload.get("configuration_suggestions") or {}).get(
                        "training_method"
                    ) or ""
                ).strip().lower()
                transition_level = str(
                    (payload.get("branch_transition") or {}).get("level") or "none"
                )
                if (
                    previous_method
                    and suggested_method
                    and suggested_method != previous_method
                    and transition_level not in {"method", "base_model"}
                ):
                    raise HTTPException(
                        409,
                        {
                            "error": "training method branch changed before exhaustion",
                            "previous_method": previous_method,
                            "next_method": suggested_method,
                            "advice": (
                                "Retain the active method, or provide a complete "
                                "method/base_model branch_transition with Validation evidence."
                            ),
                        },
                    )
                baseline_exists = any(
                    isinstance(row, dict)
                    and row.get("source") == "baseline"
                    and row.get("base_model") == base_model
                    and isinstance(row.get("score"), (int, float))
                    and not isinstance(row.get("score"), bool)
                    for row in (owning_run.history or [])
                )
                if not baseline_exists:
                    raise HTTPException(
                        409,
                        {
                            "error": "base-model baseline is missing",
                            "base_model": base_model,
                            "advice": (
                                "Run baseline Inference and Evaluation for this "
                                "model lineage before creating Train."
                            ),
                        },
                    )
        inputs = validate_bindings(
            agent_id=body.agent_id, payload=payload, inputs=body.inputs or {}
        ) if body.input_format == "typed" else {}
        if inputs:
            from zevo.engine.run.scheduler.bindings import resolve_input_bindings
            inputs = await resolve_input_bindings(db, run_id=run_id, inputs=inputs)

        # A resolved path proves that an artifact exists; it does not prove the
        # artifact belongs to this iteration's experiment. Enforce the semantic
        # graph here so an old Data product or another iteration's checkpoint
        # cannot be wired into an otherwise valid execution Ticket.
        if (
            body.input_format == "typed"
            and owning_run is not None
            and owning_run.mode != "single_stage"
        ):
            async def source_ticket(name: str, expected_agent: str) -> Ticket:
                binding = (inputs or {}).get(name) or {}
                source_id = str(binding.get("source_ticket_id") or "")
                if not source_id:
                    raise ValueError(
                        f"pipeline input {name!r} must bind a {expected_agent} Ticket"
                    )
                source = (await db.execute(
                    select(Ticket).where(Ticket.id == source_id, Ticket.run_id == run_id)
                )).scalar_one_or_none()
                if source is None or source.agent_id != expected_agent:
                    raise ValueError(
                        f"pipeline input {name!r} must come from a {expected_agent} Ticket"
                    )
                return source

            if body.agent_id in {"train", "inference"}:
                infra_ticket = await source_ticket("device_info", "infrastructure")
                purpose = str((infra_ticket.payload or {}).get("purpose") or "")
                if purpose != body.agent_id:
                    raise ValueError(
                        f"{body.agent_id} requires an Infrastructure plan with "
                        f"purpose={body.agent_id!r}, got {purpose!r}"
                    )

            if body.agent_id == "train" and payload.get("operation") == "train":
                data_ticket = await source_ticket("training_dataset", "data")
                validation_data_ticket = await source_ticket(
                    "validation_dataset", "data",
                )
                if int(data_ticket.iteration or 0) > int(body.iteration or 0):
                    raise ValueError("Train cannot use Data prepared for a later iteration")
                if validation_data_ticket.id != data_ticket.id:
                    raise ValueError(
                        "Train and Validation datasets must come from the same "
                        "versioned Run Data ticket"
                    )
                current_revisions = (await db.execute(select(Ticket.id).where(
                    Ticket.run_id == run_id,
                    Ticket.agent_id == "data",
                    Ticket.lane == "optimization",
                    Ticket.iteration == int(body.iteration or 0),
                    Ticket.status.in_(("succeeded", "degraded")),
                ).order_by(Ticket.created_at.desc()))).scalars().all()
                if current_revisions and data_ticket.id != current_revisions[0]:
                    raise ValueError(
                        "this iteration produced a revised Data artifact; Train must "
                        f"bind {current_revisions[0]} instead of an older recipe"
                    )
                data_product = (await db.execute(
                    select(WorkProduct).where(
                        WorkProduct.ticket_id == data_ticket.id,
                        WorkProduct.role == "training_dataset",
                    ).order_by(WorkProduct.created_at.desc()).limit(1)
                )).scalar_one_or_none()
                data_signature = str(
                    ((data_product.meta or {}).get("data_signature") if data_product else "")
                    or ""
                )
                if not is_sha256(data_signature):
                    raise ValueError(
                        "Train requires a verified data_signature from its Data ticket"
                    )
                payload["data_signature"] = data_signature
                payload = TrainPayload.model_validate(payload).model_dump()
                baseline_config = await source_ticket("inference_config", "inference")
                if (
                    int(baseline_config.iteration or 0) > int(body.iteration or 0)
                    or (baseline_config.payload or {}).get("model_source") != "base_model"
                    or str((baseline_config.payload or {}).get("base_model") or "")
                    != str(payload.get("base_model") or "")
                ):
                    raise ValueError(
                        "Train must use the matching base-model lineage's baseline "
                        "inference_config.yaml"
                    )
                iteration = int(body.iteration or 0)
                if payload.get("model_source") == "base_model":
                    if "parent_checkpoint" in (inputs or {}) or "parent_train_config" in (inputs or {}):
                        raise ValueError(
                            "a baseline Train branch cannot bind parent checkpoint artifacts"
                        )
                else:
                    if iteration < 2:
                        raise ValueError("a checkpoint branch is valid only after iteration 1")
                    parent = await source_ticket("parent_checkpoint", "train")
                    parent_config = await source_ticket("parent_train_config", "train")
                    if (
                        int(parent.iteration or 0) >= iteration
                        or int(parent.iteration or 0) < 1
                        or parent.id != parent_config.id
                        or str((parent.payload or {}).get("base_model") or "")
                        != str(payload.get("base_model") or "")
                    ):
                        raise ValueError(
                            "Train must bind checkpoint and config from the same "
                            "successful earlier Train iteration and base-model "
                            "lineage in this Run"
                        )
            elif body.agent_id == "inference" and payload.get("model_source") == "checkpoint":
                train_ticket = await source_ticket("checkpoint", "train")
                if int(train_ticket.iteration or 0) != int(body.iteration or 0):
                    raise ValueError("checkpoint must come from Train in the same iteration")
                baseline_config = await source_ticket("inference_config", "inference")
                if (
                    int(baseline_config.iteration or 0) > int(body.iteration or 0)
                    or (baseline_config.payload or {}).get("model_source") != "base_model"
                    or str((baseline_config.payload or {}).get("base_model") or "")
                    != str(payload.get("base_model") or "")
                ):
                    raise ValueError(
                        "trained-model Inference must reuse the matching base-model "
                        "lineage's baseline inference_config.yaml"
                    )
                if str((train_ticket.payload or {}).get("base_model") or "") != str(
                    payload.get("base_model") or ""
                ):
                    raise ValueError(
                        "checkpoint Inference cannot cross base-model lineages"
                    )
                if "predict_script" not in (inputs or {}):
                    baseline_script = (await db.execute(
                        select(WorkProduct).where(
                            WorkProduct.ticket_id == baseline_config.id,
                            WorkProduct.role == "script",
                        ).order_by(WorkProduct.created_at.desc()).limit(1)
                    )).scalar_one_or_none()
                    if baseline_script is not None and str(baseline_script.path or ""):
                        inputs["predict_script"] = {
                            "artifact_role": "script",
                            "source_ticket_id": baseline_config.id,
                            "work_product_id": baseline_script.id,
                            "path": baseline_script.path,
                        }
                if "predict_script" in (inputs or {}):
                    script_ticket = await source_ticket("predict_script", "inference")
                    if (
                        int(script_ticket.iteration or 0) > int(body.iteration or 0)
                        or (script_ticket.payload or {}).get("model_source") != "base_model"
                        or str((script_ticket.payload or {}).get("base_model") or "")
                        != str(payload.get("base_model") or "")
                    ):
                        raise ValueError(
                            "candidate Inference may reuse predict.py only from the "
                            "matching base-model lineage's baseline Inference"
                        )
            elif body.agent_id == "inference" and payload.get("model_source") == "base_model":
                profile_ticket = await source_ticket("inference_data_profile", "data")
                if int(profile_ticket.iteration or 0) != 0:
                    raise ValueError(
                        "baseline Inference must use the iteration-0 Data "
                        "inference_data_profile"
                    )
                previous_baselines = (await db.execute(select(Ticket).where(
                    Ticket.run_id == run_id,
                    Ticket.agent_id == "inference",
                    Ticket.lane == "optimization",
                    Ticket.status.in_(("succeeded", "degraded")),
                ).order_by(Ticket.created_at))).scalars().all()
                previous_models = [
                    str((candidate.payload or {}).get("base_model") or "")
                    for candidate in previous_baselines
                    if (candidate.payload or {}).get("model_source") == "base_model"
                ]
                selected_model = str(payload.get("base_model") or "")
                transition_level = str(
                    (payload.get("branch_transition") or {}).get("level") or "none"
                )
                if (
                    previous_models
                    and selected_model != previous_models[-1]
                    and transition_level != "base_model"
                ):
                    raise ValueError(
                        "a new Base-model branch requires a complete base_model "
                        "branch_transition with Validation exhaustion evidence"
                    )
                if not previous_models and transition_level != "none":
                    raise ValueError(
                        "the initial Base-model selection is not an exhausted-branch transition"
                    )
            elif body.agent_id == "registry":
                existing_registry = (await db.execute(select(Ticket).where(
                    Ticket.run_id == run_id,
                    Ticket.agent_id == "registry",
                ).limit(1))).scalar_one_or_none()
                if existing_registry is not None:
                    raise ValueError(
                        "Registry is a once-per-Run finalization stage; repair "
                        f"the existing Ticket {existing_registry.id} instead of creating another"
                    )

                score_rows = (await db.execute(select(ScoreEvent).where(
                    ScoreEvent.run_id == run_id,
                    ScoreEvent.split == "validation",
                    ScoreEvent.source == "trained",
                ).order_by(ScoreEvent.ts.asc()))).scalars().all()
                champion = None
                for score_row in score_rows:
                    if champion is None or is_better(
                        float(score_row.score),
                        float(champion.score),
                        owning_run.validation_metric_direction,
                    ):
                        champion = score_row
                if champion is None:
                    raise ValueError(
                        "final Registry requires at least one measured trained candidate"
                    )
                if int(body.iteration or 0) != int(champion.iteration or 0):
                    raise ValueError(
                        "Registry must bind the engine-selected Validation champion: "
                        f"iteration {champion.iteration}, not iteration {body.iteration}"
                    )
                train_ticket = await source_ticket("checkpoint", "train")
                train_config_ticket = await source_ticket("train_config", "train")
                eval_ticket = await source_ticket("metrics", "evaluation")
                if (
                    int(train_ticket.iteration or 0) != int(body.iteration or 0)
                    or train_config_ticket.id != train_ticket.id
                    or int(eval_ticket.iteration or 0) != int(body.iteration or 0)
                ):
                    raise ValueError(
                        "Registry checkpoint and metrics must come from the same iteration"
                    )
                prediction_binding = (eval_ticket.inputs or {}).get("predictions") or {}
                inference_id = str(prediction_binding.get("source_ticket_id") or "")
                inference_ticket = (await db.execute(
                    select(Ticket).where(
                        Ticket.id == inference_id,
                        Ticket.run_id == run_id,
                        Ticket.agent_id == "inference",
                    )
                )).scalar_one_or_none()
                if (
                    inference_ticket is None
                    or (inference_ticket.payload or {}).get("model_source") != "checkpoint"
                    or int(inference_ticket.iteration or 0) != int(body.iteration or 0)
                ):
                    raise ValueError(
                        "Registry metrics must come from checkpoint Inference in the same iteration"
                    )
                evaluated_checkpoint = (
                    (inference_ticket.inputs or {}).get("checkpoint") or {}
                )
                if str(evaluated_checkpoint.get("source_ticket_id") or "") != train_ticket.id:
                    raise ValueError(
                        "Registry metrics must evaluate the exact Train checkpoint being registered"
                    )
                registry_checkpoint = (inputs or {}).get("checkpoint") or {}
                evaluated_product_id = str(
                    evaluated_checkpoint.get("work_product_id") or ""
                )
                registry_product_id = str(
                    registry_checkpoint.get("work_product_id") or ""
                )
                if (
                    evaluated_product_id
                    and registry_product_id
                    and evaluated_product_id != registry_product_id
                ):
                    raise ValueError(
                        "Registry metrics must evaluate the exact checkpoint "
                        "WorkProduct being registered"
                    )

                checkpoint_products = (await db.execute(
                    select(WorkProduct).where(
                        WorkProduct.ticket_id == train_ticket.id,
                        WorkProduct.role == "checkpoint",
                    ).order_by(WorkProduct.created_at.desc())
                )).scalars().all()
                checkpoint_product = next((
                    product for product in checkpoint_products
                    if registry_product_id and product.id == registry_product_id
                ), None)
                if checkpoint_product is None:
                    checkpoint_product = next((
                        product for product in checkpoint_products
                        if (product.meta or {}).get("checkpoint_kind") == "final"
                    ), checkpoint_products[0] if checkpoint_products else None)
                checkpoint_meta = dict(checkpoint_product.meta or {}) if checkpoint_product else {}
                actual_method = str(checkpoint_meta.get("training_method") or "").strip()
                actual_base_model = str(checkpoint_meta.get("base_model") or "").strip()
                if not actual_method or not actual_base_model:
                    raise ValueError(
                        "Registry requires the successful Train result's method and base model"
                    )
                payload.update({
                    "training_method": actual_method,
                    "base_model": actual_base_model,
                    "task_objective": owning_run.task_objective,
                        "metric": owning_run.validation_metric,
                        "metric_direction": owning_run.validation_metric_direction,
                        "checkpoint_is_remote": (
                            str(checkpoint_meta.get("location") or "") == "remote"
                        ),
                    })
                if payload["checkpoint_is_remote"]:
                    await source_ticket("device_info", "infrastructure")
                elif "device_info" in (inputs or {}):
                    await source_ticket("device_info", "infrastructure")

                training_data = (train_ticket.inputs or {}).get("training_dataset") or {}
                data_id = str(training_data.get("source_ticket_id") or "")
                data_product = (await db.execute(
                    select(WorkProduct).where(
                        WorkProduct.ticket_id == data_id,
                        WorkProduct.role == "training_dataset",
                    ).order_by(WorkProduct.created_at.desc()).limit(1)
                )).scalar_one_or_none()
                resolved_source = str(
                    ((data_product.meta or {}).get("dataset_source") if data_product else "")
                    or ""
                ).strip()
                if not resolved_source:
                    raise ValueError(
                        "Registry requires resolved data provenance from the Train checkpoint's Data ticket"
                    )
                # Provenance is engine-stamped from the exact Data→Train chain;
                # it is not a second planner-authored version of the source.
                payload["dataset_source"] = resolved_source
                expected_tag = model_tag_for_run(run_id)
                existing_model = await db.get(RegistryModel, expected_tag)
                if existing_model is not None:
                    raise ValueError(
                        f"Run {run_id} already has final Registry model {expected_tag}"
                    )
                payload = RegistryPayload.model_validate(payload).model_dump(mode="json")

    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    if body.run_id and body.agent_id == "evaluation":
        pred = (inputs or {}).get("predictions") or {}
        source_id = str(pred.get("source_ticket_id") or "")
        mirror_source = (await db.execute(
            select(Ticket).where(Ticket.id == source_id)
        )).scalar_one_or_none()
        if (
            mirror_source is None
            or mirror_source.run_id != run_id
            or mirror_source.agent_id != "inference"
            or mirror_source.lane != "optimization"
            or int(mirror_source.iteration or 0) != int(body.iteration or 0)
            or mirror_source.status not in ("succeeded", "degraded")
        ):
            raise HTTPException(
                422,
                "an Evaluation ticket must bind predictions from the succeeded "
                "optimization Inference ticket in the same run and iteration",
            )

    defer_wakeup = await _wakeup_waits_for_supervisor(
        db,
        run=owning_run,
        assignee=body.agent_id,
    )

    tk = Ticket(
        id=tid,
        run_id=run_id,
        agent_id=body.agent_id,
        status="queued",
        input_format=body.input_format,
        lane="optimization",
        iteration=body.iteration,
        payload=payload,
        customization=(
            effective_customization.model_copy(update={"parameters": {}}).model_dump()
            if effective_customization else {}
        ),
        inputs=inputs,
        summary="",
    )
    db.add(tk)
    db.add(TicketNotice(
        ticket_id=tid, code="ticket.created", severity="info",
        body=f"Ticket created via /api/tickets for agent {body.agent_id}.",
    ))
    if defer_wakeup:
        db.add(TicketNotice(
            ticket_id=tid,
            code="ticket.awaiting_supervisor_completion",
            severity="info",
            body=(
                "Ticket accepted and awaiting the active Orchestrator "
                "heartbeat's validated completion before execution."
            ),
        ))
    await db.commit()
    await db.refresh(tk)

    if not defer_wakeup:
        # Ordinary direct/standalone assignment.  Pipeline handoffs created by
        # an active Orchestrator are released by the runner only after that
        # heartbeat has finished and its SupervisorAction has been validated.
        from zevo.engine.run.wakeup import queue_wakeup
        await queue_wakeup(
            db, agent_id=body.agent_id, ticket_id=tid,
            source="assignment",
            reason=f"ticket {tid} created and assigned to {body.agent_id}",
        )

    return CreateTicketResponse.model_validate(_to_dto(tk).model_dump())


@router.get("/tickets/{ticket_id}", response_model=TicketDetail)
async def get_ticket(
    ticket_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> TicketDetail:
    t = (await db.execute(select(Ticket).where(Ticket.id == ticket_id))).scalar_one_or_none()
    if t is None:
        raise HTTPException(404, f"ticket {ticket_id} not found")
    # An Agent caller cannot opt into held-out state. The trusted dashboard can
    # observe it live; everyone else sees it only after the owning Run ends.
    if t.lane == "held_out_test" and not is_trusted_ui_request(request):
        run_status = (await db.execute(
            select(Run.status).where(Run.id == t.run_id)
        )).scalar_one_or_none()
        if run_status not in TERMINAL_RUN_STATUSES:
            raise HTTPException(404, f"ticket {ticket_id} not found")
    messages = (
        await db.execute(
            select(TicketMessage).where(TicketMessage.ticket_id == ticket_id).order_by(TicketMessage.created_at)
        )
    ).scalars().all()
    notices = (
        await db.execute(
            select(TicketNotice).where(TicketNotice.ticket_id == ticket_id).order_by(TicketNotice.created_at)
        )
    ).scalars().all()
    results = (
        await db.execute(
            select(HeartbeatResult).where(HeartbeatResult.ticket_id == ticket_id).order_by(HeartbeatResult.created_at)
        )
    ).scalars().all()
    wps = (
        await db.execute(select(WorkProduct).where(WorkProduct.ticket_id == ticket_id))
    ).scalars().all()
    events = (
        await db.execute(
            select(ExecutionEvent).where(ExecutionEvent.ticket_id == ticket_id).order_by(ExecutionEvent.ts)
        )
    ).scalars().all()
    heartbeats = (
        await db.execute(
            select(HeartbeatRun).where(HeartbeatRun.ticket_id == ticket_id).order_by(HeartbeatRun.started_at)
        )
    ).scalars().all()
    base = _to_dto(t)
    return TicketDetail(
        **base.model_dump(),
        messages=[
            MessageDTO(id=m.id, author=m.author, body=_host_text(m.body),
                       triggered_wakeup_id=m.triggered_wakeup_id or "",
                       created_at=m.created_at.isoformat() if m.created_at else "")
            for m in messages
        ],
        notices=[
            NoticeDTO(id=n.id, code=n.code, severity=n.severity,
                      body=_host_text(n.body),
                      created_at=n.created_at.isoformat() if n.created_at else "")
            for n in notices
        ],
        results=[
            ResultDTO(id=r.id, heartbeat_id=r.heartbeat_id or "",
                      agent_id=r.agent_id, status=r.status,
                      output=r.output or {},
                      created_at=r.created_at.isoformat() if r.created_at else "")
            for r in results
        ],
        work_products=[
            WorkProductDTO(id=w.id, role=w.role, path=w.path,
                           local_path=_host_path(w.path or ""),
                           meta=w.meta or {},
                           created_at=w.created_at.isoformat() if w.created_at else "")
            for w in wps
        ],
        execution_events=[
            ExecutionEventDTO(heartbeat_id=e.heartbeat_id,
                              attempt_id=e.attempt_id,
                              event_type=e.event_type,
                              phase=e.phase, current_step=e.current_step,
                              total_steps=e.total_steps, loss=e.loss,
                              extras=e.extras or {},
                              ts=e.ts.isoformat() if e.ts else "")
            for e in events
        ],
        resolved_configs={h.id: h.resolved_config or {} for h in heartbeats if h.resolved_config},
    )


class MessageBody(StrictBody):
    body: str
    author: str = "user"
    # When true (default for non-agent authors), posting this message
    # also queues a wakeup so the assigned agent re-runs the ticket
    # with the new conversation context. Agents posting their own
    # messages should leave this False to avoid self-loops. It is ignored
    # once the run is finished -- see post_message.
    wake_agent: bool = True


def _external_stage_message_body(
    body: str, *, scheduler_state: str, ticket_status: str,
) -> str:
    """Do not let an Agent call a live/deferred Slurm stage ``Done``.

    The runner is the only component that knows whether the typed Result and
    every claimed artifact were accepted. Agents still POST progress messages,
    but a pre-validation ``Done:`` is therefore only Waiting/Running.
    """
    if ticket_status in TERMINAL_TICKET_STATUSES or not re.match(
        r"^\s*Done\s*:", body, flags=re.IGNORECASE,
    ):
        return body
    state = str(scheduler_state or "").strip().upper().split()[0].rstrip("+")
    prefix = "Waiting" if state in {"", "PENDING", "CONFIGURING"} else "Running"
    return re.sub(
        r"^(\s*)Done\s*:", rf"\1{prefix}:", body,
        count=1, flags=re.IGNORECASE,
    )


@router.post("/tickets/{ticket_id}/messages", response_model=MessageDTO)
async def post_message(
    ticket_id: str,
    payload: MessageBody,
    db: AsyncSession = Depends(get_db),
) -> MessageDTO:
    t = (await db.execute(select(Ticket).where(Ticket.id == ticket_id))).scalar_one_or_none()
    if t is None:
        raise HTTPException(404, f"ticket {ticket_id} not found")
    body = payload.body
    author = (payload.author or "").strip()
    if author == t.agent_id and re.match(r"^\s*Done\s*:", body, re.IGNORECASE):
        stage_rows = (await db.execute(
            select(InfraInstance)
            .where(
                InfraInstance.ticket_id == ticket_id,
                InfraInstance.provider == "cluster",
                InfraInstance.instance_id != "",
            )
            .order_by(InfraInstance.created_at.desc())
            .limit(5)
        )).scalars().all()
        stage = next(
            (row for row in stage_rows if bool((row.meta or {}).get("stage_job"))),
            None,
        )
        if stage is not None:
            body = _external_stage_message_body(
                body,
                scheduler_state=str((stage.meta or {}).get("scheduler_state") or ""),
                ticket_status=t.status,
            )
    message = TicketMessage(ticket_id=ticket_id, author=payload.author, body=body)
    db.add(message)
    await db.commit()
    await db.refresh(message)

    # If the message is from a human (not the assigned agent itself),
    # treat it as a conversational follow-up: queue a wakeup so the
    # agent runs again with the new context. The runner injects the
    # latest messages into the prompt so the agent actually sees the
    # follow-up.
    #
    # An EMPTY author is never a human follow-up — it's a malformed agent
    # self-message (e.g. a driver that didn't set $AGENT_ID). Re-waking on it
    # creates a self-perpetuating loop (the re-run posts another blank-author
    # Starting/Done message → another wakeup → ...). So require a real author.
    #
    # A finished run is a record, not a conversation. Waking an agent on one
    # re-runs a stage that already happened, against whatever driver and model
    # are configured NOW, and writes the result over the original -- a one-line
    # note on a completed Ticket must not turn `succeeded` into `failed`
    # and made a run that was entirely claude_cli look like it had used
    # bedrock. Messages on a terminal run are still recorded; they just do not
    # re-animate it. `ticket rerun` is the way to say you meant it.
    run = (await db.execute(select(Run).where(Run.id == t.run_id))).scalar_one_or_none()
    run_is_terminal = run is not None and run.status in TERMINAL_RUN_STATUSES
    woke = bool(payload.wake_agent and author and author != t.agent_id
                and not run_is_terminal)
    if woke:
        # G.4 — if the ticket was paused waiting for clarification,
        # the user's reply unblocks it. Flip back to `queued` BEFORE
        # queueing the wakeup so the daemon picks it up cleanly.
        if t.status == "awaiting_input":
            t.status = "queued"
            await db.commit()
        from zevo.engine.run.wakeup import queue_wakeup
        wakeup = await queue_wakeup(
            db,
            agent_id=t.agent_id,
            ticket_id=ticket_id,
            source="on_demand",
            reason=f"message by {payload.author}",
        )
        message.triggered_wakeup_id = str(getattr(wakeup, "id", "") or "")
        await db.commit()

    return MessageDTO(id=message.id, author=message.author, body=message.body,
                      created_at=message.created_at.isoformat() if message.created_at else "",
                      triggered_wakeup_id=message.triggered_wakeup_id or "")


class ClarifyBody(StrictBody):
    question: str
    # Optional: which fields of the input the agent needs clarified.
    # Surfaces as a checklist in the UI so the user knows what to fill in.
    fields_needed: list[str] = Field(default_factory=list)


@router.post("/tickets/{ticket_id}/clarify")
async def clarify_ticket(
    ticket_id: str, body: ClarifyBody,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Called by an AGENT when it needs the user's input to proceed.

    Marks the ticket awaiting_input, posts the question as a message, and
    halts the run-side wakeup. The user replies via the normal /messages
    endpoint with author!=agent_id; that automatically
    queues a wakeup that resumes the agent with the new context.

    Use this instead of failing silently. Spamming clarifications is
    worse than failing — only call this when the user really must
    decide something the agent can't (file path ambiguous, budget too
    low for the only viable model, etc.).
    """
    t = (await db.execute(select(Ticket).where(Ticket.id == ticket_id))).scalar_one_or_none()
    if t is None:
        raise HTTPException(404, f"ticket {ticket_id} not found")
    if t.status in TERMINAL_TICKET_STATUSES:
        raise HTTPException(
            400,
            f"ticket {ticket_id} is already terminal (status={t.status}); "
            "cannot ask for clarification.",
        )
    # Compose the message so it stands out in the timeline.
    fields_part = ""
    if body.fields_needed:
        fields_part = "\n\nFields needed: " + ", ".join(f"`{f}`" for f in body.fields_needed)
    db.add(TicketMessage(
        ticket_id=ticket_id,
        author=t.agent_id,
        body=f"__CLARIFY__\n\n**Need your input to proceed.** {body.question}{fields_part}",
    ))
    t.status = "awaiting_input"
    await db.commit()
    return {
        "status": "awaiting_input",
        "ticket_id": ticket_id,
        "note": (
            "Run is paused on this ticket. Post a message on "
            f"/tickets/{ticket_id}/messages to answer; the agent will "
            "resume with your reply in context."
        ),
    }


class HeartbeatBody(StrictBody):
    driver: str = ""
    model: str = ""
    work_dir: str = ""


@router.post("/tickets/{ticket_id}/heartbeat")
async def trigger_heartbeat(
    ticket_id: str,
    body: HeartbeatBody,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    t = (await db.execute(select(Ticket).where(Ticket.id == ticket_id))).scalar_one_or_none()
    if t is None:
        raise HTTPException(404, f"ticket {ticket_id} not found")

    from zevo.engine.run.wakeup import queue_wakeup
    w = await queue_wakeup(
        db, agent_id=t.agent_id, ticket_id=ticket_id,
        source="on_demand",
        reason=f"per-ticket heartbeat (driver={body.driver!r}, model={body.model!r})",
        payload={"driver": body.driver, "model": body.model,
                 "work_dir": body.work_dir or work_dir_root()},
    )
    return {"status": w.status, "ticket_id": ticket_id, "wakeup_id": w.id}


@router.post("/tickets/{ticket_id}/progress")
async def post_progress(
    ticket_id: str,
    body: dict[str, Any] = Body(...),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Live progress sink for remote training / inference.

    This is the fallback sink when a remote stage cannot continuously tee its
    log into the local Ticket work directory. Each marker is written immediately
    so the UI's 3s poll picks it up live. The runner normally watches local logs
    itself and still performs a terminal sweep for durability.

    Body mirrors a marker: `{"kind":"attempt","attempt_id":"<uuid>"}`,
    `{"kind":"progress","attempt_id":"<uuid>","step":N,"total":M,"loss":..,
    "lr":..,...}`, `{"kind":"phase","phase":"training"}`, or
    `{"kind":"config", ...resolved hyper-params...}`. Best-effort: never 404s on
    a missing ticket (the run may have just been cancelled) — returns ok=false.
    """
    exists = (await db.execute(
        select(Ticket.id).where(Ticket.id == ticket_id)
    )).scalar_one_or_none()
    if exists is None:
        return {"ok": False, "reason": "ticket not found"}

    kind = str(body.get("kind") or "progress")
    heartbeat = (await db.execute(
        select(HeartbeatRun).where(HeartbeatRun.ticket_id == ticket_id)
        .order_by(desc(HeartbeatRun.started_at)).limit(1)
    )).scalar_one_or_none()
    if heartbeat is None:
        return {"ok": False, "reason": "no heartbeat for progress"}
    heartbeat_id = heartbeat.id

    # The telemetry callback supplies a process UUID on every attempt marker
    # and progress row. Phase-only producers fall back to the most recently
    # declared attempt, while non-training stages use their heartbeat as the
    # single execution identity.
    attempt_id = str(body.get("attempt_id") or "").strip()
    attempt_id_is_valid = bool(re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        attempt_id,
    ))
    if attempt_id and not attempt_id_is_valid:
        return {"ok": False, "reason": "attempt_id must be a UUID"}
    if kind == "attempt":
        if not attempt_id:
            return {"ok": False, "reason": "attempt_id must be a UUID"}
    elif not attempt_id:
        attempt_id = str((await db.execute(
            select(ExecutionEvent.attempt_id).where(
                ExecutionEvent.heartbeat_id == heartbeat_id,
                ExecutionEvent.event_type == "attempt",
            ).order_by(desc(ExecutionEvent.ts)).limit(1)
        )).scalar_one_or_none() or heartbeat_id)

    if kind == "attempt":
        existing_attempt = (await db.execute(select(ExecutionEvent.id).where(
            ExecutionEvent.heartbeat_id == heartbeat_id,
            ExecutionEvent.event_type == "attempt",
            ExecutionEvent.attempt_id == attempt_id,
        ).limit(1))).scalar_one_or_none()
        if existing_attempt is not None:
            return {"ok": True, "deduplicated": True}
        row = ExecutionEvent(
            ticket_id=ticket_id,
            heartbeat_id=heartbeat_id,
            attempt_id=attempt_id,
            event_type="attempt",
            phase=str(body.get("phase") or "train"),
            current_step=0,
            total_steps=0,
            loss=-1.0,
            extras={},
        )
    elif kind == "phase":
        row = ExecutionEvent(ticket_id=ticket_id, heartbeat_id=heartbeat_id,
                             attempt_id=attempt_id, event_type="phase",
                             phase=str(body.get("phase", "")), current_step=0,
                             total_steps=0, loss=-1.0, extras={})
    elif kind == "config":
        cfg = {k: v for k, v in body.items()
               if k not in {"kind", "owner", "attempt_id", "t"}}
        heartbeat.resolved_config = cfg
        await db.commit()
        return {"ok": True}
    else:  # progress
        reserved = {
            "kind", "owner", "attempt_id", "t", "step", "total",
            "current_step", "total_steps", "phase",
        }
        extras = {k: v for k, v in body.items() if k not in reserved}
        phase = str(body.get("phase", ""))
        current_step = int(body.get("step", body.get("current_step", 0)) or 0)
        existing = (await db.execute(select(ExecutionEvent).where(
            ExecutionEvent.heartbeat_id == heartbeat_id,
            ExecutionEvent.attempt_id == attempt_id,
            ExecutionEvent.event_type == "progress",
            ExecutionEvent.phase == phase,
            ExecutionEvent.current_step == current_step,
        ).limit(1))).scalar_one_or_none()
        if existing is not None:
            existing.total_steps = max(
                int(existing.total_steps or 0),
                int(body.get("total", body.get("total_steps", 0)) or 0),
            )
            explicit_loss = body.get("loss")
            if explicit_loss is not None or float(existing.loss or -1.0) < 0:
                existing.loss = float(
                    explicit_loss
                    if explicit_loss is not None
                    else body.get("train_loss", body.get("eval_loss", -1.0))
                )
            existing.extras = {**dict(existing.extras or {}), **extras}
            await db.commit()
            return {"ok": True, "deduplicated": True}
        row = ExecutionEvent(
            ticket_id=ticket_id,
            heartbeat_id=heartbeat_id,
            attempt_id=attempt_id,
            event_type="progress",
            phase=phase,
            current_step=current_step,
            total_steps=int(body.get("total", body.get("total_steps", 0)) or 0),
            loss=float(body.get("loss", -1.0)),
            extras=extras,
        )
    db.add(row)
    await db.commit()
    return {"ok": True}


@router.post("/tickets/{ticket_id}/cancel")
async def cancel_ticket(
    ticket_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Cancel a ticket: kill any live heartbeat and mark it cancelled."""
    t = (await db.execute(select(Ticket).where(Ticket.id == ticket_id))).scalar_one_or_none()
    if t is None:
        raise HTTPException(404, f"ticket {ticket_id} not found")
    if t.status in TERMINAL_TICKET_STATUSES:
        # A terminal DB status does not prove a remote child process exited.
        # An operator may deliberately press Cancel again to reap such a leak.
        from zevo.engine.run.remote_jobs import cancel_ticket_remote_job
        remote_job = await cancel_ticket_remote_job(db, t)
        return {
            "status": t.status,
            "ticket_id": ticket_id,
            "remote_job": remote_job,
            "note": "already terminal; exact remote Ticket cleanup was rechecked",
        }

    # Set the ticket status to cancelled. The scheduler's runner flusher
    # polls this flag every ~3s and SIGTERMs the agent subprocess when
    # it sees it. We can't kill the process from the backend container
    # directly (separate process), so the DB flag is the signal.
    t.status = "cancelled"
    t.error_message = "cancelled by user"
    await db.commit()
    from zevo.engine.run.remote_jobs import cancel_ticket_remote_job
    remote_job = await cancel_ticket_remote_job(db, t)
    return {"status": "cancelled", "ticket_id": ticket_id,
            "remote_job": remote_job,
            "note": "scheduler will stop the local Agent and remote Ticket-owned task"}


# ───────────────────────── retry / rerun (J.2) ───────────────────────────────


class TicketRetryStatus(BaseModel):
    ticket_id: str
    ticket_status: TicketStatus
    verdict: str         # transient | structural | cancelled | unknown
    retryable: bool
    reason: str
    code: str = "unknown"
    description: str = ""
    recovery: str = ""
    last_error: str = ""
    last_exit_code: int = 0


@router.get("/tickets/{ticket_id}/retry-status", response_model=TicketRetryStatus)
async def get_retry_status(
    ticket_id: str, db: AsyncSession = Depends(get_db),
) -> TicketRetryStatus:
    """Classify whether this ticket is safe to rerun. UI hides the
    Rerun button when verdict='structural' (rerunning won't help)."""
    from zevo.db import HeartbeatRun
    from zevo.engine.run.retry_policy import classify

    t = (await db.execute(select(Ticket).where(Ticket.id == ticket_id))).scalar_one_or_none()
    if t is None:
        raise HTTPException(404, f"ticket {ticket_id} not found")

    # Look at the LATEST heartbeat for this ticket's error.
    hb = (await db.execute(
        select(HeartbeatRun).where(HeartbeatRun.ticket_id == ticket_id)
        .order_by(desc(HeartbeatRun.started_at)).limit(1)
    )).scalar_one_or_none()
    err = (hb.error_message if hb else "") or t.error_message or ""
    exit_code = hb.exit_code if hb else 0

    cls = classify(err, exit_code)
    return TicketRetryStatus(
        ticket_id=ticket_id, ticket_status=t.status,
        verdict=cls.verdict, retryable=cls.retryable, reason=cls.reason,
        code=cls.code, description=cls.description, recovery=cls.recovery,
        last_error=err[:1000], last_exit_code=exit_code,
    )


class RerunBody(StrictBody):
    strategy: Literal["fresh", "from_checkpoint"] = "fresh"
    actor: str = "anonymous"
    # If set, merged into ticket.payload BEFORE the rerun. Lets the
    # operator change a hyperparameter without editing the DB by hand.
    override_payload: dict = Field(default_factory=dict)
    # Set True to rerun regardless of the classifier's verdict. Default
    # False refuses to rerun when verdict='structural' (rerunning the
    # same inputs will fail the same way).
    force: bool = False


class RerunResponse(BaseModel):
    status: Literal["queued", "skipped", "refused"]
    ticket_id: str
    new_status: TicketStatus
    note: str
    classification: TicketRetryStatus


@router.post("/tickets/{ticket_id}/rerun", response_model=RerunResponse)
async def rerun_ticket(
    ticket_id: str, body: RerunBody,
    db: AsyncSession = Depends(get_db),
) -> RerunResponse:
    """Re-run any terminal ticket (failed, cancelled, succeeded, degraded, skipped).

    strategy="fresh"          flips status back to `queued`, clears
                              error_message, optionally merges
                              override_payload, queues a wakeup.
    strategy="from_checkpoint" same flip but preserves the existing
                              WorkProduct (if any). Downstream Tickets
                              that already resolved its binding keep working;
                              the agent's job is now just to re-emit a
                              fresh result on top.

    Refuses with 409 when the classifier returns verdict='structural'
    UNLESS force=True. The orchestrator/UI should surface the verdict +
    reason so users know what to fix.
    """
    from zevo.db import HeartbeatRun, WorkProduct
    from zevo.engine.run.retry_policy import classify

    t = (await db.execute(select(Ticket).where(Ticket.id == ticket_id))).scalar_one_or_none()
    if t is None:
        raise HTTPException(404, f"ticket {ticket_id} not found")
    if t.status in ("queued", "running", "repairing", "awaiting_input", "waiting_external"):
        # Already runnable / running — no rerun semantics make sense.
        return RerunResponse(
            status="skipped", ticket_id=ticket_id, new_status=t.status,
            note=f"ticket is {t.status}; nothing to rerun",
            classification=TicketRetryStatus(
                ticket_id=ticket_id, ticket_status=t.status,
                verdict="unknown", retryable=False,
                reason="ticket is not in a terminal state",
                code="not_applicable",
            ),
        )

    # Classify (re-using the same logic the UI sees)
    hb = (await db.execute(
        select(HeartbeatRun).where(HeartbeatRun.ticket_id == ticket_id)
        .order_by(desc(HeartbeatRun.started_at)).limit(1)
    )).scalar_one_or_none()
    err = (hb.error_message if hb else "") or t.error_message or ""
    exit_code = hb.exit_code if hb else 0
    cls = classify(err, exit_code)

    if cls.verdict == "structural" and not body.force:
        raise HTTPException(
            409,
            {
                "error": "refusing to rerun a structural failure",
                "verdict": cls.verdict,
                "reason": cls.reason,
                "advice": (
                    "Edit the upstream ticket / payload first, then "
                    "either rerun with force=true or fix the input."
                ),
            },
        )

    # Optionally drop the WorkProduct (fresh strategy) so the runner
    # doesn't think the work is done.
    if body.strategy == "fresh":
        existing_wps = (await db.execute(
            select(WorkProduct).where(WorkProduct.ticket_id == ticket_id)
        )).scalars().all()
        for wp in existing_wps:
            await db.delete(wp)

    # Merge override_payload if any
    if body.override_payload:
        new_payload = dict(t.payload or {})
        new_payload.update(body.override_payload)
        try:
            t.payload = validate_stored_payload(
                agent_id=t.agent_id,
                input_format=t.input_format,
                payload=new_payload,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    prev_status = t.status
    t.status = "queued"
    t.error_message = ""
    t.summary = ""
    t.repair_attempts = 0
    t.repair_route = ""
    await db.commit()

    # Audit + queue wakeup
    from zevo.engine.observe.audit import audit
    from zevo.engine.run.wakeup import queue_wakeup
    await audit(
        db, event_type="ticket.rerun",
        target_type="ticket", target_id=ticket_id, actor=body.actor,
        summary=f"{prev_status} → queued  (strategy={body.strategy})"
                + (f"  override={list(body.override_payload)}" if body.override_payload else ""),
        before={"status": prev_status},
        after={"status": "queued", "strategy": body.strategy,
               "override_payload_keys": sorted(body.override_payload.keys())},
    )
    w = await queue_wakeup(
        db, agent_id=t.agent_id, ticket_id=ticket_id,
        source="retry",
        reason=f"rerun by {body.actor} (strategy={body.strategy}, prior={prev_status})",
    )

    return RerunResponse(
        status="queued", ticket_id=ticket_id, new_status="queued",
        note=f"queued wakeup {w.id[:8]} ({w.status}); strategy={body.strategy}",
        classification=TicketRetryStatus(
            ticket_id=ticket_id, ticket_status="queued",
            verdict=cls.verdict, retryable=cls.retryable,
            reason=cls.reason, last_error=err[:1000], last_exit_code=exit_code,
            code=cls.code, description=cls.description, recovery=cls.recovery,
        ),
    )
