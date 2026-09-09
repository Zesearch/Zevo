"""GET /agents, GET /agents/{id} (detail), GET /agents/{id}/instructions,
GET /agents/{id}/heartbeats, GET /agents/{id}/tickets,
POST /agents/{id}/invoke."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from zevo.contracts._base import StrictBody
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.api.database import get_db
from zevo.api.ui_access import is_trusted_ui_request
from zevo.engine.agent.loader import (
    INPUT_FORMAT_FREEFORM,
    INPUT_FORMAT_TYPED,
    load_agent,
)
from zevo.db import Agent, HeartbeatRun, Run, Ticket
from zevo.contracts.tickets import TERMINAL_RUN_STATUSES


router = APIRouter()


_SYSTEM_RUNNER_IDS = frozenset({"evaluation"})


def _require_public_agent(agent_id: str) -> None:
    """Keep deterministic runners out of callable/configurable Agent details."""
    if agent_id in _SYSTEM_RUNNER_IDS:
        raise HTTPException(404, f"agent {agent_id} not found")


class SkillCardDTO(BaseModel):
    """One method in an agent's pool: the slug, the payload id, and the
    one-liner from the card's own frontmatter."""

    name: str
    method: str
    description: str


class AgentDTO(BaseModel):
    id: str
    name: str
    title: str
    reports_to: str
    default_driver: str
    default_model: str
    sandbox: str = "none"
    output_schema: str
    skills: list[str] = Field(default_factory=list)  # playbook skill names
    # The same pool with each card's own description, so the console can say
    # what a skill IS. A bare list of slugs told you `lora-sft` exists and
    # nothing about when it is the right choice.
    skill_cards: list["SkillCardDTO"] = Field(default_factory=list)
    identity_path: str = ""


class HeartbeatDTO(BaseModel):
    id: str
    ticket_id: str
    driver: str
    model: str
    operation: str = ""
    activation_phase: str = ""
    started_at: str
    finished_at: str = ""
    exit_code: int
    error_message: str = ""


class AgentStats(BaseModel):
    total_heartbeats: int
    success_count: int
    failure_count: int
    last_finished_at: str = ""
    open_ticket_count: int
    ticket_count: int = 0      # every ticket ever assigned to this agent


class TicketSummaryDTO(BaseModel):
    id: str
    run_id: str
    input_format: str
    lane: str
    iteration: int
    status: str
    summary: str
    created_at: str


class AgentDetailDTO(AgentDTO):
    stats: AgentStats
    recent_heartbeats: list[HeartbeatDTO]
    recent_tickets: list[TicketSummaryDTO]


def _skill_cards(agent_id: str) -> list["SkillCardDTO"]:
    """The agent's method pool, each card with what it is for.

    Read from the blueprint; defensive so a load error never breaks the list.
    """
    try:
        return [
            SkillCardDTO(name=s.name, method=s.method, description=s.description)
            for s in load_agent(agent_id).skills
        ]
    except Exception:
        return []


def _agent_dto(r: Agent) -> AgentDTO:
    cards = _skill_cards(r.id)  # one blueprint load, reused for both fields
    if r.id == "evaluation":
        default_driver = r.default_driver or "evaluation_runner"
        default_model = r.default_model or "no-llm"
    else:
        blueprint = load_agent(r.id)
        default_driver = r.default_driver or blueprint.default_driver
        default_model = r.default_model or blueprint.default_model
    return AgentDTO(
        id=r.id, name=r.name, title=r.title, reports_to=r.reports_to,
        default_driver=default_driver,
        default_model=default_model,
        sandbox=r.sandbox,
        output_schema=r.output_schema,
        skills=[c.name for c in cards],
        skill_cards=cards,
        identity_path=r.identity_path or "",
    )


def _heartbeat_dto(h: HeartbeatRun) -> HeartbeatDTO:
    return HeartbeatDTO(
        id=h.id, ticket_id=h.ticket_id, driver=h.driver, model=h.model,
        operation=h.operation or "",
        activation_phase=h.activation_phase or "",
        started_at=h.started_at.isoformat() if h.started_at else "",
        finished_at=h.finished_at.isoformat() if h.finished_at else "",
        exit_code=h.exit_code, error_message=h.error_message or "",
    )


def _ticket_summary(t: Ticket) -> TicketSummaryDTO:
    return TicketSummaryDTO(
        id=t.id, run_id=t.run_id, input_format=t.input_format,
        lane=t.lane, iteration=t.iteration,
        status=t.status, summary=(t.summary or "")[:200],
        created_at=t.created_at.isoformat() if t.created_at else "",
    )


def _visible_ticket_ids():
    return (
        select(Ticket.id)
        .join(Run, Run.id == Ticket.run_id)
        .where(
            (Ticket.lane != "held_out_test")
            | Run.status.in_(TERMINAL_RUN_STATUSES)
        )
    )


@router.get("/agents", response_model=list[AgentDTO])
async def list_agents(db: AsyncSession = Depends(get_db)) -> list[AgentDTO]:
    rows = (await db.execute(
        select(Agent).order_by(Agent.id)
    )).scalars().all()
    return [_agent_dto(r) for r in rows]


@router.get("/agents/{agent_id}", response_model=AgentDetailDTO)
async def get_agent(
    agent_id: str, request: Request, db: AsyncSession = Depends(get_db),
) -> AgentDetailDTO:
    _require_public_agent(agent_id)
    r = (await db.execute(select(Agent).where(Agent.id == agent_id))).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"agent {agent_id} not found")
    trusted_ui = is_trusted_ui_request(request)
    recent_hb_stmt = (
        select(HeartbeatRun)
        .where(HeartbeatRun.agent_id == agent_id)
        .order_by(desc(HeartbeatRun.started_at))
        .limit(10)
    )
    all_hb_stmt = select(HeartbeatRun).where(HeartbeatRun.agent_id == agent_id)
    recent_ticket_stmt = (
        select(Ticket).where(Ticket.agent_id == agent_id)
        .order_by(desc(Ticket.created_at)).limit(10)
    )
    open_stmt = select(Ticket).where(
        Ticket.agent_id == agent_id,
        Ticket.status.in_(["queued", "running", "repairing", "awaiting_input", "waiting_external"]),
    )
    total_stmt = select(func.count()).select_from(Ticket).where(Ticket.agent_id == agent_id)
    if not trusted_ui:
        visible = _visible_ticket_ids()
        recent_hb_stmt = recent_hb_stmt.where(HeartbeatRun.ticket_id.in_(visible))
        all_hb_stmt = all_hb_stmt.where(HeartbeatRun.ticket_id.in_(visible))
        recent_ticket_stmt = recent_ticket_stmt.where(Ticket.id.in_(visible))
        open_stmt = open_stmt.where(Ticket.id.in_(visible))
        total_stmt = total_stmt.where(Ticket.id.in_(visible))

    hbs = (
        await db.execute(
            recent_hb_stmt
        )
    ).scalars().all()
    all_hbs = (
        await db.execute(all_hb_stmt)
    ).scalars().all()
    tickets_recent = (
        await db.execute(recent_ticket_stmt)
    ).scalars().all()
    open_count = (
        await db.execute(open_stmt)
    ).scalars().all()
    ticket_total = (
        await db.execute(total_stmt)
    ).scalar_one()

    success = sum(1 for h in all_hbs if h.exit_code == 0)
    failure = sum(1 for h in all_hbs if h.exit_code not in (0, -1))
    last_finished = max(
        (h.finished_at for h in all_hbs if h.finished_at), default=None
    )

    base = _agent_dto(r)
    return AgentDetailDTO(
        **base.model_dump(),
        stats=AgentStats(
            total_heartbeats=len(all_hbs),
            success_count=success,
            failure_count=failure,
            last_finished_at=last_finished.isoformat() if last_finished else "",
            open_ticket_count=len(open_count),
            ticket_count=int(ticket_total or 0),
        ),
        recent_heartbeats=[_heartbeat_dto(h) for h in hbs],
        recent_tickets=[_ticket_summary(t) for t in tickets_recent],
    )


class PatchAgentBody(StrictBody):
    default_driver: str | None = None
    default_model: str | None = None
    sandbox: str | None = None


@router.patch("/agents/{agent_id}", response_model=AgentDTO)
async def patch_agent(
    agent_id: str,
    body: PatchAgentBody,
    db: AsyncSession = Depends(get_db),
) -> AgentDTO:
    """Update an agent's runtime config (default driver + model).

    The runner reads these from the DB row at heartbeat-start, so the
    change takes effect on the agent's NEXT invocation (no restart).
    Pass `default_driver=""` to revert to the identity.md frontmatter
    default (seeder will refill on next start).
    """
    _require_public_agent(agent_id)
    r = (await db.execute(select(Agent).where(Agent.id == agent_id))).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"agent {agent_id} not found")

    before = {"default_driver": r.default_driver, "default_model": r.default_model}

    if body.default_driver is not None:
        if body.default_driver:
            # Validate that the requested driver name is known. Catches
            # typos ("clade_cli") before the next heartbeat would fail.
            from zevo.engine.agent.drivers import is_known_driver_name
            if not is_known_driver_name(body.default_driver):
                raise HTTPException(400, f"unknown driver {body.default_driver!r}")
        r.default_driver = body.default_driver
    if body.default_model is not None:
        r.default_model = body.default_model
    if body.sandbox is not None:
        sb = (body.sandbox or "").strip().lower()
        if sb not in ("none", "openshell"):
            raise HTTPException(400, f"invalid sandbox {body.sandbox!r} (use 'none' or 'openshell')")
        effective_driver = (
            body.default_driver if body.default_driver is not None else r.default_driver
        ) or load_agent(agent_id).default_driver
        if sb == "openshell" and effective_driver != "claude_cli":
            raise HTTPException(
                400,
                "OpenShell requires the claude_cli driver",
            )
        if sb == "openshell" and agent_id != "orchestrator":
            raise HTTPException(
                400,
                "OpenShell is supported only for the orchestrator; worker agents "
                "require cross-ticket files or SSH that the current sandbox policy "
                "intentionally does not expose",
            )
        r.sandbox = sb

    if r.sandbox == "openshell" and (
        (r.default_driver or load_agent(agent_id).default_driver) != "claude_cli"
        or agent_id != "orchestrator"
    ):
        raise HTTPException(
            400,
            "OpenShell requires the claude_cli orchestrator",
        )

    await db.commit()
    await db.refresh(r)

    from zevo.engine.observe.audit import audit
    await audit(
        db, event_type="agent.patch", target_type="agent", target_id=agent_id,
        summary=f"{before['default_driver']!r}/{before['default_model']!r} → "
                f"{r.default_driver!r}/{r.default_model!r}",
        before=before,
        after={"default_driver": r.default_driver, "default_model": r.default_model},
    )

    return _agent_dto(r)


@router.get("/agents/{agent_id}/instructions")
async def get_instructions(
    agent_id: str, input_format: str = INPUT_FORMAT_TYPED,
    db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    _require_public_agent(agent_id)
    r = (await db.execute(select(Agent).where(Agent.id == agent_id))).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"agent {agent_id} not found")
    # Return the ASSEMBLED system prompt (identity + goal + shared commons +
    # platform + the mode-selected invocation doc), not a single source file --
    # the prompt is now composed from several docs under playbook/agents/<id>/. Resolving
    # via load_agent also avoids drift when the stored definition path changes.
    try:
        selected = (
            input_format if input_format == INPUT_FORMAT_FREEFORM
            else INPUT_FORMAT_TYPED
        )
        bp = load_agent(agent_id, input_format=selected)
    except (FileNotFoundError, ValueError, KeyError, TypeError, ImportError) as e:
        raise HTTPException(404, f"agent definition for {agent_id} unreadable: {e}")
    return {
        "agent_id": agent_id,
        "identity_path": str(bp.identity_path),
        "input_format": selected,
        "content": bp.instructions,
    }


@router.get("/agents/{agent_id}/heartbeats", response_model=list[HeartbeatDTO])
async def list_heartbeats(
    agent_id: str, request: Request, limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> list[HeartbeatDTO]:
    _require_public_agent(agent_id)
    stmt = select(HeartbeatRun).where(HeartbeatRun.agent_id == agent_id)
    if not is_trusted_ui_request(request):
        stmt = stmt.where(HeartbeatRun.ticket_id.in_(_visible_ticket_ids()))
    rows = (await db.execute(
        stmt.order_by(desc(HeartbeatRun.started_at)).limit(limit)
    )).scalars().all()
    return [_heartbeat_dto(h) for h in rows]


@router.get("/agents/{agent_id}/tickets", response_model=list[TicketSummaryDTO])
async def list_agent_tickets(
    agent_id: str, request: Request, limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> list[TicketSummaryDTO]:
    _require_public_agent(agent_id)
    stmt = select(Ticket).where(Ticket.agent_id == agent_id)
    if not is_trusted_ui_request(request):
        stmt = stmt.where(Ticket.id.in_(_visible_ticket_ids()))
    rows = (await db.execute(
        stmt.order_by(desc(Ticket.created_at)).limit(limit)
    )).scalars().all()
    return [_ticket_summary(r) for r in rows]


class InvokeBody(StrictBody):
    ticket_id: str = ""  # if empty, picks the oldest unstarted ticket for this agent
    driver: str = ""     # override the identity.md default
    model: str = ""


class InvokeResponse(BaseModel):
    status: str
    agent_id: str
    ticket_id: str
    detail: str = ""


@router.post("/agents/{agent_id}/invoke", response_model=InvokeResponse)
async def invoke_agent(
    agent_id: str,
    body: InvokeBody,
    db: AsyncSession = Depends(get_db),
) -> InvokeResponse:
    """Enqueue a wakeup for this agent.

    If ticket_id is provided, the wakeup targets that ticket.
    Otherwise enqueues a no-ticket wakeup; the daemon picks the oldest
    queued ticket from the agent's inbox.
    """
    _require_public_agent(agent_id)
    r = (await db.execute(select(Agent).where(Agent.id == agent_id))).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"agent {agent_id} not found")

    target_ticket_id: str | None = body.ticket_id or None
    if target_ticket_id is None:
        # Sanity: report no_work if the inbox is empty so the UI shows
        # something useful, but ALSO enqueue a cron-style wakeup so the
        # daemon checks on the next tick.
        candidate = (
            await db.execute(
                select(Ticket)
                .where(Ticket.agent_id == agent_id, Ticket.status == "queued")
                .order_by(Ticket.created_at)
                .limit(1)
            )
        ).scalar_one_or_none()
        if candidate is None:
            return InvokeResponse(
                status="no_work",
                agent_id=agent_id,
                ticket_id="",
                detail=f"No queued tickets assigned to {agent_id}.",
            )
        target_ticket_id = candidate.id

    from zevo.engine.run.wakeup import queue_wakeup
    w = await queue_wakeup(
        db, agent_id=agent_id, ticket_id=target_ticket_id,
        source="on_demand",
        reason=f"UI/CLI invoke (driver={body.driver!r}, model={body.model!r})",
        payload={"driver": body.driver, "model": body.model},
    )
    return InvokeResponse(
        status=w.status,  # 'queued' or 'coalesced'
        agent_id=agent_id,
        ticket_id=target_ticket_id or "",
        detail=f"Enqueued wakeup {w.id[:8]} ({w.status}). Daemon will pick it up.",
    )
