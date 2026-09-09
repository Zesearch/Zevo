"""Registry handoff isolation and supervisor completion regressions."""

import datetime as dt

import pytest
import pytest_asyncio
import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import (
    AgentWakeupRequest, Base, HeartbeatRun, RegistryModel, Run, Ticket,
    TicketNotice,
)
from zevo.api.routers.shared.tickets import _wakeup_waits_for_supervisor
from zevo.contracts.orchestrator import SupervisorAction
from zevo.engine.persistence import upsert_registry_from_manifest
from zevo.engine.run.runner import _release_emitted_child
from zevo.engine.run.scheduler.reconciler import _close_finished_runs


def _ago(seconds: float) -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=seconds)


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        yield s
    await engine.dispose()


def _ticket(tid, run_id, agent, *, status="succeeded", updated_ago=100):
    return Ticket(
        id=tid, run_id=run_id, agent_id=agent, status=status,
        payload={},
        updated_at=_ago(updated_ago),
    )


# ───────────────── serial supervisor → child handoff ────────────────────────


@pytest.mark.asyncio
async def test_child_created_during_supervisor_heartbeat_is_not_woken(session):
    run = Run(
        id="serial-run", metric="accuracy", agent_objective="o",
        status="running", mode="full_pipeline",
        supervisor_ticket_id="orchestrate-serial-001",
    )
    supervisor = _ticket(
        "orchestrate-serial-001", "serial-run", "orchestrator", status="running",
    )
    session.add_all([run, supervisor])
    session.add(HeartbeatRun(
        id="active-supervisor-heartbeat",
        ticket_id=supervisor.id,
        agent_id="orchestrator",
        driver="stub",
        model="stub",
        started_at=dt.datetime.now(dt.timezone.utc),
        finished_at=None,
    ))
    await session.commit()

    assert await _wakeup_waits_for_supervisor(
        session, run=run, assignee="inference",
    ) is True
    assert (await session.execute(select(AgentWakeupRequest))).scalars().all() == []


@pytest.mark.asyncio
async def test_completed_supervisor_action_releases_exactly_its_child(session):
    run = Run(
        id="serial-run", metric="accuracy", agent_objective="o",
        status="running", mode="full_pipeline",
        supervisor_ticket_id="orchestrate-serial-001",
    )
    supervisor = _ticket(
        "orchestrate-serial-001", "serial-run", "orchestrator",
        status="succeeded",
    )
    child = _ticket(
        "inference-serial-001", "serial-run", "inference", status="queued",
    )
    session.add_all([run, supervisor, child, TicketNotice(
        ticket_id=child.id,
        code="ticket.awaiting_supervisor_completion",
        severity="info",
        body="waiting",
    )])
    await session.commit()

    await _release_emitted_child(
        session,
        supervisor=supervisor,
        output=SupervisorAction(
            ticket_id=supervisor.id,
            action="emit_ticket",
            child_ticket_id=child.id,
            summary="start inference",
        ),
        heartbeat_id="finished-supervisor-heartbeat",
    )

    wakes = (await session.execute(select(AgentWakeupRequest))).scalars().all()
    assert len(wakes) == 1
    assert wakes[0].ticket_id == child.id
    assert wakes[0].status == "queued"
    assert wakes[0].source == "handoff"
    assert wakes[0].trigger_detail == "supervisor_heartbeat:finished-supervisor-heartbeat"
    notice = (await session.execute(select(TicketNotice).where(
        TicketNotice.ticket_id == child.id,
    ))).scalar_one()
    assert notice.code == "ticket.supervisor_handoff_released"


@pytest.mark.asyncio
async def test_failed_supervisor_does_not_release_its_declared_child(session):
    supervisor = _ticket(
        "orchestrate-failed-001", "failed-run", "orchestrator", status="failed",
    )
    child = _ticket(
        "data-failed-001", "failed-run", "data", status="queued",
    )
    session.add_all([supervisor, child])
    await session.commit()

    await _release_emitted_child(
        session,
        supervisor=supervisor,
        output=SupervisorAction(
            ticket_id=supervisor.id,
            action="emit_ticket",
            child_ticket_id=child.id,
            summary="POST succeeded but the activation later failed",
        ),
        heartbeat_id="failed-supervisor-heartbeat",
    )

    assert (await session.execute(select(AgentWakeupRequest))).scalars().all() == []


# ───────────────── Ticket-local Registry manifest mirror ─────────────────────


def _yaml(tmp_path, entries: dict) -> str:
    p = tmp_path / "registry.yaml"
    complete = {
        tag: {
            "ticket_id": f"registry-{str(body.get('run_id') or 'none')[:8]}-001",
            "iteration": 1,
            "base_model": "base",
            "training_method": "full_sft",
            "dataset_source": "data/files/train.jsonl",
            "task_objective": "improve the model",
            "metric": "accuracy",
            "metric_direction": "max",
            "model_path": f"/app/data/runs/{body.get('run_id', 'none')}/models/{tag}",
            "eval": {"score": 0.5},
            "retention": "retained",
            "registered_at": "2026-08-19T12:00:00Z",
            **body,
        }
        for tag, body in entries.items()
    }
    p.write_text(yaml.safe_dump({"models": complete}, sort_keys=False), encoding="utf-8")
    return str(p)


@pytest.mark.asyncio
async def test_manifest_rejects_an_unrelated_extra_entry(session, tmp_path):
    """A Ticket manifest is one snapshot, never a second global registry."""
    session.add(Run(metric="accuracy", id="live", agent_objective="o", status="running"))
    await session.commit()

    path = _yaml(tmp_path, {
        "M-live": {"run_id": "live", "iteration": 1, "base_model": "b", "training_method": "full_sft"},
        "M-gone": {"run_id": "gone"},
    })
    with pytest.raises(ValueError, match="exactly"):
        await upsert_registry_from_manifest(
            session, registry_manifest_path=path, run_id="live"
        )


@pytest.mark.asyncio
async def test_mirror_requires_an_explicit_producing_run(session, tmp_path):
    """An unscoped mirror could let historical entries interfere with one another."""
    session.add(Run(metric="accuracy", id="live", agent_objective="o", status="running"))
    await session.commit()
    path = _yaml(tmp_path, {f"v-{i}": {"run_id": f"gone-{i}", "iteration": i} for i in range(3)})

    with pytest.raises(ValueError, match="producing run_id"):
        await upsert_registry_from_manifest(
            session, registry_manifest_path=path, run_id=""
        )


@pytest.mark.asyncio
async def test_mirror_replaces_the_stable_champion_row(session, tmp_path):
    """A winning challenger updates every champion fact under one stable tag."""
    session.add(Run(metric="accuracy", id="live", agent_objective="o", status="running"))
    await session.commit()
    path = _yaml(tmp_path, {
        "M-live": {
            "run_id": "live", "iteration": 1, "base_model": "b", "training_method": "full_sft",
            "model_path": "/app/data/runs/live/models/M-live",
            "eval": {"score": 0.4},
        },
    })

    assert await upsert_registry_from_manifest(
        session, registry_manifest_path=path, run_id="live"
    ) == 1
    row = await session.get(RegistryModel, "M-live")
    assert row.model_path == "data/runs/live/models/M-live"

    # A better challenger replaces the one current model rather than adding a
    # second RegistryModel row.
    with open(path, encoding="utf-8") as fh:
        body = yaml.safe_load(fh)
    body["models"]["M-live"].update({
        "iteration": 2,
        "base_model": "b2",
        "training_method": "lora_sft",
        "model_path": "/app/data/runs/live/models/M-live",
        "eval": {"score": 0.7},
    })
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(body, fh, sort_keys=False)

    assert await upsert_registry_from_manifest(
        session, registry_manifest_path=path, run_id="live"
    ) == 0
    await session.refresh(row)
    assert row.base_model == "b2"
    assert row.training_method == "lora_sft"
    assert row.eval == {"score": 0.7}
    assert row.model_path == "data/runs/live/models/M-live"
    assert row.iteration == 2

# ───────────────── link 5: the reconciler repairs, it does not guess ─────────


@pytest.mark.asyncio
async def test_a_completion_the_supervisor_never_saw_wakes_it_instead_of_closing(session):
    """The exact shape of the failure: every ticket terminal, a model registered,
    and one child that finished AFTER the supervisor last ran."""
    session.add(Run(metric="accuracy", id="r1", agent_objective="o", status="running",
                    supervisor_ticket_id="orchestrate-r1-001",
                    registry_version_tag="v1"))
    session.add(_ticket("orchestrate-r1-001", "r1", "orchestrator", updated_ago=300))
    session.add(_ticket("train-001", "r1", "train", updated_ago=400))
    session.add(_ticket("registry-001", "r1", "registry", updated_ago=10))  # after
    session.add(RegistryModel(metric="accuracy", metric_direction="max", version_tag="v1", run_id="r1", base_model="b",
                              training_method="full_sft", eval={}))
    await session.commit()

    counts = await _close_finished_runs(session)

    # Nothing closed: the run is still open, waiting on the wake this test is
    # about. `degraded` joined the tally when a failed ticket stopped being
    # outvoted by a registered model.
    assert counts == {"success": 0, "degraded": 0, "failed": 0, "halted": 0}
    assert (await session.get(Run, "r1")).status == "running"   # left open
    wakes = (await session.execute(
        select(AgentWakeupRequest).where(
            AgentWakeupRequest.ticket_id == "orchestrate-r1-001")
    )).scalars().all()
    assert len(wakes) == 1
    assert "registry-001" in wakes[0].reason


@pytest.mark.asyncio
async def test_a_run_the_supervisor_has_seen_out_requests_explicit_finalization(session):
    """Terminal workers are not permission for reconciliation to invent success."""
    session.add(Run(metric="accuracy", id="r2", agent_objective="o", status="running",
                    supervisor_ticket_id="orchestrate-r2-001",
                    registry_version_tag="v2"))
    session.add(_ticket("orchestrate-r2-001", "r2", "orchestrator", updated_ago=10))
    session.add(_ticket("train-002", "r2", "train", updated_ago=400))
    session.add(_ticket("registry-002", "r2", "registry", updated_ago=100))
    session.add(RegistryModel(metric="accuracy", metric_direction="max", version_tag="v2", run_id="r2", base_model="b",
                              training_method="full_sft", eval={}))
    await session.commit()

    counts = await _close_finished_runs(session)
    assert counts == {"success": 0, "degraded": 0, "failed": 0, "halted": 0}
    assert (await session.get(Run, "r2")).status == "running"
    wake = (await session.execute(select(AgentWakeupRequest).where(
        AgentWakeupRequest.ticket_id == "orchestrate-r2-001",
        AgentWakeupRequest.trigger_detail == "finalize_run",
    ))).scalar_one()
    assert wake.source == "reconciler"


@pytest.mark.asyncio
async def test_repair_does_not_storm(session):
    """queue_wakeup coalesces, so repeated ticks over an unrepaired run leave one
    pending request rather than one per tick."""
    session.add(Run(metric="accuracy", id="r3", agent_objective="o", status="running",
                    supervisor_ticket_id="orchestrate-r3-001"))
    session.add(_ticket("orchestrate-r3-001", "r3", "orchestrator", updated_ago=300))
    session.add(_ticket("registry-003", "r3", "registry", updated_ago=10))
    await session.commit()

    for _ in range(3):
        await _close_finished_runs(session)

    wakes = (await session.execute(
        select(AgentWakeupRequest).where(
            AgentWakeupRequest.ticket_id == "orchestrate-r3-001",
            AgentWakeupRequest.status == "queued")
    )).scalars().all()
    assert len(wakes) == 1


@pytest.mark.asyncio
async def test_a_run_with_no_supervisor_is_unaffected(session):
    """Single-agent runs have no supervisor to owe anything."""
    session.add(Run(metric="accuracy", id="r4", agent_objective="o", status="running"))
    session.add(_ticket("train-004", "r4", "train", updated_ago=10))
    await session.commit()

    counts = await _close_finished_runs(session)
    assert counts["halted"] == 1   # terminal, no model registered
    assert (await session.get(Run, "r4")).status == "halted"
