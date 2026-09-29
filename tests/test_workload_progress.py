"""Preparation output must not complete a benchmark or replace a train curve."""
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from starlette.requests import Request

from zevo.api.routers.shared.runs import _benchmark_progress
from zevo.api.routers.shared.tickets import get_ticket, post_progress
from zevo.db import Base, ExecutionEvent, HeartbeatRun, Run, Ticket
from zevo.engine.run.workload_execution import register_execution, bind_progress, classify_event
from zevo.engine.run.scheduler.reconciler import _persist_slurm_execution_marker


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session
    await engine.dispose()


async def setup(db, agent, provider="cluster"):
    run = Run(id="r", metric="accuracy", gpu_provider=provider, status="running", holdout={
        "validation_sets": [{"name": f"Benchmark {i}"} for i in range(6)],
    })
    ticket = Ticket(id=f"{agent}-r-001", run_id=run.id, agent_id=agent,
                    status="running", lane="optimization", iteration=0,
                    payload={"model_source": "base_model"})
    heartbeat = HeartbeatRun(id="h", ticket_id=ticket.id, agent_id=agent,
                             activation_phase="submit", driver="claude_cli", model="test")
    db.add_all([run, ticket, heartbeat])
    await db.commit()
    return run, ticket, heartbeat


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["train", "inference"])
async def test_api_cannot_claim_scheduler_source_but_real_job_progress_is_visible(db, agent):
    _, ticket, heartbeat = await setup(db, agent)
    attempt = "550e8400-e29b-41d4-a716-446655440000"
    marker = {"attempt_id": attempt, "step": 10, "total": 10, "loss": .2,
              "benchmark_id": "validation:0", "telemetry_source": "scheduler"}
    result = await post_progress(ticket.id, marker, db)
    assert result["ignored"] is True
    assert not (await db.execute(select(ExecutionEvent))).scalars().all()

    # Unattributed historical rehearsal and its attempt must not replace the
    # actual runtime curve in the ticket detail response.
    for kind in ("attempt", "progress"):
        db.add(ExecutionEvent(ticket_id=ticket.id, heartbeat_id=heartbeat.id,
                             attempt_id="old-test", event_type=kind,
                             phase="done", current_step=10, total_steps=10,
                             extras={"benchmark_id": "validation:1"}))
    for kind in ("attempt", "progress"):
        assert await _persist_slurm_execution_marker(
            db, ticket_id=ticket.id, heartbeat=heartbeat, kind=kind,
            payload=marker, phase="train", attempt_id=attempt,
        )
    await db.commit()
    detail = await get_ticket(ticket.id, Request({"type": "http"}), db)
    assert len(detail.execution_events) == 4
    verified = [e for e in detail.execution_events if e.extras["execution_purpose"] == "workload"]
    historical = [e for e in detail.execution_events if e.extras["execution_purpose"] == "unverified"]
    assert len(verified) == len(historical) == 2
    assert {e.attempt_id for e in verified} == {attempt}
    assert {e.attempt_id for e in historical} == {"old-test"}


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["train", "inference"])
@pytest.mark.parametrize("provider", ["cluster", "cloud", "instance"])
async def test_preflight_does_not_enter_progress_or_change_summary(db, agent, provider):
    _, ticket, heartbeat = await setup(db, agent, provider)
    ticket.summary = "Preparing workload"
    marker = {"progress_scope": "preflight", "step": 1, "total": 1,
              "loss": .01, "benchmark_id": "validation:0",
              "benchmark_name": "Benchmark 0", "benchmark_index": 1,
              "benchmark_total": 6}
    assert (await post_progress(ticket.id, marker, db))["ignored"] is True
    assert not await _persist_slurm_execution_marker(
        db, ticket_id=ticket.id, heartbeat=heartbeat, kind="progress",
        payload=marker, phase="preflight", attempt_id="probe",
    )
    assert ticket.summary == "Preparing workload"
    assert not (await db.execute(select(ExecutionEvent))).scalars().all()


@pytest.mark.asyncio
async def test_four_historical_rehearsal_completions_do_not_count(db):
    run, ticket, _ = await setup(db, "inference")
    events = [ExecutionEvent(
        ticket_id=ticket.id, heartbeat_id="h", attempt_id="rehearsal",
        event_type="progress", phase="done", current_step=100, total_steps=100,
        extras={"benchmark_id": f"validation:{i}"},
        ts=datetime.now(timezone.utc),
    ) for i in (0, 1, 4, 5)]
    progress = _benchmark_progress(run, [ticket], reveal_holdout=False, completion_events=events)
    assert progress["validation"]["inference_completed"] == 0
    events.append(ExecutionEvent(
        ticket_id=ticket.id, event_type="progress", current_step=100, total_steps=100,
        extras={"benchmark_id": "validation:0", "execution_id": "formal", "execution_purpose": "workload", "benchmark_complete": True},
    ))
    progress = _benchmark_progress(run, [ticket], reveal_holdout=False, completion_events=events)
    assert progress["validation"]["inference_completed"] == 1
    # Successful, validated ticket results remain authoritative for old runs.
    ticket.status = "succeeded"
    progress = _benchmark_progress(run, [ticket], reveal_holdout=False, completion_events=events)
    assert progress["validation"]["inference_completed"] == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["cloud", "instance"])
async def test_direct_execution_progress_remains_supported(db, provider):
    _, ticket, heartbeat = await setup(db, "train", provider)
    execution = await register_execution(
        db, execution_id="formal", ticket_id=ticket.id, heartbeat_id=heartbeat.id,
        runtime_key=provider, log_path="/work/train.log",
    )
    result = await post_progress(ticket.id, {
        "execution_id": execution.id, "step": 1, "total": 10, "loss": 1.2,
    }, db)
    assert result == {"ok": True}
    detail = await get_ticket(ticket.id, Request({"type": "http"}), db)
    assert detail.execution_events[0].loss == 1.2


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["cluster", "cloud", "instance"])
async def test_execution_identity_is_not_inferred_from_agent_claims(db, provider):
    _, ticket, hb = await setup(db, "train", provider)
    claimed = {"execution_id": "fake", "execution_purpose": "workload", "benchmark_complete": True, "step": 10, "total": 10}
    assert bind_progress(claimed, None).get("execution_id") is None
    assert (await post_progress(ticket.id, claimed, db))["ignored"]
    execution = await register_execution(db, execution_id="real", ticket_id=ticket.id,
                                         heartbeat_id=hb.id, runtime_key=provider, log_path="/work/train.log")
    event = ExecutionEvent(ticket_id="another-ticket", heartbeat_id=hb.id,
                           extras=bind_progress(claimed, execution))
    assert classify_event(event, {execution.id: execution})["execution_purpose"] == "unverified"
    with pytest.raises(ValueError, match="identity"):
        await register_execution(db, execution_id="real", ticket_id="another-ticket",
                                 heartbeat_id=hb.id, runtime_key=provider, log_path="/work/train.log")
