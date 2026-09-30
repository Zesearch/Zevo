"""User instructions survive Ticket turns and get a visible agent decision."""
from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.database import get_db
from zevo.api.routers.shared import runs as runs_router
from zevo.api.routers.shared.tickets import MessageBody, post_message
from zevo.db import AgentWakeupRequest, Base, Run, RunInstruction, Ticket
from zevo.engine.agent.drivers._prompt import _conversation_block
from zevo.engine.run.runner import _run_instruction_context
from zevo.engine.run.steering import instruction_gate_active


@pytest_asyncio.fixture
async def client_and_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    app = FastAPI()
    app.include_router(runs_router.router, prefix="/api")

    async def database_override():
        async with Session() as session:
            yield session

    app.dependency_overrides[get_db] = database_override
    async with Session() as session:
        session.add(Run(
            id="run-1", task_name="task", status="running",
            mode="full_pipeline", metric="accuracy",
            supervisor_ticket_id="orchestrate-001",
        ))
        session.add(Ticket(
            id="orchestrate-001", run_id="run-1", agent_id="orchestrator",
            status="succeeded", payload={},
        ))
        session.add(Ticket(
            id="train-001", run_id="run-1", agent_id="train",
            status="waiting_external", payload={},
        ))
        session.add(Ticket(
            id="test-001", run_id="run-1", agent_id="inference",
            lane="held_out_test", status="running", payload={},
        ))
        await session.commit()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, Session
    await engine.dispose()


@pytest.mark.asyncio
async def test_run_instruction_wakes_supervisor_and_persists_decision(client_and_session) -> None:
    client, Session = client_and_session
    posted = await client.post(
        "/api/runs/run-1/instructions",
        json={"body": "  Use fewer GPUs when safe  ", "source_ticket_id": "train-001"},
    )
    assert posted.status_code == 200
    instruction = posted.json()
    assert instruction["body"] == "Use fewer GPUs when safe"
    assert instruction["status"] == "queued"
    assert instruction["source_ticket_id"] == "train-001"

    async with Session() as session:
        wakeups = (await session.execute(select(AgentWakeupRequest))).scalars().all()
        assert len(wakeups) == 1
        assert wakeups[0].ticket_id == "orchestrate-001"
        run = await session.get(Run, "run-1")
        supervisor = await session.get(Ticket, "orchestrate-001")
        specialist = await session.get(Ticket, "train-001")
        assert await instruction_gate_active(session, "run-1") is True
        assert await _run_instruction_context(session, run=run, ticket=specialist) == []
        context = await _run_instruction_context(session, run=run, ticket=supervisor)
        assert [item["body"] for item in context] == ["Use fewer GPUs when safe"]
        assert (await session.get(RunInstruction, instruction["id"])).status == "delivered"
        prompt = _conversation_block(context)
        assert "RUN USER INSTRUCTIONS" in prompt
        assert instruction["id"] in prompt
        assert "PATCH /api/runs/{run_id}/instructions/{instruction_id}" in prompt

    decision = await client.patch(
        f"/api/runs/run-1/instructions/{instruction['id']}",
        json={"status": "scheduled", "agent_response": "I will change the next safe step."},
    )
    assert decision.status_code == 200
    assert decision.json()["status"] == "scheduled"
    listed = await client.get("/api/runs/run-1/instructions")
    assert listed.status_code == 200
    assert listed.json()[0]["agent_response"] == "I will change the next safe step."

    async with Session() as session:
        assert await instruction_gate_active(session, "run-1") is False
        context = await _run_instruction_context(
            session,
            run=await session.get(Run, "run-1"),
            ticket=await session.get(Ticket, "orchestrate-001"),
        )
        assert context[0]["status"] == "scheduled"

    applied = await client.patch(
        f"/api/runs/run-1/instructions/{instruction['id']}",
        json={"status": "applied", "agent_response": "The requested change is in the next plan."},
    )
    assert applied.status_code == 200
    async with Session() as session:
        assert await _run_instruction_context(
            session,
            run=await session.get(Run, "run-1"),
            ticket=await session.get(Ticket, "orchestrate-001"),
        ) == []


@pytest.mark.asyncio
async def test_needs_input_keeps_run_instruction_gate_closed(client_and_session) -> None:
    client, Session = client_and_session
    posted = await client.post(
        "/api/runs/run-1/instructions", json={"body": "Change the next step"},
    )
    instruction_id = posted.json()["id"]
    decision = await client.patch(
        f"/api/runs/run-1/instructions/{instruction_id}",
        json={"status": "needs_input", "agent_response": "Which checkpoint?"},
    )
    assert decision.status_code == 200
    async with Session() as session:
        assert await instruction_gate_active(session, "run-1") is True


@pytest.mark.asyncio
async def test_orchestrator_routes_instruction_as_a_quiesced_specialist_wake(
    client_and_session,
) -> None:
    client, Session = client_and_session
    posted = await client.post(
        "/api/runs/run-1/instructions", json={"body": "Apply this to Train now"},
    )
    instruction_id = posted.json()["id"]
    async with Session() as session:
        instruction = await session.get(RunInstruction, instruction_id)
        instruction.status = "delivered"
        await session.commit()
        await post_message(
            "train-001",
            MessageBody(
                body="Apply this to Train now",
                author="orchestrator",
                run_instruction_id=instruction_id,
                instruction_action="restart_activation",
            ),
            session,
        )
        ticket = await session.get(Ticket, "train-001")
        assert ticket.status == "queued"
        wakeups = (await session.execute(
            select(AgentWakeupRequest).where(
                AgentWakeupRequest.ticket_id == "train-001"
            )
        )).scalars().all()
        assert len(wakeups) == 1
        assert wakeups[0].payload == {"run_instruction_id": instruction_id, "instruction_action": "restart_activation"}


@pytest.mark.asyncio
async def test_ended_run_rejects_new_instruction(client_and_session) -> None:
    client, Session = client_and_session
    async with Session() as session:
        run = await session.get(Run, "run-1")
        run.status = "cancelled"
        await session.commit()
    response = await client.post(
        "/api/runs/run-1/instructions", json={"body": "Start again"},
    )
    assert response.status_code == 409
    async with Session() as session:
        assert (await session.execute(select(RunInstruction))).scalars().all() == []


@pytest.mark.asyncio
async def test_private_test_ticket_cannot_be_instruction_context(client_and_session) -> None:
    client, _Session = client_and_session
    response = await client.post(
        "/api/runs/run-1/instructions",
        json={"body": "Please review this step", "source_ticket_id": "test-001"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_cancel", [False, True])
async def test_replacement_sigterm_does_not_cancel_ticket_or_consume_repair(
    client_and_session, monkeypatch, explicit_cancel,
):
    from datetime import datetime, timezone
    from zevo.db import HeartbeatRun
    from zevo.engine.run.activation_control import finish_superseded_activation
    from zevo.engine.run.scheduler.reconciler import _close_finished_runs
    from zevo.engine.run import remote_jobs

    client, Session = client_and_session
    posted = await client.post('/api/runs/run-1/instructions', json={'body': 'Replace the current execution'})
    instruction_id = posted.json()['id']
    calls = []

    async def stop(db, ticket, *, preserve_allocation=False):
        calls.append(preserve_allocation)
        return {'attempted': True, 'ok': True}

    monkeypatch.setattr(remote_jobs, 'cancel_ticket_remote_job', stop)
    async with Session() as runner:
        ticket = await runner.get(Ticket, 'train-001')
        ticket.status = 'running'
        ticket.repair_attempts = 2
        hb = HeartbeatRun(id='old-activation', ticket_id=ticket.id, agent_id='train', driver='test', model='test')
        runner.add(hb)
        await runner.commit()
        async with Session() as api:
            await post_message(ticket.id, MessageBody(
                body='Replace the current execution', author='orchestrator',
                run_instruction_id=instruction_id, instruction_action='restart_activation',
            ), api)
            current = await api.get(Ticket, ticket.id)
            assert current.status == 'running'
            assert (await api.get(Run, 'run-1')).status == 'running'
            if explicit_cancel:
                current.status = 'cancelled'
                await api.commit()
        hb.finished_at = datetime.now(timezone.utc)
        hb.exit_code = 1
        hb.error_message = 'RuntimeError: claude exec cancelled (signal SIGTERM)'
        hb.estimated_cost_usd = 1.25
        assert await finish_superseded_activation(runner, ticket, hb) is (not explicit_cancel)
        await runner.commit()
        assert ticket.status == ('cancelled' if explicit_cancel else 'queued')
        assert ticket.repair_attempts == 2
        assert hb.superseded_by_instruction_id == instruction_id
        assert hb.estimated_cost_usd == 1.25
        assert hb.finished_at is not None
        if not explicit_cancel:
            await _close_finished_runs(runner)
            assert (await runner.get(Run, 'run-1')).status == 'running'
    assert calls == []  # Commit the handoff before process cleanup.


@pytest.mark.asyncio
async def test_instruction_continue_does_not_stop_execution(client_and_session, monkeypatch):
    from zevo.engine.run import remote_jobs
    client, Session = client_and_session
    posted = await client.post('/api/runs/run-1/instructions', json={'body': 'Use this context later'})

    async def unexpected_stop(*args, **kwargs):
        pytest.fail('continuing an instruction must not stop the workload')

    monkeypatch.setattr(remote_jobs, 'cancel_ticket_remote_job', unexpected_stop)
    async with Session() as db:
        await post_message('train-001', MessageBody(
            body='Use this context later', author='orchestrator',
            run_instruction_id=posted.json()['id'],
        ), db)
        assert (await db.get(Ticket, 'train-001')).status == 'waiting_external'
