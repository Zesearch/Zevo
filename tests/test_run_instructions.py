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
from zevo.db import InfraInstance
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


@pytest.mark.asyncio
async def test_resize_persists_request_and_suppresses_old_watcher(client_and_session):
    from datetime import datetime, timezone
    from zevo.engine.run.runner import _slurm_watcher_job_is_current, _activate_failed_workload_repair
    client, Session = client_and_session
    posted = await client.post('/api/runs/run-1/instructions', json={'body': 'Use 4 GPUs'})
    async with Session() as db:
        run = await db.get(Run, 'run-1')
        run.gpu_provider = 'cluster'
        run.num_gpus = 4
        old = InfraInstance(
            id='old-job', run_id='run-1', ticket_id='train-001',
            provider='cluster', instance_id='123', status='failed', gpu_count=1,
            released_at=datetime.now(timezone.utc),
            meta={'resource_request': True, 'submission_committed': True,
                  'scheduler_state': 'FAILED'},
        )
        db.add(old)
        await db.commit()
        await post_message('train-001', MessageBody(
            body='Use 4 GPUs', author='orchestrator',
            run_instruction_id=posted.json()['id'], instruction_action='restart_activation',
            requested_gpus=4,
        ), db)
        ticket = await db.get(Ticket, 'train-001')
        from zevo.engine.run.steering import requested_slurm_gpus
        assert await requested_slurm_gpus(db, ticket.id) == 4
        assert old.meta['superseded_by_instruction_id'] == posted.json()['id']
        assert not await _slurm_watcher_job_is_current(db, ticket, {'job_id': '123'})
        assert not await _activate_failed_workload_repair(db, ticket)
        assert ticket.repair_attempts == 0
        assert ticket.status == 'queued'


@pytest.mark.asyncio
@pytest.mark.parametrize('provider,maximum', [('cloud', 4), ('cluster', 2)])
async def test_invalid_resize_does_not_change_ticket(client_and_session, provider, maximum):
    from fastapi import HTTPException
    client, Session = client_and_session
    posted = await client.post('/api/runs/run-1/instructions', json={'body': 'Use 4 GPUs'})
    async with Session() as db:
        run = await db.get(Run, 'run-1')
        run.gpu_provider, run.num_gpus = provider, maximum
        await db.commit()
        with pytest.raises(HTTPException) as error:
            await post_message('train-001', MessageBody(
                body='Use 4 GPUs', author='orchestrator', requested_gpus=4,
                run_instruction_id=posted.json()['id'], instruction_action='restart_activation',
            ), db)
        assert error.value.status_code == 422
        ticket = await db.get(Ticket, 'train-001')
        assert ticket.status == 'waiting_external'
        assert not (ticket.customization or {}).get('slurm_requested_gpus')


@pytest.mark.asyncio
async def test_resize_finds_retained_owner_after_workload_alias_finished(client_and_session):
    from datetime import datetime, timezone
    from zevo.engine.run.remote_jobs import allocation_for_resize
    _, Session = client_and_session
    async with Session() as db:
        owner = InfraInstance(
            id='owner', run_id='run-1', ticket_id='infra-001', provider='cluster',
            instance_id='123', status='ready', gpu_count=1,
            meta={'resource_request': True, 'monitor_terminal': True},
        )
        alias = InfraInstance(
            id='alias', run_id='run-1', ticket_id='train-001', provider='cluster',
            instance_id='123', status='released', gpu_count=1,
            released_at=datetime.now(timezone.utc),
            meta={'resource_request': True, 'allocation_owner_row_id': 'owner'},
        )
        db.add_all([owner, alias])
        await db.commit()
        ticket = await db.get(Ticket, 'train-001')
        assert (await allocation_for_resize(db, ticket)).id == owner.id
        other = InfraInstance(
            id='other', run_id='run-1', ticket_id='test-001', provider='cluster',
            instance_id='123', status='ready', meta={'resource_request': True},
        )
        db.add(other)
        await db.commit()
        with pytest.raises(ValueError, match='another stage'):
            await allocation_for_resize(db, ticket)
        other.released_at = datetime.now(timezone.utc)
        owner.released_at = datetime.now(timezone.utc)
        await db.commit()
        assert await allocation_for_resize(db, ticket) is None


@pytest.mark.asyncio
async def test_gpu_stage_scope_reaches_future_optimization_tickets_only(client_and_session):
    from zevo.engine.run.steering import requested_slurm_gpus
    client, Session = client_and_session
    posted = await client.post('/api/runs/run-1/instructions', json={'body': 'Use 4 GPUs for inference and training'})
    async with Session() as db:
        run = await db.get(Run, 'run-1')
        run.gpu_provider, run.num_gpus = 'cluster', 4
        await db.commit()
        await post_message('train-001', MessageBody(
            body='Use 4 GPUs', author='orchestrator', requested_gpus=4,
            gpu_stages=['train', 'inference'], run_instruction_id=posted.json()['id'],
            instruction_action='restart_activation',
        ), db)
        db.add(Ticket(id='future-infer', run_id='run-1', agent_id='inference', status='queued'))
        db.add(Ticket(id='other-data', run_id='run-1', agent_id='data', status='queued'))
        await db.commit()
        assert await requested_slurm_gpus(db, 'future-infer') == 4
        assert await requested_slurm_gpus(db, 'test-001') == 0
        assert await requested_slurm_gpus(db, 'other-data') == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("already_cancelled", [False, True])
async def test_user_directed_cancellation_records_replacement(client_and_session, monkeypatch, already_cancelled):
    from datetime import timedelta
    from zevo.api.routers.shared.tickets import CancelTicketBody, cancel_ticket
    from zevo.engine.run import remote_jobs
    from zevo.engine.run.outcome import unresolved_ticket_issues

    _, Session = client_and_session
    cleanup = []

    async def stop(db, ticket):
        cleanup.append(ticket.id)
        return {"ok": True}

    monkeypatch.setattr(remote_jobs, "cancel_ticket_remote_job", stop)
    async with Session() as db:
        old = await db.get(Ticket, "train-001")
        old.iteration = 6
        if already_cancelled:
            old.status = "cancelled"
        new = Ticket(id="train-new", run_id=old.run_id, agent_id="train",
                     status="queued", iteration=7, payload={},
                     created_at=old.created_at + timedelta(seconds=1))
        db.add(new)
        db.add(RunInstruction(id="change-plan", run_id=old.run_id, body="Cancel this train and revise Data", status="scheduled"))
        await db.commit()
        body = CancelTicketBody(instruction_id="change-plan", replacement_ticket_id=new.id,
                                reason="User requested a revised training plan")
        result = await cancel_ticket(old.id, db=db, body=body)
        assert result["status"] == "cancelled"
        run = await db.get(Run, old.run_id)
        assert run.lifecycle["cancelled_ticket_replacements"][old.id]["instruction_id"] == "change-plan"
        assert unresolved_ticket_issues([old, new], run=run)
        new.status = "succeeded"
        assert unresolved_ticket_issues([old, new], run=run) == []
        assert old.status == "cancelled"
        assert cleanup == [old.id]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["other_run", "other_stage", "other_lane", "self", "missing_instruction", "declined", "failed_original"])
async def test_cancel_rejects_invalid_replacement(client_and_session, monkeypatch, case):
    from datetime import timedelta
    from fastapi import HTTPException
    from zevo.api.routers.shared.tickets import CancelTicketBody, cancel_ticket
    from zevo.engine.run import remote_jobs

    _, Session = client_and_session
    async def unexpected_stop(*args, **kwargs):
        pytest.fail("Invalid replacement must not cancel remote work")
    monkeypatch.setattr(remote_jobs, "cancel_ticket_remote_job", unexpected_stop)
    async with Session() as db:
        old = await db.get(Ticket, "train-001")
        new = Ticket(id="train-new", run_id=old.run_id, agent_id="train", status="succeeded",
                     iteration=1, payload={}, created_at=old.created_at + timedelta(seconds=1))
        if case == "other_run":
            db.add(Run(id="other", task_name="task", status="running", metric="accuracy"))
            new.run_id = "other"
        if case == "other_stage": new.agent_id = "data"
        if case == "other_lane": new.lane = "held_out_test"
        if case == "failed_original": old.status = "failed"
        db.add(new)
        db.add(RunInstruction(id="change-plan", run_id=old.run_id, body="Revise the plan",
                              status="declined" if case == "declined" else "scheduled"))
        await db.commit()
        original_status = old.status
        with pytest.raises(HTTPException) as error:
            await cancel_ticket(old.id, db=db, body=CancelTicketBody(
                instruction_id="missing" if case == "missing_instruction" else "change-plan",
                replacement_ticket_id=old.id if case == "self" else new.id, reason="Revise plan"))
        assert error.value.status_code in {409, 422}
        assert old.status == original_status
        assert not (await db.get(Run, old.run_id)).lifecycle.get("cancelled_ticket_replacements")


@pytest.mark.asyncio
async def test_replaced_cancellation_allows_clean_final_run_status(client_and_session):
    from datetime import timedelta
    from zevo.api.routers.shared.runs import PatchRunRequest, patch_run

    _, Session = client_and_session
    async with Session() as db:
        run = await db.get(Run, "run-1")
        run.summary = "Champion saved after the revised plan completed."
        run.registry_version_tag = "M-run-1"
        run.iterations_completed = 7
        run.lifecycle = {"cancelled_ticket_replacements": {
            "train-001": {"instruction_id": "change-plan", "replacement_ticket_id": "train-new",
                          "reason": "User requested the revised plan"}}}
        old = await db.get(Ticket, "train-001")
        old.status = "cancelled"
        old.iteration = 6
        (await db.get(Ticket, "test-001")).status = "succeeded"
        db.add(Ticket(id="train-new", run_id=run.id, agent_id="train", iteration=7,
                      status="succeeded", payload={}, created_at=old.created_at + timedelta(seconds=1)))
        db.add(Ticket(id="registry-new", run_id=run.id, agent_id="registry",
                      status="succeeded", payload={}))
        await db.commit()
        result = await patch_run(run.id, PatchRunRequest(status="success"), db)
        assert result.status == "success"
        assert run.halted_reason == ""
        assert run.lifecycle["terminal_outcome"]["issues"] == []
        assert old.status == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [None, {}])
async def test_plain_cancel_accepts_empty_request_body(client_and_session, monkeypatch, payload):
    from zevo.api.routers.shared import tickets as tickets_router
    from zevo.engine.run import remote_jobs

    client, Session = client_and_session
    client._transport.app.include_router(tickets_router.router, prefix="/api")
    async def stop(db, ticket):
        return {"ok": True}
    monkeypatch.setattr(remote_jobs, "cancel_ticket_remote_job", stop)
    response = await client.post("/api/tickets/train-001/cancel", json=payload)
    assert response.status_code == 200, response.text
    async with Session() as db:
        assert (await db.get(Ticket, "train-001")).status == "cancelled"
        assert not (await db.get(Run, "run-1")).lifecycle.get("cancelled_ticket_replacements")


@pytest.mark.parametrize("payload", [{"instruction_id": "i"}, {"replacement_ticket_id": "t", "reason": "r"}, {"instruction_id": "i", "replacement_ticket_id": "t", "reason": " "}])
def test_cancel_rejects_partial_replacement_body(payload):
    from pydantic import ValidationError
    from zevo.api.routers.shared.tickets import CancelTicketBody
    with pytest.raises(ValidationError):
        CancelTicketBody.model_validate(payload)
