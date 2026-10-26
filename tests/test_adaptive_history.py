import pytest
from zevo.engine.run.adaptive_history import legacy_decision_context

RID = "a" * 32
REQUESTS = {RID: {"status": "decided", "ticket_id": "train-1", "step": 552,
                  "decision": {"action": "continue", "next_step": 1104}}}
COMMAND = 'curl -X POST "$API/api/runs/$RUN/adaptive-training/train-1/decision" --data-binary @body.json'
OUTPUT = '{"status":"decided","decision":{"request_id":"' + RID + '","action":"continue","next_step":1104,"reason":"truncated'


def test_truncated_accepted_response_restores_exact_request():
    context = legacy_decision_context(COMMAND, OUTPUT, REQUESTS)
    assert context == {"request_id": RID, "train_ticket_id": "train-1", "step": 552,
                       "action": "continue", "next_step": 1104}


@pytest.mark.parametrize("command,output,requests,error", [
    (COMMAND.replace("POST", "GET"), OUTPUT, REQUESTS, False),
    (COMMAND.replace("train-1", "train-2"), OUTPUT, REQUESTS, False),
    (COMMAND, 'HTTP=422 {"detail":"reason too long"}', REQUESTS, False),
    (COMMAND, OUTPUT.replace("1104", "1656"), REQUESTS, False),
    (COMMAND, OUTPUT, REQUESTS, True),
    (COMMAND, OUTPUT, {}, False),
    (COMMAND, OUTPUT, {RID: {**REQUESTS[RID], "status": "pending"}}, False),
])
def test_unaccepted_or_mismatched_records_cannot_become_decisions(command, output, requests, error):
    assert legacy_decision_context(command, output, requests, is_error=error) is None


@pytest.mark.asyncio
async def test_api_projection_pairs_calls_with_results_without_mutating_history():
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from zevo.db import Base, Run, Ticket, HeartbeatRun, TranscriptEvent
    from zevo.api.routers.ui.heartbeats import _mid_training_contexts
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        run = Run(id="run", task_name="task", metric="accuracy", status="running",
                  lifecycle={"adaptive_training_requests": REQUESTS})
        db.add(run)
        db.add(Ticket(id="orch", run_id="run", agent_id="orchestrator", status="succeeded"))
        first = HeartbeatRun(id="first", ticket_id="orch", agent_id="orchestrator", driver="stub", model="stub")
        other = HeartbeatRun(id="other", ticket_id="orch", agent_id="orchestrator", driver="stub", model="stub")
        db.add_all([first, other])
        await db.commit()
        db.add_all([
            TranscriptEvent(heartbeat_id="first", seq=1, type="tool_call", payload={"tool_use_id":"call", "input":{"command":COMMAND}}),
            TranscriptEvent(heartbeat_id="first", seq=2, type="tool_result", payload={"tool_use_id":"call", "output":OUTPUT, "is_error":False}),
            TranscriptEvent(heartbeat_id="other", seq=1, type="tool_result", payload={"tool_use_id":"call", "output":OUTPUT}),
        ])
        await db.commit()
        result = await _mid_training_contexts(db, [first, other])
        assert list(result) == ["first"]
        assert result["first"]["step"] == 552
        assert run.lifecycle == {"adaptive_training_requests": REQUESTS}
        assert not first.resolved_config
    await engine.dispose()
