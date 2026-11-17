"""Data recovery must not consume an iteration before any Train is created."""
import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.routers.shared.tickets import (
    _guard_data_revision_before_train, _optimization_data_tickets,
)
from zevo.db.models import Base, Run, Ticket


def ticket(tid, agent, iteration, status, run_id="r"):
    return Ticket(id=tid, run_id=run_id, agent_id=agent, iteration=iteration,
                  status=status, lane="optimization", input_format="typed",
                  payload={"operation": "prepare_run_data" if agent == "data" else "train"})


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        session.add_all([Run(id="r", metric="accuracy"), Run(id="other", metric="accuracy")])
        session.add_all([ticket("base", "data", 0, "succeeded"),
                         ticket("previous-train", "train", 3, "succeeded"),
                         ticket("partial-data", "data", 4, "succeeded")])
        await session.commit()
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_partial_data_can_be_replaced_in_same_untrained_iteration(db):
    # Reproduces 7d9dfe58: succeeded base-only Data at iteration 4, Train only at 3.
    db.add(ticket("unrelated-train", "train", 4, "succeeded", run_id="other"))
    await db.commit()
    await _guard_data_revision_before_train(db, "r", 4, await _optimization_data_tickets(db, "r"))
    db.add(ticket("replacement-data", "data", 4, "succeeded"))
    await db.commit()
    rows = await _optimization_data_tickets(db, "r")
    assert {row.id for row in rows if row.iteration == 4} == {"partial-data", "replacement-data"}
    await _guard_data_revision_before_train(db, "r", 4, rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["queued", "running", "repairing", "waiting_external", "awaiting_input"])
async def test_active_data_cannot_be_replaced_concurrently(db, status):
    db.add(ticket("active-data", "data", 4, status))
    await db.commit()
    with pytest.raises(HTTPException) as exc:
        await _guard_data_revision_before_train(db, "r", 4, await _optimization_data_tickets(db, "r"))
    assert exc.value.status_code == 409
    assert exc.value.detail["ticket_id"] == "active-data"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["queued", "running", "succeeded", "failed", "cancelled"])
async def test_train_freezes_iteration_data_even_if_not_running(db, status):
    db.add(ticket("bound-train", "train", 4, status))
    await db.commit()
    with pytest.raises(HTTPException) as exc:
        await _guard_data_revision_before_train(db, "r", 4, await _optimization_data_tickets(db, "r"))
    assert exc.value.status_code == 409
    assert exc.value.detail["ticket_id"] == "bound-train"


async def failed_input_train(db, tid="blocked"):
    from zevo.db import HeartbeatResult
    t = ticket(tid, "train", 4, "failed")
    t.repair_route = "orchestrator"
    db.add(t)
    db.add(HeartbeatResult(ticket_id=tid, agent_id="train", status="failed",
        output={"status": "failed", "failure_recovery": {"scope": "upstream_input"}}))
    await db.commit()
    return t


@pytest.mark.asyncio
async def test_preflight_input_failure_allows_revision_without_mutating_old_train(db):
    t = await failed_input_train(db)
    t.inputs = {"training_dataset": {"source_ticket_id": "partial-data"}}
    await db.commit()
    await _guard_data_revision_before_train(db, "r", 4, await _optimization_data_tickets(db, "r"))
    db.add(ticket("repair-data", "data", 4, "succeeded"))
    await db.commit()
    assert t.status == "failed"
    assert t.inputs["training_dataset"]["source_ticket_id"] == "partial-data"
    await _guard_data_revision_before_train(db, "r", 4, await _optimization_data_tickets(db, "r"))


@pytest.mark.asyncio
@pytest.mark.parametrize("evidence", ["allocation", "allocation_without_job", "workload", "step", "checkpoint", "result_checkpoint", "continuation", "live_heartbeat", "later_train", "earlier_train", "pending_retry"])
async def test_input_failure_cannot_hide_execution_or_another_train(db, evidence):
    from datetime import datetime, timezone, timedelta
    from zevo.db import (HeartbeatRun, InfraInstance, WorkloadExecution,
                         ExecutionEvent, WorkProduct, HeartbeatResult, AgentWakeupRequest)
    await failed_input_train(db)
    hb = HeartbeatRun(id="hb", ticket_id="blocked", agent_id="train", driver="test",
                      model="test", finished_at=datetime.now(timezone.utc))
    db.add(hb)
    if evidence.startswith("allocation"):
        db.add(InfraInstance(ticket_id="blocked", run_id="r", provider="cluster",
            instance_id="123" if evidence == "allocation" else "", status="failed"))
    elif evidence == "workload":
        db.add(WorkloadExecution(ticket_id="blocked", heartbeat_id="hb", runtime_key="train", log_path="log"))
    elif evidence == "step":
        db.add(ExecutionEvent(ticket_id="blocked", heartbeat_id="hb", attempt_id="a",
                              phase="train", event_type="progress", current_step=1))
    elif evidence == "checkpoint":
        db.add(WorkProduct(ticket_id="blocked", role="checkpoint", path="model"))
    elif evidence in {"result_checkpoint", "continuation"}:
        # Historical output must block even when the latest result is preflight failure.
        db.add(HeartbeatResult(ticket_id="blocked", status="failed",
            created_at=datetime.now(timezone.utc)-timedelta(hours=1),
            output={"checkpoint_path": "model"} if evidence == "result_checkpoint" else {"continuation": {"checkpoint": "model"}}))
    elif evidence == "pending_retry":
        db.add(AgentWakeupRequest(ticket_id="blocked", agent_id="train", source="retry", status="queued"))
    elif evidence == "live_heartbeat":
        hb.finished_at = None
    else:
        other = ticket("other-attempt", "train", 4, "queued" if evidence == "later_train" else "succeeded")
        if evidence == "earlier_train":
            other.created_at = datetime.now(timezone.utc)-timedelta(hours=1)
        db.add(other)
    await db.commit()
    with pytest.raises(HTTPException) as exc:
        await _guard_data_revision_before_train(db, "r", 4, await _optimization_data_tickets(db, "r"))
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_latest_failure_must_explicitly_request_upstream_input_repair(db):
    from zevo.db import HeartbeatResult
    await failed_input_train(db)
    db.add(HeartbeatResult(ticket_id="blocked", status="failed",
        output={"status": "failed", "failure_recovery": {"scope": "local_execution"}}))
    await db.commit()
    with pytest.raises(HTTPException):
        await _guard_data_revision_before_train(db, "r", 4, await _optimization_data_tickets(db, "r"))


@pytest.mark.asyncio
async def test_preflight_exception_still_blocks_concurrent_data(db):
    await failed_input_train(db)
    db.add(ticket("repair-in-progress", "data", 4, "running"))
    await db.commit()
    with pytest.raises(HTTPException) as exc:
        await _guard_data_revision_before_train(db, "r", 4, await _optimization_data_tickets(db, "r"))
    assert exc.value.detail["ticket_id"] == "repair-in-progress"
