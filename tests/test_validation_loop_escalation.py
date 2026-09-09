"""A `queued` ticket whose wakeup fails typed-payload validation must NOT loop.

Typed-payload validation (validate_stored_payload, and the per-agent *TaskInput
builders that enforce the same contract) runs BEFORE the agent executes. A
ticket that fails it used to stay `queued`: the wakeup was marked failed but the
ticket was untouched, and the alarm-clock cron re-enqueued the identical payload
every ~5 min forever -- no escalation, no re-plan, and on a rented GPU an
unbounded idle burn (hit live: an inference ticket's payload failed
InferenceTaskInput validation every wakeup).

These tests prove the fix:
  * the runner escalates each such wakeup to the orchestrator and, after N
    consecutive failures, fails the ticket terminally instead of looping;
  * run_ticket actually routes a build-time ValidationError through that path;
  * the reconciler is a safety net for a stuck-queued ticket the runner's
    terminal step never reached.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import zevo.engine.run.runner as runner_mod
from zevo.contracts.inference import InferenceTaskInput
from zevo.db.models import AgentWakeupRequest, Base, Run, Ticket
from zevo.engine.run.failure_policy import (
    MAX_PAYLOAD_VALIDATION_ATTEMPTS,
    PAYLOAD_VALIDATION_FAILURE_SIGNATURE,
)
from zevo.engine.run.runner import (
    StoredPayloadValidationError,
    _escalate_stored_payload_validation_failure,
    run_ticket,
)
from zevo.engine.run.scheduler.reconciler import (
    _fail_validation_looping_tickets,
    reconcile_runs_and_tickets,
)


async def _session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


_RUN = dict(
    task_name="t", task_objective="o", agent_objective="ao",
    metric="accuracy", metric_direction="max",
    validation_metric="accuracy", validation_metric_direction="max",
    status="running",
)


def _a_validation_error() -> ValidationError:
    """A real pydantic ValidationError, exactly like a bad stored payload."""
    try:
        InferenceTaskInput()  # missing every required field
    except ValidationError as exc:
        return exc
    raise AssertionError("InferenceTaskInput() should have failed validation")


async def _seed_run_with_child(
    db, *, child_status: str = "queued", with_supervisor: bool = True,
) -> None:
    db.add(Run(
        id="r1",
        supervisor_ticket_id="orch-1" if with_supervisor else "",
        **_RUN,
    ))
    if with_supervisor:
        db.add(Ticket(
            id="orch-1", run_id="r1", agent_id="orchestrator",
            status="queued", payload={}, summary="", error_message="",
        ))
    db.add(Ticket(
        id="infer-1", run_id="r1", agent_id="inference", status=child_status,
        payload={"garbage": True}, summary="", error_message="",
    ))
    await db.commit()


async def _add_failed_validation_wakeups(db, ticket_id: str, n: int) -> None:
    for i in range(n):
        db.add(AgentWakeupRequest(
            id=f"w-{ticket_id}-{i}", agent_id="inference", ticket_id=ticket_id,
            source="cron", status="failed",
            reason=f"boom | {PAYLOAD_VALIDATION_FAILURE_SIGNATURE}: bad payload",
        ))
    await db.commit()


async def _orch_wakeups(db) -> list[AgentWakeupRequest]:
    return list((await db.execute(
        select(AgentWakeupRequest).where(
            AgentWakeupRequest.agent_id == "orchestrator",
            AgentWakeupRequest.status == "queued",
        )
    )).scalars().all())


# --------------------------------------------------------------------------
# runner escalation + bounded counter
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_failures_escalate_and_do_not_loop_terminally() -> None:
    """Below the cap: escalate to the orchestrator, keep the ticket queued
    (recoverable), and raise so the wakeup is recorded failed."""
    Session = await _session()
    async with Session() as db:
        await _seed_run_with_child(db)
        tk = await db.get(Ticket, "infer-1")
        run = await db.get(Run, "r1")

        with pytest.raises(StoredPayloadValidationError):
            await _escalate_stored_payload_validation_failure(
                db, tk, run, _a_validation_error()
            )

        tk = await db.get(Ticket, "infer-1")
        assert tk.status == "queued"  # not terminal yet -- orchestrator may fix it
        assert tk.repair_route == "orchestrator"
        assert PAYLOAD_VALIDATION_FAILURE_SIGNATURE in tk.error_message
        # Escalated: the orchestrator was woken with the error as feedback.
        woke = await _orch_wakeups(db)
        assert len(woke) == 1
        assert woke[0].ticket_id == "orch-1"
        assert "infer-1" in woke[0].reason


@pytest.mark.asyncio
async def test_fails_terminally_after_n_consecutive_validation_failures() -> None:
    """At the cap the ticket is FAILED terminally -- not left queued to loop."""
    Session = await _session()
    async with Session() as db:
        await _seed_run_with_child(db)
        # Prior failures push this activation to the Nth (the cap).
        await _add_failed_validation_wakeups(
            db, "infer-1", MAX_PAYLOAD_VALIDATION_ATTEMPTS - 1
        )
        tk = await db.get(Ticket, "infer-1")
        run = await db.get(Run, "r1")

        with pytest.raises(StoredPayloadValidationError):
            await _escalate_stored_payload_validation_failure(
                db, tk, run, _a_validation_error()
            )

        tk = await db.get(Ticket, "infer-1")
        assert tk.status == "failed"  # terminal -- the cron can no longer re-drive it
        assert tk.repair_route == "orchestrator"
        assert len(await _orch_wakeups(db)) == 1  # still escalated


@pytest.mark.asyncio
async def test_run_ticket_routes_build_validation_error_through_escalation(
    monkeypatch, tmp_path,
) -> None:
    """The wiring: a ValidationError from building the typed input is caught by
    run_ticket, escalated, and the ticket is not left silently queued."""
    Session = await _session()
    async with Session() as db:
        await _seed_run_with_child(db)

        async def _boom(*a, **k):
            raise _a_validation_error()

        # Patch the typed-input builder to fail exactly as an invalid stored
        # payload would (InferenceTaskInput construction).
        monkeypatch.setattr(runner_mod, "_build_input", _boom)

        with pytest.raises(StoredPayloadValidationError):
            await run_ticket(
                db, ticket_id="infer-1", work_dir_root=str(tmp_path),
            )

        tk = await db.get(Ticket, "infer-1")
        # Never flipped to running (build failed before execution) and now
        # escalated rather than silently re-queued.
        assert tk.status == "queued"
        assert tk.repair_route == "orchestrator"
        assert PAYLOAD_VALIDATION_FAILURE_SIGNATURE in tk.error_message
        assert len(await _orch_wakeups(db)) == 1


@pytest.mark.asyncio
async def test_no_supervisor_still_bounds_without_escalation() -> None:
    """A run with no supervisor has nowhere to escalate, but the terminal cap
    still breaks the loop."""
    Session = await _session()
    async with Session() as db:
        await _seed_run_with_child(db, with_supervisor=False)
        await _add_failed_validation_wakeups(
            db, "infer-1", MAX_PAYLOAD_VALIDATION_ATTEMPTS - 1
        )
        tk = await db.get(Ticket, "infer-1")
        run = await db.get(Run, "r1")

        with pytest.raises(StoredPayloadValidationError):
            await _escalate_stored_payload_validation_failure(
                db, tk, run, _a_validation_error()
            )

        tk = await db.get(Ticket, "infer-1")
        assert tk.status == "failed"
        assert await _orch_wakeups(db) == []  # nobody to wake, but no loop either


# --------------------------------------------------------------------------
# reconciler safety net
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reconciler_fails_ticket_stuck_with_repeated_validation_failures() -> None:
    Session = await _session()
    async with Session() as db:
        await _seed_run_with_child(db)
        await _add_failed_validation_wakeups(
            db, "infer-1", MAX_PAYLOAD_VALIDATION_ATTEMPTS
        )

        n = await _fail_validation_looping_tickets(db)
        assert n == 1

        tk = await db.get(Ticket, "infer-1")
        assert tk.status == "failed"
        assert tk.repair_route == "orchestrator"
        woke = await _orch_wakeups(db)
        assert len(woke) == 1
        assert "infer-1" in woke[0].reason


@pytest.mark.asyncio
async def test_reconciler_leaves_ticket_below_the_cap_alone() -> None:
    Session = await _session()
    async with Session() as db:
        await _seed_run_with_child(db)
        await _add_failed_validation_wakeups(
            db, "infer-1", MAX_PAYLOAD_VALIDATION_ATTEMPTS - 1
        )

        assert await _fail_validation_looping_tickets(db) == 0
        tk = await db.get(Ticket, "infer-1")
        assert tk.status == "queued"  # not enough failures yet
        assert await _orch_wakeups(db) == []


@pytest.mark.asyncio
async def test_reconciler_skips_ticket_whose_run_is_terminal() -> None:
    Session = await _session()
    async with Session() as db:
        await _seed_run_with_child(db)
        run = await db.get(Run, "r1")
        run.status = "failed"
        await db.commit()
        await _add_failed_validation_wakeups(
            db, "infer-1", MAX_PAYLOAD_VALIDATION_ATTEMPTS
        )

        assert await _fail_validation_looping_tickets(db) == 0
        tk = await db.get(Ticket, "infer-1")
        assert tk.status == "queued"  # untouched -- the run is already over


@pytest.mark.asyncio
async def test_reconcile_entrypoint_reports_validation_failures(monkeypatch) -> None:
    """The safety net is wired into the full reconciler pass and reported."""
    Session = await _session()

    def _factory():
        return Session

    monkeypatch.setattr(
        "zevo.engine.run.scheduler.reconciler.get_session_factory", _factory
    )
    async with Session() as db:
        await _seed_run_with_child(db)
        await _add_failed_validation_wakeups(
            db, "infer-1", MAX_PAYLOAD_VALIDATION_ATTEMPTS
        )

    report = await reconcile_runs_and_tickets(stale_ticket_seconds=1)
    assert report["validation_tickets_failed"] == 1

    async with Session() as db:
        tk = await db.get(Ticket, "infer-1")
        assert tk.status == "failed"
