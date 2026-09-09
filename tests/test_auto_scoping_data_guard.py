"""Auto mode's `scope_problem` Ticket must not count as an iteration-0 Data Ticket.

Live failure this pins: after scoping succeeded and the supervisor kicked off
the normal pipeline, the orchestrator's `prepare_run_data` work order was
refused with 409 "iteration 0 Data is already settled or active" because the
duplicate-Data guard counted the succeeded `scope-<run8>-001` row (a
`data`-agent Ticket on the optimization lane) as an existing iteration-0 Data
Ticket. Baseline inference then deadlocked on the missing data profile and the
run failed. The guard must see through scoping tickets; a genuine duplicate
iteration-0 Data Ticket must still be refused.
"""
from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.routers.shared.tickets import _optimization_data_tickets
from zevo.db.models import Base, Run, Ticket
from zevo.engine.run.scoping import is_scoping_ticket


async def _session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


_RUN = dict(
    task_name="auto-t", task_objective="o", agent_objective="ao",
    metric="accuracy", metric_direction="max",
    validation_metric="accuracy", validation_metric_direction="max",
    status="running", mode="auto",
)


def _ticket(tid: str, operation: str, status: str = "succeeded") -> Ticket:
    return Ticket(
        id=tid, run_id="r1", agent_id="data", status=status,
        input_format="typed", lane="optimization", iteration=0,
        payload={"operation": operation}, summary="", error_message="",
    )


@pytest.mark.asyncio
async def test_succeeded_scoping_ticket_does_not_block_iteration0_data() -> None:
    Session = await _session()
    async with Session() as db:
        db.add(Run(id="r1", **_RUN))
        db.add(_ticket("scope-r1000000-001", "scope_problem"))
        await db.commit()
        # Only a scoping ticket exists -> iteration 0 prepare_run_data must be
        # allowed, i.e. the guard sees NO existing Data tickets.
        assert await _optimization_data_tickets(db, "r1") == []


@pytest.mark.asyncio
async def test_real_data_ticket_still_counts_and_scoping_is_excluded() -> None:
    Session = await _session()
    async with Session() as db:
        db.add(Run(id="r1", **_RUN))
        db.add(_ticket("scope-r1000000-001", "scope_problem"))
        db.add(_ticket("data-r1000000-001", "prepare_run_data"))
        await db.commit()
        live = await _optimization_data_tickets(db, "r1")
        # The genuine Data ticket is counted (so a second iteration-0 one would
        # still be refused), the scoping row is not.
        assert [t.id for t in live] == ["data-r1000000-001"]
        assert not any(is_scoping_ticket(t) for t in live)


@pytest.mark.asyncio
async def test_terminal_scoping_and_data_tickets_are_ignored_alike() -> None:
    Session = await _session()
    async with Session() as db:
        db.add(Run(id="r1", **_RUN))
        db.add(_ticket("scope-r1000000-001", "scope_problem", status="failed"))
        db.add(_ticket("data-r1000000-001", "prepare_run_data", status="cancelled"))
        await db.commit()
        assert await _optimization_data_tickets(db, "r1") == []
