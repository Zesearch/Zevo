"""Canonical ArtifactBinding resolution tests."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
import pytest_asyncio
import sqlalchemy.event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from zevo.db.models import Base, Run, Ticket, WorkProduct
from zevo.engine.run.scheduler.bindings import resolve_input_bindings


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        yield session
    await engine.dispose()


async def _seed(session: AsyncSession) -> None:
    session.add(Run(metric="accuracy", id="r1", task_name="t", agent_objective="t", status="running"))
    session.add(Ticket(
        id="data-001", run_id="r1", agent_id="data", status="succeeded",
        input_format="typed", lane="optimization", iteration=0,
        payload={}, customization={}, inputs={},
    ))
    session.add(Ticket(
        id="data-002", run_id="r1", agent_id="data", status="failed",
        input_format="typed", lane="optimization", iteration=0,
        payload={}, customization={}, inputs={},
    ))
    session.add(WorkProduct(
        ticket_id="data-001", role="training_dataset",
        path="/data/data-001/dataset.jsonl", meta={},
        created_at=datetime.now(timezone.utc),
    ))
    session.add(WorkProduct(
        ticket_id="data-001", role="log", path="/data/data-001/data.log", meta={},
    ))
    await session.commit()


def _binding(ticket: str, role: str) -> dict:
    return {
        "artifact_role": role, "source_ticket_id": ticket,
        "work_product_id": "", "path": "",
    }


@pytest.mark.asyncio
async def test_resolves_exact_ticket_and_role(session: AsyncSession) -> None:
    await _seed(session)
    out = await resolve_input_bindings(
        session, run_id="r1",
        inputs={"training_dataset": _binding("data-001", "training_dataset")},
    )
    assert out["training_dataset"]["path"] == "/data/data-001/dataset.jsonl"
    assert out["training_dataset"]["work_product_id"]


@pytest.mark.asyncio
async def test_exact_work_product_selects_intermediate_but_default_selects_final(
    session: AsyncSession,
) -> None:
    await _seed(session)
    session.add(Ticket(
        id="train-001", run_id="r1", agent_id="train", status="succeeded",
        input_format="typed", lane="optimization", iteration=1,
        payload={}, customization={}, inputs={},
    ))
    intermediate = WorkProduct(
        id="checkpoint-intermediate", ticket_id="train-001", role="checkpoint",
        path="/remote/train-001/intermediate/epoch-1",
        meta={"checkpoint_kind": "intermediate", "epoch": 1.0},
    )
    final = WorkProduct(
        id="checkpoint-final", ticket_id="train-001", role="checkpoint",
        path="/remote/train-001/model",
        meta={"checkpoint_kind": "final"},
    )
    session.add_all([final, intermediate])
    await session.commit()

    default = await resolve_input_bindings(
        session, run_id="r1",
        inputs={"parent_checkpoint": _binding("train-001", "checkpoint")},
    )
    assert default["parent_checkpoint"]["work_product_id"] == "checkpoint-final"

    exact_binding = _binding("train-001", "checkpoint")
    exact_binding["work_product_id"] = "checkpoint-intermediate"
    exact = await resolve_input_bindings(
        session, run_id="r1", inputs={"parent_checkpoint": exact_binding},
    )
    assert exact["parent_checkpoint"]["path"].endswith("epoch-1")

    wrong_role = _binding("train-001", "checkpoint")
    wrong_role["work_product_id"] = "checkpoint-final"
    wrong_role["artifact_role"] = "log"
    with pytest.raises(ValueError, match="does not match"):
        await resolve_input_bindings(
            session, run_id="r1", inputs={"parent_checkpoint": wrong_role},
        )


@pytest.mark.asyncio
async def test_rejects_non_usable_source(session: AsyncSession) -> None:
    await _seed(session)
    with pytest.raises(ValueError, match="not succeeded/degraded"):
        await resolve_input_bindings(
            session, run_id="r1",
            inputs={"training_dataset": _binding("data-002", "training_dataset")},
        )


@pytest.mark.asyncio
async def test_never_substitutes_another_role(session: AsyncSession) -> None:
    await _seed(session)
    with pytest.raises(ValueError, match="artifact role 'predictions'"):
        await resolve_input_bindings(
            session, run_id="r1",
            inputs={"predictions": _binding("data-001", "predictions")},
        )


@pytest.mark.asyncio
async def test_literal_path_needs_no_database_query(session: AsyncSession) -> None:
    await _seed(session)
    query_count = 0

    @sqlalchemy.event.listens_for(session.bind.sync_engine, "before_cursor_execute")
    def _count(conn, cursor, statement, *args, **kwargs):
        nonlocal query_count
        if "work_products" in statement.lower() or "from tickets" in statement.lower():
            query_count += 1

    literal = {
        "artifact_role": "predictions", "source_ticket_id": "",
        "work_product_id": "", "path": "/uploads/predictions.csv",
    }
    out = await resolve_input_bindings(session, run_id="r1", inputs={"predictions": literal})
    assert out["predictions"] == literal
    assert query_count == 0
