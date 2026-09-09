"""Run-local Agent memory isolation, applicability, and lifecycle."""
from __future__ import annotations

import json

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from zevo.contracts.memory import MemoryDetail, MemoryUpdate
from zevo.db.models import AgentMemoryEntry, Base, Run, Ticket, TicketNotice
from zevo.engine.agent.memory import load_memory_context, persist_memory_updates


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        yield db
    await engine.dispose()


def _run(run_id: str, *, status: str = "running") -> Run:
    return Run(
        id=run_id,
        task_name="memory-test",
        agent_objective="test run-scoped continuity",
        metric="accuracy",
        status=status,
        gpu_provider="instance",
        generation_backend="vllm",
        holdout={
            "validation_set": "/tmp/validation.csv",
            "test_set": "/private/test.csv",
            "test_public": "/private/test_public.csv",
            "test_sample_submission": "/private/sample.csv",
        },
    )


def _ticket(
    ticket_id: str,
    run_id: str,
    agent_id: str = "train",
    *,
    lane: str = "optimization",
) -> Ticket:
    return Ticket(
        id=ticket_id,
        run_id=run_id,
        agent_id=agent_id,
        status="running",
        lane=lane,
        iteration=1,
        payload={
            "operation": "train" if agent_id == "train" else "run_inference",
            "base_model": "Qwen/Qwen3-0.6B",
            "training_method_pin": "lora_sft" if agent_id == "train" else "",
        },
        customization={},
        inputs={
            "inference_config": {
                "artifact_role": "inference_config",
                "source_ticket_id": "infer-config-a",
            },
        },
    )


def _update(
    key: str,
    summary: str = "Observed and verified during execution.",
    *,
    visibility: str = "agent_local",
    details: dict | None = None,
    kind: str = "runtime_finding",
) -> MemoryUpdate:
    wire_details = [
        MemoryDetail(
            key=str(detail_key),
            value=(
                detail_value
                if isinstance(detail_value, str)
                else json.dumps(detail_value, sort_keys=True)
            ),
        )
        for detail_key, detail_value in (details or {}).items()
    ]
    return MemoryUpdate(
        kind=kind,
        key=key,
        summary=summary,
        details=wire_details,
        visibility=visibility,
    )


def test_memory_update_is_strict_and_normalized() -> None:
    update = _update("  CUDA OOM ", summary="  reduce   batch pressure  ")
    assert update.key == "cuda_oom"
    assert update.summary == "reduce batch pressure"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        MemoryUpdate(
            kind="pitfall",
            key="bad",
            summary="bad",
            details=[],
            visibility="agent_local",
            unknown=True,
        )
    with pytest.raises(ValidationError, match="6000"):
        _update("too_large", details={"raw": "x" * 7000})
    with pytest.raises(ValidationError, match="list"):
        MemoryUpdate(
            kind="pitfall",
            key="old_open_map",
            summary="Old open dictionaries are no longer a valid wire shape.",
            details={"path": "/tmp/model"},
            visibility="agent_local",
        )
    with pytest.raises(ValidationError, match="duplicate keys"):
        MemoryUpdate(
            kind="pitfall",
            key="duplicate_details",
            summary="Duplicate detail keys would make persistence ambiguous.",
            details=[
                {"key": "version", "value": "one"},
                {"key": "version", "value": "two"},
            ],
            visibility="agent_local",
        )


@pytest.mark.asyncio
async def test_memory_is_isolated_by_run_agent_and_lane(session: AsyncSession) -> None:
    run_a = _run("run-a")
    run_b = _run("run-b")
    source = _ticket("train-a", "run-a")
    other_run = _ticket("train-b", "run-b")
    other_agent = _ticket("infer-a", "run-a", "inference")
    held_out = _ticket("held-a", "run-a", lane="held_out_test")
    orchestrator = _ticket("orch-a", "run-a", "orchestrator")
    session.add_all([run_a, run_b, source, other_run, other_agent, held_out, orchestrator])
    await session.flush()

    stored, rejected = await persist_memory_updates(
        session,
        run=run_a,
        ticket=source,
        updates=[_update(
            "allocator", visibility="shared_candidate", kind="verified_fact",
            details={"probe": "idle"},
        )],
    )
    assert (stored, rejected) == (1, [])
    await session.flush()

    own = await load_memory_context(session, run=run_a, ticket=source)
    assert [entry.key for entry in own.entries] == ["allocator"]
    assert own.entries[0].details == {"probe": "idle"}
    assert (await load_memory_context(
        session, run=run_b, ticket=other_run,
    )).entries == []
    assert (await load_memory_context(
        session, run=run_a, ticket=other_agent,
    )).entries == []
    assert (await load_memory_context(
        session, run=run_a, ticket=held_out,
    )).entries == []
    shared = await load_memory_context(session, run=run_a, ticket=orchestrator)
    assert [entry.key for entry in shared.entries] == ["allocator"]


@pytest.mark.asyncio
async def test_local_memory_is_not_shared_and_contract_drift_filters_it(
    session: AsyncSession,
) -> None:
    run = _run("run-a")
    train = _ticket("train-a", "run-a")
    orchestrator = _ticket("orch-a", "run-a", "orchestrator")
    session.add_all([run, train, orchestrator])
    await session.flush()
    await persist_memory_updates(
        session, run=run, ticket=train, updates=[_update("local_only")],
    )
    await session.flush()
    assert (await load_memory_context(
        session, run=run, ticket=orchestrator,
    )).entries == []

    # A different model lineage is genuinely incompatible. Transient Ticket
    # bindings and method choices are deliberately not applicability keys.
    train.payload = {**train.payload, "base_model": "Other/Model"}
    assert (await load_memory_context(session, run=run, ticket=train)).entries == []


@pytest.mark.asyncio
async def test_specialist_memory_survives_iteration_recipe_and_binding_changes(
    session: AsyncSession,
) -> None:
    """Run-local continuity follows the Specialist, not transient Ticket JSON."""
    run = _run("run-a")
    first = _ticket("train-a", "run-a")
    session.add_all([run, first])
    await session.flush()
    await persist_memory_updates(
        session, run=run, ticket=first,
        updates=[_update("vllm_version_flag", "Use the installed-version flag.")],
    )
    await session.flush()

    second = _ticket("train-b", "run-a")
    second.iteration = 2
    second.payload = {
        **first.payload,
        "training_method_pin": "",
        "configuration_suggestions": {"training_method": "full_sft"},
    }
    second.inputs = {
        "inference_config": {
            "artifact_role": "inference_config",
            "source_ticket_id": "infer-config-b",
        },
    }
    session.add(second)
    await session.flush()

    context = await load_memory_context(session, run=run, ticket=second)
    assert [entry.key for entry in context.entries] == ["vllm_version_flag"]


@pytest.mark.asyncio
async def test_matching_update_supersedes_without_deleting_history(
    session: AsyncSession,
) -> None:
    run = _run("run-a")
    train = _ticket("train-a", "run-a")
    session.add_all([run, train])
    await session.flush()
    await persist_memory_updates(
        session, run=run, ticket=train,
        updates=[_update("batch_limit", "batch two was stable")],
    )
    await session.flush()
    await persist_memory_updates(
        session, run=run, ticket=train,
        updates=[_update("batch_limit", "batch four was stable")],
    )
    await session.flush()

    rows = (await session.execute(
        select(AgentMemoryEntry).order_by(AgentMemoryEntry.created_at)
    )).scalars().all()
    assert len(rows) == 2
    assert {row.status for row in rows} == {"active", "superseded"}
    active = next(row for row in rows if row.status == "active")
    old = next(row for row in rows if row.status == "superseded")
    assert active.supersedes_id == old.id
    context = await load_memory_context(session, run=run, ticket=train)
    assert [entry.summary for entry in context.entries] == ["batch four was stable"]


@pytest.mark.asyncio
async def test_rejects_secrets_test_assets_and_duplicate_result_keys(
    session: AsyncSession,
) -> None:
    run = _run("run-a")
    train = _ticket("train-a", "run-a")
    session.add_all([run, train])
    await session.flush()
    stored, rejected = await persist_memory_updates(
        session,
        run=run,
        ticket=train,
        updates=[
            _update("validation_ok", details={"path": "/tmp/validation.csv"}),
            _update("secret", details={"api_key": "do-not-store"}),
            _update("test_leak", details={"path": "/private/test.csv"}),
            _update("validation_ok", "duplicate"),
        ],
    )
    assert stored == 1
    assert len(rejected) == 3
    await session.flush()
    notices = (await session.execute(select(TicketNotice))).scalars().all()
    assert len(notices) == 3


@pytest.mark.asyncio
async def test_terminal_run_is_audit_only(session: AsyncSession) -> None:
    run = _run("run-a")
    train = _ticket("train-a", "run-a")
    session.add_all([run, train])
    await session.flush()
    await persist_memory_updates(
        session, run=run, ticket=train, updates=[_update("before_finish")],
    )
    await session.flush()
    run.status = "success"
    assert (await load_memory_context(session, run=run, ticket=train)).entries == []
    stored, rejected = await persist_memory_updates(
        session, run=run, ticket=train, updates=[_update("after_finish")],
    )
    assert (stored, rejected) == (0, [])
