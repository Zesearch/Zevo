"""Tests for zevo.engine.run.retry_policy + /tickets/{id}/rerun endpoint."""
from __future__ import annotations

import pytest
import pytest_asyncio
from datetime import datetime, timezone

from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import (
    Base, AuditEvent, AgentWakeupRequest, HeartbeatRun, Run, Ticket, WorkProduct,
)
from zevo.engine.run.retry_policy import classify, failure_catalog


def test_executable_failure_catalog_is_complete_and_unambiguous() -> None:
    modes = failure_catalog()
    assert len({mode.code for mode in modes}) == len(modes)
    assert all(mode.pattern.pattern for mode in modes)
    assert all(mode.reason and mode.description and mode.recovery for mode in modes)


# ────────────────────────── classify() unit table ───────────────────────────


@pytest.mark.parametrize("err,exit_code,expected_verdict,expected_retryable", [
    # ── transient ────────────────────────────────────────────────────────────
    ("CUDA out of memory: 16 GB free, needed 24 GB", 1, "transient", True),
    ("OOM detected during forward pass",             1, "transient", True),
    ("Connection timed out after 30s",               1, "transient", True),
    ("HTTPSConnectionPool: Read timed out",          1, "transient", True),
    ("429 Too Many Requests — rate limit hit",       1, "transient", True),
    ("openai.RateLimitError: rate limit",            1, "transient", True),
    ("503 Service Unavailable — try again later",    1, "transient", True),
    ("Bad Gateway from upstream",                    1, "transient", True),
    ("ssh: connect to host 1.2.3.4 port 22: Operation timed out", 1, "transient", True),

    # ── structural (don't auto-retry) ────────────────────────────────────────
    ("ValidationError: 9 validation errors for DataResult\nstatus\n  Field required", 1, "structural", False),
    ("unresolved input binding 'training_dataset': source ticket data-016 has no work product with artifact role 'training_dataset'", 1, "structural", False),
    ("inputs unresolvable: source ticket data-016 failed", 1, "structural", False),
    ("KeyError: train ticket 'train-001' payload missing required field 'prompt_framing'", 1, "structural", False),
    ("no parseable trailing JSON line in agent output", 1, "structural", False),
    ("unsupported file type '.bin'",                    1, "structural", False),
    ("[Errno 2] No such file or directory: '/tmp/missing.csv'", 1, "structural", False),
    ("runtime experiment contract violated: inference_data_profile", 1, "structural", False),
    ("INVALID TrainRunConfig: template mismatch", 1, "structural", False),

    # ── cancelled (operator decision) ────────────────────────────────────────
    ("cancelled by user",                            1, "cancelled", True),
    ("received SIGTERM during eval",                 1, "cancelled", True),
    ("",                                            143, "cancelled", True),

    # ── unknown (last-resort: allow rerun, flag uncertainty) ────────────────
    ("RuntimeError: something nobody planned for",   1, "unknown", True),
])
def test_classify_buckets(err, exit_code, expected_verdict, expected_retryable):
    out = classify(err, exit_code)
    assert out.verdict == expected_verdict, (
        f"err={err!r} classified as {out.verdict!r} ({out.reason}); expected {expected_verdict}"
    )
    assert out.retryable == expected_retryable


def test_classify_succeeded_ticket_is_not_retryable() -> None:
    out = classify("", 0)
    assert out.verdict == "unknown"
    assert out.retryable is False


@pytest.mark.parametrize("error,code", [
    ("runtime experiment contract violated: inference_data_profile.answer_fields_removed", "inference_data_profile"),
    ("INVALID TrainRunConfig: template mismatch", "specialist_contract_validation"),
    ("ssh: connect to host gpu.example port 22: Connection refused", "ssh_unreachable"),
    ("504 Gateway Timeout", "provider_5xx"),
])
def test_classify_uses_the_specific_executable_catalog_rule(error: str, code: str) -> None:
    assert classify(error, 1).code == code


# ─────────────────────────── rerun endpoint ────────────────────────────────


@pytest_asyncio.fixture
async def app_client():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    from fastapi import FastAPI
    from zevo.api.routers.shared import tickets as tickets_router
    from zevo.api.database import get_db

    app = FastAPI()
    app.include_router(tickets_router.router, prefix="/api")

    async def _override_get_db():
        s = Session()
        try:
            yield s
        finally:
            await s.close()

    app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, Session

    await engine.dispose()


async def _seed_failed_ticket(
    Session, *, ticket_id: str, error_message: str, exit_code: int = 1,
    with_workproduct: bool = False,
) -> None:
    async with Session() as s:
        s.add(Run(metric="accuracy", id="r-retry", task_name="t", agent_objective="t", status="failed",
                  mode="single_stage",
                  summary="", halted_reason="",
                  registry_version_tag="",
                  started_at=datetime.now(timezone.utc),
                  max_cost_usd=0.0))
        s.add(Ticket(id=ticket_id, run_id="r-retry", agent_id="data",
                     status="failed",
                     summary="", error_message=error_message,
                     payload={
                         "operation": "prepare_run_data",
                         "dataset_source": "/tmp/raw.jsonl",
                         "dataset": "/tmp/raw.jsonl",
                         "training_method": "full_sft",
                     }))
        s.add(HeartbeatRun(
            ticket_id=ticket_id, agent_id="data",
            driver="claude_cli", model="claude-opus-4-7",
            stdout_path="", stderr_path="",
            error_message=error_message, exit_code=exit_code,
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        ))
        if with_workproduct:
            s.add(WorkProduct(ticket_id=ticket_id, role="dataset",
                              path="/tmp/x.jsonl", meta={},
                              created_at=datetime.now(timezone.utc)))
        await s.commit()


@pytest.mark.asyncio
async def test_retry_status_returns_classifier_verdict(app_client) -> None:
    client, Session = app_client
    await _seed_failed_ticket(Session, ticket_id="t1",
                              error_message="CUDA out of memory: needed 24 GB")
    r = await client.get("/api/tickets/t1/retry-status")
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "transient"
    assert body["retryable"] is True
    assert "OOM" in body["reason"]


@pytest.mark.asyncio
async def test_rerun_fresh_flips_status_and_clears_workproduct(app_client) -> None:
    client, Session = app_client
    await _seed_failed_ticket(Session, ticket_id="t2",
                              error_message="Connection timed out",
                              with_workproduct=True)
    r = await client.post("/api/tickets/t2/rerun",
                          json={"strategy": "fresh", "actor": "operator"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "queued"
    assert body["new_status"] == "queued"

    # work_product should be gone after fresh rerun
    async with Session() as s:
        from sqlalchemy import select as _select
        wps = (await s.execute(_select(WorkProduct).where(WorkProduct.ticket_id == "t2"))).scalars().all()
        assert len(wps) == 0
        # ticket back to queued
        tk = (await s.execute(_select(Ticket).where(Ticket.id == "t2"))).scalar_one()
        assert tk.status == "queued"
        assert tk.error_message == ""
        # wakeup queued
        wks = (await s.execute(_select(AgentWakeupRequest).where(AgentWakeupRequest.ticket_id == "t2"))).scalars().all()
        assert len(wks) == 1
        assert wks[0].source == "retry"


@pytest.mark.asyncio
async def test_rerun_from_checkpoint_preserves_workproduct(app_client) -> None:
    client, Session = app_client
    await _seed_failed_ticket(Session, ticket_id="t3",
                              error_message="Connection timed out",
                              with_workproduct=True)
    r = await client.post("/api/tickets/t3/rerun",
                          json={"strategy": "from_checkpoint"})
    assert r.status_code == 200

    async with Session() as s:
        from sqlalchemy import select as _select
        wps = (await s.execute(_select(WorkProduct).where(WorkProduct.ticket_id == "t3"))).scalars().all()
        assert len(wps) == 1, "from_checkpoint must preserve work_product"


@pytest.mark.asyncio
async def test_rerun_refuses_structural_without_force(app_client) -> None:
    client, Session = app_client
    await _seed_failed_ticket(Session, ticket_id="t4",
                              error_message="ValidationError: 9 validation errors for DataResult\nstatus\n  Field required")
    r = await client.post("/api/tickets/t4/rerun", json={"strategy": "fresh"})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["verdict"] == "structural"
    assert "structural" in detail["error"].lower() or "structural" in detail.get("verdict", "")


@pytest.mark.asyncio
async def test_rerun_force_overrides_structural_refusal(app_client) -> None:
    client, Session = app_client
    await _seed_failed_ticket(Session, ticket_id="t5",
                              error_message="ValidationError: ... for DataResult")
    r = await client.post("/api/tickets/t5/rerun",
                          json={"strategy": "fresh", "force": True})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_rerun_with_override_payload_merges(app_client) -> None:
    client, Session = app_client
    await _seed_failed_ticket(Session, ticket_id="t6",
                              error_message="Connection timed out")
    r = await client.post("/api/tickets/t6/rerun", json={
        "strategy": "fresh",
        "override_payload": {
            "configuration_suggestions": {"target_size": 500},
        },
    })
    assert r.status_code == 200

    async with Session() as s:
        from sqlalchemy import select as _select
        tk = (await s.execute(_select(Ticket).where(Ticket.id == "t6"))).scalar_one()
    assert tk.payload["configuration_suggestions"]["target_size"] == 500
    # original keys preserved
    assert tk.payload["dataset"] == "/tmp/raw.jsonl"


@pytest.mark.asyncio
async def test_rerun_skips_already_running_ticket(app_client) -> None:
    """If the ticket is still queued/running, /rerun is a no-op."""
    client, Session = app_client
    async with Session() as s:
        s.add(Run(metric="accuracy", id="r-skip", task_name="t", agent_objective="t", status="running",
                  summary="", halted_reason="",
                  registry_version_tag="",
                  started_at=datetime.now(timezone.utc),
                  max_cost_usd=0.0))
        s.add(Ticket(id="t7", run_id="r-skip", agent_id="data",
                     status="queued",
                     summary="", error_message="", payload={}))
        await s.commit()
    r = await client.post("/api/tickets/t7/rerun", json={"strategy": "fresh"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "skipped"


@pytest.mark.asyncio
async def test_rerun_writes_audit_event(app_client) -> None:
    client, Session = app_client
    await _seed_failed_ticket(Session, ticket_id="t8",
                              error_message="Connection timed out")
    await client.post("/api/tickets/t8/rerun",
                      json={"strategy": "fresh", "actor": "operator"})

    async with Session() as s:
        from sqlalchemy import select as _select
        rows = (await s.execute(
            _select(AuditEvent).where(AuditEvent.target_id == "t8")
        )).scalars().all()
    assert len(rows) == 1
    assert rows[0].event_type == "ticket.rerun"
    assert rows[0].actor == "operator"
    assert "failed → queued" in rows[0].summary
