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


@pytest.mark.asyncio
@pytest.mark.parametrize('summary_bytes', [0, 92831, 2 * 1024 * 1024])
async def test_terminal_stdout_replay_recovers_tail_once_and_preserves_phase_order(db, monkeypatch, summary_bytes):
    import asyncio
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from zevo.engine.run.scheduler import reconciler as r
    from zevo.db import InfraInstance

    run, ticket, heartbeat = await setup(db, 'inference')
    now = datetime.now(timezone.utc).timestamp()
    lines = [f'__PHASE__:{ticket.id}:setup@{now}\n',
             f'__PHASE__:{ticket.id}:load_model@{now}\n']
    for i in range(3):
        lines.append(f'__PROGRESS__:{ticket.id}:' + json.dumps({
            'phase': 'generate', 'benchmark_name': f'Benchmark {i}',
            'benchmark_id': f'validation:{i}', 'benchmark_complete': True,
            'step': 10, 'total': 10, 't': now + i + 1,
        }) + '\n')
    # Foreign ticket markers must not leak into this execution.
    lines.append('__PHASE__:inference-other-001:done@' + str(now) + '\n')
    calls = []

    async def spawn(*args, **kwargs):
        calls.append(args[-1])
        stdout = asyncio.StreamReader()
        stdout.feed_data(lines[0].encode() + b'x' * summary_bytes + b'\n'
                         + ''.join(lines[1:]).encode())
        stdout.feed_eof()
        return SimpleNamespace(stdout=stdout, returncode=0,
                               wait=AsyncMock(return_value=0),
                               communicate=AsyncMock(return_value=(b'', b'')))

    monkeypatch.setattr(r.asyncio, 'create_subprocess_exec', spawn)
    monkeypatch.setattr(r, '_slurm_connection', AsyncMock(return_value={'host': 'host', 'user': 'u', 'key': '/test/key'}))
    row = InfraInstance(id='execution', instance_id='123', provider='cluster', ticket_id=ticket.id,
                        run_id=run.id, status='ready')
    meta = {'submission_heartbeat_id': heartbeat.id,
            'remote_workdir': '/work/ticket', 'log_path': '/work/ticket/out.log'}
    # Simulate the live stream recording only the first two phases/benchmarks.
    for kind, payload in [m for line in lines[:4] for m in r.scan_text(line) if m]:
        await r._persist_slurm_execution_marker(
            db, ticket_id=ticket.id, heartbeat=heartbeat, kind=kind, payload=payload,
            phase=payload.get('phase', ''), attempt_id=heartbeat.id,
            execution_id=row.id, runtime_key='cluster:123', log_path=meta['log_path'])
    await db.flush()
    live = asyncio.create_task(asyncio.Event().wait())
    r._SLURM_STATUS_STREAMS[row.id] = live
    await r._drain_terminal_telemetry(db, row=row, run=run, ticket=ticket, meta=meta)
    assert live.cancelled()
    assert row.id not in r._SLURM_STATUS_STREAMS
    assert meta.get('telemetry_drained'), meta
    await db.flush()
    events = (await db.execute(select(ExecutionEvent))).scalars().all()
    assert len(events) == 5
    assert sum(e.extras.get('benchmark_complete', False) for e in events) == 3
    assert all(e.extras['execution_id'] == row.id for e in events)
    assert calls == ['exec cat -- /work/ticket/out.log']
    await r._drain_terminal_telemetry(db, row=row, run=run, ticket=ticket, meta=meta)
    assert len(calls) == 1
    # Retry after a scheduler restart is still idempotent.
    meta.pop('telemetry_drained')
    await r._drain_terminal_telemetry(db, row=row, run=run, ticket=ticket, meta=meta)
    await db.flush()
    assert len((await db.execute(select(ExecutionEvent))).scalars().all()) == 5
    phases = {e.phase: e for e in events if e.event_type == 'phase'}
    assert phases['setup'].extras['telemetry_sequence'] == 1
    assert phases['load_model'].extras['telemetry_sequence'] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [OSError('unreachable'), TimeoutError()])
async def test_terminal_drain_failure_is_bounded_and_retryable(db, monkeypatch, error):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from zevo.engine.run.scheduler import reconciler as r
    run, ticket, heartbeat = await setup(db, 'inference')
    monkeypatch.setattr(r, '_slurm_connection', AsyncMock(return_value={'host': 'host', 'user': 'u', 'key': '/test/key'}))
    stream = AsyncMock(side_effect=error)
    monkeypatch.setattr(r, '_stream_slurm_status_events', stream)
    meta = {'submission_heartbeat_id': heartbeat.id,
            'remote_workdir': '/work/ticket', 'log_path': '/work/ticket/out.log'}
    row = SimpleNamespace(id='execution', instance_id='123')
    await r._drain_terminal_telemetry(db, row=row, run=run, ticket=ticket, meta=meta)
    assert not meta.get('telemetry_drained')
    assert meta['telemetry_drain_error']
    assert meta['telemetry_drain_retry_at']
    # Failed telemetry must neither raise into Collect nor busy-loop SSH retries.
    await r._drain_terminal_telemetry(db, row=row, run=run, ticket=ticket, meta=meta)
    assert stream.await_count == 1


@pytest.mark.asyncio
async def test_config_replay_preserves_engine_owned_activation_metadata(db):
    _, ticket, heartbeat = await setup(db, "inference")
    heartbeat.resolved_config = {"execution_phase": "submit", "measurement_revision": "r2"}
    await _persist_slurm_execution_marker(
        db, ticket_id=ticket.id, heartbeat=heartbeat, kind="config",
        payload={"batch_size": 64, "execution_phase": "collect", "telemetry_sequence": 1},
        phase="setup", attempt_id=heartbeat.id)
    assert heartbeat.resolved_config["execution_phase"] == "submit"
    assert heartbeat.resolved_config["measurement_revision"] == "r2"
    assert heartbeat.resolved_config["batch_size"] == 64
    assert "telemetry_sequence" not in heartbeat.resolved_config


@pytest.mark.asyncio
async def test_live_slurm_stream_keeps_progress_after_large_pool_summary(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from zevo.engine.run.scheduler import reconciler as r

    ticket_id = 'train-long-line-001'
    stdout = asyncio.StreamReader()  # Keep asyncio's real default 64 KiB limit.
    stdout.feed_data(
        f'__PHASE__:{ticket_id}:pool_selection@1793399589\n'.encode()
        + b'x' * 92831 + b'\n'
        + f'__PHASE__:{ticket_id}:training@1793400403\n'.encode()
        + f'__PROGRESS__:{ticket_id}:{{"step":460,"total":556,"t":1793414535}}'.encode()
    )
    stdout.feed_eof()  # Also preserve the final record without a newline.
    proc = SimpleNamespace(stdout=stdout, returncode=0,
                           communicate=AsyncMock(return_value=(b'', b'')))
    spawn = AsyncMock(return_value=proc)
    record = AsyncMock()
    monkeypatch.setattr(r.asyncio, 'create_subprocess_exec', spawn)
    monkeypatch.setattr(r, '_record_slurm_execution_marker', record)

    await r._stream_slurm_status_events(
        row_id='execution', job_id='123', ticket_id=ticket_id,
        heartbeat_id='heartbeat', connection={'host': 'cluster', 'user': 'u', 'key': '/test/key'},
        status_path='/work/status', telemetry_path='/work/out.log',
    )

    events = [call.kwargs for call in record.await_args_list]
    assert [event['kind'] for event in events] == ['phase', 'phase', 'progress']
    assert [event['payload']['telemetry_sequence'] for event in events] == [1, 2, 3]
    assert events[-1]['phase'] == 'training'
    assert events[-1]['payload']['step'] == 460
    assert events[-1]['payload']['total'] == 556
    assert all(event['row_id'] == 'execution' for event in events)
