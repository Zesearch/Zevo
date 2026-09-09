"""A long run is not a dead run.

The stuck-ticket sweep used `heartbeat.started_at` to decide whether the runner
had crashed. That timestamp never changes, so the test really asked "has this
been going a while?" — and answered yes for every training job that outlived the
cutoff, failing healthy tickets on a timer while the GPU was still working. The
three longest training activations in this system ran 2932s, 2353s and 2054s
against a 1800s cutoff; at the moment the old rule fired, all three had been
silent for under twenty seconds.

Liveness now comes from the transcript, which an agent writes continuously for
as long as it is working.
"""

import datetime as dt
import json

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import (
    AgentWakeupRequest, Base, ExecutionEvent, HeartbeatRun, InfraInstance, Run,
    Ticket, TicketMessage, TranscriptEvent, WorkProduct,
)
from zevo.api.routers.shared.runs import _finish_run_heartbeats
from zevo.engine.run.scheduler.reconciler import (
    _close_stale_terminal_heartbeats,
    _next_slurm_poll_interval,
    _parse_slurm_status_event,
    _persist_slurm_execution_marker,
    _record_slurm_status_event,
    _reconcile_slurm_stage_jobs,
    _release_run_instances,
    _slurm_telemetry_path,
    _sweep_stuck_tickets,
)

CUTOFF = 1800


def test_slurm_status_event_parser_accepts_only_the_exact_job() -> None:
    running = _parse_slurm_status_event(
        "ZEVO_SLURM_EVENT_V1|12345|RUNNING||2026-09-03T04:29:00Z",
        expected_job_id="12345",
    )
    assert running is not None
    assert running[0] == "RUNNING"
    assert running[1] is None
    assert running[2].tzinfo is not None

    exited = _parse_slurm_status_event(
        "ZEVO_SLURM_EVENT_V1|12345|EXITED|7|2026-09-03T04:30:00Z",
        expected_job_id="12345",
    )
    assert exited is not None
    assert exited[:2] == ("EXITED", 7)
    assert _parse_slurm_status_event(
        "ZEVO_SLURM_EVENT_V1|99999|RUNNING||2026-09-03T04:29:00Z",
        expected_job_id="12345",
    ) is None


def test_slurm_poll_interval_resets_queue_backoff_when_job_starts() -> None:
    assert _next_slurm_poll_interval(
        previous_state="PENDING", state="RUNNING", agent_id="inference",
        previous_interval=300,
    ) == 30
    assert _next_slurm_poll_interval(
        previous_state="PENDING", state="RUNNING", agent_id="train",
        previous_interval=300,
    ) == 60


def test_slurm_pending_backoff_is_stage_aware() -> None:
    assert _next_slurm_poll_interval(
        previous_state="PENDING", state="PENDING", agent_id="inference",
        previous_interval=120,
    ) == 120
    assert _next_slurm_poll_interval(
        previous_state="PENDING", state="PENDING", agent_id="train",
        previous_interval=240,
    ) == 300
    assert _next_slurm_poll_interval(
        previous_state="PENDING", state="PENDING", agent_id="inference",
        previous_interval=None,
    ) == 30


def test_slurm_telemetry_path_must_stay_inside_ticket_workspace() -> None:
    root = "/remote/run/train-001"
    assert _slurm_telemetry_path({
        "remote_workdir": root,
        "log_path": root + "/slurm-123.out",
    }) == root + "/slurm-123.out"
    assert _slurm_telemetry_path({
        "remote_workdir": root,
        "log_path": root + "/../secret",
    }) == ""
    assert _slurm_telemetry_path({
        "remote_workdir": root,
        "log_path": "/etc/passwd",
    }) == ""


def _ago(seconds: float) -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=seconds)


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        s.add(Run(metric="accuracy", id="r1", agent_objective="liveness", status="running"))
        await s.commit()
        yield s
    await engine.dispose()


async def _mk(session, ticket_id, *, started_ago, chatter_ago=None, finished=False):
    """One in-progress ticket with a live heartbeat, optionally still talking."""
    session.add(Ticket(id=ticket_id, run_id="r1", agent_id="train",
                       status="running", payload={}))
    hb = HeartbeatRun(
        id=f"hb-{ticket_id}", ticket_id=ticket_id, agent_id="train",
        driver="claude_cli", model="m", started_at=_ago(started_ago),
        finished_at=_ago(0) if finished else None,
        exit_code=0 if finished else -1, stdout_path="", stderr_path="",
        error_message="",
    )
    session.add(hb)
    if chatter_ago is not None:
        session.add(TranscriptEvent(
            heartbeat_id=hb.id, seq=1, type="tool_call", payload={},
            ts=_ago(chatter_ago),
        ))
    await session.commit()


@pytest.mark.asyncio
async def test_remote_training_markers_restore_and_merge_live_chart_rows(session):
    ticket_id = "train-telemetry-001"
    heartbeat = HeartbeatRun(
        id="hb-train-telemetry", ticket_id=ticket_id, agent_id="train",
        driver="claude_cli", model="m", activation_phase="submit",
        stdout_path="", stderr_path="", error_message="",
    )
    session.add_all([
        Ticket(
            id=ticket_id, run_id="r1", agent_id="train",
            status="running", payload={},
        ),
        heartbeat,
    ])
    await session.commit()
    emitted = dt.datetime.now(dt.timezone.utc).timestamp()
    attempt_id = "550e8400-e29b-41d4-a716-446655440000"

    await _persist_slurm_execution_marker(
        session, ticket_id=ticket_id, heartbeat=heartbeat, kind="attempt",
        payload={"attempt_id": attempt_id, "t": emitted}, phase="train",
        attempt_id=attempt_id,
    )
    await _persist_slurm_execution_marker(
        session, ticket_id=ticket_id, heartbeat=heartbeat, kind="progress",
        payload={
            "attempt_id": attempt_id, "step": 20, "total": 100,
            "loss": 1.2, "learning_rate": 2e-5, "t": emitted,
        },
        phase="train", attempt_id=attempt_id,
    )
    # Trainer emits evaluation separately at the same step. It enriches the
    # chart row instead of erasing its training loss or creating a duplicate.
    await _persist_slurm_execution_marker(
        session, ticket_id=ticket_id, heartbeat=heartbeat, kind="progress",
        payload={
            "attempt_id": attempt_id, "step": 20, "total": 100,
            "eval_loss": 0.8, "eval_mean_token_accuracy": 0.7,
            "t": emitted + 1,
        },
        phase="train", attempt_id=attempt_id,
    )
    # Reconnecting `tail -n +1` replays the log; persistence remains idempotent.
    await _persist_slurm_execution_marker(
        session, ticket_id=ticket_id, heartbeat=heartbeat, kind="progress",
        payload={
            "attempt_id": attempt_id, "step": 20, "total": 100,
            "loss": 1.2, "learning_rate": 2e-5, "t": emitted,
        },
        phase="train", attempt_id=attempt_id,
    )
    await session.commit()

    rows = (await session.execute(
        select(ExecutionEvent).where(ExecutionEvent.ticket_id == ticket_id)
    )).scalars().all()
    attempts = [row for row in rows if row.event_type == "attempt"]
    progress = [row for row in rows if row.event_type == "progress"]
    assert len(attempts) == 1
    assert len(progress) == 1
    assert progress[0].loss == pytest.approx(1.2)
    assert progress[0].extras["loss"] == pytest.approx(1.2)
    assert progress[0].extras["eval_loss"] == pytest.approx(0.8)
    assert progress[0].extras["eval_mean_token_accuracy"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_a_long_run_that_is_still_talking_survives(session):
    # The regression: 49 minutes in, well past the cutoff, but it spoke 3s ago.
    await _mk(session, "train-009", started_ago=2932, chatter_ago=3)
    swept = await _sweep_stuck_tickets(session, CUTOFF)
    assert swept == 0
    tk = await session.get(Ticket, "train-009")
    assert tk.status == "running"


@pytest.mark.asyncio
async def test_a_run_that_went_quiet_is_still_swept(session):
    # The case the sweep exists for: the heartbeat is open and nothing has come
    # out of it for longer than the cutoff.
    await _mk(session, "train-030", started_ago=4000, chatter_ago=2400)
    swept = await _sweep_stuck_tickets(session, CUTOFF)
    assert swept == 1
    tk = await session.get(Ticket, "train-030")
    assert tk.status == "failed"
    assert "silent for" in tk.error_message


@pytest.mark.asyncio
async def test_an_activation_that_never_spoke_falls_back_to_its_start(session):
    # No transcript at all — the one case `started_at` was always right about.
    await _mk(session, "train-031", started_ago=4000, chatter_ago=None)
    assert await _sweep_stuck_tickets(session, CUTOFF) == 1
    assert (await session.get(Ticket, "train-031")).status == "failed"


@pytest.mark.asyncio
async def test_a_young_silent_activation_is_left_alone(session):
    # Startup is quiet — model load, ssh, dataset download — and that is normal
    # for a while. Only quiet PAST the cutoff is a symptom.
    await _mk(session, "train-032", started_ago=600, chatter_ago=None)
    assert await _sweep_stuck_tickets(session, CUTOFF) == 0


@pytest.mark.asyncio
async def test_a_finished_heartbeat_still_sweeps_its_ticket(session):
    # Unchanged path: the runner closed the heartbeat but the ticket was left
    # running, which is a real crash-after-close.
    await _mk(session, "train-033", started_ago=100, chatter_ago=10, finished=True)
    assert await _sweep_stuck_tickets(session, CUTOFF) == 1
    assert (await session.get(Ticket, "train-033")).status == "failed"


@pytest.mark.asyncio
async def test_the_sweep_stays_idempotent(session):
    await _mk(session, "train-034", started_ago=4000, chatter_ago=3900)
    assert await _sweep_stuck_tickets(session, CUTOFF) == 1
    assert await _sweep_stuck_tickets(session, CUTOFF) == 0


@pytest.mark.asyncio
async def test_stale_heartbeat_on_terminal_run_is_closed(session):
    await _mk(session, "train-terminal", started_ago=4000, chatter_ago=3900)
    run = await session.get(Run, "r1")
    run.status = "cancelled"
    await session.commit()

    assert await _close_stale_terminal_heartbeats(session, CUTOFF) == 1
    heartbeat = await session.get(HeartbeatRun, "hb-train-terminal")
    assert heartbeat.finished_at is not None
    assert heartbeat.exit_code == 130
    assert await _close_stale_terminal_heartbeats(session, CUTOFF) == 0


@pytest.mark.asyncio
async def test_cancel_path_closes_open_heartbeat_immediately(session):
    await _mk(session, "train-cancelled", started_ago=60, chatter_ago=1)

    assert await _finish_run_heartbeats(
        session,
        "r1",
        reason="cancelled by user (run cancelled)",
    ) == 1
    await session.commit()

    heartbeat = await session.get(HeartbeatRun, "hb-train-cancelled")
    assert heartbeat.finished_at is not None
    assert heartbeat.exit_code == 130


@pytest.mark.asyncio
async def test_fresh_heartbeat_on_terminal_run_gets_a_grace_period(session):
    await _mk(session, "train-finishing", started_ago=60, chatter_ago=2)
    run = await session.get(Run, "r1")
    run.status = "success"
    await session.commit()

    assert await _close_stale_terminal_heartbeats(session, CUTOFF) == 0
    heartbeat = await session.get(HeartbeatRun, "hb-train-finishing")
    assert heartbeat.finished_at is None


@pytest.mark.asyncio
async def test_normal_finalization_honors_explicit_manual_release(session, tmp_path):
    """auto_release=false is an intentional lifecycle choice, not a hint that
    reconciliation may silently ignore."""
    path = tmp_path / "device_info.json"
    path.write_text(json.dumps({
        "provider": "cloud", "cloud_backend": "vastai",
        "instance_id": "cloud-manual-1", "auto_release": False,
        "ssh": {"host": "gpu.example", "port": 22, "user": "root", "key_path": "/tmp/key"},
        "gpu": {"has_gpu": True, "gpu_count": 1},
    }), encoding="utf-8")
    session.add(Ticket(
        id="infra-manual", run_id="r1", agent_id="infrastructure",
        status="succeeded", payload={}
    ))
    session.add(WorkProduct(
        ticket_id="infra-manual", role="device_info", path=str(path), meta={},
    ))
    row = InfraInstance(
        instance_id="cloud-manual-1", provider="cloud", status="ready",
        run_id="r1", ticket_id="infra-manual", dph=0.1,
        meta={"backend": "vastai", "auto_release": False},
    )
    session.add(row)
    await session.commit()

    run = await session.get(Run, "r1")
    assert await _release_run_instances(session, run) == 0
    await session.refresh(row)
    assert row.status == "ready"
    assert row.released_at is None


@pytest.mark.asyncio
async def test_slurm_watcher_wakes_deferred_ticket_only_at_terminal(
    session, monkeypatch,
) -> None:
    import zevo.engine.run.scheduler.reconciler as reconciler

    ticket = Ticket(
        id="train-slurm-001", run_id="r1", agent_id="train",
        status="waiting_external", payload={}, inputs={},
    )
    job = InfraInstance(
        instance_id="12345", provider="cluster", status="provisioning",
        run_id="r1", ticket_id=ticket.id,
        meta={
            "stage_job": True,
            "scheduler_state": "PENDING",
            "submission_committed": True,
        },
    )
    session.add_all([ticket, job])
    await session.commit()

    async def connection(*_args, **_kwargs):
        return {"host": "cluster", "port": 22, "user": "u"}

    async def completed(*_args, **_kwargs):
        return "COMPLETED", "0:0", "None", None

    monkeypatch.setattr(reconciler, "_slurm_connection", connection)
    monkeypatch.setattr(reconciler, "_query_slurm_job", completed)

    checked, resumed = await _reconcile_slurm_stage_jobs(session)
    assert (checked, resumed) == (1, 1)
    await session.refresh(ticket)
    await session.refresh(job)
    assert ticket.status == "queued"
    assert job.status == "released"
    assert job.ready_at == job.created_at
    wakeups = (await session.execute(
        select(AgentWakeupRequest).where(
            AgentWakeupRequest.ticket_id == ticket.id,
        )
    )).scalars().all()
    assert len(wakeups) == 1
    assert wakeups[0].source == "slurm_watcher"


@pytest.mark.asyncio
async def test_slurm_watcher_exposes_running_until_collect(session, monkeypatch) -> None:
    import zevo.engine.run.scheduler.reconciler as reconciler

    ticket = Ticket(
        id="infer-slurm-running", run_id="r1", agent_id="inference",
        status="waiting_external", payload={}, inputs={},
    )
    job = InfraInstance(
        instance_id="12347", provider="cluster", status="provisioning",
        run_id="r1", ticket_id=ticket.id,
        meta={
            "stage_job": True,
            "scheduler_state": "PENDING",
            "submission_committed": True,
        },
    )
    session.add_all([ticket, job])
    await session.commit()

    async def connection(*_args, **_kwargs):
        return {"host": "cluster", "port": 22, "user": "u"}

    async def running(*_args, **_kwargs):
        return "RUNNING", "0:0", "None", _ago(5)

    monkeypatch.setattr(reconciler, "_slurm_connection", connection)
    monkeypatch.setattr(reconciler, "_query_slurm_job", running)

    assert await _reconcile_slurm_stage_jobs(session) == (1, 0)
    await session.refresh(ticket)
    await session.refresh(job)
    assert ticket.status == "running"
    assert job.status == "ready"
    assert job.ready_at is not None
    assert job.meta["monitor_interval_seconds"] == 30
    messages = (await session.execute(select(TicketMessage).where(
        TicketMessage.ticket_id == ticket.id,
    ))).scalars().all()
    assert [message.body for message in messages] == [
        "Running: Slurm job 12347 started; Zevo will collect and validate its "
        "outputs after it finishes."
    ]

    # A finished submission heartbeat is normal while Slurm owns the work.
    session.add(HeartbeatRun(
        id="hb-slurm-running", ticket_id=ticket.id, agent_id="inference",
        driver="claude_cli", model="m", started_at=_ago(30),
        finished_at=_ago(20), exit_code=0, stdout_path="", stderr_path="",
        error_message="",
    ))
    await session.commit()
    assert await _sweep_stuck_tickets(session, CUTOFF) == 0
    await session.refresh(ticket)
    assert ticket.status == "running"

    async def completed(*_args, **_kwargs):
        return "COMPLETED", "0:0", "None", _ago(1)

    job.meta = {**job.meta, "monitor_next_at": _ago(1).isoformat()}
    session.add(job)
    await session.commit()
    monkeypatch.setattr(reconciler, "_query_slurm_job", completed)
    assert await _reconcile_slurm_stage_jobs(session) == (1, 1)
    await session.refresh(ticket)
    await session.refresh(job)
    assert ticket.status == "queued"
    assert job.status == "released"
    assert job.meta["collect_wakeup_queued"] is True
    # Terminal reconciliation is idempotent and never publishes Done; collect
    # still has to validate the actual outputs first.
    assert await _reconcile_slurm_stage_jobs(session) == (0, 0)
    messages = (await session.execute(select(TicketMessage).where(
        TicketMessage.ticket_id == ticket.id,
    ))).scalars().all()
    assert not any(message.body.startswith("Done:") for message in messages)


@pytest.mark.asyncio
async def test_slurm_status_stream_persists_running_immediately(
    session, monkeypatch,
) -> None:
    import zevo.engine.run.scheduler.reconciler as reconciler

    ticket = Ticket(
        id="infer-stream-running", run_id="r1", agent_id="inference",
        status="waiting_external", payload={}, inputs={},
    )
    job = InfraInstance(
        id="infra-stream-running", instance_id="22347", provider="cluster",
        status="provisioning", run_id="r1", ticket_id=ticket.id,
        meta={
            "stage_job": True,
            "scheduler_state": "PENDING",
            "monitor_interval_seconds": 120,
            "submission_committed": True,
        },
    )
    session.add_all([ticket, job])
    await session.commit()
    Session = async_sessionmaker(session.bind, expire_on_commit=False)
    monkeypatch.setattr(reconciler, "get_session_factory", lambda: Session)

    occurred_at = _ago(2)
    # Correct a scheduler timestamp that was rendered in site-local time but
    # arrived without a UTC offset before the event stream observed the job.
    job.ready_at = _ago(3600)
    await session.commit()
    assert await _record_slurm_status_event(
        row_id=job.id, job_id=job.instance_id, event="RUNNING",
        exit_code=None, occurred_at=occurred_at,
    ) is True

    ticket_id = ticket.id
    job_row_id = job.id
    session.expire_all()
    ticket = await session.get(Ticket, ticket_id)
    job = await session.get(InfraInstance, job_row_id)
    assert ticket.status == "running"
    assert job.status == "ready"
    ready_at = (
        job.ready_at
        if job.ready_at.tzinfo is not None
        else job.ready_at.replace(tzinfo=dt.timezone.utc)
    )
    assert abs((ready_at - occurred_at).total_seconds()) < 0.01
    assert job.meta["scheduler_state"] == "RUNNING"
    assert job.meta["status_stream_observed"] is True
    assert job.meta["monitor_interval_seconds"] == 30
    messages = (await session.execute(select(TicketMessage).where(
        TicketMessage.ticket_id == ticket.id,
    ))).scalars().all()
    assert len(messages) == 1
    assert messages[0].body.startswith("Running: Slurm job 22347 started")


@pytest.mark.asyncio
async def test_slurm_status_stream_prompts_terminal_confirmation(
    session, monkeypatch,
) -> None:
    import zevo.engine.run.scheduler.reconciler as reconciler

    ticket = Ticket(
        id="train-stream-exited", run_id="r1", agent_id="train",
        status="running", payload={}, inputs={},
    )
    job = InfraInstance(
        id="infra-stream-exited", instance_id="32347", provider="cluster",
        status="ready", run_id="r1", ticket_id=ticket.id,
        meta={
            "stage_job": True,
            "scheduler_state": "RUNNING",
            "external_ticket_running": True,
            "submission_committed": True,
        },
    )
    session.add_all([ticket, job])
    await session.commit()
    Session = async_sessionmaker(session.bind, expire_on_commit=False)
    monkeypatch.setattr(reconciler, "get_session_factory", lambda: Session)

    assert await _record_slurm_status_event(
        row_id=job.id, job_id=job.instance_id, event="EXITED",
        exit_code=137, occurred_at=_ago(1),
    ) is True

    job_row_id = job.id
    session.expire_all()
    job = await session.get(InfraInstance, job_row_id)
    assert job.status == "ready"
    assert job.released_at is None
    assert job.meta["scheduler_state"] == "RUNNING"
    assert job.meta["scheduler_event_exit_pending"] is True
    assert job.meta["scheduler_event_exit_code"] == 137
    assert job.meta["monitor_interval_seconds"] == 5


@pytest.mark.asyncio
async def test_watcher_does_not_own_an_uncommitted_submission(
    session, monkeypatch,
) -> None:
    import zevo.engine.run.scheduler.reconciler as reconciler

    ticket = Ticket(
        id="infer-submit-active", run_id="r1", agent_id="inference",
        status="running", payload={}, inputs={},
    )
    job = InfraInstance(
        id="infra-submit-active", instance_id="42347", provider="cluster",
        status="provisioning", run_id="r1", ticket_id=ticket.id,
        meta={
            "stage_job": True,
            "scheduler_state": "PENDING",
            "status_path": "/remote/infer-submit-active/.zevo-slurm-status",
        },
    )
    session.add_all([ticket, job])
    await session.commit()

    async def must_not_connect(*_args, **_kwargs):
        raise AssertionError("watcher started before runner committed submission")

    monkeypatch.setattr(reconciler, "_slurm_connection", must_not_connect)
    assert await _reconcile_slurm_stage_jobs(session) == (0, 0)
    await session.refresh(ticket)
    await session.refresh(job)
    assert ticket.status == "running"
    assert job.status == "provisioning"


@pytest.mark.asyncio
async def test_committed_fast_job_queues_collect_after_submit_heartbeat(
    session,
) -> None:
    ticket = Ticket(
        id="infer-fast-complete", run_id="r1", agent_id="inference",
        status="running", payload={}, inputs={},
    )
    job = InfraInstance(
        id="infra-fast-complete", instance_id="52347", provider="cluster",
        status="released", run_id="r1", ticket_id=ticket.id,
        released_at=_ago(1),
        meta={
            "stage_job": True,
            "scheduler_state": "COMPLETED",
            "submission_committed": True,
        },
    )
    session.add_all([ticket, job])
    await session.commit()

    assert await _reconcile_slurm_stage_jobs(session) == (0, 1)
    await session.refresh(ticket)
    await session.refresh(job)
    assert ticket.status == "queued"
    assert job.meta["collect_wakeup_queued"] is True
    assert await _sweep_stuck_tickets(session, CUTOFF) == 0


@pytest.mark.asyncio
async def test_slurm_watcher_honors_persisted_backoff(session, monkeypatch) -> None:
    import zevo.engine.run.scheduler.reconciler as reconciler

    ticket = Ticket(
        id="infer-slurm-001", run_id="r1", agent_id="inference",
        status="waiting_external", payload={}, inputs={},
    )
    job = InfraInstance(
        instance_id="12346", provider="cluster", status="provisioning",
        run_id="r1", ticket_id=ticket.id,
        meta={
            "stage_job": True,
            "scheduler_state": "PENDING",
            "monitor_next_at": (_ago(-120)).isoformat(),
            "submission_committed": True,
        },
    )
    session.add_all([ticket, job])
    await session.commit()

    async def must_not_poll(*_args, **_kwargs):
        raise AssertionError("watcher ignored monitor_next_at")

    monkeypatch.setattr(reconciler, "_query_slurm_job", must_not_poll)
    assert await _reconcile_slurm_stage_jobs(session) == (0, 0)
    await session.refresh(ticket)
    assert ticket.status == "waiting_external"
