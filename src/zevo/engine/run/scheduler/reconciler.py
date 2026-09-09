"""Run/ticket reconciler -- keep the dashboard honest.

The dashboard reads `runs.status` and `tickets.status` straight from the
DB. If the scheduler crashes mid-flight (process killed, container
restart, transient asyncpg error), tickets stay `running` and runs
stay `running` forever, even though no agent is actually working on
them. The dashboard then shows stale "RUNNING" pills that are pure lies.

This module is the bookkeeping job that closes the loop:

  1. Stuck-ticket sweep -- any ticket in `running` whose latest
     heartbeat is finished (or whose updated_at is older than
     `stale_ticket_seconds`) gets demoted to `failed` with a clear
     reason. Re-running the ticket is the agent/operator's call; we
     just stop pretending it's alive.

  2. Run-status repair -- an all-terminal pipeline is handed back to its
     supervisor for one explicit final decision. Only the supervisor may mark a
     clean `success`; the reconciler may report `degraded`, `failed`, or
     `halted` when that decision cannot be completed honestly.

The reconciler is idempotent: running it twice is a no-op the second
time. It is safe to call from the daemon's main loop on every tick and
also from a one-shot CLI for manual cleanup.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

from sqlalchemy import func, select, update as sa_update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.contracts.tickets import TERMINAL_RUN_STATUSES, TERMINAL_TICKET_STATUSES
from zevo.contracts.infrastructure import (
    SLURM_STATUS_EVENT_PREFIX,
    SLURM_STATUS_FILENAME,
)
from zevo.engine.observe.run_metrics import incomplete_journal_entries
from zevo.engine.run.failure_policy import MAX_REPAIR_ATTEMPTS
from zevo.engine.observe.markers import scan_text
from zevo.engine.ssh_auth import ssh_base_args
from zevo.providers import resolve_ssh_key
from zevo.db import (
    AgentWakeupRequest,
    ExecutionEvent,
    GpuLease,
    HeartbeatRun,
    InfraInstance,
    Run,
    SshHost,
    Ticket,
    TicketMessage,
    TranscriptEvent,
    WorkProduct,
    get_session_factory,
)


log = logging.getLogger(__name__)

_RECONCILER_TAG = "[reconciler]"
_SLURM_INITIAL_POLL_SECONDS = 30
_SLURM_PENDING_MAX_POLL_SECONDS = {
    # Inference jobs are commonly short enough to start and finish inside a
    # generic five-minute queue backoff. Keep their queue-state view fresher.
    "inference": 120,
    # Training jobs can remain queued for hours, so retain a polite ceiling for
    # shared schedulers.
    "train": 300,
}
_SLURM_DEFAULT_PENDING_MAX_POLL_SECONDS = 300
_SLURM_RUNNING_POLL_SECONDS = {
    # Once Slurm owns a running allocation, never carry the queue backoff into
    # execution. These bounds are also the maximum terminal-state visibility
    # lag before Zevo starts collection.
    "inference": 30,
    "evaluation": 30,
    "train": 60,
}
_SLURM_DEFAULT_RUNNING_POLL_SECONDS = 60
_SLURM_EXIT_CONFIRM_POLL_SECONDS = 5
_SLURM_TERMINAL_STATES = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
}
_SLURM_STATUS_STREAMS: dict[str, asyncio.Task[None]] = {}


def _meta_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _aware(parsed)


def _normalise_slurm_state(value: str) -> str:
    # sacct may suffix a terminal state with '+' when its display column was
    # truncated. Array/step details are deliberately ignored: Zevo registers
    # and watches the parent finite job it submitted.
    return value.strip().upper().split()[0].rstrip("+") if value.strip() else ""


def _parse_slurm_status_event(
    line: str, *, expected_job_id: str,
) -> tuple[str, int | None, datetime] | None:
    """Parse one authenticated job-local lifecycle event from the SSH stream."""
    parts = line.strip().split("|")
    if len(parts) != 5 or parts[0] != SLURM_STATUS_EVENT_PREFIX:
        return None
    _, job_id, event, raw_exit_code, raw_time = parts
    if job_id != expected_job_id or event not in {"RUNNING", "EXITED"}:
        return None
    occurred_at = _meta_datetime(raw_time)
    if occurred_at is None:
        return None
    exit_code: int | None = None
    if event == "EXITED":
        try:
            exit_code = int(raw_exit_code)
        except ValueError:
            return None
    return event, exit_code, occurred_at


def _slurm_status_path(meta: dict[str, Any]) -> str:
    path = str(meta.get("status_path") or "").strip()
    if (
        not path.startswith("/")
        or "\n" in path
        or "\r" in path
        or not path.endswith("/" + SLURM_STATUS_FILENAME)
    ):
        return ""
    return path


def _slurm_telemetry_path(meta: dict[str, Any]) -> str:
    """Return a safe Ticket-local remote stdout path, or ``""``.

    ``log_path`` is resolved and stamped by the engine from the Slurm contract
    after the concrete JOBID exists. Keep the containment check as defence in
    depth: a stage log is accepted only when it is an absolute child of the
    exact remote Ticket workspace stamped into the registered Slurm job.
    """
    raw_path = str(meta.get("log_path") or "").strip()
    raw_root = str(meta.get("remote_workdir") or "").strip()
    if (
        not raw_path.startswith("/")
        or not raw_root.startswith("/")
        or "\n" in raw_path
        or "\r" in raw_path
    ):
        return ""
    path = PurePosixPath(raw_path)
    root = PurePosixPath(raw_root)
    if ".." in path.parts or path == root:
        return ""
    try:
        path.relative_to(root)
    except ValueError:
        return ""
    return str(path)


def _marker_datetime(value: object) -> datetime:
    """Use a sane emitter timestamp, otherwise the scheduler receive time."""
    now = datetime.now(timezone.utc)
    if value is None:
        return now
    try:
        emitted = datetime.fromtimestamp(float(value), timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return now
    return emitted if abs((now - emitted).total_seconds()) <= 86400 else now


async def _persist_slurm_execution_marker(
    session: AsyncSession,
    *,
    ticket_id: str,
    heartbeat: HeartbeatRun,
    kind: str,
    payload: dict[str, Any],
    phase: str,
    attempt_id: str,
) -> bool:
    """Idempotently turn one remote stage marker into UI telemetry."""
    occurred_at = _marker_datetime(payload.get("t"))
    attempt_id = str(payload.get("attempt_id") or attempt_id or heartbeat.id)

    if kind == "config":
        heartbeat.resolved_config = {
            key: value for key, value in payload.items()
            if key not in {"kind", "owner", "attempt_id", "t"}
        }
        return True

    if kind == "attempt":
        existing = (await session.execute(select(ExecutionEvent.id).where(
            ExecutionEvent.heartbeat_id == heartbeat.id,
            ExecutionEvent.event_type == "attempt",
            ExecutionEvent.attempt_id == attempt_id,
        ).limit(1))).scalar_one_or_none()
        if existing is not None:
            return False
        session.add(ExecutionEvent(
            ticket_id=ticket_id,
            heartbeat_id=heartbeat.id,
            attempt_id=attempt_id,
            event_type="attempt",
            phase=phase or "train",
            current_step=0,
            total_steps=0,
            loss=-1.0,
            extras={},
            ts=occurred_at,
        ))
        return True

    if kind == "phase":
        name = str(payload.get("phase") or phase or "")
        if not name:
            return False
        existing = (await session.execute(select(ExecutionEvent.id).where(
            ExecutionEvent.heartbeat_id == heartbeat.id,
            ExecutionEvent.attempt_id == attempt_id,
            ExecutionEvent.event_type == "phase",
            ExecutionEvent.phase == name,
            ExecutionEvent.ts == occurred_at,
        ).limit(1))).scalar_one_or_none()
        if existing is not None:
            return False
        session.add(ExecutionEvent(
            ticket_id=ticket_id,
            heartbeat_id=heartbeat.id,
            attempt_id=attempt_id,
            event_type="phase",
            phase=name,
            current_step=0,
            total_steps=0,
            loss=-1.0,
            extras={},
            ts=occurred_at,
        ))
        return True

    if kind != "progress":
        return False

    try:
        current_step = int(payload.get("step", payload.get("current_step", 0)) or 0)
        total_steps = int(payload.get("total", payload.get("total_steps", 0)) or 0)
    except (TypeError, ValueError):
        return False
    name = str(payload.get("phase") or phase or "")
    loss = -1.0
    for key in ("loss", "train_loss", "eval_loss"):
        try:
            candidate = float(payload.get(key))
        except (TypeError, ValueError):
            continue
        if candidate >= 0:
            loss = candidate
            break
    reserved = {
        "kind", "owner", "attempt_id", "t", "step", "total",
        "current_step", "total_steps", "phase",
    }
    extras = {key: value for key, value in payload.items() if key not in reserved}
    existing = (await session.execute(select(ExecutionEvent).where(
        ExecutionEvent.heartbeat_id == heartbeat.id,
        ExecutionEvent.attempt_id == attempt_id,
        ExecutionEvent.event_type == "progress",
        ExecutionEvent.phase == name,
        ExecutionEvent.current_step == current_step,
    ).limit(1))).scalar_one_or_none()
    if existing is not None:
        existing.total_steps = max(int(existing.total_steps or 0), total_steps)
        # An eval record commonly follows the training record for the same
        # step. Merge eval_* into extras without replacing the training loss.
        if "loss" in payload or float(existing.loss or -1.0) < 0:
            existing.loss = loss
        existing.extras = {**dict(existing.extras or {}), **extras}
        existing.ts = max(_aware(existing.ts), occurred_at)
        return False
    session.add(ExecutionEvent(
        ticket_id=ticket_id,
        heartbeat_id=heartbeat.id,
        attempt_id=attempt_id,
        event_type="progress",
        phase=name,
        current_step=current_step,
        total_steps=total_steps,
        loss=loss,
        extras=extras,
        ts=occurred_at,
    ))
    return True


def _next_slurm_poll_interval(
    *, previous_state: str, state: str, agent_id: str,
    previous_interval: object,
) -> int:
    """Choose a state- and stage-aware interval for one exact Slurm job.

    Queue checks back off because queue latency can be long and shared Slurm
    controllers should not be polled aggressively. Running checks have a
    bounded cadence based on the stage's expected scale. Most importantly, a
    PENDING interval is never inherited after the job starts running.
    """
    normalized_state = _normalise_slurm_state(state)
    normalized_previous = _normalise_slurm_state(previous_state)
    stage = str(agent_id or "").strip().lower()

    if normalized_state == "RUNNING":
        return _SLURM_RUNNING_POLL_SECONDS.get(
            stage, _SLURM_DEFAULT_RUNNING_POLL_SECONDS,
        )

    if normalized_state == "PENDING":
        ceiling = _SLURM_PENDING_MAX_POLL_SECONDS.get(
            stage, _SLURM_DEFAULT_PENDING_MAX_POLL_SECONDS,
        )
        try:
            prior = int(previous_interval or 0)
        except (TypeError, ValueError):
            prior = 0
        if normalized_previous != "PENDING" or prior <= 0:
            return _SLURM_INITIAL_POLL_SECONDS
        return min(ceiling, max(_SLURM_INITIAL_POLL_SECONDS, prior * 2))

    return _SLURM_INITIAL_POLL_SECONDS


async def _slurm_connection(run: Run, session: AsyncSession) -> dict[str, Any] | None:
    selected = await session.get(SshHost, run.ssh_host_id) if run.ssh_host_id else None
    if selected is not None:
        return {
            "host": selected.host,
            "port": int(selected.port or 22),
            "user": selected.username,
            "key": selected.key_path or "",
            "password": selected.password_path or "",
            "env_setup": selected.env_setup or "",
        }
    host = os.environ.get("ZEVO_CLUSTER_SSH_HOST", "").strip()
    if not host:
        return None
    password = os.environ.get("ZEVO_CLUSTER_SSH_PASSWORD_FILE", "").strip()
    return {
        "host": host,
        "port": int(os.environ.get("ZEVO_CLUSTER_SSH_PORT", "22") or "22"),
        "user": os.environ.get("ZEVO_CLUSTER_SSH_USER", "").strip(),
        "key": "" if password else resolve_ssh_key("ZEVO_CLUSTER_SSH_KEY"),
        "password": password,
        "env_setup": os.environ.get("ZEVO_CLUSTER_ENV_SETUP", "").strip(),
    }


def _mark_slurm_job_running(
    *, session: AsyncSession, row: InfraInstance, ticket: Ticket,
    meta: dict[str, Any], observed_at: datetime,
) -> None:
    """Apply a RUNNING observation once, whether streamed or polled."""
    row.status = "ready"
    # Job-local events use `date -u` and are the most precise start signal.
    # They also correct site `sacct Start` values rendered without an offset
    # and previously interpreted as UTC by the backend.
    row.ready_at = (
        observed_at
        if bool(meta.get("status_stream_observed"))
        else row.ready_at or observed_at
    )
    if ticket.status == "waiting_external":
        ticket.status = "running"
        meta["external_ticket_running"] = True
    if not bool(meta.get("running_announced")):
        session.add(TicketMessage(
            ticket_id=ticket.id,
            author="system",
            body=(
                f"Running: Slurm job {row.instance_id} started; Zevo will "
                "collect and validate its outputs after it finishes."
            ),
        ))
        meta["running_announced"] = True


async def _record_slurm_status_event(
    *, row_id: str, job_id: str, event: str,
    exit_code: int | None, occurred_at: datetime,
) -> bool:
    """Persist one event received from the job's append-only status file."""
    Session = get_session_factory()
    async with Session() as session:
        result = (await session.execute(
            select(InfraInstance, Ticket)
            .join(Ticket, Ticket.id == InfraInstance.ticket_id)
            .where(InfraInstance.id == row_id)
        )).first()
        if result is None:
            return False
        row, ticket = result
        if (
            row.instance_id != job_id
            or row.released_at is not None
            or ticket.status in TERMINAL_TICKET_STATUSES
        ):
            return False
        meta = dict(row.meta or {})
        if not bool(meta.get("submission_committed")):
            return False
        if _normalise_slurm_state(
            str(meta.get("scheduler_state") or "")
        ) in _SLURM_TERMINAL_STATES:
            return False

        now = datetime.now(timezone.utc)
        meta["status_event_at"] = occurred_at.isoformat()
        meta["status_stream_observed"] = True
        if event == "RUNNING":
            previous_state = str(meta.get("scheduler_state") or "PENDING")
            meta["scheduler_state"] = "RUNNING"
            meta.pop("scheduler_event_exit_pending", None)
            interval = _next_slurm_poll_interval(
                previous_state=previous_state,
                state="RUNNING",
                agent_id=ticket.agent_id,
                previous_interval=meta.get("monitor_interval_seconds"),
            )
            meta["monitor_interval_seconds"] = interval
            meta["monitor_last_at"] = now.isoformat()
            meta["monitor_next_at"] = (
                now + _dt.timedelta(seconds=interval)
            ).isoformat()
            _mark_slurm_job_running(
                session=session, row=row, ticket=ticket, meta=meta,
                observed_at=occurred_at,
            )
        elif event == "EXITED":
            # The job-local trap knows the process exit code immediately, but
            # sacct remains authoritative for TIMEOUT/PREEMPTED/NODE_FAIL. Ask
            # the normal watcher to confirm promptly instead of guessing.
            meta["scheduler_event_exit_pending"] = True
            meta["scheduler_event_exit_code"] = int(exit_code or 0)
            meta["scheduler_event_exit_at"] = occurred_at.isoformat()
            meta["monitor_interval_seconds"] = _SLURM_EXIT_CONFIRM_POLL_SECONDS
            meta["monitor_next_at"] = now.isoformat()
        else:
            return False
        row.meta = meta
        await session.commit()
        return True


async def _record_slurm_execution_marker(
    *, row_id: str, job_id: str, ticket_id: str, heartbeat_id: str,
    kind: str, payload: dict[str, Any], phase: str, attempt_id: str,
) -> bool:
    """Persist one authenticated marker read from this job's remote stdout."""
    Session = get_session_factory()
    async with Session() as session:
        result = (await session.execute(
            select(InfraInstance, Ticket)
            .join(Ticket, Ticket.id == InfraInstance.ticket_id)
            .where(InfraInstance.id == row_id)
        )).first()
        heartbeat = await session.get(HeartbeatRun, heartbeat_id)
        if result is None or heartbeat is None:
            return False
        row, ticket = result
        meta = dict(row.meta or {})
        owner = str(payload.get("owner") or "")
        if (
            row.instance_id != job_id
            or ticket.id != ticket_id
            or heartbeat.ticket_id != ticket_id
            or str(meta.get("submission_heartbeat_id") or "") != heartbeat_id
            or not bool(meta.get("submission_committed"))
            or (owner and owner != ticket_id)
        ):
            return False
        clean_payload = dict(payload)
        clean_payload.pop("owner", None)
        changed = await _persist_slurm_execution_marker(
            session,
            ticket_id=ticket_id,
            heartbeat=heartbeat,
            kind=kind,
            payload=clean_payload,
            phase=phase,
            attempt_id=attempt_id,
        )
        await session.commit()
        return changed


async def _stream_slurm_status_events(
    *, row_id: str, job_id: str, ticket_id: str, heartbeat_id: str,
    connection: dict[str, Any], status_path: str, telemetry_path: str = "",
) -> None:
    """Stream lifecycle plus Train/Inference telemetry over one SSH session."""
    remote_paths = [status_path]
    if telemetry_path:
        remote_paths.append(telemetry_path)
    tail_command = "exec tail -q -n +1 -F -- " + " ".join(
        shlex.quote(path) for path in remote_paths
    )
    args = [
        *ssh_base_args(
            key_path=str(connection.get("key") or ""),
            password_path=str(connection.get("password") or ""),
            port=int(connection.get("port") or 22),
            connect_timeout=15,
        ),
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        f"{connection.get('user')}@{connection.get('host')}",
        tail_command,
    ]
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        if proc.stdout is None:
            return
        current_phase = ""
        current_attempt = heartbeat_id
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace")
            parsed = _parse_slurm_status_event(
                line,
                expected_job_id=job_id,
            )
            if parsed is not None:
                event, exit_code, occurred_at = parsed
                await _record_slurm_status_event(
                    row_id=row_id,
                    job_id=job_id,
                    event=event,
                    exit_code=exit_code,
                    occurred_at=occurred_at,
                )
            if not telemetry_path:
                continue
            for marker in scan_text(line):
                if marker is None:
                    continue
                kind, payload = marker
                owner = str(payload.get("owner") or "")
                if owner != ticket_id:
                    continue
                if kind == "attempt":
                    current_attempt = str(payload.get("attempt_id") or current_attempt)
                elif kind == "phase":
                    current_phase = str(payload.get("phase") or current_phase)
                    payload.setdefault("attempt_id", current_attempt)
                elif kind == "progress":
                    payload.setdefault("attempt_id", current_attempt)
                    payload.setdefault("phase", current_phase)
                await _record_slurm_execution_marker(
                    row_id=row_id,
                    job_id=job_id,
                    ticket_id=ticket_id,
                    heartbeat_id=heartbeat_id,
                    kind=kind,
                    payload=payload,
                    phase=current_phase,
                    attempt_id=current_attempt,
                )
    finally:
        if proc.returncode is None:
            proc.terminate()
        try:
            await asyncio.wait_for(proc.communicate(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()


def _forget_slurm_status_stream(
    row_id: str, task: asyncio.Task[None],
) -> None:
    if _SLURM_STATUS_STREAMS.get(row_id) is task:
        _SLURM_STATUS_STREAMS.pop(row_id, None)
    if task.cancelled():
        return
    try:
        error = task.exception()
    except asyncio.CancelledError:
        return
    if error is not None:
        log.warning(
            "[slurm-status] stream for row %s ended: %s", row_id, error,
        )


async def _ensure_slurm_status_stream(
    *, row: InfraInstance, run: Run, ticket: Ticket, meta: dict[str, Any],
    session: AsyncSession,
) -> bool:
    """Start the event stream once; a later reconcile reconnects on failure."""
    status_path = _slurm_status_path(meta)
    if not status_path:
        return False
    existing = _SLURM_STATUS_STREAMS.get(row.id)
    if existing is not None and not existing.done():
        return True
    connection = await _slurm_connection(run, session)
    if connection is None or not connection.get("host") or not connection.get("user"):
        return False
    heartbeat_id = str(meta.get("submission_heartbeat_id") or "").strip()
    if not heartbeat_id:
        return False
    telemetry_path = (
        _slurm_telemetry_path(meta)
        if ticket.agent_id in {"train", "inference"}
        else ""
    )
    task = asyncio.create_task(
        _stream_slurm_status_events(
            row_id=row.id,
            job_id=row.instance_id,
            ticket_id=ticket.id,
            heartbeat_id=heartbeat_id,
            connection=connection,
            status_path=status_path,
            telemetry_path=telemetry_path,
        ),
        name=f"slurm-status-{row.id}",
    )
    _SLURM_STATUS_STREAMS[row.id] = task
    task.add_done_callback(
        lambda finished, row_id=row.id: _forget_slurm_status_stream(row_id, finished)
    )
    return True


def _cancel_inactive_slurm_status_streams(active_row_ids: set[str]) -> None:
    for row_id, task in list(_SLURM_STATUS_STREAMS.items()):
        if row_id not in active_row_ids:
            _SLURM_STATUS_STREAMS.pop(row_id, None)
            task.cancel()


async def _query_slurm_job(
    *, connection: dict[str, Any], job_id: str,
) -> tuple[str, str, str, datetime | None]:
    """Return state, exit code, reason, and actual start for one exact job."""
    if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
        raise ValueError(f"invalid Slurm job id {job_id!r}")
    bootstrap = (
        "if ! command -v squeue >/dev/null 2>&1 && "
        "[ -f /etc/profile.d/modules.sh ]; then "
        "source /etc/profile.d/modules.sh; module load default-environment; fi"
    )
    command = (
        "set +u; " + bootstrap + "; "
        + f"q=$(squeue -h -j {job_id} -o %T 2>/dev/null | head -n 1 || true); "
        + f"a=$(TZ=UTC sacct -n -X -j {job_id} --format=State,ExitCode,Start -P "
          "2>/dev/null | awk 'NF {print; exit}' || true); "
        + f"r=$(scontrol show job -o {job_id} 2>/dev/null | "
          "sed -n 's/.*Reason=\\([^ ]*\\).*/\\1/p' | head -n 1 || true); "
        + "printf 'QUEUE=%s\\nACCOUNTING=%s\\nREASON=%s\\n' \"$q\" \"$a\" \"$r\""
    )
    args = [
        *ssh_base_args(
            key_path=str(connection.get("key") or ""),
            password_path=str(connection.get("password") or ""),
            port=int(connection.get("port") or 22),
            connect_timeout=15,
        ),
        f"{connection.get('user')}@{connection.get('host')}",
        command,
    ]
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    if proc.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()[:1000]
        raise RuntimeError(detail or f"SSH scheduler query exited {proc.returncode}")
    values: dict[str, str] = {}
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep and key in {"QUEUE", "ACCOUNTING", "REASON"}:
            values[key] = value.strip()
    accounting = values.get("ACCOUNTING", "")
    acct_parts = accounting.split("|")
    acct_state = acct_parts[0] if acct_parts else ""
    exit_code = acct_parts[1] if len(acct_parts) > 1 else ""
    started_at = _meta_datetime(acct_parts[2] if len(acct_parts) > 2 else "")
    state = _normalise_slurm_state(values.get("QUEUE", ""))
    if not state:
        state = _normalise_slurm_state(acct_state)
    return state, exit_code.strip(), values.get("REASON", "").strip(), started_at


async def _reconcile_slurm_stage_jobs(
    session: AsyncSession,
) -> tuple[int, int]:
    """Observe finite cluster jobs and wake their paused Tickets at terminal.

    This is deliberately a backend responsibility. An Agent submits one file,
    registers its JOBID, and exits; it does not spend tokens or hold a runner
    while Slurm queues/runs. A quiet persistent SSH stream consumes job-local
    RUNNING/EXITED events in near real time. State-aware, low-frequency Slurm
    queries remain the correctness fallback for legacy jobs, disconnected
    streams, and scheduler terminal details that a shell exit trap cannot know.
    """
    rows = (await session.execute(
        select(InfraInstance, Run, Ticket)
        .join(Run, Run.id == InfraInstance.run_id)
        .join(Ticket, Ticket.id == InfraInstance.ticket_id)
        .where(
            InfraInstance.provider == "cluster",
            InfraInstance.instance_id != "",
            Run.status.notin_(list(TERMINAL_RUN_STATUSES)),
            Ticket.status.notin_(list(TERMINAL_TICKET_STATUSES)),
        )
        .order_by(InfraInstance.ticket_id, InfraInstance.created_at.desc())
    )).all()

    # A Train continuation can register another JOBID for the same Ticket. Only
    # the newest row owns the current activation; an old completed job must not
    # wake the Ticket beside its newer running continuation.
    newest: list[tuple[InfraInstance, Run, Ticket]] = []
    seen_tickets: set[str] = set()
    for row, run, ticket in rows:
        if ticket.id in seen_tickets:
            continue
        seen_tickets.add(ticket.id)
        newest.append((row, run, ticket))

    stream_row_ids = {
        row.id
        for row, _run, _ticket in newest
        if _normalise_slurm_state(
            str((row.meta or {}).get("scheduler_state") or "")
        ) not in _SLURM_TERMINAL_STATES
        and bool((row.meta or {}).get("submission_committed"))
        and not bool((row.meta or {}).get("scheduler_event_exit_pending"))
        and bool(_slurm_status_path(dict(row.meta or {})))
    }
    _cancel_inactive_slurm_status_streams(stream_row_ids)
    for row, run, ticket in newest:
        if row.id not in stream_row_ids:
            continue
        try:
            await _ensure_slurm_status_stream(
                row=row, run=run, ticket=ticket,
                meta=dict(row.meta or {}), session=session,
            )
        except Exception as exc:
            # A broken event stream never owns correctness. The persisted
            # low-frequency Slurm query below remains the fallback.
            log.warning(
                "[slurm-status] could not start stream for row %s: %s",
                row.id, exc,
            )

    now = datetime.now(timezone.utc)
    checked = 0
    resumed = 0
    for row, run, ticket in newest:
        meta = dict(row.meta or {})
        # Registering a JOBID can happen while its submit activation is still
        # validating generated files and the typed Result. Until the runner
        # commits that handoff, the activation—not this watcher—owns the Ticket.
        if not bool(meta.get("submission_committed")):
            continue
        state = _normalise_slurm_state(str(meta.get("scheduler_state") or ""))
        previous_state = state
        terminal = state in _SLURM_TERMINAL_STATES
        exit_event_pending = bool(meta.get("scheduler_event_exit_pending"))
        due_at = _meta_datetime(meta.get("monitor_next_at"))
        if not terminal and due_at is not None and due_at > now:
            continue

        if not terminal:
            connection = await _slurm_connection(run, session)
            if connection is None or not connection.get("host") or not connection.get("user"):
                meta["monitor_error"] = "cluster SSH connection is unavailable"
            else:
                try:
                    state, exit_code, reason, started_at = await _query_slurm_job(
                        connection=connection, job_id=row.instance_id,
                    )
                    checked += 1
                    meta.pop("monitor_error", None)
                    if state:
                        meta["scheduler_state"] = state
                    if exit_code:
                        meta["scheduler_exit_code"] = exit_code
                    if reason and reason.lower() not in {"none", "(null)"}:
                        meta["scheduler_reason"] = reason
                    if started_at is not None:
                        row.ready_at = row.ready_at or started_at
                except Exception as exc:  # transient SSH/scheduler failures retry
                    meta["monitor_error"] = str(exc)[:1000]

            state = _normalise_slurm_state(str(meta.get("scheduler_state") or ""))
            if state == "PENDING":
                queued_for = max(0.0, (now - _aware(row.created_at)).total_seconds())
                queue_limit = max(0.0, float(run.max_queue_wait_hours or 0.0)) * 3600
                if queue_limit and queued_for >= queue_limit:
                    connection = connection or await _slurm_connection(run, session)
                    if connection is not None:
                        try:
                            cancel = await asyncio.create_subprocess_exec(
                                *ssh_base_args(
                                    key_path=str(connection.get("key") or ""),
                                    password_path=str(connection.get("password") or ""),
                                    port=int(connection.get("port") or 22),
                                ),
                                f"{connection.get('user')}@{connection.get('host')}",
                                f"scancel {row.instance_id}",
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                            )
                            await asyncio.wait_for(cancel.communicate(), timeout=30)
                        except Exception as exc:
                            meta["monitor_error"] = f"queue timeout scancel failed: {exc}"[:1000]
                    state = "TIMEOUT"
                    meta["scheduler_state"] = state
                    meta["scheduler_reason"] = (
                        f"maximum queue wait of {queue_limit / 3600:g} hours exceeded"
                    )

            terminal = state in _SLURM_TERMINAL_STATES
            exit_event_at = _meta_datetime(meta.get("scheduler_event_exit_at"))
            recent_exit_event = bool(
                exit_event_pending
                and exit_event_at is not None
                and (now - exit_event_at).total_seconds() < 60
            )
            interval = (
                _SLURM_EXIT_CONFIRM_POLL_SECONDS
                if recent_exit_event and not terminal
                else _next_slurm_poll_interval(
                    previous_state=previous_state,
                    state=state,
                    agent_id=ticket.agent_id,
                    previous_interval=meta.get("monitor_interval_seconds"),
                )
            )
            meta["monitor_interval_seconds"] = interval
            meta["monitor_last_at"] = now.isoformat()
            meta["monitor_next_at"] = (now + _dt.timedelta(seconds=interval)).isoformat()

        if state == "RUNNING":
            _mark_slurm_job_running(
                session=session, row=row, ticket=ticket, meta=meta,
                observed_at=row.ready_at or now,
            )
        elif state == "PENDING":
            row.status = "provisioning"

        if terminal:
            meta.pop("scheduler_event_exit_pending", None)
            # If a very short job completed between two checks, do not label its
            # whole lifetime as queue wait. Regular jobs get the actual first
            # observed RUNNING timestamp above.
            if state == "COMPLETED" and row.ready_at is None:
                row.ready_at = row.created_at
            row.status = "released" if state == "COMPLETED" else "failed"
            row.released_at = row.released_at or now
            row.release_reason = str(
                meta.get("scheduler_reason") or f"Slurm job {state.lower()}"
            )[:2000]
            meta["monitor_terminal"] = True
            can_queue_collect = (
                bool(meta.get("submission_committed"))
                and ticket.status in {"waiting_external", "running"}
            )
            if can_queue_collect and not bool(meta.get("collect_wakeup_queued")):
                ticket.status = "queued"
                ticket.summary = f"Slurm job {row.instance_id} finished: {state}"
                ticket.error_message = ""
                meta["collect_wakeup_queued"] = True
                row.meta = meta
                await session.flush()
                from zevo.engine.run.wakeup import queue_wakeup
                await queue_wakeup(
                    session,
                    agent_id=ticket.agent_id,
                    ticket_id=ticket.id,
                    source="slurm_watcher",
                    trigger_detail=f"job_{state.lower()}",
                    reason=(
                        f"finite Slurm job {row.instance_id} reached {state}; "
                        "collect and validate its outputs"
                    ),
                    payload={"job_id": row.instance_id, "scheduler_state": state},
                )
                resumed += 1
                continue
        row.meta = meta

    await session.commit()
    return checked, resumed


def _aware(t: datetime) -> datetime:
    """UTC-aware, whichever backend produced it. Postgres returns aware
    datetimes, SQLite naive ones, and comparing the two raises."""
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


async def _last_sign_of_life(session: AsyncSession, hb: HeartbeatRun) -> datetime:
    """When this activation was last observed doing something.

    NOT `started_at`. That is a constant, so comparing it against a cutoff asks
    "has this run been going a long time?" rather than "is it still alive?", and
    the two answers differ exactly where it matters: a training job that runs
    longer than the cutoff is declared crashed while it is still training. The
    longest training activations in this system run well past an hour; the cutoff
    is half of that, so healthy work was being failed on a timer.

    An agent writes transcript events continuously — the runner's flusher pushes
    them to the DB as they happen so the UI can stream them — and they span the
    whole activation. That makes the newest one a real liveness signal, with no
    new column to maintain and nothing extra to write: if an activation has gone
    quiet for longer than the cutoff, it is genuinely stuck, however long it has
    been running.

    Falls back to `started_at` for an activation that has not managed to emit
    anything yet, which is the case the cutoff was always right about.
    """
    # Both sides need the same awareness before they can be compared: Postgres
    # returns aware datetimes and SQLite (local dev, tests) returns naive ones,
    # and mixing them raises rather than comparing wrong — which is how this got
    # caught, but only because something compared them.
    started = _aware(hb.started_at)
    last = (await session.execute(
        select(func.max(TranscriptEvent.ts)).where(TranscriptEvent.heartbeat_id == hb.id)
    )).scalar_one_or_none()
    return started if last is None else max(_aware(last), started)


async def _sweep_stuck_tickets(
    session: AsyncSession, stale_ticket_seconds: int
) -> int:
    """Mark `running` tickets whose runner is gone as `failed`.

    'Runner is gone' = (a) latest heartbeat exists and is finished, or
    (b) no heartbeat at all AND ticket.updated_at is older than the
    stale cutoff. (a) covers the "crashed mid-run" case; (b) covers
    the "wakeup picked up the ticket, flipped status, then crashed
    before writing a heartbeat" edge case.
    """
    cutoff = datetime.now(timezone.utc) - _dt.timedelta(seconds=stale_ticket_seconds)
    running = (await session.execute(
        select(Ticket).where(Ticket.status == "running")
    )).scalars().all()

    swept: list[Ticket] = []
    for tk in running:
        # A finite Slurm job has no live Agent heartbeat by design. The backend
        # watcher owns it between submit and collect, so a finished submission
        # heartbeat is evidence of correct deferral, not a crashed runner.
        stage_rows = (await session.execute(
            select(InfraInstance)
            .where(
                InfraInstance.ticket_id == tk.id,
                InfraInstance.provider == "cluster",
                InfraInstance.instance_id != "",
                InfraInstance.released_at.is_(None),
            )
            .order_by(InfraInstance.created_at.desc())
            .limit(5)
        )).scalars().all()
        live_stage = next((
            row for row in stage_rows
            if bool((row.meta or {}).get("stage_job"))
            and _normalise_slurm_state(
                str((row.meta or {}).get("scheduler_state") or "")
            ) not in _SLURM_TERMINAL_STATES
        ), None)
        if live_stage is not None:
            continue

        latest_hb = (await session.execute(
            select(HeartbeatRun)
            .where(HeartbeatRun.ticket_id == tk.id)
            .order_by(HeartbeatRun.started_at.desc())
            .limit(1)
        )).scalar_one_or_none()

        runner_gone = False
        reason = ""
        if latest_hb is not None and latest_hb.finished_at is not None:
            runner_gone = True
            reason = (
                f"{_RECONCILER_TAG} heartbeat {latest_hb.id[:8]} "
                f"finished at {latest_hb.finished_at.isoformat()} "
                f"(exit={latest_hb.exit_code}) but ticket left running"
            )
        elif latest_hb is not None and latest_hb.finished_at is None:
            # Heartbeat row written but never marked finished. Whether that means
            # a crash depends on whether the activation is still SAYING anything
            # — see `_last_sign_of_life`. A long job that is still emitting is
            # working, not stuck.
            alive_at = await _last_sign_of_life(session, latest_hb)
            if alive_at < cutoff:
                runner_gone = True
                now = datetime.now(timezone.utc)
                quiet = int((now - alive_at).total_seconds())
                age = int((now - _aware(latest_hb.started_at)).total_seconds())
                reason = (
                    f"{_RECONCILER_TAG} heartbeat {latest_hb.id[:8]} started "
                    f"{age}s ago, silent for {quiet}s (stale_cutoff="
                    f"{stale_ticket_seconds}s); runner crashed mid-execution"
                )
        elif latest_hb is None and _aware(tk.updated_at) < cutoff:
            runner_gone = True
            age = int((datetime.now(timezone.utc) - _aware(tk.updated_at)).total_seconds())
            reason = (
                f"{_RECONCILER_TAG} no heartbeat ever recorded and "
                f"ticket stuck running for {age}s "
                f"(stale_cutoff={stale_ticket_seconds}s)"
            )

        if runner_gone:
            tk.status = "failed"
            tk.error_message = reason
            if not tk.summary:
                tk.summary = "reconciler: runner gone"
            swept.append(tk)

    if swept:
        await session.commit()
        # A crashed runner (SSH drop / process death) can leave the remote
        # trainer alive, holding GPUs indefinitely on a cluster/instance
        # allocation — cloud self-heals via auto-release on close, but Slurm
        # steps keep squatting (the "predict.py held two B200s for 90 min"
        # failure). Reap the orphaned remote work for each swept ticket:
        # cancel_run_remote_jobs self-filters to train/inference and step-kills
        # zevo-<ticket> (never the allocation). Best-effort — never raise into
        # the sweep.
        try:
            from zevo.engine.run.remote_jobs import cancel_run_remote_jobs
            await cancel_run_remote_jobs(session, swept)
        except Exception as e:
            log.warning("%s remote-job reap after sweep failed: %s", _RECONCILER_TAG, e)
    return len(swept)


async def _reactivate_stuck_repairing_tickets(
    session: AsyncSession, stale_ticket_seconds: int
) -> int:
    """Re-queue the retry wakeup for repairing tickets whose reactivation was lost.

    A ticket enters 'repairing' and the runner immediately queues a source=retry
    wakeup to re-run it inside the same ticket. If the scheduler dies between the
    status flip and that wakeup landing (or the row is otherwise lost), the
    ticket sits 'repairing' forever and its run never closes -- the stuck-ticket
    sweep only reaps 'running'. Detect it exactly: a 'repairing' ticket older
    than the stale cutoff with NO queued/running wakeup in flight has a lost
    reactivation. Re-queue it (queue_wakeup coalesces, so this cannot storm)
    rather than fail it, which would waste the repair attempt.
    """
    from zevo.engine.run.wakeup import queue_wakeup

    cutoff = datetime.now(timezone.utc) - _dt.timedelta(seconds=stale_ticket_seconds)
    repairing = (await session.execute(
        select(Ticket).where(Ticket.status == "repairing")
    )).scalars().all()

    requeued = 0
    for tk in repairing:
        if _aware(tk.updated_at) >= cutoff:
            continue  # just flipped -- its retry wakeup is probably still draining
        active = (await session.execute(
            select(func.count()).select_from(AgentWakeupRequest).where(
                AgentWakeupRequest.ticket_id == tk.id,
                AgentWakeupRequest.status.in_(("queued", "running")),
            )
        )).scalar_one()
        if int(active or 0) > 0:
            continue  # a reactivation is already in flight; leave it alone
        run = await session.get(Run, tk.run_id)
        if run is None or run.status in TERMINAL_RUN_STATUSES:
            continue  # nothing to reactivate into
        log.warning(
            "%s reactivating stuck repairing ticket %s (attempt %d/%d); "
            "the original retry wakeup was lost",
            _RECONCILER_TAG, tk.id, int(tk.repair_attempts or 0), MAX_REPAIR_ATTEMPTS,
        )
        await queue_wakeup(
            session,
            agent_id=tk.agent_id,
            ticket_id=tk.id,
            source="retry",
            trigger_detail=f"repair_{int(tk.repair_attempts or 0)}",
            reason=(
                f"{_RECONCILER_TAG} reactivating stuck repairing ticket "
                f"(attempt {int(tk.repair_attempts or 0)}/{MAX_REPAIR_ATTEMPTS}); "
                "the original retry wakeup was lost"
            ),
            payload={"repair_attempt": int(tk.repair_attempts or 0)},
        )
        requeued += 1
    return requeued


async def _fail_validation_looping_tickets(session: AsyncSession) -> int:
    """Safety net: a `queued` ticket whose wakeups keep failing typed-payload
    validation must never idle-burn a GPU forever.

    The runner already escalates each such wakeup and, after
    MAX_PAYLOAD_VALIDATION_ATTEMPTS, fails the ticket terminally (see
    ``_escalate_stored_payload_validation_failure``). This is the backstop for
    when that terminal step never runs -- its final wakeup was never re-drained,
    the runner crashed before committing, or a mid-run restart lost it -- leaving
    the ticket `queued` while the alarm-clock cron re-enqueues the identical
    payload every tick. Detect it from the same evidence the runner uses: a
    `queued` ticket with at least MAX_PAYLOAD_VALIDATION_ATTEMPTS wakeups marked
    `failed` and stamped with the validation signature. Fail it and wake the run
    orchestrator so it can re-emit a corrected child.
    """
    from zevo.engine.run.failure_policy import (
        MAX_PAYLOAD_VALIDATION_ATTEMPTS,
        PAYLOAD_VALIDATION_FAILURE_SIGNATURE,
    )
    from zevo.engine.run.wakeup import queue_wakeup

    queued = (await session.execute(
        select(Ticket).where(Ticket.status == "queued")
    )).scalars().all()

    failed = 0
    for tk in queued:
        n = (await session.execute(
            select(func.count()).select_from(AgentWakeupRequest).where(
                AgentWakeupRequest.ticket_id == tk.id,
                AgentWakeupRequest.status == "failed",
                AgentWakeupRequest.reason.contains(
                    PAYLOAD_VALIDATION_FAILURE_SIGNATURE
                ),
            )
        )).scalar_one()
        if int(n or 0) < MAX_PAYLOAD_VALIDATION_ATTEMPTS:
            continue
        run = await session.get(Run, tk.run_id)
        if run is None or run.status in TERMINAL_RUN_STATUSES:
            continue  # nothing to escalate into
        reason = (
            f"{_RECONCILER_TAG} ticket {tk.id} stuck queued with {int(n)} "
            f"typed-payload validation failures "
            f"(>= {MAX_PAYLOAD_VALIDATION_ATTEMPTS}); failing it terminally so a "
            "cron re-enqueue can no longer idle-burn a GPU on an invalid payload"
        )
        log.warning("%s", reason)
        tk.status = "failed"
        tk.repair_route = "orchestrator"
        tk.error_message = reason[:2000]
        if not tk.summary:
            tk.summary = "reconciler: payload validation looped"
        await session.commit()
        # Escalate to the orchestrator so the run can recover with a corrected
        # child rather than silently losing this stage.
        if tk.agent_id != "orchestrator" and run.supervisor_ticket_id:
            await queue_wakeup(
                session,
                agent_id="orchestrator",
                ticket_id=run.supervisor_ticket_id,
                source="handoff",
                trigger_detail=f"payload_validation_failed:{tk.id}"[:128],
                reason=(
                    f"{_RECONCILER_TAG} child ticket {tk.id} failed typed-payload "
                    f"validation {int(n)}× and was failed; re-emit a corrected "
                    "ticket"
                ),
            )
        failed += 1
    return failed


async def _watchdog_halt_over_budget_runs(session: AsyncSession) -> int:
    """Enforce each run's cost / runtime / iteration caps as a safety net.

    The orchestrator is expected to stop itself when a cap is hit (it sees the
    same budget snapshot every wake), but if it misbehaves, wedges, or loops,
    nothing else bounds spend or wall-clock. This is the backstop: for every
    non-terminal run with a cap set, if the canonical budget snapshot reports it
    over cost or over time -- or it has reached its iteration budget -- halt the
    run and fail its in-flight tickets.

    Marking the run terminal makes the runner short-circuit (see run_ticket's
    terminal-run guard) and the resource-cleanup pass reap its GPUs; here we also
    directly reap remote jobs for the tickets we fail, exactly as the
    stuck-ticket sweep does, so an orphaned Slurm/cloud trainer can't keep
    burning money after the cap that was supposed to stop it.
    """
    from zevo.engine.cost.budget import snapshot_for_run

    open_runs = (await session.execute(
        select(Run).where(Run.status.in_(["planning", "running"]))
    )).scalars().all()

    halted = 0
    reaped_tickets: list[Ticket] = []
    for r in open_runs:
        cap_cost = float(r.max_cost_usd or 0.0)
        cap_time = float(r.max_runtime_hours or 0.0)
        cap_iters = int(r.iteration_budget or 0)
        if cap_cost <= 0 and cap_time <= 0 and cap_iters <= 0:
            continue  # nothing to enforce -- an unbounded run is the user's call
        if r.cancel_requested_at is not None:
            # Already stopping; halting it now would reap the box under the
            # checkpoint copy the operator asked for.
            continue

        snap = await snapshot_for_run(session, r.id)
        reasons: list[str] = []
        if cap_cost > 0 and snap.over_budget:
            reasons.append(f"cost ${snap.spent_usd:.2f} >= cap ${cap_cost:.2f}")
        if cap_time > 0 and snap.over_time_limit:
            reasons.append(
                f"runtime {snap.elapsed_runtime_hours:.2f}h >= cap {cap_time:.2f}h"
            )
        completed = int(r.iterations_completed or 0)
        if cap_iters > 0 and completed >= cap_iters:
            reasons.append(f"iterations {completed} >= budget {cap_iters}")
        if not reasons:
            continue

        r.status = "halted"
        r.halted_reason = f"{_RECONCILER_TAG} budget watchdog: " + "; ".join(reasons)
        r.finished_at = datetime.now(timezone.utc)
        halted += 1
        log.warning("%s halting run %s over budget: %s",
                    _RECONCILER_TAG, r.id[:8], "; ".join(reasons))

        # Fail whatever is still in flight so the runner + sweeps stop working it.
        tickets = (await session.execute(
            select(Ticket).where(Ticket.run_id == r.id)
        )).scalars().all()
        for tk in tickets:
            if tk.status not in TERMINAL_TICKET_STATUSES:
                tk.status = "failed"
                tk.error_message = r.halted_reason
                if not tk.summary:
                    tk.summary = "reconciler: run over budget"
                reaped_tickets.append(tk)

    if halted:
        await session.commit()
        # cancel_run_remote_jobs self-filters to train/inference and step-kills
        # zevo-<ticket> (never the shared allocation). Best-effort.
        if reaped_tickets:
            try:
                from zevo.engine.run.remote_jobs import cancel_run_remote_jobs
                await cancel_run_remote_jobs(session, reaped_tickets)
            except Exception as e:
                log.warning("%s remote-job reap after watchdog halt failed: %s",
                            _RECONCILER_TAG, e)
    return halted


async def _close_stale_terminal_heartbeats(
    session: AsyncSession, stale_heartbeat_seconds: int
) -> int:
    """Backstop orphan activations attached to an already-terminal Run.

    The API cancellation path closes these immediately. This sweep covers
    terminal decisions made elsewhere and historical rows left open by a
    runner crash. A grace period protects a supervisor that has just patched
    its Run terminal but is still writing its final result.
    """
    cutoff = datetime.now(timezone.utc) - _dt.timedelta(
        seconds=stale_heartbeat_seconds
    )
    rows = (await session.execute(
        select(HeartbeatRun, Run.status)
        .join(Ticket, Ticket.id == HeartbeatRun.ticket_id)
        .join(Run, Run.id == Ticket.run_id)
        .where(
            HeartbeatRun.finished_at.is_(None),
            Run.status.in_(TERMINAL_RUN_STATUSES),
        )
    )).all()

    closed = 0
    now = datetime.now(timezone.utc)
    for heartbeat, run_status in rows:
        if await _last_sign_of_life(session, heartbeat) >= cutoff:
            continue
        heartbeat.finished_at = now
        heartbeat.exit_code = 130 if run_status == "cancelled" else 1
        if not heartbeat.error_message:
            heartbeat.error_message = (
                f"{_RECONCILER_TAG} closed stale activation after Run became "
                f"{run_status}"
            )
        closed += 1
    if closed:
        await session.commit()
    return closed


def decide_run_status(*, has_registered_model: bool, failed_ticket_ids: list[str]) -> str:
    """What a reconciler-owned close may report.

    A registered model used to win outright, so a run that produced one closed
    as `success` no matter what had failed along the way — and the thing most
    likely to settle last is the multi-hop held-out branch, even though it
    starts alongside Validation, because it still needs the GPU. Run 28845478 lost its final test
    measurement to an expired allocation, scored its last iteration on
    validation alone, and reported a green success.

    It is not a write-off either — seven iterations ran and a model came out.
    `degraded` is exactly that: real work, incomplete.
    """
    # `success` is deliberately absent. It is a product decision made by the
    # Orchestrator with a complete Journal and final summary, never inferred
    # from the incidental fact that all worker Tickets are terminal.
    if has_registered_model:
        return "degraded"
    return "failed" if failed_ticket_ids else "halted"


async def _close_finished_runs(session: AsyncSession) -> dict[str, int]:
    """Close out runs whose tickets are all in terminal state.

    Returns counts by terminal status applied: {"success", "failed",
    "halted"}.
    """
    open_runs = (await session.execute(
        select(Run).where(Run.status.in_(["planning", "running"]))
    )).scalars().all()

    counts: dict[str, int] = {"success": 0, "degraded": 0, "failed": 0, "halted": 0}
    closed_runs: list[Run] = []
    for r in open_runs:
        if r.cancel_requested_at is not None:
            continue  # closing is the cancel-rescue task's job, after the copy
        tickets = (await session.execute(
            select(Ticket).where(Ticket.run_id == r.id)
        )).scalars().all()
        if not tickets:
            # Run with no tickets at all -- safe to halt.
            r.status = "halted"
            r.halted_reason = f"{_RECONCILER_TAG} no tickets exist for this run"
            r.finished_at = datetime.now(timezone.utc)
            counts["halted"] += 1
            closed_runs.append(r)
            continue

        statuses = {t.status for t in tickets}
        non_terminal = statuses - TERMINAL_TICKET_STATUSES
        if non_terminal:
            # Still work pending (including external scheduler wait) -- leave alone.
            continue

        # All tickets terminal — but that is not the same as the run being over.
        # This pipeline is reactive: a child finishing is what wakes the
        # supervisor to decide the next step, and between the child committing
        # and the supervisor being queued the run momentarily looks finished. If
        # that handoff is ever LOST — it was, when a failed registry-file mirror
        # left the session unusable and every statement after it raised — the run
        # sits here looking complete while the decision it was waiting on was
        # never made. Closing it then reports a run that stopped after one
        # iteration as a success nobody chose.
        #
        # Detect it exactly rather than on a timer: a child updated after the
        # supervisor last ran is a completion the supervisor has not seen. Repair
        # it by asking again (queue_wakeup coalesces, so this cannot storm), and
        # leave the run open. Once the supervisor runs, it is the newest thing in
        # the run and this stops matching.
        supervisor = next((t for t in tickets if t.agent_id == "orchestrator"), None)
        if supervisor is not None:
            sup_seen = _aware(supervisor.updated_at)
            unseen = [
                t for t in tickets
                if t.agent_id != "orchestrator" and _aware(t.updated_at) > sup_seen
            ]
            if unseen:
                newest = max(unseen, key=lambda t: _aware(t.updated_at))
                log.warning(
                    "%s run %s looks finished but %s completed after the supervisor "
                    "last ran — the handoff was lost; waking it instead of closing",
                    _RECONCILER_TAG, r.id[:8], newest.id,
                )
                from zevo.engine.run.wakeup import queue_wakeup
                await queue_wakeup(
                    session, agent_id="orchestrator", ticket_id=supervisor.id,
                    source="handoff",
                    reason=(f"reconciler: {newest.id} finished ({newest.status}) but the "
                            f"supervisor was never woken for it"),
                )
                await session.commit()
                continue

            # A successful supervisor activation cannot silently end a Run
            # after Evaluation wrote only the factual half of its Journal row.
            # Wake the same supervisor to supply the four-part narrative; ticket
            # creation and terminal Run PATCHes enforce the same boundary.
            incomplete = incomplete_journal_entries(r.history)
            if incomplete and supervisor.status in ("succeeded", "degraded", "skipped"):
                log.warning(
                    "%s run %s has incomplete Journal rows %s; waking the "
                    "supervisor instead of closing",
                    _RECONCILER_TAG, r.id[:8], incomplete,
                )
                from zevo.engine.run.wakeup import queue_wakeup
                await queue_wakeup(
                    session, agent_id="orchestrator", ticket_id=supervisor.id,
                    source="handoff",
                    reason=(
                        "reconciler: complete Journal fields action, result, "
                        f"analysis, and next for {incomplete}"
                    ),
                )
                continue

        failed_ids = sorted(t.id for t in tickets if t.status == "failed")

        # A clean run is not over merely because every worker Ticket is over.
        # Give the supervisor one explicit finalization wake. If it completes
        # that wake without PATCHing a terminal decision, close honestly as an
        # incomplete/degraded run instead of silently manufacturing success or
        # paying for an unbounded series of LLM retries.
        if supervisor is not None and not failed_ids:
            finalization = (await session.execute(
                select(AgentWakeupRequest)
                .where(
                    AgentWakeupRequest.ticket_id == supervisor.id,
                    AgentWakeupRequest.source == "reconciler",
                    AgentWakeupRequest.trigger_detail == "finalize_run",
                )
                .order_by(AgentWakeupRequest.created_at.desc())
                .limit(1)
            )).scalar_one_or_none()
            if finalization is None:
                from zevo.engine.run.wakeup import queue_wakeup
                await queue_wakeup(
                    session, agent_id="orchestrator", ticket_id=supervisor.id,
                    source="reconciler", trigger_detail="finalize_run",
                    reason=(
                        "all child Tickets are terminal; write the final Journal "
                        "and consolidated summary, then explicitly PATCH the Run "
                        "to success or a truthful non-success terminal status"
                    ),
                )
                continue
            if finalization.status in ("queued", "running"):
                continue

            has_model = await _run_produced_registered_model(session, r)
            new_status = "degraded" if has_model else "halted"
            r.status = new_status
            r.halted_reason = (
                f"{_RECONCILER_TAG} supervisor finalization wake "
                f"{finalization.id[:8]} ended as {finalization.status} without "
                "an explicit terminal Run decision"
            )
            r.finished_at = datetime.now(timezone.utc)
            counts[new_status] += 1
            closed_runs.append(r)
            continue

        # Decide degraded/failed/halted. Clean success is never inferred here.
        #
        # A registered model used to win outright, so a run that produced one
        # closed as `success` no matter what had failed along the way — and the
        # thing most likely to settle last is the multi-hop held-out branch,
        # even though it starts alongside Validation, because it still needs
        # the GPU. Run 28845478 ended with its final test measurement lost to an
        # expired allocation, its last iteration scored on validation only, and a green
        # "success" on the page.
        #
        # It is not a write-off either: seven iterations ran and a model came
        # out. `degraded` is that: real work, incomplete.
        new_status = decide_run_status(
            has_registered_model=await _run_produced_registered_model(session, r),
            failed_ticket_ids=failed_ids,
        )

        r.status = new_status
        if not r.halted_reason:
            if new_status == "degraded":
                # Name what did not finish. "degraded" on its own tells a reader
                # something is missing without saying which number to distrust.
                r.halted_reason = (
                    f"{_RECONCILER_TAG} a model was registered, but "
                    f"{len(failed_ids)} ticket(s) failed: "
                    f"{', '.join(failed_ids[:5])}"
                    + (f" (+{len(failed_ids) - 5} more)" if len(failed_ids) > 5 else "")
                )
            elif new_status == "failed":
                r.halted_reason = (
                    f"{_RECONCILER_TAG} all tickets terminal, "
                    f"failed: {', '.join(failed_ids[:5])}"
                    + (f" (+{len(failed_ids) - 5} more)" if len(failed_ids) > 5 else "")
                )
            else:
                r.halted_reason = (
                    f"{_RECONCILER_TAG} all tickets terminal but no model "
                    f"registered (supervisor never closed the loop)"
                )
        r.finished_at = datetime.now(timezone.utc)
        counts[new_status] += 1
        closed_runs.append(r)

    if any(counts.values()):
        await session.commit()
    # Auto-release rented GPUs for the runs we just closed. The reactive
    # model dropped System A's dispatcher, which used to fire a
    # release_remote ticket in its finally block -- without this, Vast.ai
    # instances leak and bill indefinitely after a run ends.
    if closed_runs:
        released = 0
        for r in closed_runs:
            released += await _release_run_instances(session, r)
            # `instance` allocations are the user's — _release_run_instances is a
            # no-op there — but a train/inference srun STEP this run started can
            # keep holding GPUs after close. Step-kill zevo-<ticket> (never the
            # allocation) so the cards free for the next run. cloud self-destroys
            # and cluster scancels its holder job, so this is only needed for
            # instance. Best-effort.
            if r.gpu_provider == "instance":
                try:
                    tks = (await session.execute(
                        select(Ticket).where(Ticket.run_id == r.id)
                    )).scalars().all()
                    from zevo.engine.run.remote_jobs import cancel_run_remote_jobs
                    await cancel_run_remote_jobs(session, tks)
                except Exception as e:
                    log.warning("[reconciler] instance step reap for run %s failed: %s", r.id[:8], e)
        if released:
            log.info("[reconciler] released %d system-owned GPU resource(s) for closed runs", released)
        # Fixed `instance` hosts are operator-owned and never destroyed, but the
        # CARDS this run held have to go back or the next run cannot be
        # placed. Separate from the loop above precisely because the release
        # there is a no-op for this provider.
        cards = await _release_gpu_leases(session, [r.id for r in closed_runs])
        if cards:
            log.info("[reconciler] released %d GPU lease(s) for closed runs", cards)
    return counts


async def _release_gpu_leases(session: AsyncSession, run_ids: list[str]) -> int:
    """Hand back every leased GPU the closed runs were holding."""
    if not run_ids:
        return 0
    try:
        res = await session.execute(
            sa_update(GpuLease)
            .where(GpuLease.run_id.in_(run_ids), GpuLease.released_at.is_(None))
            .values(
                released_at=datetime.now(timezone.utc),
                release_reason="run ended (reconciler)",
            )
        )
        await session.commit()
        return int(res.rowcount or 0)
    except Exception as e:  # cleanup must never crash the reconciler
        log.warning("[reconciler] GPU lease release failed: %s", e)
        await session.rollback()
        return 0


async def _release_run_instances(session: AsyncSession, run: Run) -> int:
    """Free system-owned GPUs marked for automatic release when a run ends.

    Reads the `device_info` work products (the infra agent writes
    device_info.json with `provider` + `instance_id`) and, per provider,
    **destroys** the Vast.ai instance (cloud). Finite cluster stage jobs are
    discovered from open InfraInstance rows and scancelled if still active.
    An explicit `auto_release=false` is preserved on ordinary
    completion; cancellation/deletion use their force-cleanup path. Failures
    are logged, never raised — cleanup must not crash the reconciler.
    """
    import json as _json
    from pathlib import Path as _Path

    wps = (await session.execute(
        select(WorkProduct)
        .join(Ticket, Ticket.id == WorkProduct.ticket_id)
        .where(Ticket.run_id == run.id, WorkProduct.role == "device_info")
    )).scalars().all()

    # Successful device contracts always name vastai|lambda. Empty remains
    # possible only when acquisition failed after bookkeeping but before a
    # valid device artifact; terminal cleanup treats it as a leak backstop.
    cloud: dict[str, str] = {}
    cluster_jobs: set[str] = set()
    manual_release: set[str] = set()
    # (ssh block, remote per-run workdir) to `rm -rf` once the run has ended —
    # artifacts were SCP'd back to the local ./runs during the run.
    remote_cleanup: list[tuple[dict, str]] = []
    for wp in wps:
        try:
            info = _json.loads(_Path(wp.path).read_text(encoding="utf-8"))
        except Exception:
            continue
        prov = str(info.get("provider", "") or "").lower()
        # Remote per-run work dir for either SSH-backed route.
        route = info.get("cluster") or info.get("instance") or {}
        wd = str(route.get("workdir", "") or "").strip()
        sshb = info.get("ssh") or {}
        if wd and sshb.get("host") and prov in ("cluster", "instance"):
            remote_cleanup.append((sshb, wd))
        iid = str(info.get("instance_id", "") or "").strip()
        if not iid:
            continue
        if prov == "instance":
            continue  # fixed operator-owned host — never power it off
        if iid in manual_release:
            continue
        if info.get("auto_release") is not True:
            manual_release.add(iid)
            cloud.pop(iid, None)
            cluster_jobs.discard(iid)
            continue
        if prov == "cluster":
            cluster_jobs.add(iid)
        elif prov == "cloud":
            cloud.setdefault(iid, str(info.get("cloud_backend", "") or "").lower())
        else:
            log.warning(
                "[reconciler] device_info for run %s names unknown provider %r "
                "(expected cluster|cloud|instance); not releasing %s",
                run.id[:8], prov, iid)

    # Also pick up instances the agent registered via POST /infra/instances but
    # that are not lifecycle handles in device_info.json — notably a finite
    # Slurm stage job recorded right after `sbatch` while still PENDING, when the run is cancelled before
    # the GPU is even allocated. Without this those jobs leak in the queue.
    infra_rows = (await session.execute(
        select(InfraInstance).where(
            InfraInstance.run_id == run.id,
            InfraInstance.instance_id != "",
            InfraInstance.released_at.is_(None),
        )
    )).scalars().all()
    for inst in infra_rows:
        prov = (inst.provider or "").lower()
        if prov == "instance":
            continue  # legacy row only; the operator manages the fixed host
        if inst.instance_id in manual_release:
            continue
        if (inst.meta or {}).get("auto_release") is not True:
            manual_release.add(inst.instance_id)
            cloud.pop(inst.instance_id, None)
            cluster_jobs.discard(inst.instance_id)
            continue
        if prov == "cluster":
            cluster_jobs.add(inst.instance_id)
        elif prov == "cloud":
            cloud.setdefault(inst.instance_id, str((inst.meta or {}).get("backend", "") or "").lower())
        else:
            log.warning(
                "[reconciler] infra row %s names unknown provider %r "
                "(expected cluster|cloud|instance); not releasing %s",
                inst.id, prov, inst.instance_id)

    released = 0

    # Destroy each cloud box on its backend, falling back to the other if the
    # hint was missing/wrong (destroy is idempotent; 'already gone' is harmless).
    async def _destroy_cloud(iid: str, hint: str) -> bool:
        order = ["lambda", "vastai"] if hint == "lambda" else ["vastai", "lambda"]
        for name in order:
            try:
                if name == "lambda":
                    from zevo.providers.lambda_labs import LambdaCloudProvider as _P
                else:
                    from zevo.providers.vastai import VastAIProvider as _P
                if await _P().destroy_instance(iid):
                    log.info("[reconciler] released %s instance %s (run %s ended)", name, iid, run.id[:8])
                    return True
            except Exception as e:
                log.warning("[reconciler] %s release of %s failed: %s", name, iid, e)
        return False

    destroyed: set[str] = set()
    for iid, hint in cloud.items():
        if await _destroy_cloud(iid, hint):
            destroyed.add(iid)
            released += 1

    if cluster_jobs:
        selected_connection = (
            await session.get(SshHost, run.ssh_host_id)
            if run.ssh_host_id else None
        )
        host = (
            selected_connection.host if selected_connection is not None
            else os.environ.get("ZEVO_CLUSTER_SSH_HOST", "").strip()
        )
        if host:
            key = (
                selected_connection.key_path if selected_connection is not None
                else resolve_ssh_key("ZEVO_CLUSTER_SSH_KEY")
            )
            password_path = (
                selected_connection.password_path
                if selected_connection is not None
                else os.environ.get("ZEVO_CLUSTER_SSH_PASSWORD_FILE", "").strip()
            )
            if password_path:
                key = ""
            port = (
                selected_connection.port if selected_connection is not None
                else os.environ.get("ZEVO_CLUSTER_SSH_PORT", "22") or "22"
            )
            user = (
                selected_connection.username if selected_connection is not None
                else os.environ.get("ZEVO_CLUSTER_SSH_USER", "root")
            )
            for jid in cluster_jobs:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *ssh_base_args(
                            key_path=key,
                            password_path=password_path,
                            port=int(port),
                        ),
                        f"{user}@{host}", f"scancel {jid}",
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    )
                    await asyncio.wait_for(proc.communicate(), timeout=30)
                    released += 1
                    log.info("[reconciler] scancelled Slurm job %s (run %s ended)", jid, run.id[:8])
                except Exception as e:
                    log.warning("[reconciler] failed to scancel job %s: %s", jid, e)

    # Delete the remote per-run work dir now that the run ended (its artifacts
    # were SCP'd back to ./runs during the run). Guard: the path MUST contain
    # this run's id, so we never rm a shared/parent dir. Shared caches live
    # outside the run dir so they survive for the next run.
    import shlex as _shlex
    for sshb, wd in remote_cleanup:
        if not wd or run.id not in wd or wd.count("/") < 2:
            continue
        try:
            key = str(sshb.get("key_path") or "").strip()
            password_path = str(sshb.get("password_path") or "").strip()
            port = str(sshb.get("port") or 22)
            user = str(sshb.get("user") or "root")
            host = str(sshb.get("host") or "")
            args = [
                *ssh_base_args(
                    key_path=key,
                    password_path=password_path,
                    port=int(port),
                ),
                f"{user}@{host}", f"rm -rf {_shlex.quote(wd)}",
            ]
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await asyncio.wait_for(proc.communicate(), timeout=30)
            log.info("[reconciler] removed remote work dir %s (run %s ended)", wd, run.id[:8])
        except Exception as e:
            log.warning("[reconciler] failed to remove remote dir %s: %s", wd, e)

    # Mark the registered rows released so the leak detector / `hardware`
    # stop showing them (idempotent: scancel of a finished job is harmless).
    # A CLOUD row whose destroy FAILED on both backends stays active on
    # purpose — the box is still billing, and marking it released is exactly
    # what hides it from the stuck-infra detector (the playbook contract says
    # the bookkeeping row outlives a failed destruction).
    if infra_rows:
        now = datetime.now(timezone.utc)
        for inst in infra_rows:
            if inst.instance_id in manual_release:
                continue
            if (
                inst.provider == "cloud"
                and inst.instance_id in cloud
                and inst.instance_id not in destroyed
            ):
                log.warning(
                    "[reconciler] cloud instance %s could not be destroyed; "
                    "leaving its row active so leak detection stays honest",
                    inst.instance_id)
                continue
            inst.status = "released"
            inst.released_at = now
            if not inst.release_reason:
                inst.release_reason = "run ended"
        await session.commit()
    return released


async def _run_produced_registered_model(
    session: AsyncSession, run: Run
) -> bool:
    """Did this run produce a registry entry?

    Source of truth #1: the run row already has a registry_version_tag.
    Source of truth #2: a work_product with role='registry_entry' attached
    to any of the run's tickets.
    """
    if run.registry_version_tag:
        return True
    n = (await session.execute(
        select(func.count(WorkProduct.id))
        .join(Ticket, Ticket.id == WorkProduct.ticket_id)
        .where(Ticket.run_id == run.id, WorkProduct.role == "registry_entry")
    )).scalar_one()
    return int(n) > 0


async def _cleanup_terminal_resources(session: AsyncSession) -> int:
    """Release resources for Runs closed explicitly through the API.

    Reconciler-owned closes already clean up in `_close_finished_runs`. A clean
    success now comes from the supervisor's PATCH while its heartbeat is still
    running, so it never appears in that function's `closed_runs` list. Select
    only terminal Runs that still own a live lease/instance; after cleanup the
    predicate becomes false, making this pass idempotent.
    """
    live_lease = select(GpuLease.id).where(
        GpuLease.run_id == Run.id, GpuLease.released_at.is_(None),
    ).exists()
    live_instance = select(InfraInstance.id).where(
        InfraInstance.run_id == Run.id, InfraInstance.released_at.is_(None),
    ).exists()
    runs = (await session.execute(
        select(Run).where(
            Run.status.in_(TERMINAL_RUN_STATUSES),
            live_lease | live_instance,
        )
    )).scalars().all()
    if not runs:
        return 0
    released = 0
    for run in runs:
        released += await _release_run_instances(session, run)
    released += await _release_gpu_leases(session, [run.id for run in runs])
    return released


async def reconcile_runs_and_tickets(
    *, stale_ticket_seconds: int
) -> dict[str, Any]:
    """Run the full reconciliation pass.

    Returns a dict suitable for log printing, e.g.
        {"tickets_failed": 2, "runs_closed": {"success": 0, "failed": 1, "halted": 1}}
    """
    Session = get_session_factory()
    report: dict[str, Any] = {
        "tickets_failed": 0,
        "repairing_reactivated": 0,
        "validation_tickets_failed": 0,
        "runs_halted_over_budget": 0,
        "cancel_rescues_started": 0,
        "heartbeats_closed": 0,
        "slurm_jobs_checked": 0,
        "slurm_tickets_resumed": 0,
        "runs_closed": {"success": 0, "degraded": 0, "failed": 0, "halted": 0},
        "resources_released": 0,
    }
    try:
        async with Session() as s:
            checked, resumed = await _reconcile_slurm_stage_jobs(s)
            report["slurm_jobs_checked"] = checked
            report["slurm_tickets_resumed"] = resumed
        async with Session() as s:
            report["tickets_failed"] = await _sweep_stuck_tickets(s, stale_ticket_seconds)
        # Recover repairing tickets whose retry wakeup was lost, so their run
        # doesn't wedge forever with pending-but-dead work.
        async with Session() as s:
            report["repairing_reactivated"] = await _reactivate_stuck_repairing_tickets(
                s, stale_ticket_seconds
            )
        # Backstop the runner's payload-validation bound: fail any `queued`
        # ticket whose wakeups keep failing typed-payload validation, so a cron
        # re-enqueue can never idle-burn a GPU on an invalid payload forever.
        async with Session() as s:
            report["validation_tickets_failed"] = (
                await _fail_validation_looping_tickets(s)
            )
        # Runs whose cancel is waiting on a checkpoint copy: start the copy
        # (once) here; the task closes the run and the cleanup pass below reaps
        # the box on a later tick.
        async with Session() as s:
            from zevo.engine.run.cancel_rescue import complete_requested_cancels
            report["cancel_rescues_started"] = await complete_requested_cancels(s)
        # Enforce cost/runtime/iteration caps BEFORE closing runs, so a run this
        # pass halts also gets its resources reaped by _cleanup_terminal_resources.
        async with Session() as s:
            report["runs_halted_over_budget"] = await _watchdog_halt_over_budget_runs(s)
        async with Session() as s:
            report["heartbeats_closed"] = await _close_stale_terminal_heartbeats(
                s, stale_ticket_seconds
            )
        async with Session() as s:
            report["runs_closed"] = await _close_finished_runs(s)
        async with Session() as s:
            report["resources_released"] = await _cleanup_terminal_resources(s)
    except SQLAlchemyError:
        log.exception("reconciler aborted (DB error); will retry next tick")
    return report
