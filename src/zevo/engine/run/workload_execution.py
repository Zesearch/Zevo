"""Execution identity assigned by backend collectors, never by transcript text."""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from zevo.db import WorkloadExecution
from zevo.engine.run.benchmark_telemetry import is_preflight_progress


async def register_execution(
    db: AsyncSession, *, execution_id: str, ticket_id: str,
    heartbeat_id: str, runtime_key: str, log_path: str,
) -> WorkloadExecution:
    row = await db.get(WorkloadExecution, execution_id)
    identity = (ticket_id, heartbeat_id, runtime_key, log_path)
    if row is None:
        row = WorkloadExecution(id=execution_id, ticket_id=ticket_id,
                                heartbeat_id=heartbeat_id, runtime_key=runtime_key,
                                log_path=log_path, purpose="workload")
        db.add(row)
        await db.flush()
    elif (row.ticket_id, row.heartbeat_id, row.runtime_key, row.log_path) != identity:
        raise ValueError("workload execution identity does not match its registration")
    return row


def bind_progress(payload: dict, execution: WorkloadExecution | None) -> dict:
    """Overwrite identity claims using the collector's registered execution."""
    clean = {k: v for k, v in payload.items()
             if k not in {"telemetry_source", "execution_id", "execution_purpose"}}
    if execution is not None:
        clean.update(execution_id=execution.id, execution_purpose=execution.purpose)
    return clean


def classify_event(event, executions: dict[str, WorkloadExecution]) -> dict:
    """Keep historical records visible without inferring completion from unknown origins."""
    marker = dict(event.extras or {})
    identity = str(marker.get("execution_id") or "")
    execution = executions.get(identity)
    if is_preflight_progress(marker):
        marker["execution_purpose"] = "diagnostic"
    elif (execution is not None and execution.ticket_id == event.ticket_id
          and execution.heartbeat_id == event.heartbeat_id):
        marker["execution_purpose"] = execution.purpose
    else:
        marker["execution_purpose"] = "unverified"
    return marker
