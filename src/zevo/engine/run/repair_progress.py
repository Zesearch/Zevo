"""Bound retries using execution identity and artifact bytes, never agent prose."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlalchemy import or_, select

from zevo.db import ExecutionEvent, HeartbeatResult, InfraInstance, TicketNotice, WorkProduct, WorkloadExecution
from zevo.engine.run.workload_execution import classify_event


def _diagnostic_role(name):
    # These are mutable summaries, not executable inputs or validated outputs.
    # Keep generation_diagnostics and length_calibration: they are required
    # inference artifacts whose correction can repair a rejected result.
    role = name.removesuffix('_path')
    return role in {
        'log', 'stdout', 'stderr', 'adaptive_training_report', 'training_diagnostics',
        'diagnostic_report', 'repair_report', 'probe_report',
    } or role.endswith(('_log', '_repair_report', '_probe_report'))


def _paths(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if _diagnostic_role(key) or key in {
                'notes', 'error_message', 'failure_recovery', 'memory_updates',
            }:
                continue
            is_artifact = key.endswith('_path') or key in {
                'evaluation_script', 'scoring_set', 'sample_submission',
            }
            if is_artifact and isinstance(child, str) and child:
                yield child
            else:
                yield from _paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from _paths(child)


def _identity(path):
    p = Path(path)
    try:
        if not p.is_file():
            return 'missing'
        if p.stat().st_size > 8 * 1024 * 1024:
            st = p.stat()
            return [st.st_size, st.st_mtime_ns]
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return 'unreadable'


def execution_retry_authorized(meta):
    request = meta.get('execution_repair') or meta.get('inference_execution_retry') or {}
    return bool(request.get('action', 'resubmit') == 'resubmit' and request.get('reason')
                and request.get('execution_attempt') == int(meta.get('execution_attempt') or 1))


def collect_repair_authorized(meta):
    request = meta.get('execution_repair') or {}
    return bool(request.get('action') == 'collect' and request.get('reason')
                and request.get('execution_attempt') == int(meta.get('execution_attempt') or 1))


def record_execution_repair(row, *, action, reason, heartbeat_id):
    """Bind permission to this execution only; a later attempt needs a new decision."""
    meta = dict(row.meta or {})
    meta.pop('inference_execution_retry', None)
    row.meta = {**meta, 'execution_repair': {
        'action': action, 'reason': reason.strip(), 'heartbeat_id': heartbeat_id,
        'execution_attempt': int(meta.get('execution_attempt') or 1),
    }}


def verify_repair_training_config(previous, path):
    if previous is None or not execution_retry_authorized(previous.meta or {}):
        return
    expected = (previous.meta or {}).get('training_config_sha256')
    if not expected or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError('execution repair requires the exact registered training configuration')


async def _measured_progress(session, ticket):
    """Only backend-bound workload steps count; timestamps/probe chatter do not."""
    executions = (await session.scalars(select(WorkloadExecution).where(
        WorkloadExecution.ticket_id == ticket.id))).all()
    if not executions:
        return {}
    registrations = {row.id: row for row in executions}
    events = (await session.scalars(select(ExecutionEvent).where(
        ExecutionEvent.ticket_id == ticket.id,
        ExecutionEvent.event_type == 'progress',
        ExecutionEvent.heartbeat_id.in_({row.heartbeat_id for row in executions}),
    ))).all()
    progress = {}
    for event in events:
        marker = classify_event(event, registrations)
        if marker.get('execution_purpose') != 'workload' or event.current_step <= 0:
            continue
        key = json.dumps([event.phase, marker.get('benchmark_id', '')])
        progress[key] = max(progress.get(key, 0), event.current_step)
    return progress


async def repeated_without_progress(session, ticket, error, *, result_output=None):
    job = (await session.execute(select(InfraInstance).where(
        InfraInstance.ticket_id == ticket.id,
        # A failed cloud allocation placeholder is not a new execution.
        or_(InfraInstance.provider == 'cluster', InfraInstance.instance_id != ''),
    ).order_by(InfraInstance.created_at.desc()).limit(1))).scalar_one_or_none()
    meta = (job.meta or {}) if job else {}
    # The runner has not persisted the current HeartbeatResult yet. Inspect it
    # directly rather than judging this attempt using the preceding result.
    if result_output is None:
        results = (await session.execute(select(HeartbeatResult).where(
            HeartbeatResult.ticket_id == ticket.id
        ).order_by(HeartbeatResult.created_at.desc()).limit(1))).scalars().all()
        result_output = results[0].output if results else {}
    paths = set(_paths(result_output))
    # Any agent may fail before publishing a WorkProduct. Bound input artifacts
    # are still evidence, including infrastructure scripts and registry inputs.
    paths.update(_paths(ticket.payload or {}))
    products = (await session.execute(select(WorkProduct).where(
        WorkProduct.ticket_id == ticket.id))).scalars().all()
    paths.update(p.path for p in products if p.path and not _diagnostic_role(p.role))
    paths.update(str(Path(p).with_name('length_calibration.json')) for p in list(paths) if p.endswith(('.yaml', '.yml')))
    # Rewritten notes and log timestamps are not new validation evidence.
    paths = {p for p in paths if not p.endswith(('.log', '.out', '.err'))}
    # Train/Inference retain their legacy completed-collection grouping. Data
    # and Evaluation distinguish different errors even on the same completed job.
    completed = (ticket.agent_id in {'train', 'inference'}
                 and str(meta.get('scheduler_state', '')).upper() == 'COMPLETED')
    # Structured recovery descriptions are prose, not fresh execution evidence.
    # Legacy non-collection errors retain their old grouping for compatibility.
    error_group = ('structured_failure' if result_output.get('failure_recovery')
                   else 'completed_collect' if completed else ' '.join(error.split()))
    evidence = {'error': error_group,
                'retry_requested': execution_retry_authorized(meta),
                'job': [job.id, job.instance_id, meta.get('execution_attempt'), meta.get('submission_heartbeat_id')] if job else None,
                'files': {p: _identity(p) for p in sorted(paths)},
                'progress': await _measured_progress(session, ticket)}
    fingerprint = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    previous = (await session.execute(select(TicketNotice).where(
        TicketNotice.ticket_id == ticket.id, TicketNotice.code == 'repair.progress'
    ).order_by(TicketNotice.created_at.desc()).limit(1))).scalar_one_or_none()
    # A reusable ticket may have completed successfully since its last repair.
    # Those old failures are no longer a consecutive streak. Deferred/waiting
    # activations neither increment nor clear the streak.
    if previous is not None:
        recovered = await session.scalar(select(HeartbeatResult.id).where(
            HeartbeatResult.ticket_id == ticket.id,
            HeartbeatResult.status.in_(['succeeded', 'degraded']),
            HeartbeatResult.created_at > previous.created_at,
        ).limit(1))
        if recovered is not None:
            previous = None
    old = json.loads(previous.body) if previous else {}
    count = old.get('count', 0) + 1 if old.get('fingerprint') == fingerprint else 1
    session.add(TicketNotice(ticket_id=ticket.id, code='repair.progress', severity='info',
                            body=json.dumps({'fingerprint': fingerprint, 'count': count,
                                             'progress': evidence['progress']})))
    from zevo.engine.run.failure_policy import MAX_UNCHANGED_FAILURES
    return count >= MAX_UNCHANGED_FAILURES
