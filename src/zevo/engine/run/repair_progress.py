"""Bound retries using execution identity and artifact bytes, never agent prose."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlalchemy import select

from zevo.db import HeartbeatResult, InfraInstance, TicketNotice, WorkProduct


def _paths(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith('_path') and isinstance(child, str) and child:
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
    request = meta.get('inference_execution_retry') or {}
    return bool(request.get('reason') and request.get('execution_attempt') == int(meta.get('execution_attempt') or 1))


async def repeated_without_progress(session, ticket, error):
    job = (await session.execute(select(InfraInstance).where(
        InfraInstance.ticket_id == ticket.id, InfraInstance.provider == 'cluster'
    ).order_by(InfraInstance.created_at.desc()).limit(1))).scalar_one_or_none()
    meta = (job.meta or {}) if job else {}
    results = (await session.execute(select(HeartbeatResult).where(
        HeartbeatResult.ticket_id == ticket.id
    ).order_by(HeartbeatResult.created_at.desc()).limit(1))).scalars().all()
    paths = set(_paths(results[0].output)) if results else set()
    products = (await session.execute(select(WorkProduct).where(
        WorkProduct.ticket_id == ticket.id))).scalars().all()
    paths.update(p.path for p in products if p.path)
    paths.update(str(Path(p).with_name('length_calibration.json')) for p in list(paths) if p.endswith(('.yaml', '.yml')))
    # Rewritten notes and log timestamps are not new validation evidence.
    paths = {p for p in paths if not p.endswith(('.log', '.out', '.err'))}
    completed = str(meta.get('scheduler_state', '')).upper() == 'COMPLETED'
    evidence = {'error': 'completed_collect' if completed else ' '.join(error.split()),
                'retry_requested': execution_retry_authorized(meta),
                'job': [job.id, meta.get('execution_attempt'), meta.get('submission_heartbeat_id')] if job else None,
                'files': {p: _identity(p) for p in sorted(paths)}}
    fingerprint = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    previous = (await session.execute(select(TicketNotice).where(
        TicketNotice.ticket_id == ticket.id, TicketNotice.code == 'repair.progress'
    ).order_by(TicketNotice.created_at.desc()).limit(1))).scalar_one_or_none()
    old = json.loads(previous.body) if previous else {}
    count = old.get('count', 0) + 1 if old.get('fingerprint') == fingerprint else 1
    session.add(TicketNotice(ticket_id=ticket.id, code='repair.progress', severity='info',
                            body=json.dumps({'fingerprint': fingerprint, 'count': count})))
    return count >= 3
