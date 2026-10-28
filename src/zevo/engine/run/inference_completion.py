"""Commit artifact-verified inference completion independently of allocation health."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import hashlib

from sqlalchemy import select

from zevo.db import ExecutionEvent, InfraInstance, WorkloadExecution
from zevo.engine.run.workload_execution import register_execution


def file_digest(path: str) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


async def record_completion(session, *, ticket, heartbeat, inp, result, primary_name: str):
    """Call only after all result/config/prediction validators accept the result.

    Never change scheduler state or mark an allocation reusable. If an exact
    GPU execution cannot be established, retain completion on a separate result
    identity rather than assigning it to an arbitrary retry.
    """
    from zevo.engine.run.runner import _slurm_stage_implementation_sha256
    script = str(result.slurm_script_path or '')
    row = None
    bookkeeping_id = getattr(inp.slurm_job, 'bookkeeping_row_id', '')
    if bookkeeping_id and script and Path(script).is_file():
        script_hash = file_digest(script)
        implementation_hash = _slurm_stage_implementation_sha256(
            SimpleNamespace(stage='inference', script_path=script))
        rows = (await session.execute(select(InfraInstance).where(
            InfraInstance.ticket_id == ticket.id, InfraInstance.id == bookkeeping_id,
        ))).scalars().all()
        matches = [item for item in rows if (item.meta or {}).get('script_sha256') == script_hash
                   and (item.meta or {}).get('implementation_sha256') == implementation_hash]
        if len(matches) == 1:
            row = matches[0]
    execution = await session.get(WorkloadExecution, row.id) if row else None
    if execution is None:
        execution = await register_execution(session,
            execution_id=f'verified-{heartbeat.id}', ticket_id=ticket.id,
            heartbeat_id=heartbeat.id, runtime_key='verified_result', log_path=result.log_path or '')
    now = datetime.now(timezone.utc)
    members = [(inp.benchmark_id or ('test:0' if ticket.lane == 'held_out_test' else 'validation:0'), primary_name, result)]
    assigned = {member.name: member for member in inp.suite_members}
    members.extend((assigned[item.name].benchmark_id, item.name, item) for item in result.suite_members)
    manifests = []
    for index, (identity, name, item) in enumerate(members):
        manifests.append({'benchmark_id': identity, 'name': name, 'rows': item.n_rows,
            'predictions_sha256': file_digest(item.predictions_path),
            'diagnostics_sha256': file_digest(item.generation_diagnostics_path),
            'config_sha256': file_digest(item.inference_config_path)})
        phase = f'verified_inference:{identity}'
        existing = (await session.execute(select(ExecutionEvent).where(
            ExecutionEvent.ticket_id == ticket.id, ExecutionEvent.attempt_id == execution.id,
            ExecutionEvent.phase == phase, ExecutionEvent.event_type == 'progress',
        ))).scalars().first()
        if existing is not None:
            continue
        session.add(ExecutionEvent(ticket_id=ticket.id, heartbeat_id=execution.heartbeat_id,
            attempt_id=execution.id, event_type='progress', phase=phase,
            current_step=item.n_rows, total_steps=item.n_rows, loss=-1.0, ts=now,
            extras={'execution_id': execution.id, 'execution_purpose': 'workload',
                    'benchmark_id': identity, 'benchmark_name': name,
                    'benchmark_index': index + 1, 'benchmark_total': len(members),
                    'benchmark_complete': True, 'artifact_verified': True,
                    'result_heartbeat_id': heartbeat.id}))
    if row:
        row.meta = {**(row.meta or {}), 'verified_workload_result': {
            'heartbeat_id': heartbeat.id, 'verified_at': now.isoformat(),
            'script_sha256': script_hash, 'implementation_sha256': implementation_hash,
            'members': manifests,
        }}
    if row and inp.slurm_job.enabled:
        heartbeat.resolved_config = {**(heartbeat.resolved_config or {}),
                                     'execution_phase': 'collect'}
